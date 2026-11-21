"""Human-confirmed natural-language intake; no Agent/model/action side effects."""
from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from backend.agent.collaboration.event_parser import parse_content_to_event
from backend.event_ingestion import EventIngestionError, ingest_event
from backend.regional.repository import SQLiteRegionalRepository
from backend.regional.resolver import EventLocationBindingService

router = APIRouter(prefix="/events/intake", tags=["Confirmed Event Intake"])
EventType = Literal["accident", "congestion", "illegal_parking", "wrong_way",
                    "pedestrian_intrusion", "signal_fault", "vehicle_stopped", "construction_block"]


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=3000)

    @field_validator("description")
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip():
            raise ValueError("请先描述交通事件")
        return value.strip()


class ConfirmedFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eventType: EventType
    vehicleCount: StrictInt | None = Field(default=None, ge=0, le=100000)
    injuredCount: StrictInt | None = Field(default=None, ge=0, le=100000)
    fatalityCount: StrictInt | None = Field(default=None, ge=0, le=100000)
    avgSpeed: float | None = Field(default=None, ge=0, le=500, allow_inf_nan=False)
    queueLength: float | None = Field(default=None, ge=0, le=1000000, allow_inf_nan=False)
    duration: float | None = Field(default=None, ge=0, le=31536000, allow_inf_nan=False)
    isMainRoad: StrictBool = False
    nearbySchool: StrictBool = False
    nearbyHospital: StrictBool = False
    weather: Literal["unknown", "clear", "rain", "snow", "fog"] = "unknown"
    timePeriod: Literal["unknown", "off_peak", "morning_peak", "evening_peak"] = "unknown"

    @field_validator("avgSpeed", "queueLength", "duration", mode="before")
    @classmethod
    def numeric_observation(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("观测值须为数字，未提供请填 null")
        return value


class ConfirmRequest(PreviewRequest):
    requestId: str = Field(min_length=8, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    confirmed: StrictBool
    regionId: str = Field(min_length=1, max_length=100)
    roadId: str = Field(min_length=1, max_length=100)
    facts: ConfirmedFacts
    dataReality: Literal["user_reported", "synthetic"] = "user_reported"

    @field_validator("confirmed")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("请先核对并确认登记信息")
        return value


def _casualty_counts(description: str) -> tuple[int | None, int | None]:
    """Extract explicit reports conservatively; absence of information is unknown.

    Conflicting reports or an uncertain clause require human confirmation.
    This is bounded rule extraction, not a general Chinese language parser.
    """
    values: list[set[int]] = [set(), set()]
    number = r"\d+|[零〇一二两三四五六七八九十]+"
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))

    def count(text):
        if text.isdecimal():
            return int(text)
        if text in digits:
            return digits[text]
        if "十" in text:
            left, right = text.split("十", 1)
            if (not left or left in digits) and (not right or right in digits):
                return digits.get(left, 1) * 10 + digits.get(right, 0)
        return None

    for clause in re.split(r"[，,。；;\n]", description):
        if re.search(r"未知|不详|不明|待核实|待确认|未确认|未核实|未证实|未发现|未见|未报告|不确定|并非|不是|(?:不能|无法|尚未|暂未).*?确认|(?:暂无|未收到).*?(?:信息|消息|报告|通报)", clause):
            continue
        if any(word in clause for word in ("无人伤亡", "无人员伤亡", "没有人员伤亡")):
            values[0].add(0)
            values[1].add(0)
        for index, words in enumerate(("受伤|轻伤|重伤", "死亡|遇难")):
            patterns = [rf"({number})\s*(?:人|名)(?:{words})",
                        rf"(?:{words})(?:人数)?\s*(?:为|有|共|约)?\s*({number})\s*(?:人|名)?"]
            for pattern in patterns:
                for match in re.finditer(pattern, clause):
                    value = count(match.group(1))
                    if value is not None:
                        values[index].add(value)
    return tuple(next(iter(items)) if len(items) == 1 else None for items in values)


def preview_description(description: str) -> dict:
    parsed = parse_content_to_event(description)
    kind = ""
    if parsed.get("hasAccident"):
        kind = "accident"
    elif any(w in description for w in ("拥堵", "堵车", "排队", "缓行", "堵了")):
        kind = "congestion"
    elif any(w in description for w in ("信号灯故障", "信号灯异常", "红绿灯故障", "灯控故障")):
        kind = "signal_fault"
    else:
        for words, candidate in [(("违停", "违法停车"), "illegal_parking"),
                                  (("逆行",), "wrong_way"),
                                  (("行人闯入",), "pedestrian_intrusion"),
                                  (("车辆停驶", "车辆滞留"), "vehicle_stopped"),
                                  (("施工占道",), "construction_block")]:
            if any(word in description for word in words):
                kind = candidate
                break
    injured, fatality = _casualty_counts(description)
    road = parsed.get("roadName", "")
    if road in ("未命名路段", "未命名主干道", "未知路段"):
        road = ""
    def measurement(pattern):
        match = re.search(pattern, description, re.I)
        return float(match.group(1)) if match else None

    speed = measurement(r"(?:平均车速|车速|时速|速度)\s*(?:约|为|是)?\s*(\d+(?:\.\d+)?)")
    if speed is None:
        speed = measurement(r"(\d+(?:\.\d+)?)\s*km/?h")
    queue = measurement(r"排队(?:长度)?\s*(?:约|为|是)?\s*(\d+(?:\.\d+)?)\s*(?:米|m)")
    duration_match = re.search(r"持续(?:了)?\s*(?:约|为)?\s*(\d+(?:\.\d+)?)\s*(秒|分钟|分|小时)", description)
    duration = (float(duration_match.group(1)) * {"秒": 1, "分钟": 60, "分": 60, "小时": 3600}[duration_match.group(2)]
                if duration_match else None)
    facts = {
        "eventType": kind, "vehicleCount": parsed.get("vehicleCount"),
        "injuredCount": injured, "fatalityCount": fatality,
        "avgSpeed": speed, "queueLength": queue, "duration": duration,
        "isMainRoad": parsed["isMainRoad"],
        "nearbySchool": parsed["nearbySchool"], "nearbyHospital": parsed["nearbyHospital"],
        "weather": parsed["weather"] if any(w in description for w in ("晴", "雨", "雪", "雾")) else "unknown",
        "timePeriod": parsed["timePeriod"] if any(w in description for w in ("早高峰", "晚高峰", "平峰", "非高峰")) else "unknown",
    }
    return {"description": description, "facts": facts, "suggestedRoadName": road,
            "unknownFields": [key for key, value in facts.items() if value is None],
            "warnings": ["这是规则提取的待确认草稿，尚未入库。请核对事件类型并选择具体区域和道路。",
                         "留空表示未知；填写 0 表示明确报告为零。登记确认不等于现场已独立核验。"]}


@router.post("/preview")
def preview(body: PreviewRequest):
    return preview_description(body.description)


@router.post("/confirm")
def confirm(body: ConfirmRequest):
    repo = SQLiteRegionalRepository()
    region = repo.get_region(body.regionId)
    road = repo.get_road(body.roadId)
    if (not region or region.get("status") != "active" or not road
            or road.get("regionId") != body.regionId or road.get("status") != "active"):
        raise HTTPException(422, detail={"message": "请选择同一区域内有效的规范道路"})
    # A selected road must still resolve uniquely. Do this before any event write.
    from backend.regional.resolver import EventLocationResolver
    resolution = EventLocationResolver(repo).resolve({"roadName": road["name"]}, region_id=body.regionId)
    if resolution.get("status") != "resolved" or resolution.get("roadId") != body.roadId:
        raise HTTPException(409, detail={"message": "道路名称存在歧义，请先完善区域道路资料"})
    reported = {"description": body.description, "source": "user_confirmed_report",
                "independentlyVerified": False, "injuredCount": body.facts.injuredCount,
                "fatalityCount": body.facts.fatalityCount}
    event = body.facts.model_dump(exclude={"injuredCount", "fatalityCount"})
    event.update(roadName=road["name"], regionId=body.regionId,
                 confidence=None, reportedFacts=reported)
    # This is an operator-confirmed report, not a detector confidence estimate.
    metadata = {"sourceType": "manual_confirmed_report", "dataReality": body.dataReality,
                "description": body.description, "reportedFacts": reported,
                "canonicalRegionId": body.regionId, "canonicalRoadId": body.roadId,
                "confirmationMeaning": "人工确认登记内容，不代表独立现场核验或动作批准"}
    parsed_facts = preview_description(body.description)["facts"]
    confirmed_facts = body.facts.model_dump()
    metadata["intakeAudit"] = {
        "parserVersion": "rules-20261111-2", "parsedFacts": parsed_facts,
        "confirmedFacts": confirmed_facts,
        "corrections": {key: {"parsed": parsed_facts.get(key), "confirmed": value}
                        for key, value in confirmed_facts.items() if parsed_facts.get(key) != value}}
    try:
        result = ingest_event(source="manual_confirmed_report", source_event_id=body.requestId,
                              event=event, source_metadata=metadata,
                              allow_unknown_measurements=True, create_only=True)
    except EventIngestionError as exc:
        raise HTTPException(409 if exc.code == "ingestion_conflict" else 422,
                            detail={"code": exc.code, "message": exc.message}) from exc
    # Retry retains the same event even if a previous binding attempt failed.
    binding = EventLocationBindingService(repo).resolve_and_bind(result["eventId"], region_id=body.regionId)
    return {**result, "locationBinding": binding.get("binding"),
            "analysisStarted": False,
            "next": {"analysisUrl": f"/?view=multi&eventId={result['eventId']}",
                     "eventUrl": f"/?view=simulation&eventId={result['eventId']}"}}
