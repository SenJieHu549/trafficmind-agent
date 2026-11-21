"""Unknown measurements must survive SQLite and authoritative projection."""
from copy import deepcopy
import sqlite3

import pytest

from backend.tools import db_tools
from backend.tools.event_identity import compact_event_context, hydrate_authoritative_event


METRICS = ("avgSpeed", "queueLength", "duration")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "measurements.db"
    monkeypatch.setattr(db_tools, "DB_PATH", str(path))
    return path


def result_for(event_id, values):
    return {"eventId": event_id, "standardEvent": {
        "eventId": event_id, "eventType": "accident", "eventTypeCn": "accident",
        "roadName": "isolated road", **values}, "riskScore": 45, "riskLevel": "medium"}


@pytest.mark.parametrize("value", [None, "", "unknown", "invalid", True, False, float("inf")])
def test_unknown_or_invalid_measurement_stays_null_through_readback(isolated_db, value):
    assert db_tools.save_event_analysis(result_for("unknown", dict.fromkeys(METRICS, value)))
    record = db_tools.get_event_by_id("unknown")
    context = compact_event_context(hydrate_authoritative_event("unknown"))
    for field in METRICS:
        assert record[field] is None
        assert context[field] is None


@pytest.mark.parametrize("value,expected", [(0, 0.0), ("0", 0.0), (12.5, 12.5), (" 12.5 ", 12.5)])
def test_known_values_including_zero_are_not_unknown(isolated_db, value, expected):
    assert db_tools.save_event_analysis(result_for("known", dict.fromkeys(METRICS, value)))
    context = compact_event_context(hydrate_authoritative_event("known"))
    assert all(context[field] == expected and context[field] is not None for field in METRICS)


def test_missing_fields_and_known_to_unknown_upsert(isolated_db):
    assert db_tools.save_event_analysis(result_for("change", dict.fromkeys(METRICS, 10)))
    assert db_tools.save_event_analysis(result_for("change", {}))
    assert all(db_tools.get_event_by_id("change")[field] is None for field in METRICS)
    assert all(compact_event_context(hydrate_authoritative_event("change"))[field] is None for field in METRICS)


def test_legacy_default_zero_schema_accepts_explicit_null_without_rewriting_history(isolated_db, monkeypatch):
    old_columns = [(name, "REAL DEFAULT 0" if name in METRICS else ddl)
                   for name, ddl in db_tools._MIGRATION_COLUMNS]
    with monkeypatch.context() as legacy:
        legacy.setattr(db_tools, "_MIGRATION_COLUMNS", old_columns)
        db_tools.init_db()
    assert db_tools.save_event_analysis(result_for("historic-zero", dict.fromkeys(METRICS, 0)))
    assert db_tools.save_event_analysis(result_for("new-unknown", {}))
    db_tools.init_db()
    with sqlite3.connect(isolated_db) as conn:
        rows = dict((row[0], row[1:]) for row in conn.execute(
            "SELECT eventId, avgSpeed, queueLength, duration FROM event_records"))
    assert rows["historic-zero"] == (0.0, 0.0, 0.0)
    assert rows["new-unknown"] == (None, None, None)


def test_new_schema_measurement_defaults_are_null(isolated_db):
    db_tools.init_db()
    with sqlite3.connect(isolated_db) as conn:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(event_records)")}
    assert all(columns[field][4] in (None, "NULL") for field in METRICS)


def test_authoritative_null_does_not_fall_back_to_stale_raw_measurement():
    snapshot = {"eventId": "current", "avgSpeed": None, "queueLength": None, "duration": None,
                "rawEvent": dict.fromkeys(METRICS, 25),
                "fullResult": {"standardEvent": dict.fromkeys(METRICS, 30)}}
    original = deepcopy(snapshot)
    assert all(compact_event_context(snapshot)[field] is None for field in METRICS)
    assert snapshot == original
