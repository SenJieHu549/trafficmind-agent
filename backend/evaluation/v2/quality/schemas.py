"""Strict public Judge contract. Invalid output is an error, never zero stars."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claimId: str
    quote: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    evidenceIds: list[str] = Field(default_factory=list)


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: StrictInt | None = Field(ge=1, le=5)
    reason: str = Field(min_length=1)
    issues: list[Issue] = Field(default_factory=list)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unableToJudge: StrictBool
    unableReason: str | None
    reasoning: DimensionScore
    evidence: DimensionScore
    feasibility: DimensionScore

    @model_validator(mode="after")
    def consistent_availability(self):
        values = [self.reasoning.score, self.evidence.score, self.feasibility.score]
        if self.unableToJudge:
            if any(v is not None for v in values) or not self.unableReason:
                raise ValueError("unable verdict requires null scores and a reason")
        elif any(v is None for v in values) or self.unableReason is not None:
            raise ValueError("available verdict requires all scores and null unableReason")
        return self


def validate_locations(verdict, payload):
    claims = {c["claimId"]: c["text"] for c in payload["claims"]}
    evidence = {e["evidenceId"] for e in payload["evidence"]}
    for dimension in (verdict.reasoning, verdict.evidence, verdict.feasibility):
        for issue in dimension.issues:
            if issue.claimId not in claims or issue.quote not in claims[issue.claimId]:
                raise ValueError("judge_issue_does_not_locate_candidate_text")
            if set(issue.evidenceIds) - evidence:
                raise ValueError("judge_issue_uses_unknown_evidence")
    return verdict
