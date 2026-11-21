"""Paired denominators and score coverage for the small-set experiment."""
import pytest
from backend.evaluation.v2.ablation_report import compare, metrics


def row(passed=True, quality=4, fallback=False):
    return {"taskPassed":passed, "quality":quality, "degraded":fallback,
            "failureReason":"unknown_evidence_reference" if fallback else None,
            "judgeStatus":"scored", "toolCalls":0, "llmCalls":1}


def test_fallback_stays_in_task_denominator_but_has_no_fake_zero_quality():
    r = metrics([row(),row(False,None,True)])
    assert r["taskPassRate"] == .5 and r["qualityScoredN"] == 1
    assert r["qualityMeanConditional"] == 4 and r["invalidReferenceCount"] == 1


def test_quality_deltas_use_same_cases_and_show_regressions():
    before={"a":row(),"b":row(False,None,True)}
    after={"a":row(False,3),"b":row(True,5)}
    r=compare(before,after,["a","b"])
    assert r["taskPassDeltaPercentagePoints"] == 0
    assert r["pairedQualityN"] == 1 and r["pairedQualityMeanDelta"] == -1
    assert r["taskRegressions"] == ["a"] and r["taskRecoveries"] == ["b"]


def test_missing_case_cannot_silently_shrink_the_denominator():
    with pytest.raises(ValueError,match="unpaired_task_denominators"):
        compare({"a":row()},{"a":row()},["a","b"])
