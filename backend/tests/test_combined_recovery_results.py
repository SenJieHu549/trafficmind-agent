import pytest

from backend.evaluation.v2.combined_recovery_results import merge_counts


def test_combined_rate_uses_counts_instead_of_average_of_rates():
    result = merge_counts({"trials": 120, "recovered": 114}, {"trials": 18, "recovered": 18})
    assert result == {"trials": 138, "recovered": 132, "recoveryRate": 132/138}
    assert result["recoveryRate"] != (.95 + 1)/2


def test_empty_denominator_is_not_reported_as_perfect_recovery():
    assert merge_counts({"trials": 0, "recovered": 0})["recoveryRate"] is None


def test_impossible_recovery_counts_are_rejected():
    with pytest.raises(ValueError):
        merge_counts({"trials": 18, "recovered": 19})
