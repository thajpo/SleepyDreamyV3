import pytest

from scripts.evaluate_cartpole_checkpoints import summarize_returns


def test_summarize_returns_preserves_range_and_solved_fraction():
    summary = summarize_returns([100.0, 500.0, 500.0, 300.0])

    assert summary == {
        "episodes": 4,
        "mean_return": 350.0,
        "median_return": 400.0,
        "min_return": 100.0,
        "max_return": 500.0,
        "solved_fraction": 0.5,
    }


def test_summarize_returns_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        summarize_returns([])
