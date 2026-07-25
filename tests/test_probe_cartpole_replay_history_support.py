import pytest

from scripts.probe_cartpole_replay_history_support import (
    csv_fieldnames,
    select_actionable_rows,
    summarize_support_rows,
    validate_evidence_pair,
)


def _row(index: int, preference: int) -> dict:
    return {
        "sample_index": index // 2,
        "start_collector_id": 0,
        "start_episode_id": index // 4,
        "start_offset": 0,
        "real_policy_pref": preference,
    }


def test_actionable_selection_is_deterministic_and_excludes_ties():
    rows = [_row(index, -1 if index % 3 == 0 else index % 2) for index in range(20)]

    first = select_actionable_rows(rows, cap=5, seed=17)
    second = select_actionable_rows(rows, cap=5, seed=17)

    assert first == second
    assert len(first) == 5
    assert all(row["real_policy_pref"] >= 0 for row in first)


def test_support_summary_preserves_occurrence_and_unique_start_counts():
    rows = [_row(0, -1), _row(1, 1), _row(2, 0), _row(3, 1)]
    selected = select_actionable_rows(rows, cap=10, seed=17)

    summary = summarize_support_rows(rows, selected)

    assert summary["sampled_rows"] == 4
    assert summary["sampled_actionable_rows"] == 3
    assert summary["sampled_actionable_fraction"] == 0.75
    assert summary["sampled_real_policy_pref_hist"] == {"-1": 1, "1": 2, "0": 1}
    assert summary["sampled_sequence_starts"] == 2
    assert summary["unique_replay_sequence_starts"] == 1


def test_csv_schema_includes_fields_added_only_to_selected_rows():
    rows = [{"state": 1}, {"state": 2, "policy_q": 3}]

    assert csv_fieldnames(rows) == ["state", "policy_q"]


def test_cross_step_evidence_requires_explicit_opt_in():
    validate_evidence_pair(3_000, 3_000, allow_cross_step=False)
    validate_evidence_pair(3_000, 3_500, allow_cross_step=True)

    with pytest.raises(ValueError, match="--allow-cross-step"):
        validate_evidence_pair(3_000, 3_500, allow_cross_step=False)
