from scripts.probe_cartpole_final_policy_trace import (
    select_final_only_keys,
    summarize_actor_cross,
)


def _labels(preferences):
    return {
        (0, timestep): {"real_policy_pref": str(preference)}
        for timestep, preference in enumerate(preferences)
    }


def test_final_only_selection_is_fixed_and_excludes_shared_actionable_rows():
    solved = _labels([-1, -1, 0, -1, 1, -1])
    final = _labels([0, 1, 0, -1, 1, 1])

    first = select_final_only_keys(solved, final, cap=2, seed=23)
    second = select_final_only_keys(solved, final, cap=2, seed=23)

    assert first == second
    assert set(first).issubset({(0, 0), (0, 1), (0, 5)})


def test_actor_cross_summary_identifies_representation_transfer():
    rows = [
        {
            "sample_index": 0,
            "t": 4,
            "first_action": 0,
            "next_depth": 1,
            "solved_solved": 1,
            "final_solved": 1,
            "solved_final": 0,
            "final_final": 0,
        },
        {
            "sample_index": 0,
            "t": 4,
            "first_action": 0,
            "next_depth": 2,
            "solved_solved": 1,
            "final_solved": 0,
            "solved_final": 0,
            "final_final": 0,
        },
    ]

    summary = summarize_actor_cross(rows)

    assert summary["branches_with_divergence"] == 1
    assert summary["first_divergence_depth_histogram"] == {"1": 1}
    assert summary["changed_rows_representation_swap_transfers_final"] == 1.0
    assert summary["changed_rows_actor_swap_transfers_final"] == 0.5
