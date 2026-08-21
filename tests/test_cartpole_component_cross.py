import csv

import numpy as np
import pytest

from scripts.probe_cartpole_component_cross import (
    cell_name,
    component_cells,
    load_fixed_label_rows,
    validate_fixed_label_state,
)


def _label_row(sample_index: int, timestep: int) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "t": timestep,
        "x": 0.1,
        "x_dot": 0.2,
        "theta": 0.3,
        "theta_dot": 0.4,
        "real_policy_score_0": 10.0,
        "real_policy_score_1": 11.0,
        "real_policy_pref": 1,
    }


def _write_labels(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_component_matrix_is_complete_unique_and_stable():
    cells = component_cells()
    names = [cell_name(cell) for cell in cells]

    assert len(cells) == 16
    assert len(set(names)) == 16
    assert cells[0] == {
        "representation": "solved",
        "heads": "solved",
        "critic": "solved",
        "actor": "solved",
    }
    assert cells[-1] == {
        "representation": "final",
        "heads": "final",
        "critic": "final",
        "actor": "final",
    }


def test_fixed_labels_are_keyed_by_replay_occurrence(tmp_path):
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(2, 4), _label_row(2, 5)])

    labels = load_fixed_label_rows(path)

    assert set(labels) == {(2, 4), (2, 5)}
    validate_fixed_label_state(
        labels[(2, 4)], np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    )


def test_fixed_labels_reject_duplicates(tmp_path):
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(2, 4), _label_row(2, 4)])

    with pytest.raises(ValueError, match="duplicate"):
        load_fixed_label_rows(path)


def test_fixed_labels_reject_state_drift(tmp_path):
    path = tmp_path / "labels.csv"
    _write_labels(path, [_label_row(2, 4)])
    labels = load_fixed_label_rows(path)

    with pytest.raises(ValueError, match="does not match"):
        validate_fixed_label_state(
            labels[(2, 4)],
            np.asarray([0.1, 0.2, 0.3, 0.5], dtype=np.float32),
        )
