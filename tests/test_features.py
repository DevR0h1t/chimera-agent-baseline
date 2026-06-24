"""Tests for the per-case feature-embedding loader."""

import json

from chimera_agent_baseline.features import (
    FEATURE_ORIGINS_BY_TASK,
    FeatureStore,
    _as_vector_list,
)


def _write_features(root, case_id: str, payload: dict) -> None:
    sub = root / case_id
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "features.json").write_text(json.dumps(payload))


def test_loads_origins_as_vector_lists(tmp_path):
    _write_features(
        tmp_path,
        "T3-1",
        {
            "case_id": "T3-1",
            "mri": [[0.1, 0.2, 0.3]],
            "biopsy": [[1.0, 1.0], [2.0, 2.0]],
            "prostatectomy": [[9.0]],
        },
    )
    store = FeatureStore(tmp_path)
    assert store.list_case_ids() == ["T3-1"]
    assert store.get_origin("T3-1", "mri") == [[0.1, 0.2, 0.3]]
    assert len(store.get_origin("T3-1", "biopsy")) == 2
    assert store.get_origin("T3-1", "prostatectomy") == [[9.0]]


def test_normalises_bare_vector_to_list(tmp_path):
    # A single flat vector (not wrapped) is coerced to a one-element list.
    _write_features(tmp_path, "PT-a", {"case_id": "PT-a", "mri": [0.5, 0.6, 0.7]})
    store = FeatureStore(tmp_path)
    assert store.get_origin("PT-a", "mri") == [[0.5, 0.6, 0.7]]


def test_absent_origin_returns_empty_list(tmp_path):
    _write_features(tmp_path, "T2-1", {"case_id": "T2-1", "mri": [[1.0]], "biopsy": [[2.0]]})
    store = FeatureStore(tmp_path)
    assert store.get_origin("T2-1", "prostatectomy") == []


def test_missing_features_file_means_absent_case(tmp_path):
    (tmp_path / "PT-a").mkdir()  # no features.json
    store = FeatureStore(tmp_path)
    assert store.list_case_ids() == []
    assert store.get("PT-a") == {}


def test_as_vector_list_helper():
    assert _as_vector_list([]) == []
    assert _as_vector_list([1.0, 2.0]) == [[1.0, 2.0]]
    assert _as_vector_list([[1.0], [2.0]]) == [[1.0], [2.0]]


def test_origins_by_task_mapping():
    assert FEATURE_ORIGINS_BY_TASK[1] == ("mri",)
    assert FEATURE_ORIGINS_BY_TASK[2] == ("mri", "biopsy")
    assert FEATURE_ORIGINS_BY_TASK[3] == ("mri", "biopsy", "prostatectomy")
