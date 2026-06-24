"""Tests for the optional image-embedding predictor template."""

import json

from chimera_agent_baseline.features import FeatureStore
from chimera_agent_baseline.tools.predictor import (
    PREDICTOR_TOOL_NAME,
    make_predictor_tool,
    run_predictor,
)


def _write_features(root, case_id: str, payload: dict) -> None:
    sub = root / case_id
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "features.json").write_text(json.dumps(payload))


def test_run_predictor_stub_shape():
    out = run_predictor({"mri": [[0.1, 0.2]], "biopsy": [[0.3, 0.4]]})
    assert out["prediction"] is None
    assert "stub" in out["detail"]


def test_predictor_tool_passes_all_origins_without_raw_vectors(tmp_path):
    # Multimodal case: MRI + biopsy + prostatectomy all handed to the predictor.
    _write_features(
        tmp_path,
        "T3-1",
        {
            "case_id": "T3-1",
            "mri": [[0.1, 0.2, 0.3]],
            "biopsy": [[0.4], [0.5]],
            "prostatectomy": [[0.6]],
        },
    )
    store = FeatureStore(tmp_path)
    tool = make_predictor_tool(store)
    assert tool.__name__ == PREDICTOR_TOOL_NAME

    result = json.loads(tool(case_id="T3-1"))
    # Reports the per-origin vector counts (all three present)...
    assert result["origins"] == {"mri": 1, "biopsy": 2, "prostatectomy": 1}
    # ...but never echoes the raw vectors back to the agent.
    assert "0.1" not in json.dumps(result)


def test_run_predictor_sees_every_origin(tmp_path):
    _write_features(tmp_path, "T3-1", {"case_id": "T3-1", "mri": [[1.0]], "biopsy": [[2.0]]})
    store = FeatureStore(tmp_path)
    seen = {}

    def spy(features):
        seen.update(features)
        return {"prediction": 0.0}

    import chimera_agent_baseline.tools.predictor as pred

    orig = pred.run_predictor
    pred.run_predictor = spy
    try:
        json.loads(make_predictor_tool(store)(case_id="T3-1"))
    finally:
        pred.run_predictor = orig
    assert set(seen) == {"mri", "biopsy"}


def test_predictor_tool_notes_when_no_embeddings(tmp_path):
    (tmp_path / "PT-a").mkdir()  # no features.json
    store = FeatureStore(tmp_path)
    tool = make_predictor_tool(store)
    result = json.loads(tool(case_id="PT-a"))
    assert "No embeddings available" in result["note"]
