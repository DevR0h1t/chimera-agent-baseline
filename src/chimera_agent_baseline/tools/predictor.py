"""Task-2 image-embedding treatment predictor."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np

from chimera_agent_baseline.features import FeatureStore, Vector


log = logging.getLogger(__name__)

PREDICTOR_TOOL_NAME = "get_image_predictor"

MRI_DIM = 1024
BIOPSY_DIM = 960

_ARTIFACT = None


def _load_artifact():
    global _ARTIFACT

    if _ARTIFACT is not None:
        return _ARTIFACT

    candidates = [
        Path("model/task2_predictor.joblib"),
        Path("/opt/ml/model/task2_predictor.joblib"),
        Path("/model/task2_predictor.joblib"),
    ]

    for path in candidates:
        if path.exists():
            log.info("Loading Task-2 predictor from %s", path)
            _ARTIFACT = joblib.load(path)
            return _ARTIFACT

    raise FileNotFoundError(
        "task2_predictor.joblib not found in: "
        + ", ".join(str(p) for p in candidates)
    )


def _mean_vectors(vectors: list[Vector], dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros(dim, dtype=np.float32)

    arr = np.asarray(vectors, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != dim:
        raise ValueError(
            f"Expected embedding dimension {dim}, got {arr.shape}"
        )

    return arr.mean(axis=0)


def _featurise(features: dict[str, list[Vector]]) -> np.ndarray:
    mri = features.get("MRI image") or []
    biopsy = features.get("Biopsy slide") or []

    # Intentionally do NOT use prostatectomy embeddings.
    # They are post-treatment and could leak the target.
    mri_vec = _mean_vectors(mri, MRI_DIM)
    biopsy_vec = _mean_vectors(biopsy, BIOPSY_DIM)

    extra = np.asarray(
        [
            float(bool(mri)),
            float(bool(biopsy)),
            float(len(mri)),
            float(len(biopsy)),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [mri_vec, biopsy_vec, extra]
    ).reshape(1, -1)


def run_predictor(features: dict[str, list[Vector]]) -> dict:
    """Run the trained Task-2 treatment classifier."""

    try:
        artifact = _load_artifact()
        model = artifact["model"]

        x = _featurise(features)

        probs = model.predict_proba(x)[0]
        classes = model.classes_

        order = np.argsort(probs)[::-1]

        ranked = [
            {
                "action": str(classes[i]),
                "probability": round(float(probs[i]), 4),
            }
            for i in order
        ]

        top = ranked[0]

        return {
            "prediction": top["action"],
            "probability": top["probability"],
            "class_probabilities": {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(classes, probs)
            },
            "warning": (
                "Supporting signal only. The model was trained on 72 "
                "released Task-2 cases. watchful_waiting had only 2 "
                "training examples and its probability is unreliable."
            ),
        }

    except Exception as exc:
        log.exception("Task-2 predictor failed")
        return {
            "prediction": None,
            "error": str(exc),
        }


def make_predictor_tool(
    feature_store: FeatureStore,
) -> Callable[[str], str]:
    """Build the get_image_predictor MCP tool."""

    def get_image_predictor(case_id: str) -> str:
        features = feature_store.get(case_id)

        if not features:
            return json.dumps(
                {
                    "case_id": case_id,
                    "note": "No embeddings available for this case.",
                }
            )

        result = run_predictor(features)

        origins = {
            origin: len(vectors)
            for origin, vectors in features.items()
        }

        return json.dumps(
            {
                "case_id": case_id,
                "origins": origins,
                **result,
            }
        )

    get_image_predictor.__name__ = PREDICTOR_TOOL_NAME
    get_image_predictor.__doc__ = (
        "Task 2 supporting classifier over the patient's MRI and biopsy "
        "neural representations. Returns probabilities for the four "
        "treatment-management classes without exposing raw embeddings. "
        "Use as supporting evidence only; clinical/pathological evidence "
        "and NCCN/EAU guidance take precedence."
    )
    get_image_predictor.__annotations__ = {
        "case_id": str,
        "return": str,
    }

    return get_image_predictor
