import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_FILE = "prostate-treatment-decision.json"
FEATURE_FILE = "prostate-modality-level-neural-representations.json"

MRI_DIM = 1024
BIOPSY_DIM = 960

VALID_LABELS = {
    "active_surveillance",
    "continued_surveillance",
    "watchful_waiting",
    "active_treatment",
}


def mean_vectors(vectors, dim):
    """Mean-pool zero or more modality vectors."""
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


def featurise(payload):
    mri = payload.get("MRI image") or []
    biopsy = payload.get("Biopsy slide") or []

    # IMPORTANT:
    # Deliberately ignore "Prostatectomy slide".
    # It may encode post-treatment information and therefore leak the target.
    mri_vec = mean_vectors(mri, MRI_DIM)
    biopsy_vec = mean_vectors(biopsy, BIOPSY_DIM)

    # Preserve useful missingness / multiplicity information.
    extra = np.asarray(
        [
            float(bool(mri)),
            float(bool(biopsy)),
            float(len(mri)),
            float(len(biopsy)),
        ],
        dtype=np.float32,
    )

    return np.concatenate([mri_vec, biopsy_vec, extra])


def load_dataset(task2_dir):
    X = []
    y = []
    case_ids = []

    for case_dir in sorted(task2_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        label_path = case_dir / LABEL_FILE
        feature_path = case_dir / FEATURE_FILE

        # Only the released labelled subset is usable for supervised fitting.
        if not label_path.exists() or not feature_path.exists():
            continue

        label = json.loads(label_path.read_text())

        if not isinstance(label, str):
            print(f"Skipping {case_dir.name}: label is not a string")
            continue

        if label not in VALID_LABELS:
            print(
                f"Skipping {case_dir.name}: unexpected label {label!r}"
            )
            continue

        try:
            payload = json.loads(feature_path.read_text())
            features = featurise(payload)
        except Exception as exc:
            print(f"Skipping {case_dir.name}: {exc}")
            continue

        X.append(features)
        y.append(label)
        case_ids.append(case_dir.name)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        case_ids,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "task2_dir",
        type=Path,
        help="Path to train_release/task2",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/task2_predictor.joblib"),
    )

    args = parser.parse_args()

    X, y, case_ids = load_dataset(args.task2_dir)

    counts = Counter(y)

    print()
    print("Loaded labelled patients:", len(y))
    print("Feature matrix:", X.shape)
    print("Class counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label:24s} {count}")

    if len(y) != 72:
        print(
            f"\nWARNING: expected 72 released labels, loaded {len(y)}"
        )

    # The rarest class is watchful_waiting with only 2 examples.
    # Therefore ordinary 5-fold stratified CV is impossible.
    #
    # 2-fold guarantees that each training fold contains one WW patient.
    min_class_count = min(counts.values())
    n_splits = min(2, min_class_count)

    if n_splits < 2:
        raise RuntimeError(
            "At least two examples per class are required."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    random_state=42,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Keep the search deliberately small.
    # 72 samples does not justify a large hyperparameter sweep.
    param_grid = {
        "pca__n_components": [
            2,
            4,
            8,
            12,
            16,
        ],
        "clf__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        "clf__class_weight": [
            None,
            "balanced",
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("\nFitting PCA + multinomial logistic regression...")
    search.fit(X, y)

    print("\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    print(
        f"\nBest mean CV weighted F1: "
        f"{search.best_score_:.4f}"
    )

    best_model = search.best_estimator_

    # Sanity-check predictions.
    #
    # NOTE: because hyperparameters were selected using this same dataset,
    # these numbers are informative but optimistic. They are NOT an
    # independent validation estimate.
    oof_pred = cross_val_predict(
        best_model,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        method="predict",
    )

    weighted_f1 = f1_score(
        y,
        oof_pred,
        average="weighted",
    )

    balanced_acc = balanced_accuracy_score(
        y,
        oof_pred,
    )

    print(f"OOF weighted F1:       {weighted_f1:.4f}")
    print(f"OOF balanced accuracy: {balanced_acc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            oof_pred,
            digits=3,
            zero_division=0,
        )
    )

    print("Confusion examples:")
    for case_id, actual, predicted in zip(
        case_ids,
        y,
        oof_pred,
    ):
        if actual != predicted:
            print(
                f"  {case_id}: "
                f"{actual} -> {predicted}"
            )

    artifact = {
        "model": best_model,

        "classes": list(
            best_model.named_steps["clf"].classes_
        ),

        "n_training_cases": len(y),

        "class_counts": dict(counts),

        "feature_spec": {
            "mri": "mean pooled 1024-D",
            "biopsy": "mean pooled 960-D",
            "extra": [
                "MRI present",
                "biopsy present",
                "MRI vector count",
                "biopsy vector count",
            ],
            "prostatectomy": "INTENTIONALLY EXCLUDED - leakage risk",
        },

        "warning": (
            "watchful_waiting has only 2 released training examples; "
            "its learned probability is unreliable and should not "
            "override clinical/guideline reasoning."
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        args.output,
        compress=3,
    )

    print(
        f"\nSaved model: {args.output}"
    )

    print(
        f"Model size: "
        f"{args.output.stat().st_size / 1024**2:.2f} MiB"
    )


if __name__ == "__main__":
    main()
