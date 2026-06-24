"""Tests for the runtime case loader."""

import json
from pathlib import Path

import pytest

from chimera_agent_baseline.case_loader import load_cases


def _write_pair(root: Path, pid: str, prompt: dict, clinical: dict | None = None) -> None:
    sub = root / pid
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "prompt.json").write_text(json.dumps(prompt))
    if clinical is not None:
        (sub / "clinical.json").write_text(json.dumps(clinical))


def _stub_template(tmp_path: Path) -> Path:
    """A minimal Jinja template over the flat prompt.json fields."""
    tdir = tmp_path / "templates"
    tdir.mkdir()
    (tdir / "agent_prompt.j2").write_text("Case {{ case_id }}: PSA {{ psa }}.")
    return tdir


def test_load_cases_returns_one_entry_per_subdir(tmp_path: Path):
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "PT-a", {"case_id": "PT-a", "psa": 4.5})
    _write_pair(cases, "PT-b", {"case_id": "PT-b", "psa": 12.0})

    queries = load_cases(cases, task=1, templates_dir=tdir)
    assert [q["case_id"] for q in queries] == ["PT-a", "PT-b"]
    assert all(q["task"] == 1 for q in queries)
    assert "PSA 4.5" in queries[0]["context"]


def test_load_cases_loads_non_pt_case_dirs(tmp_path: Path):
    # Task-2 case dirs are named T2-* (not PT-*); discovery is by prompt.json, not prefix.
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "T2-001", {"case_id": "T2-001", "psa": 4.7})
    queries = load_cases(cases, task=2, templates_dir=tdir)
    assert [q["case_id"] for q in queries] == ["T2-001"]


def test_load_cases_uses_task2_default_when_payload_omits_task(tmp_path: Path):
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "PT-a", {"case_id": "PT-a", "psa": 4.5})

    queries = load_cases(cases, task=2, templates_dir=tdir)
    assert queries[0]["task"] == 2


def test_load_cases_raises_for_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope")


def test_load_cases_skips_subdirs_without_prompt(tmp_path: Path):
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "PT-a", {"case_id": "PT-a", "psa": 4.5})
    (cases / "logs").mkdir()  # no prompt.json -> not a case
    queries = load_cases(cases, task=1, templates_dir=tdir)
    assert [q["case_id"] for q in queries] == ["PT-a"]


def test_task3_renders_simplified_prompt(tmp_path: Path):
    # Uses the real template to exercise the task-3 branch (age / PSA / DRE only).
    cases = tmp_path / "cases"
    _write_pair(
        cases,
        "T3-001",
        {"case_id": "T3-001", "task": 3, "age": 64, "psa": 8.2, "dre": "Firm nodule, right base."},
    )
    ctx = load_cases(cases, task=3)[0]["context"]
    assert "64-year-old male" in ctx
    assert "PSA: 8.2 ng/mL" in ctx
    assert "Physical examination (DRE): Firm nodule, right base." in ctx
    assert "time to biochemical recurrence" in ctx
    assert "surgical pathology report" in ctx
    # The task-1/2 panel must not leak into the simplified task-3 prompt.
    assert "PI-RADS" not in ctx
    assert "csPCa" not in ctx
