"""Tests for the runtime case loader."""

import json
from pathlib import Path

import pytest

from chimera_agent_baseline.case_loader import TASK1_NAME, TASK2_NAME, load_cases


def _write_pair(root: Path, pid: str, prompt: dict, tools: dict | None = None) -> None:
    sub = root / pid
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "prompt.json").write_text(json.dumps(prompt))
    if tools is not None:
        (sub / "tools.json").write_text(json.dumps(tools))


def _stub_template(tmp_path: Path) -> Path:
    """A minimal Jinja template that just dumps a few fields."""
    tdir = tmp_path / "templates"
    tdir.mkdir()
    (tdir / "agent_prompt.j2").write_text("Case {{ case_id }}: PSA {{ current_psa.value }}.")
    return tdir


def test_load_cases_returns_one_entry_per_pt_subdir(tmp_path: Path):
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "PT-a", {"case_id": "PT-a", "current_psa": {"value": 4.5}})
    _write_pair(cases, "PT-b", {"case_id": "PT-b", "current_psa": {"value": 12.0}})

    queries = load_cases(cases, task=1, templates_dir=tdir)
    assert [q["case_id"] for q in queries] == ["PT-a", "PT-b"]
    assert all(q["task"] == TASK1_NAME for q in queries)
    assert "PSA 4.5" in queries[0]["context"]


def test_load_cases_uses_task2_default_when_payload_omits_task(tmp_path: Path):
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "PT-a", {"case_id": "PT-a", "current_psa": {"value": 4.5}})

    queries = load_cases(cases, task=2, templates_dir=tdir)
    assert queries[0]["task"] == TASK2_NAME


def test_load_cases_raises_for_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope")


def test_load_cases_skips_non_pt_subdirs(tmp_path: Path):
    tdir = _stub_template(tmp_path)
    cases = tmp_path / "cases"
    _write_pair(cases, "PT-a", {"case_id": "PT-a", "current_psa": {"value": 4.5}})
    (cases / "logs").mkdir()
    (cases / "logs" / "prompt.json").write_text("{}")
    queries = load_cases(cases, task=1, templates_dir=tdir)
    assert [q["case_id"] for q in queries] == ["PT-a"]
