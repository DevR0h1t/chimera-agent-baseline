"""Tests for the multi-task run-plan resolution."""

import json

import pytest
from omegaconf import OmegaConf

from chimera_agent_baseline.run import _run_plan


def _make_case(root, task: int, case_id: str) -> None:
    sub = root / f"task{task}" / "agent_input" / case_id
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "prompt.json").write_text(json.dumps({"case_id": case_id, "task": task}))


def _cfg(data_root, tasks):
    return OmegaConf.create({"paths": {"data_root": str(data_root)}, "agent": {"tasks": tasks}})


def test_run_plan_includes_present_tasks_skips_missing(tmp_path):
    _make_case(tmp_path, 1, "PT-a")
    _make_case(tmp_path, 2, "T2-1")  # task 3 dir absent
    plan = _run_plan(_cfg(tmp_path, [1, 2, 3]))
    assert [t for t, _ in plan] == [1, 2]
    assert all(d.is_dir() for _, d in plan)


def test_run_plan_respects_task_subset(tmp_path):
    _make_case(tmp_path, 1, "PT-a")
    _make_case(tmp_path, 2, "T2-1")
    plan = _run_plan(_cfg(tmp_path, [2]))
    assert [t for t, _ in plan] == [2]


def test_run_plan_raises_when_no_tasks_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        _run_plan(_cfg(tmp_path, [1, 2, 3]))


def test_run_plan_rejects_unknown_task(tmp_path):
    _make_case(tmp_path, 1, "PT-a")
    with pytest.raises(ValueError, match="Unknown task"):
        _run_plan(_cfg(tmp_path, [1, 9]))
