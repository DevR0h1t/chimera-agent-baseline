"""Tests for the MCP server, CaseDataStore, and ToolSpec extraction."""

import json

import pytest

from chimera_agent_baseline.mcp_server import create_server
from chimera_agent_baseline.tools.base import CaseDataStore, ToolSpec
from chimera_agent_baseline.tools.definitions import TASK1_TOOLS, TASK2_TOOLS


def _write_case(root, pid: str, payload: dict) -> None:
    sub = root / pid
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "clinical.json").write_text(json.dumps(payload))


# ── CaseDataStore ────────────────────────────────────────────────────────


class TestCaseDataStore:
    def test_loads_pair_layout(self, tmp_path):
        _write_case(tmp_path, "PT-aaa", {"case_id": "PT-aaa", "pirads": "4"})
        _write_case(tmp_path, "PT-bbb", {"case_id": "PT-bbb", "pirads": "5"})
        store = CaseDataStore(tmp_path)
        assert sorted(store.list_case_ids()) == ["PT-aaa", "PT-bbb"]

    def test_loads_non_pt_case_dirs(self, tmp_path):
        # Task-2 case dirs are named T2-* (not PT-*); discovery is by file, not prefix.
        _write_case(tmp_path, "T2-001", {"case_id": "T2-001", "pirads": "2"})
        store = CaseDataStore(tmp_path)
        assert store.list_case_ids() == ["T2-001"]

    def test_ignores_dirs_without_clinical_json(self, tmp_path):
        _write_case(tmp_path, "PT-a", {"case_id": "PT-a", "pirads": "4"})
        (tmp_path / "logs").mkdir()  # no clinical.json -> not a case
        store = CaseDataStore(tmp_path)
        assert store.list_case_ids() == ["PT-a"]

    def test_get_case_found(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x", "pirads": "3"})
        store = CaseDataStore(tmp_path)
        case = store.get_case("PT-x")
        assert case is not None
        assert case["pirads"] == "3"

    def test_get_case_missing(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x"})
        store = CaseDataStore(tmp_path)
        assert store.get_case("PT-y") is None

    def test_extract_returns_only_requested_fields(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x", "pirads": "4", "psa": 12.0, "vol": 35.0})
        store = CaseDataStore(tmp_path)
        out = store.extract("PT-x", ("pirads", "vol"))
        assert out == {"case_id": "PT-x", "pirads": "4", "vol": 35.0}

    def test_extract_omits_missing_fields(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x", "pirads": "4"})
        store = CaseDataStore(tmp_path)
        out = store.extract("PT-x", ("pirads", "psa"))
        assert out == {"case_id": "PT-x", "pirads": "4"}

    def test_extract_missing_case_raises(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x"})
        store = CaseDataStore(tmp_path)
        with pytest.raises(KeyError, match="PT-z"):
            store.extract("PT-z", ("foo",))

    def test_empty_dir(self, tmp_path):
        store = CaseDataStore(tmp_path)
        assert store.list_case_ids() == []


# ── MCP server ──────────────────────────────────────────────────────────


class TestMCPServer:
    def test_create_server_task1(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x"})
        server = create_server(str(tmp_path), tools=TASK1_TOOLS)
        assert server is not None

    def test_create_server_task2(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x"})
        server = create_server(str(tmp_path), tools=TASK2_TOOLS)
        assert server is not None

    def test_create_server_empty_dir(self, tmp_path):
        server = create_server(str(tmp_path))
        assert server is not None

    def test_task1_tools_size(self):
        assert len(TASK1_TOOLS) == 6

    def test_task2_tools_size(self):
        assert len(TASK2_TOOLS) == 6

    def test_custom_tool(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x", "psa": 4.2})
        custom = ToolSpec(name="get_custom", description="A custom tool.", fields=("psa",))
        server = create_server(str(tmp_path), tools=[custom])
        assert server is not None

    def test_tool_returns_requested_fields(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x", "radiology_report": "PI-RADS 4 lesion."})
        server = create_server(str(tmp_path), tools=TASK1_TOOLS)
        tools = server._tool_manager._tools
        result = json.loads(tools["get_mri_report"].fn(case_id="PT-x"))
        assert result["radiology_report"] == "PI-RADS 4 lesion."

    def test_tool_returns_no_data_note_when_fields_missing(self, tmp_path):
        _write_case(tmp_path, "PT-x", {"case_id": "PT-x"})  # no pathology fields
        server = create_server(str(tmp_path), tools=TASK1_TOOLS)
        tools = server._tool_manager._tools
        result = json.loads(tools["get_pathology_report"].fn(case_id="PT-x"))
        assert "note" in result
        assert "No data available" in result["note"]
