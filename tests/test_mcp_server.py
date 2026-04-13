"""Tests for the MCP server and tool definitions.

Uses test data from ``test/input/task{1,2,3}`` to verify that the data
store loads correctly, field extraction normalizes across sources, each
tool definition returns the expected data, and the MCP server can be
created without error.
"""

import json
from pathlib import Path

import pytest

from chimera_agent_baseline.mcp_server import create_server
from chimera_agent_baseline.tools.base import CaseDataStore, extract_fields
from chimera_agent_baseline.tools.definitions import (
    CLINICAL_INFO,
    FOLLOW_UP,
    GLEASON_GRADES,
    MRI_FINDINGS,
    PATHOLOGY_STAGING,
    SURGICAL_PATHOLOGY,
    TOOL_REGISTRY,
)

TEST_DATA = Path(__file__).parent.parent / "test" / "input"


# ── CaseDataStore ────────────────────────────────────────────────────────


class TestCaseDataStore:
    def test_load_task1(self):
        store = CaseDataStore(TEST_DATA / "task1")
        ids = store.list_case_ids()
        assert len(ids) == 10
        assert "rumc-001" in ids
        assert "ks-001" in ids

    def test_load_task2(self):
        store = CaseDataStore(TEST_DATA / "task2")
        assert len(store.list_case_ids()) == 10

    def test_load_task3(self):
        store = CaseDataStore(TEST_DATA / "task3")
        assert len(store.list_case_ids()) == 10

    def test_get_case_found(self):
        store = CaseDataStore(TEST_DATA / "task1")
        case = store.get_case("rumc-001")
        assert case is not None
        assert case["case_id"] == "rumc-001"

    def test_get_case_missing(self):
        store = CaseDataStore(TEST_DATA / "task1")
        assert store.get_case("nonexistent") is None

    def test_extract_missing_case_raises(self):
        store = CaseDataStore(TEST_DATA / "task1")
        with pytest.raises(KeyError, match="nonexistent"):
            store.extract("nonexistent", {"foo": ["bar"]})

    def test_empty_dir(self, tmp_path):
        store = CaseDataStore(tmp_path)
        assert store.list_case_ids() == []


# ── extract_fields ───────────────────────────────────────────────────────


class TestExtractFields:
    def test_first_matching_field_wins(self):
        case = {"GLEASON1": 4, "primary_gleason": 3}
        result = extract_fields(case, {"primary": ["primary_gleason", "GLEASON1"]})
        assert result["primary"] == 3

    def test_fallback_to_second_field(self):
        case = {"GLEASON1": 4}
        result = extract_fields(case, {"primary": ["primary_gleason", "GLEASON1"]})
        assert result["primary"] == 4

    def test_missing_field_omitted(self):
        case = {"age": 72}
        result = extract_fields(
            case,
            {
                "age": ["age"],
                "psa": ["X_RESULT", "pre_operative_PSA"],
            },
        )
        assert result["age"] == 72
        assert "psa" not in result

    def test_null_value_preserved(self):
        case = {"tertiary_gleason": None}
        result = extract_fields(case, {"tertiary": ["tertiary_gleason"]})
        assert "tertiary" in result
        assert result["tertiary"] is None


# ── Clinical info tool ───────────────────────────────────────────────────


class TestClinicalInfoTool:
    def test_rumc_case_task1(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", CLINICAL_INFO.field_mapping)
        assert result["case_id"] == "rumc-001"
        assert result["age"] == 72
        assert result["psa"] == 8.0
        assert result["source"] == "chimera"

    def test_karolinska_case_task1(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("ks-001", CLINICAL_INFO.field_mapping)
        assert result["age"] == 72  # MR_age
        assert result["psa"] == 18.1  # X_RESULT
        assert "medical_history" in result  # Anamnes

    def test_karolinska_psa_field_name(self):
        """X_RESULT should map to 'psa' for Karolinska cases."""
        store = CaseDataStore(TEST_DATA / "task2")
        result = store.extract("ks-001", CLINICAL_INFO.field_mapping)
        assert result["psa"] == 8.2


# ── Gleason grades tool ─────────────────────────────────────────────────


class TestGleasonGradesTool:
    def test_rumc_case(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", GLEASON_GRADES.field_mapping)
        assert result["primary_gleason"] == 3
        assert result["secondary_gleason"] == 5
        assert result["isup_grade"] == 4

    def test_karolinska_case(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("ks-002", GLEASON_GRADES.field_mapping)
        assert result["primary_gleason"] == 4  # GLEASON1
        assert result["secondary_gleason"] == 4  # GLEASON2
        assert result["isup_grade"] == 4

    def test_tertiary_null_when_absent(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", GLEASON_GRADES.field_mapping)
        assert result["tertiary_gleason"] is None

    def test_tertiary_present(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-002", GLEASON_GRADES.field_mapping)
        assert result["tertiary_gleason"] == 5


# ── MRI findings tool ───────────────────────────────────────────────────


class TestMRIFindingsTool:
    def test_rumc_pirads(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", MRI_FINDINGS.field_mapping)
        assert result["pirads"] == 2

    def test_karolinska_full_mri(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("ks-001", MRI_FINDINGS.field_mapping)
        assert "pirads" in result
        assert "lesion_detected" in result
        assert "prostate_volume" in result

    def test_no_mri_data(self):
        """Task 3 Karolinska cases have no MRI fields."""
        store = CaseDataStore(TEST_DATA / "task3")
        result = store.extract("ks-001", MRI_FINDINGS.field_mapping)
        assert "pirads" not in result


# ── Pathology staging tool ───────────────────────────────────────────────


class TestPathologyStagingTool:
    def test_rumc_pt_stage(self):
        store = CaseDataStore(TEST_DATA / "task2")
        result = store.extract("rumc-001", PATHOLOGY_STAGING.field_mapping)
        assert result["pt_stage"] == "pT2b"

    def test_karolinska_tnm(self):
        store = CaseDataStore(TEST_DATA / "task3")
        result = store.extract("ks-001", PATHOLOGY_STAGING.field_mapping)
        assert result["pt_stage"] == "T2"
        assert result["n_stage"] == "N0"
        assert result["m_stage"] == "M1"

    def test_no_staging_task1(self):
        """Task 1 RUMC cases have no staging fields."""
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", PATHOLOGY_STAGING.field_mapping)
        assert "pt_stage" not in result


# ── Surgical pathology tool ──────────────────────────────────────────────


class TestSurgicalPathologyTool:
    def test_rumc_task3(self):
        store = CaseDataStore(TEST_DATA / "task3")
        result = store.extract("rumc-001", SURGICAL_PATHOLOGY.field_mapping)
        assert "positive_surgical_margins" in result
        assert "capsular_penetration" in result
        assert "lymphovascular_invasion" in result

    def test_no_surgical_data_task1(self):
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", SURGICAL_PATHOLOGY.field_mapping)
        # Only case_id should match, surgical fields don't exist in task 1
        assert "positive_surgical_margins" not in result


# ── Follow-up tool ───────────────────────────────────────────────────────


class TestFollowUpTool:
    def test_bcr_positive(self):
        store = CaseDataStore(TEST_DATA / "task2")
        result = store.extract("rumc-005", FOLLOW_UP.field_mapping)
        assert result["bcr"] == 1
        assert result["bcr_psa"] == 1.39

    def test_bcr_negative(self):
        store = CaseDataStore(TEST_DATA / "task2")
        result = store.extract("rumc-001", FOLLOW_UP.field_mapping)
        assert result["bcr"] == 0
        assert result["bcr_psa"] is None

    def test_no_follow_up_task1(self):
        """Task 1 has no BCR / follow-up data."""
        store = CaseDataStore(TEST_DATA / "task1")
        result = store.extract("rumc-001", FOLLOW_UP.field_mapping)
        assert "bcr" not in result


# ── MCP server creation ─────────────────────────────────────────────────


class TestMCPServer:
    def test_create_server_task1(self):
        server = create_server(str(TEST_DATA / "task1"))
        assert server is not None

    def test_create_server_task2(self):
        server = create_server(str(TEST_DATA / "task2"))
        assert server is not None

    def test_create_server_task3(self):
        server = create_server(str(TEST_DATA / "task3"))
        assert server is not None

    def test_create_server_empty_dir(self, tmp_path):
        """Server should create even with no data (tools return errors)."""
        server = create_server(str(tmp_path))
        assert server is not None

    def test_custom_tools(self):
        from chimera_agent_baseline.tools.base import ToolSpec

        custom = ToolSpec(
            name="get_custom",
            description="A custom tool.",
            field_mapping={"case_id": ["case_id"], "psa": ["pre_operative_PSA"]},
        )
        server = create_server(str(TEST_DATA / "task1"), tools=[custom])
        assert server is not None

    def test_tool_registry_has_expected_count(self):
        assert len(TOOL_REGISTRY) == 6

    def test_action_log_records_tool_calls(self):
        """Action log records calls and clears on retrieval."""
        import chimera_agent_baseline.mcp_server as mcp_mod

        # Ensure clean state
        if mcp_mod._ACTION_LOG_PATH.exists():
            mcp_mod._ACTION_LOG_PATH.unlink()

        server = create_server(str(TEST_DATA / "task1"))
        tools = server._tool_manager._tools

        # Call a tool
        tools["get_clinical_info"].fn(case_id="rumc-001")

        # Retrieve action log
        log_json = tools["get_action_log"].fn()
        log_entries = json.loads(log_json)

        assert len(log_entries) == 1
        assert log_entries[0]["tool"] == "get_clinical_info"
        assert log_entries[0]["args"] == {"case_id": "rumc-001"}
        assert log_entries[0]["result"]["psa"] == 8.0

        # Log should be cleared after retrieval
        log_json = tools["get_action_log"].fn()
        assert json.loads(log_json) == []
