"""Base utilities for precomputed tool data.

Provides :class:`CaseDataStore` for loading and indexing clinical data, and
:class:`ToolSpec` for declaratively defining new tools via field mappings.

Adding a custom tool
--------------------
1. Define a :class:`ToolSpec` with a name, description, and field mapping::

       MY_TOOL = ToolSpec(
           name="get_my_data",
           description="Retrieve my custom data for a patient case.",
           field_mapping={
               "output_field": ["source_field_variant_a", "source_field_variant_b"],
           },
       )

2. Append it to ``TOOL_REGISTRY`` in ``definitions.py``.

The MCP server will pick it up automatically.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a precomputed data tool.

    Attributes:
        name: Tool function name (used by MCP for discovery).
        description: Human-readable description (shown to the LLM).
        field_mapping: ``{normalized_name: [source_field, ...]}`` — tried
            in order, first match in the case record wins.
    """

    name: str
    description: str
    field_mapping: dict[str, list[str]]


class CaseDataStore:
    """Loads a ``clinical-data.json`` file and indexes cases by ``case_id``."""

    def __init__(self, data_dir: str | Path):
        self._cases: dict[str, dict] = {}
        self._load(Path(data_dir))

    def _load(self, data_dir: Path) -> None:
        data_file = data_dir / "clinical-data.json"
        if not data_file.exists():
            log.warning("No clinical-data.json found in %s", data_dir)
            return
        cases = json.loads(data_file.read_text())
        for case in cases:
            case_id = case.get("case_id")
            if case_id:
                self._cases[case_id] = case
        log.info("Loaded %d cases from %s", len(self._cases), data_file)

    def get_case(self, case_id: str) -> dict | None:
        """Return raw case record, or ``None`` if not found."""
        return self._cases.get(case_id)

    def list_case_ids(self) -> list[str]:
        """Return all loaded case IDs."""
        return list(self._cases.keys())

    def extract(self, case_id: str, field_mapping: dict[str, list[str]]) -> dict:
        """Extract and normalize fields for a case.

        Only fields that exist in the source record are included in the
        result.  Fields whose source key is present but has a ``None`` value
        are preserved (e.g. ``tertiary_gleason: null``).

        Raises:
            KeyError: If *case_id* is not found.
        """
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"Case '{case_id}' not found. Available: {self.list_case_ids()}")
        return extract_fields(case, field_mapping)


def extract_fields(case: dict, field_mapping: dict[str, list[str]]) -> dict:
    """Extract and normalize fields from a single case record.

    For each *normalized_name* the *source_fields* list is tried in order;
    the first key found in *case* is used.  If none of the source fields
    exist, the key is **omitted** from the result (distinguishing "not
    available" from an explicit ``null``).
    """
    result: dict = {}
    for normalized_name, source_fields in field_mapping.items():
        for field_name in source_fields:
            if field_name in case:
                result[normalized_name] = case[field_name]
                break
    return result
