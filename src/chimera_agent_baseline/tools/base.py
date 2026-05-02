"""Base utilities for precomputed tool data.

Provides :class:`CaseDataStore` for loading per-case ``tools.json``
files and :class:`ToolSpec` for declaratively defining new tools.

Adding a custom tool
--------------------
1. Define a :class:`ToolSpec` with a name, description, and the field
   names from ``tools.json`` it should return::

       MY_TOOL = ToolSpec(
           name="get_my_data",
           description="Retrieve my custom data for a patient case.",
           fields=("my_field", "another_field"),
       )

2. Append it to ``TASK1_TOOLS`` (or ``TASK2_TOOLS``) in
   ``definitions.py``.

The MCP server picks it up automatically.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a precomputed-data tool.

    Attributes:
        name: Tool function name (used by MCP for discovery).
        description: Human-readable description (shown to the LLM).
        fields: Names of the per-case fields this tool returns. Each
            name must exist as a top-level key in ``tools.json`` for
            cases where the tool has data; missing keys are silently
            omitted from the returned payload. ``case_id`` is always
            included automatically.
    """

    name: str
    description: str
    fields: tuple[str, ...]


class CaseDataStore:
    """Loads per-patient ``tools.json`` files and indexes cases by ``case_id``.

    Expects the canonical layout::

        data_dir/
          PT-<id>/
            tools.json
            prompt.json   # ignored — read by the agent runner, not the MCP server

    """

    def __init__(self, data_dir: str | Path):
        self._cases: dict[str, dict] = {}
        self._load(Path(data_dir))

    def _load(self, data_dir: Path) -> None:
        if not data_dir.is_dir():
            log.warning("Data dir %s is not a directory", data_dir)
            return

        pair_subdirs = sorted(
            p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("PT-") and (p / "tools.json").exists()
        )
        if not pair_subdirs:
            log.warning("No PT-*/tools.json subdirectories found in %s", data_dir)
            return

        for sub in pair_subdirs:
            try:
                case = json.loads((sub / "tools.json").read_text())
            except json.JSONDecodeError:
                log.warning("Skipping %s/tools.json: invalid JSON", sub)
                continue
            case_id = case.get("case_id") or sub.name
            self._cases[case_id] = case
        log.info("Loaded %d cases from %s", len(self._cases), data_dir)

    def get_case(self, case_id: str) -> dict | None:
        """Return raw case record, or ``None`` if not found."""
        return self._cases.get(case_id)

    def list_case_ids(self) -> list[str]:
        return list(self._cases.keys())

    def extract(self, case_id: str, fields: tuple[str, ...]) -> dict:
        """Return ``{case_id, **fields_present_on_the_case}``.

        Fields not present on the case are silently omitted (so a
        biopsy-naïve case calling ``get_pathology_report`` returns just
        ``{case_id}`` with no pathology keys, not an error).

        Raises:
            KeyError: If *case_id* is not found.
        """
        case = self._cases.get(case_id)
        if case is None:
            raise KeyError(f"Case '{case_id}' not found. Available: {self.list_case_ids()}")
        out: dict = {"case_id": case.get("case_id", case_id)}
        for f in fields:
            if f in case:
                out[f] = case[f]
        return out
