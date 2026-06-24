"""Base utilities for precomputed tool data.

Provides :class:`CaseDataStore` for loading per-case ``clinical.json``
files and :class:`ToolSpec` for declaratively defining new tools.

``clinical.json`` mirrors the masked "Extended EHR view" sections of the
urologist forms (radiology / MRI report, pathology report, previous
notes, laboratory results, PSA history, family-history anamnesis). Each
tool returns one such section's value directly, so the agent must take an
action to reveal it — exactly as the urologist expanded a masked section.

Adding a custom tool
--------------------
1. Define a :class:`ToolSpec` with a name, description, and the field
   names from ``clinical.json`` it should return::

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
            name must exist as a top-level key in ``clinical.json`` for
            cases where the tool has data; missing keys are silently
            omitted from the returned payload. ``case_id`` is always
            included automatically.
    """

    name: str
    description: str
    fields: tuple[str, ...]


#: Per-case clinical data filename.
CASE_DATA_FILENAME = "clinical.json"


class CaseDataStore:
    """Loads per-patient ``clinical.json`` files and indexes cases by ``case_id``.

    Expects the canonical layout::

        data_dir/
          <case-dir>/            # e.g. PT-<id> (task 1) or T2-<n> (task 2)
            clinical.json
            prompt.json          # ignored — read by the agent runner, not the MCP server

    """

    def __init__(self, data_dir: str | Path):
        self._cases: dict[str, dict] = {}
        self._load(Path(data_dir))

    def _load(self, data_dir: Path) -> None:
        if not data_dir.is_dir():
            log.warning("Data dir %s is not a directory", data_dir)
            return

        case_subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / CASE_DATA_FILENAME).exists())
        if not case_subdirs:
            log.warning("No */%s subdirectories found in %s", CASE_DATA_FILENAME, data_dir)
            return

        for sub in case_subdirs:
            case_file = sub / CASE_DATA_FILENAME
            try:
                case = json.loads(case_file.read_text())
            except json.JSONDecodeError:
                log.warning("Skipping %s: invalid JSON", case_file)
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
