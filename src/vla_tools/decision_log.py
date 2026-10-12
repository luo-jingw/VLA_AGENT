"""Read runs/<task>/decisions.md: the decisions a person makes for a run. Read-only.

One entry per `## D-NNN` heading, followed by exactly these seven fields, each on a `- key: value` line:

    stage       S0 .. S7, the stage that waits on it
    status      pending | decided
    item        what is to be decided
    evidence    files or numbers the decision rests on
    proposal    the agent's recommendation
    decision    the person's decision; empty while pending
    written_to  the file the decision was written into; empty until the agent has done so
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Literal

FIELDS = ("stage", "status", "item", "evidence", "proposal", "decision", "written_to")
DecisionStatus = Literal["pending", "decided"]


@dataclasses.dataclass(frozen=True)
class Decision:
    id: str
    stage: str
    status: DecisionStatus
    item: str
    evidence: str
    proposal: str
    decision: str
    written_to: str

    @property
    def complete(self) -> bool:
        """Decided, and carried into the file it governs."""
        return self.status == "decided" and bool(self.decision) and bool(self.written_to)


def load_decisions(path: Path) -> list[Decision]:
    """All entries in file order; an absent file has none. A malformed entry raises ValueError naming it."""
    if not path.exists():
        return []
    entries: list[Decision] = []
    blocks = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]
    for block in blocks:
        lines = block.splitlines()
        heading = lines[0].strip()
        if not re.fullmatch(r"D-\d+", heading):
            raise ValueError(f"{path}: heading '## {heading}' is not D-NNN")
        values: dict[str, str] = {}
        for line in lines[1:]:
            match = re.fullmatch(r"- (\w+):\s?(.*)", line.strip())
            if match:
                values[match.group(1)] = match.group(2).strip()
        missing = [f for f in FIELDS if f not in values]
        if missing:
            raise ValueError(f"{path}: {heading} lacks {missing}")
        if values["status"] not in ("pending", "decided"):
            raise ValueError(f"{path}: {heading} status must be pending or decided, got {values['status']!r}")
        entries.append(Decision(id=heading, **{f: values[f] for f in FIELDS}))  # type: ignore[arg-type]
    return entries
