"""The CLEF-Flash question that decides hotdog / not_hotdog.

Defaults live in clef_schema.json next to this file; CLEF_SCHEMA=/path/to/other.json (same shape) overrides it.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("clef_schema.json")
LABELS = {"hotdog", "not_hotdog"}


@dataclass(frozen=True)
class ClefPrompt:
    question_id: str
    type: str
    instructions: str
    criteria: dict[str, str]  # option id -> description the model judges against
    state: dict[str, Any] | str
    label_map: dict[str, str] = field(default_factory=dict)  # option id -> "hotdog" | "not_hotdog"

    def question(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": dict(self.criteria)}

    def record(self, images: list) -> dict[str, Any]:
        """The input record CLEF-Flash expects: state, images and our single question."""
        return {"state": self.state, "images": images, "questions": {self.question_id: self.question()}}


def load_prompt(path: str | os.PathLike | None = None) -> ClefPrompt:
    """Load and validate the schema (explicit path, else CLEF_SCHEMA, else clef_schema.json)."""
    path = Path(path or os.environ.get("CLEF_SCHEMA") or DEFAULT_SCHEMA_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))
    criteria = data["criteria"]
    label_map = data.get("label_map") or {option: option for option in criteria}

    labels = set(label_map.values())
    if labels != LABELS:
        raise ValueError(f"{path}: label_map must map onto exactly hotdog/not_hotdog, got {sorted(labels)}")
    if set(label_map) != set(criteria):
        raise ValueError(f"{path}: label_map keys must equal criteria keys")

    return ClefPrompt(
        question_id=data.get("question_id", "food"),
        type=data.get("type", "choice"),
        instructions=data["instructions"],
        criteria=criteria,
        state=data.get("state", {"task": "Classify the food shown in the image"}),
        label_map=label_map,
    )
