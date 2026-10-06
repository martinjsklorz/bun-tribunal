"""Prompt / schema for the CLEF-Flash hotdog question.

Defaults live in clef_schema.json next to this file. Override with
CLEF_SCHEMA=/path/to/other.json (same shape). No torch import here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("clef_schema.json")


@dataclass(frozen=True)
class ClefPrompt:
    question_id: str
    type: str
    instructions: str
    criteria: dict[str, str]
    state: dict | str
    # maps model option id -> contract label ("hotdog" / "not_hotdog")
    label_map: dict[str, str] = field(default_factory=dict)

    def question(self) -> dict:
        return {
            "type": self.type,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }

    def record(self, images: list) -> dict:
        return {
            "state": self.state,
            "images": images,
            "questions": {self.question_id: self.question()},
        }


def load_prompt(path: str | os.PathLike | None = None) -> ClefPrompt:
    p = Path(path or os.environ.get("CLEF_SCHEMA") or DEFAULT_SCHEMA_PATH)
    data = json.loads(p.read_text(encoding="utf-8"))
    criteria = data["criteria"]
    label_map = data.get("label_map") or {k: k for k in criteria}
    labels = set(label_map.values())
    if labels != {"hotdog", "not_hotdog"}:
        raise ValueError(f"{p}: label_map must map onto exactly hotdog/not_hotdog, got {sorted(labels)}")
    if set(label_map) != set(criteria):
        raise ValueError(f"{p}: label_map keys must equal criteria keys")
    return ClefPrompt(
        question_id=data.get("question_id", "food"),
        type=data.get("type", "choice"),
        instructions=data["instructions"],
        criteria=criteria,
        state=data.get("state", {"task": "Classify the food shown in the image"}),
        label_map=label_map,
    )
