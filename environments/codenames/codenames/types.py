from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CardColor = Literal["Red", "Blue", "Civilian", "Assassin"]


@dataclass
class BoardState:
    words: list[str]
    key_grid: list[CardColor]
    revealed: list[CardColor | None]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BoardState":
        return cls(
            words=list(data["words"]),
            key_grid=list(data["key_grid"]),
            revealed=list(data["revealed"]),
        )


@dataclass(frozen=True)
class GuessResult:
    type: Literal["correct", "wrong", "assassin", "invalid"]
    word: str
    color: CardColor | None = None
    reason: str | None = None
