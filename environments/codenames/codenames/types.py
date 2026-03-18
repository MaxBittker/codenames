from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

CardColor = Literal["Red", "Blue", "Civilian", "Assassin"]


@dataclass
class BoardConfig:
    """Configuration for board generation with difficulty presets."""

    board_size: int = 25
    num_red: int = 8
    num_blue: int = 7
    num_civilian: int = 9
    num_assassin: int = 1
    difficulty: str | None = None

    def __post_init__(self) -> None:
        total = self.num_red + self.num_blue + self.num_civilian + self.num_assassin
        if total != self.board_size:
            raise ValueError(
                f"Color counts ({self.num_red}+{self.num_blue}+{self.num_civilian}+{self.num_assassin}={total}) "
                f"must sum to board_size ({self.board_size})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BoardConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


DIFFICULTY_PRESETS: dict[str, BoardConfig] = {
    "easy": BoardConfig(board_size=6, num_red=3, num_blue=1, num_civilian=1, num_assassin=1, difficulty="easy"),
    "medium": BoardConfig(board_size=12, num_red=5, num_blue=2, num_civilian=4, num_assassin=1, difficulty="medium"),
    "standard": BoardConfig(board_size=25, num_red=8, num_blue=7, num_civilian=9, num_assassin=1, difficulty="standard"),
    "hard": BoardConfig(board_size=25, num_red=8, num_blue=9, num_civilian=7, num_assassin=1, difficulty="hard"),
}


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
