from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Literal

CardColor = Literal["Red", "Blue", "Assassin"]


@dataclass
class BoardConfig:
    """A concrete board configuration (after sampling)."""

    board_size: int = 8
    num_red: int = 4
    num_blue: int = 3
    num_assassin: int = 1

    def __post_init__(self) -> None:
        total = self.num_red + self.num_blue + self.num_assassin
        if total != self.board_size:
            raise ValueError(
                f"Color counts ({self.num_red}+{self.num_blue}+{self.num_assassin}={total}) "
                f"must sum to board_size ({self.board_size})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BoardConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BoardSamplingConfig:
    """Defines ranges for sampling board configurations.

    Each board samples ``board_size`` uniformly from [min_board_size, max_board_size]
    and ``red_ratio`` uniformly from [min_red_ratio, max_red_ratio].  num_red is
    clamped to at least 2, num_assassin is always 1, and the rest are blue.
    """

    min_board_size: int = 4
    max_board_size: int = 16
    min_red_ratio: float = 0.3
    max_red_ratio: float = 0.6

    def sample(self, rng: random.Random) -> BoardConfig:
        board_size = rng.randint(self.min_board_size, self.max_board_size)
        red_ratio = rng.uniform(self.min_red_ratio, self.max_red_ratio)
        num_assassin = 1
        num_red = max(2, round(board_size * red_ratio))
        # Ensure at least 0 blue slots remain
        num_red = min(num_red, board_size - num_assassin)
        num_blue = board_size - num_red - num_assassin
        return BoardConfig(
            board_size=board_size,
            num_red=num_red,
            num_blue=num_blue,
            num_assassin=num_assassin,
        )


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
