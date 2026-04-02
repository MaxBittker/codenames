from __future__ import annotations

import random

from .types import BoardConfig, BoardSamplingConfig, BoardState, CardColor, GuessResult
from .wordpool import WORD_POOL


def shuffle(items: list, rng: random.Random | None = None) -> list:
    engine = rng or random
    result = list(items)
    engine.shuffle(result)
    return result


def select_words(
    pool: list[str] | None = None,
    rng: random.Random | None = None,
    count: int = 25,
) -> list[str]:
    return shuffle(pool or WORD_POOL, rng)[:count]


def generate_key_grid(
    rng: random.Random | None = None,
    config: BoardConfig | None = None,
) -> list[CardColor]:
    cfg = config or BoardConfig()
    colors: list[CardColor] = [
        *["Red"] * cfg.num_red,
        *["Blue"] * cfg.num_blue,
        *["Assassin"] * cfg.num_assassin,
    ]
    return shuffle(colors, rng)


def create_board(
    words: list[str] | None = None,
    key_grid: list[CardColor] | None = None,
    rng: random.Random | None = None,
    config: BoardConfig | None = None,
) -> BoardState:
    cfg = config or BoardConfig()
    return BoardState(
        words=[word.upper() for word in (words or select_words(rng=rng, count=cfg.board_size))],
        key_grid=list(key_grid or generate_key_grid(rng=rng, config=cfg)),
        revealed=[None] * cfg.board_size,
    )


def evaluate_guess(board: BoardState, guess_word: str) -> GuessResult:
    upper_word = guess_word.upper().strip()
    try:
        index = next(i for i, word in enumerate(board.words) if word == upper_word)
    except StopIteration:
        return GuessResult(type="invalid", word=guess_word, reason="Word not found on board")

    if board.revealed[index] is not None:
        return GuessResult(type="invalid", word=guess_word, reason="Word already revealed")

    color = board.key_grid[index]
    board.revealed[index] = color
    if color == "Red":
        return GuessResult(type="correct", word=board.words[index], color="Red")
    if color == "Assassin":
        return GuessResult(type="assassin", word=board.words[index])
    return GuessResult(type="wrong", word=board.words[index], color=color)


def count_remaining(board: BoardState, color: CardColor) -> int:
    return sum(
        1
        for revealed, actual in zip(board.revealed, board.key_grid)
        if revealed is None and actual == color
    )


def format_board_for_guesser(board: BoardState) -> str:
    """Show unrevealed words without color info (guesser can't see colors)."""
    unrevealed = [w for w, r in zip(board.words, board.revealed) if r is None]
    return "Remaining words: " + ", ".join(unrevealed)


def format_board_for_cluegiver(board: BoardState) -> str:
    red: list[str] = []
    blue: list[str] = []
    assassin = ""

    for word, color, revealed in zip(board.words, board.key_grid, board.revealed):
        if revealed is not None:
            continue
        if color == "Red":
            red.append(word)
        elif color == "Blue":
            blue.append(word)
        else:
            assassin = word

    lines = [f"RED words to find ({len(red)} remaining): {', '.join(red)}"]
    lines.append(f"BLUE words to AVOID: {', '.join(blue)}")
    lines.append(f"ASSASSIN word to AVOID: {assassin}")
    return "\n".join(lines)


