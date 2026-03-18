from __future__ import annotations

import os
import re
from random import Random
from typing import Any

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

from .game import (
    count_remaining,
    create_board,
    evaluate_guess,
    format_board_for_cluegiver,
)
from .types import BoardConfig, BoardState, DIFFICULTY_PRESETS, GuessResult


CLUEGIVER_SYSTEM_PROMPT = """You are a Codenames cluegiver (codemaster). You are playing cooperatively with a guesser to find RED words on the board.

RULES:
- You can see the board AND the secret key grid showing which words are Red, Blue, Civilian, or Assassin.
- Give a one-word clue and a number indicating how many RED words relate to that clue.
- Your guesser will try to guess that many words based on your clue.
- Your clue must be a SINGLE word (no spaces, no hyphens, no parts of board words).
- AVOID clues that could lead the guesser to Blue, Civilian, or especially the Assassin word.

STRATEGY:
- Try to connect 2-3 Red words with a clue to make progress efficiently.
- Avoid clues that could also match Blue or Assassin words.
- Consider what the guesser might think - avoid ambiguous clues.

You MUST use the give_clue tool to submit your clue."""

GUESSER_SYSTEM_PROMPT = """You are a Codenames guesser. You are cooperating with a cluegiver to find RED words.

You can see the board but NOT the secret colors. The cluegiver gave you a one-word clue and a number. The number tells you how many board words relate to that clue. You may guess up to (number + 1) words.

- Start with the word you're MOST confident about.
- Stop early if you're unsure — hitting the Assassin loses the game.
- Consider what words the cluegiver would AVOID connecting to (those are likely Blue or Assassin).

For each guess, reply with the word followed by a colon and a brief reason. Say STOP on its own line to end early.

Example:
APPLE: fruit connects to the clue "orchard"
TREE: also found in an orchard
STOP"""

GUESS_LINE_RE = re.compile(r"^([A-Za-z]+)\s*:\s*(.+)$")


def _read_prime_config() -> tuple[str | None, str | None]:
    """Read inference URL and API key from the prime CLI config."""
    try:
        from prime_cli.core.config import Config
        cfg = Config()
        return cfg.inference_url, cfg.api_key
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# LLM Guesser
# ---------------------------------------------------------------------------


class LLMGuesser:
    """Simulates the guesser role using a separate LLM call."""

    def __init__(
        self,
        model: str = "openai/gpt-5.4-nano",
        api_base: str | None = None,
        api_key: str | None = None,
        max_completion_tokens: int = 16_000,
    ):
        resolved_base, resolved_key = api_base, api_key

        # Fall back to OPENAI_API_KEY env var (uses default OpenAI base URL)
        if not resolved_key:
            resolved_key = os.environ.get("OPENAI_API_KEY")

        # Fall back to prime CLI config only if no key found yet
        if not resolved_key:
            prime_base, prime_key = _read_prime_config()
            resolved_key = prime_key
            if not resolved_base:
                resolved_base = prime_base

        client_kwargs: dict[str, Any] = {}
        if resolved_base:
            client_kwargs["base_url"] = resolved_base
        if resolved_key:
            client_kwargs["api_key"] = resolved_key

        self.client = AsyncOpenAI(**client_kwargs)
        # Strip known provider prefixes (e.g. "openai/gpt-5.4-nano" -> "gpt-5.4-nano")
        # but keep HuggingFace-style org/model names intact.
        _PROVIDER_PREFIXES = ("openai/", "anthropic/", "google/")
        lower = model.lower()
        if any(lower.startswith(p) for p in _PROVIDER_PREFIXES):
            self.model = model.split("/", 1)[-1]
        else:
            self.model = model
        self.max_completion_tokens = max_completion_tokens

    async def guess(
        self, clue_word: str, max_guesses: int, unrevealed_words: list[str]
    ) -> list[tuple[str, str]]:
        """Return a list of (word, reason) tuples."""
        word_list = ", ".join(unrevealed_words)
        user_msg = (
            f'Clue: "{clue_word}" for {max_guesses - 1}\n'
            f"You may guess up to {max_guesses} words.\n\n"
            f"Remaining words: {word_list}"
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GUESSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=self.max_completion_tokens,
            temperature=0.0,
        )

        text = (response.choices[0].message.content or "").strip()
        unrevealed_upper = {w.upper() for w in unrevealed_words}

        guesses: list[tuple[str, str]] = []
        seen: set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if line.upper() == "STOP":
                break
            match = GUESS_LINE_RE.match(line)
            if not match:
                continue
            word = match.group(1).upper()
            reason = match.group(2).strip()
            if word in unrevealed_upper and word not in seen:
                guesses.append((word, reason))
                seen.add(word)
            if len(guesses) >= max_guesses:
                break

        return guesses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_board_config(
    difficulty: str | None = None,
    board_size: int | None = None,
    num_red: int | None = None,
    num_blue: int | None = None,
    num_civilian: int | None = None,
    num_assassin: int | None = None,
) -> BoardConfig | None:
    """Build a BoardConfig from a difficulty preset and/or explicit overrides.

    Returns ``None`` when all arguments are ``None`` (use standard defaults).
    """
    if all(v is None for v in (difficulty, board_size, num_red, num_blue, num_civilian, num_assassin)):
        return None

    # Start from a preset or standard defaults
    base = DIFFICULTY_PRESETS.get(difficulty or "standard")
    assert base is not None
    bs = board_size if board_size is not None else base.board_size
    nr = num_red if num_red is not None else base.num_red
    nb = num_blue if num_blue is not None else base.num_blue
    na = num_assassin if num_assassin is not None else base.num_assassin

    if num_civilian is not None:
        nc = num_civilian
    else:
        # Auto-adjust civilians to fill remaining slots
        nc = bs - nr - nb - na

    return BoardConfig(
        board_size=bs, num_red=nr, num_blue=nb,
        num_civilian=nc, num_assassin=na, difficulty=difficulty,
    )


def _build_dataset(
    train_size: int, eval_size: int, seed: int,
    config: BoardConfig | None = None,
) -> tuple[Dataset, Dataset]:
    train_rows = [_make_row(seed + index, "train", config=config) for index in range(train_size)]
    eval_rows = [_make_row(seed + train_size + index, "eval", config=config) for index in range(eval_size)]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _make_row(seed: int, split: str, config: BoardConfig | None = None) -> dict[str, Any]:
    rng = Random(seed)
    board = create_board(rng=rng, config=config)
    info: dict[str, Any] = {"seed": seed, "split": split, "words": board.words, "key_grid": board.key_grid}
    if config is not None:
        info["board_config"] = config.to_dict()
    prompt = [{"role": "user", "content": f"Current board:\n{format_board_for_cluegiver(board)}"}]
    return {"prompt": prompt, "info": info, "answer": "", "task": split}


def _validate_clue(word: str, board: BoardState) -> str:
    normalized = word.upper().strip()
    if not normalized.isalpha():
        raise ValueError("Clue must be a single alphabetic word.")
    for board_word in board.words:
        if normalized == board_word:
            raise ValueError("Clue cannot exactly match a board word.")
        if normalized in board_word or board_word in normalized:
            raise ValueError("Clue cannot contain a board word or be contained in one.")
    return normalized


def _format_guess_result(result: GuessResult) -> str:
    if result.type == "correct":
        return f'{result.word} (Red!)'
    if result.type == "assassin":
        return f'{result.word} (Assassin - game over)'
    if result.type == "wrong":
        return f"{result.word} ({result.color} - turn over)"
    return f'{result.word} (Invalid: {result.reason})'


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CodenamesCluegiverEnv(vf.StatefulToolEnv):
    """Single-clue environment: the model gives one clue, an LLM guesser
    guesses, and the rollout ends.  The reward reflects how many RED words the
    guesser found from that single clue.

    The verifiers framework executes tools at the start of the *next* turn
    (inside ``env_response``).  We therefore need ``max_turns >= 2``:
    turn 1 generates the clue tool-call, and at the start of turn 2 the
    tool is executed.  We then set ``final_env_response`` so the framework
    skips the unnecessary second model call.
    """

    def __init__(
        self,
        guesser_model: str = "openai/gpt-5.4-nano",
        guesser_api_base: str | None = None,
        guesser_api_key: str | None = None,
        guesser_max_tokens: int = 16_000,
        max_turns: int = 2,
        **kwargs: Any,
    ):
        self.guesser = LLMGuesser(
            model=guesser_model,
            api_base=guesser_api_base,
            api_key=guesser_api_key,
            max_completion_tokens=guesser_max_tokens,
        )
        self._state_registry: dict[str, dict[str, Any]] = {}
        super().__init__(max_turns=max_turns, system_prompt=CLUEGIVER_SYSTEM_PROMPT, **kwargs)
        self.add_tool(self.give_clue, args_to_skip=["state_token"])

    async def setup_state(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        info = state.get("info", {})
        board = create_board(words=info.get("words"), key_grid=info.get("key_grid"))
        state["board"] = board.to_dict()
        state["total_red_found"] = 0
        state["assassin_hit"] = False
        state["blue_hit"] = False
        state["game_over"] = False
        state["last_clue"] = None
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Any,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> dict:
        state_token = str(id(state))
        self._state_registry[state_token] = state
        tool_args["state_token"] = state_token
        return tool_args

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs: Any) -> Any:
        """Execute tool calls and, if the game ended, set final_env_response
        so the framework skips the second model call."""
        result = await super().env_response(messages, state, **kwargs)
        if state.get("game_over", False):
            state["final_env_response"] = result
        return result

    async def is_completed(self, *args: Any, **kwargs: Any) -> bool:
        # verifiers calls this as (messages, state) or (state,) depending on version
        state = args[-1] if args else kwargs.get("state", {})
        if isinstance(state, dict) and state.get("game_over", False):
            return True
        return await super().is_completed(*args, **kwargs)

    async def give_clue(self, word: str, number: int, state_token: str) -> str:
        state = self._state_registry[state_token]
        board = BoardState.from_dict(state["board"])
        num_red = state.get("info", {}).get("board_config", {}).get("num_red", 8)
        clue_word = _validate_clue(word, board)
        clue_number = max(1, min(int(number), num_red))
        state["last_clue"] = {"word": clue_word, "number": clue_number}

        unrevealed = [w for w, r in zip(board.words, board.revealed) if r is None]
        max_guesses = clue_number + 1

        try:
            guesses = await self.guesser.guess(clue_word, max_guesses, unrevealed)
        except Exception as exc:
            state["game_over"] = True
            return f"Guesser error: {exc}"

        if not guesses:
            state["game_over"] = True
            remaining = count_remaining(board, "Red")
            return (
                f'Guesser could not guess any word for "{clue_word}" {clue_number}. '
                f"Red remaining: {remaining}/{num_red}."
            )

        results: list[tuple[GuessResult, str]] = []
        for word, reason in guesses:
            result = evaluate_guess(board, word)
            results.append((result, reason))
            if result.type != "correct":
                break

        state["board"] = board.to_dict()
        state["total_red_found"] = num_red - count_remaining(board, "Red")
        state["assassin_hit"] = any(r.type == "assassin" for r, _ in results)
        state["blue_hit"] = any(r.type == "wrong" and r.color == "Blue" for r, _ in results)
        state["game_over"] = True

        lines = [f"- {_format_guess_result(r)} — guesser reasoning: \"{reason}\"" for r, reason in results]
        red_remaining = count_remaining(board, "Red")
        return (
            f"Guesser guessed:\n"
            + "\n".join(lines)
            + f"\nRed found: {state['total_red_found']}/{num_red}, remaining: {red_remaining}/{num_red}."
        )


# ---------------------------------------------------------------------------
# Reward & metrics
# ---------------------------------------------------------------------------


async def game_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Single-clue reward — per-card additive scoring, normalized to max 2.0.

    - Assassin hit  → -1.0
    - Each red found → +2.0 / num_red
    - Blue hit      → -2.0 / num_red
    """
    num_red = state.get("info", {}).get("board_config", {}).get("num_red", 8)
    if state.get("assassin_hit", False):
        return -1.0
    per_red = 2.0 / num_red
    reward = state.get("total_red_found", 0) * per_red
    if state.get("blue_hit", False):
        reward -= per_red
    return reward


async def assassin_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return 1.0 if state.get("assassin_hit", False) else 0.0


async def red_found_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return float(state.get("total_red_found", 0))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_environment(
    train_size: int = 800,
    eval_size: int = 200,
    seed: int = 0,
    guesser_model: str = "openai/gpt-5.4-nano",
    guesser_api_base: str | None = None,
    guesser_api_key: str | None = None,
    guesser_max_tokens: int = 16_000,
    self_play: bool = False,
    max_turns: int = 2,
    difficulty: str | None = None,
    board_size: int | None = None,
    num_red: int | None = None,
    num_blue: int | None = None,
    num_civilian: int | None = None,
    num_assassin: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    config = _resolve_board_config(
        difficulty=difficulty, board_size=board_size,
        num_red=num_red, num_blue=num_blue,
        num_civilian=num_civilian, num_assassin=num_assassin,
    )
    dataset, eval_dataset = _build_dataset(
        train_size=train_size, eval_size=eval_size, seed=seed, config=config,
    )
    rubric = vf.Rubric(
        funcs=[game_reward, assassin_metric, red_found_metric],
        weights=[1.0, 0.0, 0.0],
    )

    if self_play:
        # No explicit API key/base — falls through to prime config,
        # which points at the training inference server.
        guesser_api_base = None
        guesser_api_key = None

    return CodenamesCluegiverEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        guesser_model=guesser_model,
        guesser_max_tokens=guesser_max_tokens,
        guesser_api_base=guesser_api_base,
        guesser_api_key=guesser_api_key,
        max_turns=max_turns,
        **kwargs,
    )
