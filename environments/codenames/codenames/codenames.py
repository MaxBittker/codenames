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
from .types import BoardState, GuessResult


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

Reply with one word per line, most confident first. Say STOP on its own line to end early. No explanations, no numbering, just words."""

GUESS_LINE_RE = re.compile(r"^([A-Za-z]+)$")


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
        model: str = "openai/gpt-4.1-mini",
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        resolved_base, resolved_key = api_base, api_key

        # Fall back to OPENAI_API_KEY env var
        if not resolved_key:
            resolved_key = os.environ.get("OPENAI_API_KEY")

        # Fall back to prime CLI config
        if not resolved_key or not resolved_base:
            prime_base, prime_key = _read_prime_config()
            if not resolved_key:
                resolved_key = prime_key
            if not resolved_base:
                resolved_base = prime_base

        client_kwargs: dict[str, Any] = {}
        if resolved_base:
            client_kwargs["base_url"] = resolved_base
        if resolved_key:
            client_kwargs["api_key"] = resolved_key

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model

    async def guess(
        self, clue_word: str, max_guesses: int, unrevealed_words: list[str]
    ) -> list[str]:
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
            max_tokens=100,
            temperature=0.0,
        )

        text = (response.choices[0].message.content or "").strip()
        unrevealed_upper = {w.upper() for w in unrevealed_words}

        guesses: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper() == "STOP":
                break
            match = GUESS_LINE_RE.match(line)
            if not match:
                continue
            word = match.group(1).upper()
            if word in unrevealed_upper and word not in {g.upper() for g in guesses}:
                guesses.append(word)
            if len(guesses) >= max_guesses:
                break

        return guesses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataset(train_size: int, eval_size: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = [_make_row(seed + index, "train") for index in range(train_size)]
    eval_rows = [_make_row(seed + train_size + index, "eval") for index in range(eval_size)]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _make_row(seed: int, split: str) -> dict[str, Any]:
    rng = Random(seed)
    board = create_board(rng=rng)
    info = {"seed": seed, "split": split, "words": board.words, "key_grid": board.key_grid}
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
        guesser_model: str = "openai/gpt-4.1-mini",
        guesser_api_base: str | None = None,
        guesser_api_key: str | None = None,
        max_turns: int = 2,
        **kwargs: Any,
    ):
        self.guesser = LLMGuesser(
            model=guesser_model,
            api_base=guesser_api_base,
            api_key=guesser_api_key,
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

    async def is_completed(self, state: dict[str, Any], **kwargs: Any) -> bool:
        if state.get("game_over", False):
            return True
        return await super().is_completed(state, **kwargs)

    async def give_clue(self, word: str, number: int, state_token: str) -> str:
        state = self._state_registry[state_token]
        board = BoardState.from_dict(state["board"])
        clue_word = _validate_clue(word, board)
        clue_number = max(1, min(int(number), 8))
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
                f"Red remaining: {remaining}/8."
            )

        results: list[GuessResult] = []
        for guess in guesses:
            result = evaluate_guess(board, guess)
            results.append(result)
            if result.type != "correct":
                break

        state["board"] = board.to_dict()
        state["total_red_found"] = 8 - count_remaining(board, "Red")
        state["assassin_hit"] = any(r.type == "assassin" for r in results)
        state["blue_hit"] = any(r.type == "wrong" and r.color == "Blue" for r in results)
        state["game_over"] = True

        guessed = ", ".join(_format_guess_result(r) for r in results)
        red_remaining = count_remaining(board, "Red")
        return (
            f"Guesser guessed: {guessed}. "
            f"Red found: {state['total_red_found']}/8, remaining: {red_remaining}/8."
        )


# ---------------------------------------------------------------------------
# Reward & metrics
# ---------------------------------------------------------------------------


async def game_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Single-clue reward.

    - Assassin hit  → -1.0
    - Blue hit      → (red_found / 8) - 0.125
    - Otherwise     → red_found / 8
    """
    if state.get("assassin_hit", False):
        return -1.0
    base = state.get("total_red_found", 0) / 8.0
    if state.get("blue_hit", False):
        base -= 0.125
    return base


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
    guesser_model: str = "openai/gpt-4.1-mini",
    guesser_api_base: str | None = None,
    guesser_api_key: str | None = None,
    max_turns: int = 2,
    **kwargs: Any,
) -> vf.Environment:
    dataset, eval_dataset = _build_dataset(train_size=train_size, eval_size=eval_size, seed=seed)
    rubric = vf.Rubric(
        funcs=[game_reward, assassin_metric, red_found_metric],
        weights=[1.0, 0.0, 0.0],
    )

    return CodenamesCluegiverEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        guesser_model=guesser_model,
        guesser_api_base=guesser_api_base,
        guesser_api_key=guesser_api_key,
        max_turns=max_turns,
        **kwargs,
    )
