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
from .types import BoardConfig, BoardSamplingConfig, BoardState, GuessResult


CLUEGIVER_SYSTEM_PROMPT = """You are a Codenames cluegiver (codemaster). You are playing cooperatively with a guesser to find RED words on the board.

RULES:
- You can see the board AND the secret key grid showing which words are Red, Blue, or Assassin.
- Give a one-word clue and a number indicating how many RED words relate to that clue.
- You must also declare which specific RED words you are targeting with your clue.
- Your guesser will try to guess that many words based on your clue.
- Your clue must be a SINGLE word — no spaces, no hyphens, no parts of board words, max 15 letters.
- AVOID clues that could lead the guesser to Blue or especially the Assassin word.

STRATEGY:
- Try to connect multiple Red words with a clue to make progress efficiently.
- Avoid clues that could also match Blue or Assassin words.
- Consider what the guesser might think - avoid ambiguous clues.

Respond with your reasoning inside <reasoning> tags, then your clue inside <clue> tags using the exact format below:

<reasoning>
Explain your thought process here — which Red words you want to connect, why you chose this clue, and why it avoids Blue/Assassin words.
</reasoning>
<clue>
word: YOUR_CLUE
number: N
words: TARGET1, TARGET2, ...
</clue>"""

GUESSER_SYSTEM_PROMPT = """You are a Codenames guesser. You are cooperating with a cluegiver to find RED words.

You can see the board but NOT the secret colors. Every word is either Red, Blue, or the Assassin. The cluegiver gave you a one-word clue and a number. The number tells you how many board words relate to that clue. You may guess up to (number + 1) words.

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
        model: str = "openai/gpt-4.1-mini",
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
        # Strip known provider prefixes (e.g. "openai/gpt-4.1-mini" -> "gpt-4.1-mini")
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


def _build_dataset(
    train_size: int, eval_size: int, seed: int,
    sampling: BoardSamplingConfig,
) -> tuple[Dataset, Dataset]:
    train_rows = [_make_row(seed + index, "train", sampling=sampling) for index in range(train_size)]
    eval_rows = [_make_row(seed + train_size + index, "eval", sampling=sampling) for index in range(eval_size)]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _make_row(seed: int, split: str, sampling: BoardSamplingConfig) -> dict[str, Any]:
    rng = Random(seed)
    config = sampling.sample(rng)
    board = create_board(rng=rng, config=config)
    info: dict[str, Any] = {
        "seed": seed,
        "split": split,
        "words": board.words,
        "key_grid": board.key_grid,
        "board_config": config.to_dict(),
    }
    prompt = [{"role": "user", "content": f"Current board:\n{format_board_for_cluegiver(board)}"}]
    return {"prompt": prompt, "info": info, "answer": "", "task": split}


def _validate_clue(word: str, board: BoardState) -> str:
    normalized = word.upper().strip()
    if not normalized.isalpha():
        raise ValueError("Clue must be a single alphabetic word.")
    if len(normalized) > 15:
        raise ValueError(f"Clue must be 15 letters or fewer (got {len(normalized)}). Try a shorter word.")
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


def _parse_clue_block(text: str) -> tuple[str, int, list[str]]:
    """Extract word, number, and words from a <clue> block's inner text.

    Expected format inside the block::

        word: PREDATOR
        number: 2
        words: EAGLE, HAWK

    Returns (clue_word, number, target_words).
    Raises ValueError on missing or malformed fields.
    """
    word_match = re.search(r"(?m)^word:\s*(.+)$", text)
    number_match = re.search(r"(?m)^number:\s*(\d+)", text)
    words_match = re.search(r"(?m)^words:\s*(.+)$", text)

    if not word_match:
        raise ValueError("Missing 'word:' field in <clue> block.")
    if not number_match:
        raise ValueError("Missing 'number:' field in <clue> block.")
    if not words_match:
        raise ValueError("Missing 'words:' field in <clue> block.")

    clue_word = word_match.group(1).strip()
    number = int(number_match.group(1))
    target_words = [w.strip() for w in words_match.group(1).split(",") if w.strip()]

    return clue_word, number, target_words


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

parser = vf.XMLParser(fields=["reasoning", "clue"])


class CodenamesCluegiverEnv(vf.MultiTurnEnv):
    """Single-clue environment: the model gives one clue, an LLM guesser
    guesses, and the rollout ends.  The reward reflects how many RED words the
    guesser found from that single clue.

    The model outputs XML with <reasoning> and <clue> blocks.  ``max_turns=2``
    so that turn 1 produces the model's XML output and at the start of turn 2
    ``env_response`` parses the clue, runs the guesser, and sets
    ``final_env_response`` to end the rollout.
    """

    def __init__(
        self,
        guesser_model: str = "openai/gpt-4.1-mini",
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
        super().__init__(
            max_turns=max_turns,
            system_prompt=CLUEGIVER_SYSTEM_PROMPT,
            parser=parser,
            **kwargs,
        )

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

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs: Any) -> Any:
        """Parse the model's XML output, validate the clue, run the guesser,
        and set ``final_env_response`` to end the rollout."""
        # Extract the model's last assistant message
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        if not last_content:
            result = [{"role": "user", "content": "No response received. Please provide a clue using the required XML format."}]
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            state["final_env_response"] = result
            return result

        # Parse XML fields
        parsed = self.parser.parse(last_content)
        clue_block = parsed.clue

        if not clue_block:
            result = [{"role": "user", "content": "Could not find a <clue> block in your response. Please use the required XML format."}]
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            state["final_env_response"] = result
            return result

        # Parse the clue fields from the block
        try:
            clue_word, clue_number, target_words_raw = _parse_clue_block(clue_block)
        except ValueError as exc:
            result = [{"role": "user", "content": f"Invalid clue format: {exc}"}]
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            state["final_env_response"] = result
            return result

        board = BoardState.from_dict(state["board"])
        num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)

        # Validate the clue word
        try:
            clue_word = _validate_clue(clue_word, board)
        except ValueError as exc:
            result = [{"role": "user", "content": f"Invalid clue: {exc}"}]
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            state["final_env_response"] = result
            return result

        clue_number = max(1, min(int(clue_number), num_red))
        state["last_clue"] = {"word": clue_word, "number": clue_number}

        # Store the called-shot target words (cap at clue_number)
        target_words = [w.upper().strip() for w in target_words_raw][:clue_number]
        state["target_words"] = target_words

        unrevealed = [w for w, r in zip(board.words, board.revealed) if r is None]
        max_guesses = clue_number + 1

        try:
            guesses = await self.guesser.guess(clue_word, max_guesses, unrevealed)
        except Exception as exc:
            state["game_over"] = True
            state["shots_hit"] = 0
            result = [{"role": "user", "content": f"Guesser error: {exc}"}]
            state["final_env_response"] = result
            return result

        if not guesses:
            state["game_over"] = True
            state["shots_hit"] = 0
            remaining = count_remaining(board, "Red")
            result = [{"role": "user", "content": (
                f'Guesser could not guess any word for "{clue_word}" {clue_number}. '
                f"Red remaining: {remaining}/{num_red}."
            )}]
            state["final_env_response"] = result
            return result

        results: list[tuple[GuessResult, str]] = []
        for word, reason in guesses:
            gr = evaluate_guess(board, word)
            results.append((gr, reason))
            if gr.type != "correct":
                break

        state["board"] = board.to_dict()
        state["total_red_found"] = num_red - count_remaining(board, "Red")
        state["assassin_hit"] = any(r.type == "assassin" for r, _ in results)
        state["blue_hit"] = any(r.type == "wrong" and r.color == "Blue" for r, _ in results)
        state["game_over"] = True

        # Count how many called-shot targets were correctly guessed
        correctly_guessed = {r.word for r, _ in results if r.type == "correct"}
        shots_hit = sum(1 for t in target_words if t in correctly_guessed)
        state["shots_hit"] = shots_hit

        lines = [f"- {_format_guess_result(r)} — guesser reasoning: \"{reason}\"" for r, reason in results]
        red_remaining = count_remaining(board, "Red")
        transcript = (
            f"Guesser guessed:\n"
            + "\n".join(lines)
            + f"\nRed found: {state['total_red_found']}/{num_red}, remaining: {red_remaining}/{num_red}."
            + f"\nCalled shots hit: {shots_hit}/{len(target_words)}."
        )

        result = [{"role": "user", "content": transcript}]
        state["final_env_response"] = result
        return result


# ---------------------------------------------------------------------------
# Reward & metrics
# ---------------------------------------------------------------------------


async def game_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Single-clue reward — per-card additive scoring, normalized to max 2.0.

    - Assassin hit  → -1.0
    - Each red found → +2.0 / num_red
    - Blue hit      → -1.0 / num_red  (half the per-red value)
    """
    num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)
    if state.get("assassin_hit", False):
        return -1.0
    per_red = 2.0 / num_red
    reward = state.get("total_red_found", 0) * per_red
    if state.get("blue_hit", False):
        reward -= per_red * 0.5
    return reward


async def shot_calling_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Bonus reward for correctly calling target words.

    Returns shots_hit / shots_called (1.0 if all called targets were guessed
    correctly, 0.0 if none were).  Returns 0.0 if no targets were declared.
    """
    target_words = state.get("target_words", [])
    if not target_words:
        return 0.0
    shots_hit = state.get("shots_hit", 0)
    return shots_hit / len(target_words)


async def assassin_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return 1.0 if state.get("assassin_hit", False) else 0.0


async def red_found_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return float(state.get("total_red_found", 0))


async def shots_hit_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return float(state.get("shots_hit", 0))


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
    guesser_max_tokens: int = 16_000,
    self_play: bool = False,
    max_turns: int = 2,
    min_board_size: int = 4,
    max_board_size: int = 16,
    min_red_ratio: float = 0.3,
    max_red_ratio: float = 0.6,
    **kwargs: Any,
) -> vf.Environment:
    sampling = BoardSamplingConfig(
        min_board_size=min_board_size,
        max_board_size=max_board_size,
        min_red_ratio=min_red_ratio,
        max_red_ratio=max_red_ratio,
    )
    dataset, eval_dataset = _build_dataset(
        train_size=train_size, eval_size=eval_size, seed=seed, sampling=sampling,
    )

    format_reward = parser.get_format_reward_func()
    rubric = vf.Rubric(
        funcs=[game_reward, shot_calling_reward, format_reward, assassin_metric, red_found_metric, shots_hit_metric],
        weights=[1.0, 0.5, 0.1, 0.0, 0.0, 0.0],
        parser=parser,
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
