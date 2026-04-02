from __future__ import annotations

import re
from random import Random
from typing import Any

import verifiers as vf
from datasets import Dataset

from .multiagent import Agent, MultiAgentEnv, RoundRobinProtocol
from .game import (
    count_remaining,
    create_board,
    evaluate_guess,
    format_board_for_cluegiver,
    format_board_for_guesser,
)
from .types import BoardSamplingConfig, BoardState, GuessResult


CLUEGIVER_SYSTEM_PROMPT = """You are a Codenames codemaster (cluegiver). You are cooperating with a guesser to reveal RED words on the board. You can see both the visible board words and the secret key (which words are RED, BLUE, or ASSASSIN). Use that information to give safe, useful single-word clues that connect RED words while avoiding clues that might point the guesser to BLUE or ASSASSIN words.

Required output format (strict — follow exactly):
- Your reply MUST contain a <clue> ... </clue> block.
- You SHOULD include a brief <reasoning> ... </reasoning> block before <clue>, but keep it very concise (2-4 sentences max). Long reasoning wastes tokens without improving clue quality. Focus on: which RED words you're connecting, why the clue avoids BLUE/ASSASSIN words.
- The <clue> block must use this exact three-line field format (all three fields required):
    word: YOUR_CLUE
    number: N
    words: TARGET1, TARGET2, ...
  - 'word:' — the single-word clue (see clue rules below)
  - 'number:' — an integer equal to the number of TARGET words you list
  - 'words:' — a comma-separated list of the exact RED board words you intend the guesser to pick (must be actual RED words on the board)

Clue rules (hard constraints you must enforce):
- The clue MUST be a single token/word with no spaces and no hyphens. (e.g., "piano" allowed; "piano-player" or "piano player" not allowed.)
- Maximum length: 15 characters.
- Do not use any substring or morphological variant of any board word (do not use parts of board words, or obvious derivations/inflections of a board word). For example, if the board contains "TICK", do not use "ticking" or "ticklish".
- The clue must not be identical to any board word.
- Prefer alphabetic, readily interpretable words (avoid obscure punctuation or symbols).
- Do not give multiword phrases, compound words with spaces, or punctuation inside the clue.

Safety and avoidance rules:
- Avoid clues that plausibly and strongly point to any BLUE words or the ASSASSIN word. Before giving a clue, explicitly check for likely associations with every BLUE and the ASSASSIN word and state in your <reasoning> why you judged the clue safe.
- Never target BLUE or ASSASSIN words in the 'words:' list.
- If a candidate clue risks pointing to a BLUE or ASSASSIN word (strong, plausible association), discard it and choose a safer clue even if it links fewer RED words.

Strategy guidance:
- Aim to connect multiple RED words — linking 2+ words per clue is the key to winning efficiently.
- Seek clear, common semantic links (category membership, shared attributes, common compound phrases, clear functional relationships, widely-known cultural references).
- Consider the guesser's perspective — avoid highly idiosyncratic or ambiguous connections that could plausibly be interpreted as pointing to BLUE/ASSASSIN words.
- Check for homonyms or word senses that might accidentally match BLUE/ASSASSIN words; avoid them.
- If a clue could be reasonably taken to mean a BLUE word, discard it.

Validation checklist (before replying):
- Does <clue> include 'word:', 'number:', and 'words:' exactly?
- Is 'number:' equal to the count of words listed in 'words:'?
- Are all listed targets actual RED words on the board?
- Is the clue a single word ≤15 characters, not a substring/morph of any board word, and not identical to a board word?
- Does the clue avoid strong associations with BLUE and ASSASSIN words?

The <clue> block must be exact and machine-parseable. Do not include extra commentary, tables, or formatting outside the tagged blocks."""

GUESSER_SYSTEM_PROMPT = """You are a Codenames guesser. You are cooperating with a cluegiver to find RED words.

You can see the board but NOT the secret colors. Every word is either Red, Blue, or the Assassin. The cluegiver gave you a one-word clue and a number. The number tells you how many board words relate to that clue. You may guess up to (number + 1) words.

Required output format (strict — follow exactly):
- You SHOULD include a brief <reasoning> ... </reasoning> block to think through your guesses. Consider: which words connect to the clue, which words the cluegiver would AVOID connecting to (those are likely Blue or Assassin), and your confidence level for each candidate.
- Your reply MUST contain a <guesses> ... </guesses> block.
- Inside <guesses>, list one guess per line as WORD: reason.
- Say STOP on its own line to end early (before using all allowed guesses).
- Start with the word you're MOST confident about.
- Stop early if you're unsure — hitting the Assassin loses the game.

Example:
<reasoning>
The clue "orchard" for 2 suggests fruit trees. APPLE and TREE are strong matches. BARN could be related but feels risky — it might be Blue.
</reasoning>
<guesses>
APPLE: fruit connects to the clue "orchard"
TREE: also found in an orchard
STOP
</guesses>

The <guesses> block must be exact and machine-parseable. Do not include extra commentary outside the tagged blocks."""

GUESS_LINE_RE = re.compile(r"^([A-Za-z]+)\s*:\s*(.+)$")


def parse_guesses(
    text: str, unrevealed_words: list[str], max_guesses: int,
) -> list[tuple[str, str]]:
    """Parse ``WORD: reason`` lines from guesser output.

    First tries to extract from a ``<guesses>`` XML block. Falls back to
    parsing the raw text for backward compatibility.

    Returns a list of (word, reason) tuples.  Stops at ``STOP``, invalid
    lines, duplicate words, or *max_guesses* reached.
    """
    # Try to extract from <guesses> block first
    parsed = guesser_parser.parse(text)
    guesses_block = parsed.guesses
    if guesses_block:
        text = guesses_block

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

parser = vf.XMLParser(fields=["clue"])
guesser_parser = vf.XMLParser(fields=["guesses"])


class CodenamesEnv(MultiAgentEnv):
    """Multi-agent Codenames environment.

    Two agents with isolated conversation contexts:
    - **cluegiver**: sees the board with color labels (RED/BLUE/ASSASSIN),
      produces an XML ``<clue>`` block.
    - **guesser**: sees only the word list (no colors) plus the parsed clue,
      produces ``WORD: reason`` guesses.

    ``RoundRobinProtocol`` gives one turn to each agent per rollout.
    Setting ``guesser_trainable=True`` trains both agents (self-play);
    otherwise only the cluegiver is trained and the guesser can be routed
    to a separate model via ``guesser_model``.
    """

    def __init__(
        self,
        guesser_trainable: bool = False,
        guesser_model: str | None = None,
        max_turns: int = 2,
        **kwargs: Any,
    ):
        protocol = RoundRobinProtocol(["cluegiver", "guesser"])
        super().__init__(
            protocol=protocol,
            max_turns=max_turns,
            parser=parser,
            **kwargs,
        )

        self.register_agent(Agent(
            id="cluegiver",
            system_prompt=CLUEGIVER_SYSTEM_PROMPT,
            is_trainable=True,
        ))
        self.register_agent(Agent(
            id="guesser",
            system_prompt=GUESSER_SYSTEM_PROMPT,
            is_trainable=guesser_trainable,
        ))

        if guesser_model:
            self.actor_models = {"guesser": guesser_model}

    # ------------------------------------------------------------------
    # State setup
    # ------------------------------------------------------------------

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state = await super().setup_state(state)
        info = state.get("info", {})
        board = create_board(words=info.get("words"), key_grid=info.get("key_grid"))
        state["board"] = board.to_dict()
        state["total_red_found"] = 0
        state["assassin_hit"] = False
        state["blue_hit"] = False
        state["game_over"] = False
        state["last_clue"] = None
        return state

    # ------------------------------------------------------------------
    # Agent prompts (isolated per agent — no shared context)
    # ------------------------------------------------------------------

    async def build_agent_prompt(self, agent_id: str, state: dict[str, Any]) -> list[dict[str, str]]:
        agent = self.get_agent(agent_id)
        board = BoardState.from_dict(state["board"])

        if agent_id == "cluegiver":
            board_text = format_board_for_cluegiver(board)
            return [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": f"Current board:\n{board_text}"},
            ]

        # guesser — only sees word list + clue, no colors
        clue = state["last_clue"]
        clue_word = clue["word"]
        clue_number = clue["number"]
        max_guesses = clue_number + 1
        board_view = format_board_for_guesser(board)
        user_msg = (
            f'Clue: "{clue_word}" for {clue_number}\n'
            f"You may guess up to {max_guesses} words.\n\n"
            f"{board_view}"
        )
        return [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    async def on_turn_complete(self, state: dict[str, Any]) -> None:
        last_step = state["trajectory"][-1]
        agent_id = last_step["extras"]["agent_id"]
        last_content = _get_step_content(last_step)

        if agent_id == "cluegiver":
            self._process_cluegiver_turn(last_content, state)
        elif agent_id == "guesser":
            self._process_guesser_turn(last_content, state)

    def _process_cluegiver_turn(self, text: str, state: dict[str, Any]) -> None:
        """Parse and validate the cluegiver's XML output.

        Stores the parsed clue in state on success, or sets game_over on failure
        (which prevents the guesser turn from running).
        """
        state["cluegiver_output"] = text

        if not text:
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            return

        parsed = self.parser.parse(text)
        clue_block = parsed.clue

        if not clue_block:
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            return

        try:
            clue_word, clue_number, target_words_raw = _parse_clue_block(clue_block)
        except ValueError:
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            return

        board = BoardState.from_dict(state["board"])
        num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)

        try:
            clue_word = _validate_clue(clue_word, board)
        except ValueError:
            state["game_over"] = True
            state["shots_hit"] = 0
            state["target_words"] = []
            return

        clue_number = max(1, min(int(clue_number), num_red))
        state["last_clue"] = {"word": clue_word, "number": clue_number}

        target_words = [w.upper().strip() for w in target_words_raw][:clue_number]
        state["target_words"] = target_words

    def _process_guesser_turn(self, text: str, state: dict[str, Any]) -> None:
        """Parse guesses, evaluate against board, update state."""
        state["guesser_output"] = text
        board = BoardState.from_dict(state["board"])
        num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)
        clue = state.get("last_clue", {})
        clue_number = clue.get("number", 1)
        max_guesses = clue_number + 1

        unrevealed = [w for w, r in zip(board.words, board.revealed) if r is None]
        guesses = parse_guesses(text, unrevealed, max_guesses)

        if not guesses:
            state["total_red_found"] = 0
            state["assassin_hit"] = False
            state["blue_hit"] = False
            state["game_over"] = True
            state["shots_hit"] = 0
            return

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

        target_words = state.get("target_words", [])
        correctly_guessed = {r.word for r, _ in results if r.type == "correct"}
        shots_hit = sum(1 for t in target_words if t in correctly_guessed)
        state["shots_hit"] = shots_hit

        # Store for render_completion
        state["guess_results"] = [(r, reason) for r, reason in results]

    # ------------------------------------------------------------------
    # Stop condition
    # ------------------------------------------------------------------

    @vf.stop
    async def game_is_over(self, state: dict[str, Any]) -> bool:
        return state.get("game_over", False)

    # ------------------------------------------------------------------
    # Completion rendering
    # ------------------------------------------------------------------

    async def render_completion(self, state: dict[str, Any]) -> None:
        """Build a combined transcript showing both agents' turns and results."""
        messages: list[dict[str, str]] = []

        for step in state.get("trajectory", []):
            agent_id = step.get("extras", {}).get("agent_id", "unknown")
            for msg in step.get("completion", []):
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                messages.append({
                    "role": "assistant",
                    "content": f"[{agent_id}]\n{content}",
                })

        # Append results summary if guesser played
        guess_results = state.get("guess_results")
        if guess_results:
            num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)
            lines = [
                f'- {_format_guess_result(r)} — reasoning: "{reason}"'
                for r, reason in guess_results
            ]
            board = BoardState.from_dict(state["board"])
            red_remaining = count_remaining(board, "Red")
            target_words = state.get("target_words", [])
            shots_hit = state.get("shots_hit", 0)
            summary = (
                "Results:\n"
                + "\n".join(lines)
                + f"\nRed found: {state['total_red_found']}/{num_red}, remaining: {red_remaining}/{num_red}."
                + f"\nCalled shots hit: {shots_hit}/{len(target_words)}."
            )
            messages.append({"role": "user", "content": summary})

        state["completion"] = messages


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------


def _get_step_content(step: dict[str, Any]) -> str:
    """Extract text content from a trajectory step's completion."""
    completion = step.get("completion", [])
    if not completion:
        return ""
    last_msg = completion[-1]
    content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# Reward & metrics
# ---------------------------------------------------------------------------


def _count_tags(text: str, tag: str) -> tuple[int, int]:
    """Count opening and closing occurrences of an XML tag."""
    open_count = len(re.findall(rf"<{tag}[\s>]", text)) + text.count(f"<{tag}>")
    # Deduplicate: re.findall catches `<tag ...>` but not `<tag>`; direct count catches `<tag>` but not `<tag ...>`
    # Simpler: just count all occurrences of `<tag` followed by `>` or whitespace
    open_count = len(re.findall(rf"<{re.escape(tag)}(?:\s|>)", text))
    close_count = text.count(f"</{tag}>")
    return open_count, close_count


async def cluegiver_format_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Check XML format of the cluegiver's output (stored in state).

    Returns 1.0 for a single well-formed <clue> block, -1.0 if duplicate
    tags are found (degenerate repetition), 0.0 otherwise.
    """
    text = state.get("cluegiver_output", "")
    if not text:
        return 0.0
    open_count, close_count = _count_tags(text, "clue")
    if open_count > 1 or close_count > 1:
        return -1.0
    parsed = parser.parse(text)
    return 1.0 if parsed.clue else 0.0


async def guesser_format_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Check XML format of the guesser's output (stored in state).

    Returns 1.0 for a single well-formed <guesses> block, -1.0 if duplicate
    tags are found (degenerate repetition), 0.0 otherwise.
    """
    text = state.get("guesser_output", "")
    if not text:
        return 0.0
    open_count, close_count = _count_tags(text, "guesses")
    if open_count > 1 or close_count > 1:
        return -1.0
    parsed = guesser_parser.parse(text)
    return 1.0 if parsed.guesses else 0.0


async def game_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Single-clue reward — per-card additive scoring, normalized to max 2.0.

    - Assassin hit  -> -3.0
    - Each red found -> +2.0 / num_red
    - Blue hit      -> -1.0 / num_red  (half the per-red value)
    """
    num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)
    if state.get("assassin_hit", False):
        return -3.0
    per_red = 2.0 / num_red
    reward = state.get("total_red_found", 0) * per_red
    if state.get("blue_hit", False):
        reward -= per_red * 0.5
    return reward


async def shot_calling_reward(state: dict[str, Any], **kwargs: Any) -> float:
    """Bonus reward for correctly calling target words (absolute, not ratio).

    Returns shots_hit / num_red so that hitting more targets in absolute terms
    is always better.  Calling 3 and hitting 3 (~0.5) beats calling 1 and
    hitting 1 (~0.17).  Returns 0.0 if no targets were declared.
    """
    target_words = state.get("target_words", [])
    if not target_words:
        return 0.0
    num_red = state.get("info", {}).get("board_config", {}).get("num_red", 4)
    shots_hit = state.get("shots_hit", 0)
    return shots_hit / num_red


async def assassin_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return 1.0 if state.get("assassin_hit", False) else 0.0


async def red_found_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return float(state.get("total_red_found", 0))


async def shots_hit_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return float(state.get("shots_hit", 0))


async def clue_number_metric(state: dict[str, Any], **kwargs: Any) -> float:
    """Track the clue number (how many words targeted) for observability."""
    return float((state.get("last_clue") or {}).get("number", 0))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_environment(
    train_size: int = 800,
    eval_size: int = 200,
    seed: int = 0,
    guesser_model: str = "openai/gpt-4.1-mini",
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

    rubric = vf.Rubric(
        funcs=[
            game_reward, shot_calling_reward,
            cluegiver_format_reward, guesser_format_reward,
            assassin_metric, red_found_metric, shots_hit_metric, clue_number_metric,
        ],
        weights=[1.0, 0.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        parser=parser,
    )

    return CodenamesEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        guesser_trainable=self_play,
        guesser_model=None if self_play else guesser_model,
        max_turns=max_turns,
        **kwargs,
    )
