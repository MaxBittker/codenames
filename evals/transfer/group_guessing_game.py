"""Group Guessing Game — cooperative numerical coordination.

From Goldstone et al. 2024 / Riedl 2025 (Plain variant):
M agents propose integers (0-50 each) whose sum should match a hidden target.
Agents don't see each other's guesses. After each round, all agents
receive group-level feedback: "too high" or "too low" (not the actual sum).

Paper setup (Riedl 2025, Plain variant):
- 10 agents, each guesses 0-50
- Target: random integer (implicitly 0-500, sum of 10 agents' ranges)
- Feedback: only "too high" / "too low" (no actual sum revealed)
- Prompt includes binary search strategy hint
- 200 games, evaluated twice
- Accuracy = number of games won out of 200

Reference: https://arxiv.org/abs/2510.05174
"""

from __future__ import annotations

import re
from random import Random
from typing import Any

import verifiers as vf
from datasets import Dataset


# Matches the Plain variant prompt from Riedl 2025
AGENT_SYSTEM_PROMPT = """You are playing a sum guessing game. Your goal is to help your group sum to the mystery number. Your guess range is 0 to 50. Always start with the efficient strategy in guessing games which is to use a binary search approach: guessing the midpoint of the current range. Always anchor your guess on the group feedback from previous rounds (too high / too low). End your answer with: FINAL GUESS: [0-50]"""


def _build_dataset(
    num_games: int, num_eval_games: int, num_agents: int, max_rounds: int, seed: int,
) -> tuple[Dataset, Dataset]:
    rng = Random(seed)
    train_rows = [_make_row(rng, num_agents, max_rounds) for _ in range(num_games)]
    eval_rows = [_make_row(rng, num_agents, max_rounds) for _ in range(num_eval_games)]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _make_row(rng: Random, num_agents: int, max_rounds: int) -> dict[str, Any]:
    # Target is a random number in the range of possible sums (0 to num_agents * 50)
    target = rng.randint(0, num_agents * 50)
    info = {"target": target, "num_agents": num_agents, "max_rounds": max_rounds}
    prompt = [{"role": "user", "content": "Round 1. Please make your guess."}]
    return {"prompt": prompt, "info": info, "answer": str(target)}


def _parse_guess(text: str) -> int:
    """Extract guess from 'FINAL GUESS: N' format, falling back to last integer."""
    match = re.search(r'FINAL GUESS:\s*(\d+)', text)
    if match:
        return min(50, max(0, int(match.group(1))))
    numbers = re.findall(r'\d+', text)
    if numbers:
        return min(50, max(0, int(numbers[-1])))
    return 25  # midpoint fallback


class GroupGuessingEnv(vf.MultiTurnEnv):
    """Multi-turn Group Guessing Game with all-LLM agents.

    The trained model plays as ALL M agents simultaneously (each with
    isolated conversation context). This matches the paper's setup where
    each position is an independent LLM instance of the same model.
    """

    def __init__(self, num_agents: int = 10, max_rounds: int = 50, **kwargs: Any):
        super().__init__(max_turns=max_rounds, **kwargs)
        self.num_agents = num_agents

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state = await super().setup_state(state)
        info = state.get("info", {})
        target = info["target"]
        num_agents = info.get("num_agents", self.num_agents)

        state["target"] = target
        state["num_agents"] = num_agents
        state["round"] = 0
        state["won"] = False

        state["agent_histories"] = [
            [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
            for _ in range(num_agents)
        ]

        return state

    async def env_response(
        self, messages: Any, state: dict[str, Any], **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Process all agents' guesses and return group feedback."""
        last_msg = messages[-1]
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
        agent0_guess = _parse_guess(content)

        target = state["target"]
        num_agents = state["num_agents"]
        round_num = state["round"] + 1
        state["round"] = round_num

        client = state["client"]
        model = state["model"]
        other_guesses = []

        for i in range(1, num_agents):
            history = state["agent_histories"][i]
            if round_num == 1:
                history.append({"role": "user", "content": "Round 1. Please make your guess."})

            try:
                response = await client.client.chat.completions.create(
                    model=model,
                    messages=history,
                    max_tokens=128,
                    temperature=1.0,
                )
                reply = response.choices[0].message.content or "25"
                guess = _parse_guess(reply)
                history.append({"role": "assistant", "content": reply})
            except Exception:
                guess = 25
                history.append({"role": "assistant", "content": "FINAL GUESS: 25"})

            other_guesses.append(guess)

        total = agent0_guess + sum(other_guesses)

        if total == target:
            state["won"] = True
            feedback = f"Round {round_num}: Correct! Your group's sum matched the target."
            for i in range(1, num_agents):
                state["agent_histories"][i].append({"role": "user", "content": feedback})
            state["final_env_response"] = [{"role": "user", "content": feedback}]
            return [{"role": "user", "content": feedback}]

        # Only "too high" or "too low" — do NOT reveal the actual sum
        direction = "too high" if total > target else "too low"
        feedback = f"Round {round_num}: Your group's sum was {direction}. Please make your guess for round {round_num + 1}."

        for i in range(1, num_agents):
            state["agent_histories"][i].append({"role": "user", "content": feedback})

        return [{"role": "user", "content": feedback}]

    @vf.stop
    async def game_won(self, state: dict[str, Any]) -> bool:
        return state.get("won", False)


async def game_reward(state: dict[str, Any], **kwargs: Any) -> float:
    return 1.0 if state.get("won", False) else 0.0


async def rounds_used_metric(state: dict[str, Any], **kwargs: Any) -> float:
    return float(state.get("round", 0))


def load_environment(
    num_games: int = 200,
    num_eval_games: int = 200,
    num_agents: int = 10,
    max_rounds: int = 50,
    seed: int = 42,
    **kwargs: Any,
) -> vf.Environment:
    dataset, eval_dataset = _build_dataset(
        num_games=num_games, num_eval_games=num_eval_games,
        num_agents=num_agents, max_rounds=max_rounds, seed=seed,
    )

    rubric = vf.Rubric(
        funcs=[game_reward, rounds_used_metric],
        weights=[1.0, 0.0],
    )

    return GroupGuessingEnv(
        dataset=dataset, eval_dataset=eval_dataset,
        system_prompt=AGENT_SYSTEM_PROMPT,
        rubric=rubric, num_agents=num_agents, max_rounds=max_rounds,
        **kwargs,
    )
