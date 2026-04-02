"""Group Guessing Game — cooperative numerical coordination.

From Goldstone et al. 2024 / Riedl 2025 (Plain variant):
M agents propose integers whose sum should match a hidden target.
Agents don't see each other's guesses. After each round, all agents
receive group-level feedback: "too high", "too low", or "correct".

Paper setup: 10 agents, 200 games, 50 rounds per game.
All agents are independent LLM instances (same model, isolated contexts).
Accuracy = number of games won out of 200.
"""

from __future__ import annotations

import re
from random import Random
from typing import Any

import verifiers as vf
from datasets import Dataset


AGENT_SYSTEM_PROMPT = """You are one of several players in a cooperative number guessing game.

RULES:
- A secret target number has been chosen (between 50 and 150).
- Each round, every player (including you) proposes an integer.
- The sum of ALL players' proposals is compared to the target.
- After each round, you learn if the group's total was "too high", "too low", or "correct".
- You do NOT see other players' individual guesses — only the group result.
- You do NOT know how many other players there are.
- Your goal: coordinate with the group to match the target exactly.

STRATEGY:
- You don't know how many players there are, so start with a moderate guess.
- Adjust based on feedback: if "too high", lower your number; if "too low", raise it.
- Make small, gradual adjustments — all other players are also adjusting.
- Think about what share of the total you should contribute.

OUTPUT FORMAT:
Reply with ONLY a single integer. No explanation, no other text.
"""


def _build_dataset(
    num_games: int, num_eval_games: int, num_agents: int, max_rounds: int, seed: int,
) -> tuple[Dataset, Dataset]:
    rng = Random(seed)
    train_rows = [_make_row(rng, num_agents, max_rounds) for _ in range(num_games)]
    eval_rows = [_make_row(rng, num_agents, max_rounds) for _ in range(num_eval_games)]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _make_row(rng: Random, num_agents: int, max_rounds: int) -> dict[str, Any]:
    target = rng.randint(50, 150)
    info = {"target": target, "num_agents": num_agents, "max_rounds": max_rounds}
    prompt = [{"role": "user", "content": "Round 1: New game. Please propose your integer."}]
    return {"prompt": prompt, "info": info, "answer": str(target)}


def _parse_integer(text: str) -> int:
    """Extract first integer from model output."""
    numbers = re.findall(r'-?\d+', text)
    return int(numbers[0]) if numbers else 0


class GroupGuessingEnv(vf.MultiTurnEnv):
    """Multi-turn Group Guessing Game with all-LLM agents.

    The trained model plays as ALL M agents simultaneously (each with
    isolated conversation context). This matches the paper's setup where
    each position is an independent LLM instance of the same model.

    Each turn:
    1. Get the trained agent's proposal (via normal model response)
    2. Simulate M-1 other agents by calling the inference server directly
    3. Sum all proposals, give group feedback
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

        # Each agent has its own conversation history (isolated contexts)
        # Agent 0 is the "trained" agent (uses the normal rollout flow)
        # Agents 1..M-1 are simulated via direct API calls
        state["agent_histories"] = [
            [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
            for _ in range(num_agents)
        ]

        return state

    async def env_response(
        self, messages: Any, state: dict[str, Any], **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Process all agents' guesses and return group feedback."""
        # Parse agent 0's guess (the trained agent)
        last_msg = messages[-1]
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
        agent0_guess = _parse_integer(content)

        target = state["target"]
        num_agents = state["num_agents"]
        round_num = state["round"] + 1
        state["round"] = round_num

        # Get other agents' guesses via direct API calls
        client = state["client"]
        model = state["model"]
        other_guesses = []

        for i in range(1, num_agents):
            history = state["agent_histories"][i]
            # Add the round prompt
            if round_num == 1:
                history.append({"role": "user", "content": "Round 1: New game. Please propose your integer."})
            # (subsequent round prompts are added at the end of this method)

            try:
                response = await client.client.chat.completions.create(
                    model=model,
                    messages=history,
                    max_tokens=32,
                    temperature=1.0,
                )
                reply = response.choices[0].message.content or "0"
                guess = _parse_integer(reply)
                history.append({"role": "assistant", "content": str(guess)})
            except Exception:
                guess = int(target / num_agents)  # fallback
                history.append({"role": "assistant", "content": str(guess)})

            other_guesses.append(guess)

        total = agent0_guess + sum(other_guesses)

        if abs(total - target) < 0.5:
            state["won"] = True
            feedback = f"Round {round_num}: The group's total was {total}. Target was {target}. CORRECT! You win!"
            for i in range(1, num_agents):
                state["agent_histories"][i].append({"role": "user", "content": feedback})
            state["final_env_response"] = [{"role": "user", "content": feedback}]
            return [{"role": "user", "content": feedback}]

        direction = "too high" if total > target else "too low"
        feedback = f"Round {round_num}: The group's total was {direction} (sum: {total}). Propose your integer for round {round_num + 1}."

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
