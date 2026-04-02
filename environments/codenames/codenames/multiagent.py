"""Vendored multi-agent classes from verifiers (multiagent-no-opponent-conditioning branch).

These classes are not yet in the standard verifiers release. Once they are,
this module can be replaced with imports from verifiers directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.utils.message_utils import normalize_messages


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """A participant in a multi-agent environment."""

    id: str
    system_prompt: str = ""
    is_trainable: bool = True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Agent):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.is_trainable else "frozen"
        return f"Agent(id={self.id!r}, {trainable_str})"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Protocol(ABC):
    """Abstract base class for multi-agent interaction protocols."""

    @abstractmethod
    def get_initial_agent(self, state: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_next_agent(self, state: dict[str, Any]) -> str:
        pass


class RoundRobinProtocol(Protocol):
    """Simple round-robin turn order: agents take turns in sequence."""

    def __init__(self, agent_ids: list[str]):
        if not agent_ids:
            raise ValueError("agent_ids must not be empty")
        self.agent_ids = agent_ids

    def get_initial_agent(self, state: dict[str, Any]) -> str:
        return self.agent_ids[0]

    def get_next_agent(self, state: dict[str, Any]) -> str:
        current = state["extras"].get("current_agent_id", self.agent_ids[0])
        try:
            current_idx = self.agent_ids.index(current)
        except ValueError:
            current_idx = -1
        next_idx = (current_idx + 1) % len(self.agent_ids)
        return self.agent_ids[next_idx]


# ---------------------------------------------------------------------------
# MultiAgentEnv
# ---------------------------------------------------------------------------


class MultiAgentEnv(MultiTurnEnv):
    """Base class for multi-agent environments.

    Turn order is managed by a Protocol. Each agent gets its own isolated
    conversation context built by build_agent_prompt().
    """

    agents: list[str] = []
    actor_models: dict[str, str] | None = None

    def __init__(self, protocol: Protocol, **kwargs: Any):
        super().__init__(**kwargs)
        self._protocol = protocol
        self._agent_registry: dict[str, Agent] = {}

    def register_agent(self, agent: Agent) -> None:
        self._agent_registry[agent.id] = agent
        if agent.id not in self.agents:
            self.agents.append(agent.id)

    def get_agent(self, agent_id: str) -> Agent:
        if agent_id not in self._agent_registry:
            raise KeyError(
                f"Agent '{agent_id}' not found. Did you call register_agent()?"
            )
        return self._agent_registry[agent_id]

    def get_initial_agent(self, state: dict[str, Any]) -> str:
        return self._protocol.get_initial_agent(state)

    def get_next_agent(self, state: dict[str, Any]) -> str:
        return self._protocol.get_next_agent(state)

    @abstractmethod
    async def build_agent_prompt(self, agent_id: str, state: dict[str, Any]) -> list[dict[str, str]]:
        pass

    async def on_turn_complete(self, state: dict[str, Any]) -> None:
        pass

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state = await super().setup_state(state)
        state["extras"] = state.get("extras", {})
        state["extras"]["current_agent_id"] = None
        return state

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs: Any) -> Any:
        return []

    async def add_trajectory_step(self, state: dict[str, Any], trajectory_step: dict[str, Any]) -> None:
        current_agent_id = state["extras"].get("current_agent_id")
        if current_agent_id:
            trajectory_step["extras"]["agent_id"] = current_agent_id
            agent = self.get_agent(current_agent_id)
            trajectory_step["extras"]["is_trainable"] = agent.is_trainable
        await super().add_trajectory_step(state, trajectory_step)

    async def rollout(self, input, client, model, sampling_args=None):
        state = await self.init_state(input, client, model, sampling_args)
        try:
            state = await self.setup_state(state)
        except vf.Error as e:
            state["error"] = e
            return state

        state["extras"]["current_agent_id"] = self.get_initial_agent(state)

        while not await self.is_completed(state):
            agent_id = state["extras"]["current_agent_id"]

            try:
                prompt_messages = await self.build_agent_prompt(agent_id, state)
                prompt_messages = normalize_messages(
                    prompt_messages, field_name="agent_prompt"
                )

                agent_model = (
                    self.actor_models.get(agent_id) if self.actor_models else None
                )

                # Force TITO fallback: each agent gets an isolated prompt
                # with no relation to previous turns, so token stitching
                # would produce garbage. Temporarily clear the trajectory
                # so the TITO client sees len(trajectory)==0 and falls
                # back to standard chat completions.
                saved_trajectory = state.get("trajectory", [])
                state["trajectory"] = []
                try:
                    response = await self.get_model_response(
                        state, prompt_messages, model=agent_model
                    )
                finally:
                    state["trajectory"] = saved_trajectory

                await self.add_model_response(state, prompt_messages, response)
                await self.on_turn_complete(state)

                if not await self.is_completed(state):
                    state["extras"]["current_agent_id"] = self.get_next_agent(state)

            except vf.OverlongPromptError:
                state["prompt_too_long"] = True
                state["is_truncated"] = True
                break
            except vf.Error as e:
                state["error"] = e
                break

        await self.render_completion(state)
        return state
