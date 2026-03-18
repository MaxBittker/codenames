from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch


class CodenamesDryRunTests(unittest.TestCase):
    def test_cluegiver_env_dry_run(self) -> None:
        async def run() -> None:
            from codenames.codenames import (
                CodenamesCluegiverEnv,
                assassin_metric,
                game_reward,
                red_found_metric,
            )
            from codenames.game import create_board, format_board_for_cluegiver

            with patch("codenames.codenames.LLMGuesser") as MockGuesser:
                # Set up mock guesser that returns board words
                mock_instance = MockGuesser.return_value
                mock_instance.guess = AsyncMock()

                from datasets import Dataset

                dummy_ds = Dataset.from_list(
                    [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
                )
                env = CodenamesCluegiverEnv(
                    dataset=dummy_ds,
                    eval_dataset=dummy_ds,
                    rubric=None,
                )

                # Build a state with a known board
                from random import Random

                rng = Random(5)
                board = create_board(rng=rng)

                # Find first red word on the board
                red_words = [
                    w for w, c in zip(board.words, board.key_grid) if c == "Red"
                ]
                # Mock guesser returns first red word
                mock_instance.guess.return_value = [red_words[0]]

                state = {
                    "info": {
                        "words": board.words,
                        "key_grid": board.key_grid,
                    },
                    "timing": {
                        "generation_ms": 0.0,
                        "scoring_ms": 0.0,
                        "total_ms": 0.0,
                    },
                }
                state = await env.setup_state(state)

                # Directly call give_clue with a valid clue
                args = env.update_tool_args(
                    "give_clue",
                    {"word": "TESTING", "number": 1},
                    [],
                    state,
                )
                transcript = await env.give_clue(**args)

                self.assertTrue(state["game_over"])
                self.assertIn("Red found:", transcript)

                reward = await game_reward(state=state)
                assassin = await assassin_metric(state=state)
                red_found = await red_found_metric(state=state)

                self.assertIsInstance(reward, float)
                self.assertGreater(reward, 0.0)
                self.assertEqual(assassin, 0.0)
                self.assertEqual(red_found, 1.0)

        asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(unittest.main())
