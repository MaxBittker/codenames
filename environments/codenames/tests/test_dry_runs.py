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
                mock_instance.guess.return_value = [(red_words[0], "seems related")]

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


    def test_board_config_validation_rejects_mismatch(self) -> None:
        from codenames.types import BoardConfig

        with self.assertRaises(ValueError):
            BoardConfig(board_size=10, num_red=3, num_blue=2, num_civilian=6, num_assassin=1)

    def test_easy_board_creation(self) -> None:
        from random import Random

        from codenames.game import create_board
        from codenames.types import DIFFICULTY_PRESETS

        config = DIFFICULTY_PRESETS["easy"]
        rng = Random(42)
        board = create_board(rng=rng, config=config)

        self.assertEqual(len(board.words), 6)
        self.assertEqual(len(board.key_grid), 6)
        self.assertEqual(len(board.revealed), 6)
        self.assertEqual(board.key_grid.count("Red"), 3)
        self.assertEqual(board.key_grid.count("Blue"), 1)
        self.assertEqual(board.key_grid.count("Civilian"), 1)
        self.assertEqual(board.key_grid.count("Assassin"), 1)

    def test_reward_normalization_all_reds_equals_two(self) -> None:
        """Finding all reds should yield reward 2.0 for any board config."""

        async def run() -> None:
            from codenames.codenames import game_reward

            for num_red in (3, 5, 8):
                state = {
                    "total_red_found": num_red,
                    "assassin_hit": False,
                    "blue_hit": False,
                    "info": {"board_config": {"num_red": num_red}},
                }
                reward = await game_reward(state=state)
                self.assertAlmostEqual(reward, 2.0, msg=f"num_red={num_red}")

        asyncio.run(run())

    def test_reward_backward_compat_no_board_config(self) -> None:
        """Old rows without board_config should default to num_red=8."""

        async def run() -> None:
            from codenames.codenames import game_reward

            state = {
                "total_red_found": 2,
                "assassin_hit": False,
                "blue_hit": False,
                "info": {},
            }
            reward = await game_reward(state=state)
            self.assertAlmostEqual(reward, 0.5)  # 2 * 0.25

        asyncio.run(run())

    def test_make_row_easy_board(self) -> None:
        from codenames.codenames import _make_row
        from codenames.types import DIFFICULTY_PRESETS

        config = DIFFICULTY_PRESETS["easy"]
        row = _make_row(seed=0, split="train", config=config)
        info = row["info"]

        self.assertIn("board_config", info)
        self.assertEqual(info["board_config"]["num_red"], 3)
        self.assertEqual(len(info["words"]), 6)
        self.assertEqual(len(info["key_grid"]), 6)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
