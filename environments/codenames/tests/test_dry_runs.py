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

                # Simulate a model response with XML output
                xml_response = (
                    f"<reasoning>\nTESTING is a good clue for {red_words[0]}.\n</reasoning>\n"
                    f"<clue>\nword: TESTING\nnumber: 1\nwords: {red_words[0]}\n</clue>"
                )
                messages = [
                    {"role": "user", "content": "Current board:\n..."},
                    {"role": "assistant", "content": xml_response},
                ]

                result = await env.env_response(messages, state)

                self.assertTrue(state["game_over"])
                # result is a list of message dicts
                self.assertIsInstance(result, list)
                transcript = result[0]["content"]
                self.assertIn("Red found:", transcript)
                self.assertIn("Called shots hit:", transcript)

                reward = await game_reward(state=state)
                assassin = await assassin_metric(state=state)
                red_found = await red_found_metric(state=state)

                self.assertIsInstance(reward, float)
                self.assertGreater(reward, 0.0)
                self.assertEqual(assassin, 0.0)
                self.assertEqual(red_found, 1.0)

                # Shot calling: we targeted the word the mock guesser returns
                self.assertEqual(state["shots_hit"], 1)
                self.assertEqual(state["target_words"], [red_words[0]])

        asyncio.run(run())

    def test_env_response_missing_clue_block(self) -> None:
        """env_response handles missing <clue> block gracefully."""

        async def run() -> None:
            from codenames.codenames import CodenamesCluegiverEnv
            from codenames.game import create_board

            with patch("codenames.codenames.LLMGuesser"):
                from datasets import Dataset

                dummy_ds = Dataset.from_list(
                    [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
                )
                env = CodenamesCluegiverEnv(
                    dataset=dummy_ds,
                    eval_dataset=dummy_ds,
                    rubric=None,
                )

                from random import Random

                rng = Random(5)
                board = create_board(rng=rng)

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

                # Model response without proper XML
                messages = [
                    {"role": "user", "content": "Current board:\n..."},
                    {"role": "assistant", "content": "I think EAGLE is a good clue."},
                ]

                result = await env.env_response(messages, state)

                self.assertTrue(state["game_over"])
                self.assertIn("<clue>", result[0]["content"])

        asyncio.run(run())

    def test_board_config_validation_rejects_mismatch(self) -> None:
        from codenames.types import BoardConfig

        with self.assertRaises(ValueError):
            BoardConfig(board_size=10, num_red=3, num_blue=2, num_assassin=1)

    def test_sampling_config_produces_valid_boards(self) -> None:
        from random import Random

        from codenames.game import create_board
        from codenames.types import BoardSamplingConfig

        sampling = BoardSamplingConfig(
            min_board_size=4, max_board_size=16,
            min_red_ratio=0.3, max_red_ratio=0.6,
        )
        for seed in range(50):
            rng = Random(seed)
            config = sampling.sample(rng)
            board = create_board(rng=rng, config=config)

            self.assertGreaterEqual(config.board_size, 4)
            self.assertLessEqual(config.board_size, 16)
            self.assertGreaterEqual(config.num_red, 2)
            self.assertEqual(config.num_assassin, 1)
            self.assertEqual(len(board.words), config.board_size)
            self.assertEqual(len(board.key_grid), config.board_size)
            self.assertEqual(board.key_grid.count("Red"), config.num_red)
            self.assertEqual(board.key_grid.count("Blue"), config.num_blue)
            self.assertEqual(board.key_grid.count("Assassin"), 1)

    def test_reward_normalization_all_reds_equals_two(self) -> None:
        """Finding all reds should yield reward 2.0 for any board config."""

        async def run() -> None:
            from codenames.codenames import game_reward

            for num_red in (2, 5, 10):
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
        """Old rows without board_config should default to num_red=4."""

        async def run() -> None:
            from codenames.codenames import game_reward

            state = {
                "total_red_found": 2,
                "assassin_hit": False,
                "blue_hit": False,
                "info": {},
            }
            reward = await game_reward(state=state)
            self.assertAlmostEqual(reward, 1.0)  # 2 * (2.0/4)

        asyncio.run(run())

    def test_make_row_with_sampling(self) -> None:
        from codenames.codenames import _make_row
        from codenames.types import BoardSamplingConfig

        sampling = BoardSamplingConfig(min_board_size=6, max_board_size=10)
        row = _make_row(seed=0, split="train", sampling=sampling)
        info = row["info"]

        self.assertIn("board_config", info)
        self.assertGreaterEqual(info["board_config"]["num_red"], 2)
        board_size = info["board_config"]["board_size"]
        self.assertGreaterEqual(board_size, 6)
        self.assertLessEqual(board_size, 10)
        self.assertEqual(len(info["words"]), board_size)
        self.assertEqual(len(info["key_grid"]), board_size)

    def test_parse_clue_block(self) -> None:
        from codenames.codenames import _parse_clue_block

        text = "word: PREDATOR\nnumber: 2\nwords: EAGLE, HAWK"
        word, number, words = _parse_clue_block(text)
        self.assertEqual(word, "PREDATOR")
        self.assertEqual(number, 2)
        self.assertEqual(words, ["EAGLE", "HAWK"])

    def test_parse_clue_block_missing_field(self) -> None:
        from codenames.codenames import _parse_clue_block

        with self.assertRaises(ValueError):
            _parse_clue_block("word: PREDATOR\nnumber: 2")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
