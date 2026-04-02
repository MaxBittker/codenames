from __future__ import annotations

import asyncio
import unittest


class CodenamesDryRunTests(unittest.TestCase):
    def test_single_round_red_found_game_continues(self) -> None:
        """After finding one red, game should NOT be over if reds remain."""

        async def run() -> None:
            from codenames.codenames import CodenamesEnv, game_reward, red_found_metric
            from codenames.game import create_board
            from codenames.types import BoardConfig

            from datasets import Dataset

            dummy_ds = Dataset.from_list(
                [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
            )
            env = CodenamesEnv(
                dataset=dummy_ds,
                eval_dataset=dummy_ds,
                rubric=None,
            )

            from random import Random

            rng = Random(5)
            config = BoardConfig(board_size=8, num_red=4, num_blue=3, num_assassin=1)
            board = create_board(rng=rng, config=config)

            red_words = [
                w for w, c in zip(board.words, board.key_grid) if c == "Red"
            ]

            state = {
                "info": {
                    "words": board.words,
                    "key_grid": board.key_grid,
                    "board_config": config.to_dict(),
                },
                "extras": {},
            }
            state = await env.setup_state(state)

            # Round 1: cluegiver targets first red word
            xml_response = (
                f"<reasoning>\nTESTING is a good clue for {red_words[0]}.\n</reasoning>\n"
                f"<clue>\nword: TESTING\nnumber: 1\nwords: {red_words[0]}\n</clue>"
            )
            env._process_cluegiver_turn(xml_response, state)

            self.assertFalse(state["game_over"])
            self.assertEqual(state["last_clue"]["word"], "TESTING")
            self.assertEqual(state["last_clue"]["number"], 1)

            # Guesser guesses the target red word
            guesser_response = f"<guesses>\n{red_words[0]}: seems related\nSTOP\n</guesses>"
            env._process_guesser_turn(guesser_response, state)

            # Game should NOT be over — 3 reds remaining
            self.assertFalse(state["game_over"])
            self.assertEqual(state["total_red_found"], 1)
            self.assertFalse(state["assassin_hit"])

        asyncio.run(run())

    def test_full_game_win(self) -> None:
        """Find all reds across multiple rounds — game should end with win."""

        async def run() -> None:
            from codenames.codenames import (
                CodenamesEnv,
                game_reward,
                win_metric,
                efficiency_reward,
            )
            from codenames.game import create_board
            from codenames.types import BoardConfig

            from datasets import Dataset

            dummy_ds = Dataset.from_list(
                [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
            )
            env = CodenamesEnv(
                dataset=dummy_ds,
                eval_dataset=dummy_ds,
                rubric=None,
            )

            from random import Random

            rng = Random(42)
            config = BoardConfig(board_size=4, num_red=3, num_blue=0, num_assassin=1)
            board = create_board(rng=rng, config=config)

            red_words = [
                w for w, c in zip(board.words, board.key_grid) if c == "Red"
            ]

            state = {
                "info": {
                    "words": board.words,
                    "key_grid": board.key_grid,
                    "board_config": config.to_dict(),
                },
                "extras": {},
            }
            state = await env.setup_state(state)

            # Round 1: target 2 reds
            xml1 = (
                f"<clue>\nword: LINK\nnumber: 2\nwords: {red_words[0]}, {red_words[1]}\n</clue>"
            )
            env._process_cluegiver_turn(xml1, state)
            self.assertFalse(state["game_over"])

            guess1 = f"<guesses>\n{red_words[0]}: first\n{red_words[1]}: second\nSTOP\n</guesses>"
            env._process_guesser_turn(guess1, state)

            self.assertFalse(state["game_over"])
            self.assertEqual(state["total_red_found"], 2)

            # Round 2: target last red
            xml2 = (
                f"<clue>\nword: FINAL\nnumber: 1\nwords: {red_words[2]}\n</clue>"
            )
            env._process_cluegiver_turn(xml2, state)
            self.assertFalse(state["game_over"])

            guess2 = f"<guesses>\n{red_words[2]}: last one\nSTOP\n</guesses>"
            env._process_guesser_turn(guess2, state)

            # All reds found — game over, win
            self.assertTrue(state["game_over"])
            self.assertEqual(state["total_red_found"], 3)
            self.assertFalse(state["assassin_hit"])

            reward = await game_reward(state=state)
            self.assertAlmostEqual(reward, 2.0)

            win = await win_metric(state=state)
            self.assertEqual(win, 1.0)

            eff = await efficiency_reward(state=state)
            self.assertGreater(eff, 0.0)  # won in 2 rounds for 3 reds

        asyncio.run(run())

    def test_assassin_ends_game(self) -> None:
        """Hitting the assassin should end the game immediately."""

        async def run() -> None:
            from codenames.codenames import CodenamesEnv, game_reward, assassin_metric
            from codenames.game import create_board
            from codenames.types import BoardConfig

            from datasets import Dataset

            dummy_ds = Dataset.from_list(
                [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
            )
            env = CodenamesEnv(
                dataset=dummy_ds,
                eval_dataset=dummy_ds,
                rubric=None,
            )

            from random import Random

            rng = Random(7)
            config = BoardConfig(board_size=4, num_red=3, num_blue=0, num_assassin=1)
            board = create_board(rng=rng, config=config)

            assassin_word = [
                w for w, c in zip(board.words, board.key_grid) if c == "Assassin"
            ][0]
            red_words = [
                w for w, c in zip(board.words, board.key_grid) if c == "Red"
            ]

            state = {
                "info": {
                    "words": board.words,
                    "key_grid": board.key_grid,
                    "board_config": config.to_dict(),
                },
                "extras": {},
            }
            state = await env.setup_state(state)

            xml = (
                f"<clue>\nword: BAD\nnumber: 1\nwords: {red_words[0]}\n</clue>"
            )
            env._process_cluegiver_turn(xml, state)

            # Guesser hits the assassin
            guess = f"<guesses>\n{assassin_word}: oops\n</guesses>"
            env._process_guesser_turn(guess, state)

            self.assertTrue(state["game_over"])
            self.assertTrue(state["assassin_hit"])

            reward = await game_reward(state=state)
            self.assertEqual(reward, -3.0)

            assassin = await assassin_metric(state=state)
            self.assertEqual(assassin, 1.0)

        asyncio.run(run())

    def test_blue_hit_continues_game(self) -> None:
        """Guessing a blue word ends the turn but the game continues."""

        async def run() -> None:
            from codenames.codenames import CodenamesEnv, game_reward, blue_hit_metric
            from codenames.game import create_board
            from codenames.types import BoardConfig

            from datasets import Dataset

            dummy_ds = Dataset.from_list(
                [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
            )
            env = CodenamesEnv(
                dataset=dummy_ds,
                eval_dataset=dummy_ds,
                rubric=None,
            )

            from random import Random

            rng = Random(10)
            config = BoardConfig(board_size=8, num_red=4, num_blue=3, num_assassin=1)
            board = create_board(rng=rng, config=config)

            blue_word = [
                w for w, c in zip(board.words, board.key_grid) if c == "Blue"
            ][0]
            red_words = [
                w for w, c in zip(board.words, board.key_grid) if c == "Red"
            ]

            state = {
                "info": {
                    "words": board.words,
                    "key_grid": board.key_grid,
                    "board_config": config.to_dict(),
                },
                "extras": {},
            }
            state = await env.setup_state(state)

            xml = (
                f"<clue>\nword: OOPS\nnumber: 1\nwords: {red_words[0]}\n</clue>"
            )
            env._process_cluegiver_turn(xml, state)

            # Guesser hits a blue word
            guess = f"<guesses>\n{blue_word}: wrong guess\n</guesses>"
            env._process_guesser_turn(guess, state)

            # Game should continue — blue hit is not fatal
            self.assertFalse(state["game_over"])
            self.assertFalse(state["assassin_hit"])
            self.assertEqual(state["blue_hit_count"], 1)

            blue_hits = await blue_hit_metric(state=state)
            self.assertEqual(blue_hits, 1.0)

            # Reward should have a penalty for blue hit
            reward = await game_reward(state=state)
            self.assertLess(reward, 0.0)  # 0 reds found, -0.5*per_red penalty

        asyncio.run(run())

    def test_cluegiver_missing_clue_block(self) -> None:
        """_process_cluegiver_turn handles missing <clue> block gracefully."""

        async def run() -> None:
            from codenames.codenames import CodenamesEnv
            from codenames.game import create_board

            from datasets import Dataset

            dummy_ds = Dataset.from_list(
                [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
            )
            env = CodenamesEnv(
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
                "extras": {},
            }
            state = await env.setup_state(state)

            env._process_cluegiver_turn("I think EAGLE is a good clue.", state)

            self.assertTrue(state["game_over"])
            self.assertEqual(state["target_words"], [])

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
