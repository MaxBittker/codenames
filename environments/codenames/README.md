# codenames

Multi-agent `verifiers` environment for cooperative Codenames.

Two agents — a **cluegiver** and a **guesser** — cooperate to find RED words on a board. The cluegiver sees the full color key and gives a one-word clue linking target RED words. The guesser sees only the word list (no colors) and guesses based on the clue. Both agents are trained together.

Board size and color ratios are sampled randomly per game, controlled by the sampling config.

## Quickstart

```bash
prime env install codenames
prime eval run codenames -m openai/gpt-4.1-mini -n 20 -r 2
```

## Environment Args

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `train_size` | `int` | `800` | Number of generated train boards |
| `eval_size` | `int` | `200` | Number of generated eval boards |
| `seed` | `int` | `0` | Base seed for deterministic board generation |
| `max_turns` | `int` | `2` | Max turns per rollout (1 cluegiver + 1 guesser) |
| `min_board_size` | `int` | `4` | Minimum number of words on the board |
| `max_board_size` | `int` | `16` | Maximum number of words on the board |
| `min_red_ratio` | `float` | `0.3` | Minimum fraction of board words that are RED |
| `max_red_ratio` | `float` | `0.6` | Maximum fraction of board words that are RED |

## Reward

| Component | Weight | Description |
| --------- | ------ | ----------- |
| `game_reward` | 1.0 | Assassin hit: **-1.0**. Otherwise: `+2/num_red` per red found, `-1/num_red` for blue hit |
| `shot_calling_reward` | 0.5 | `shots_hit / num_red` — bonus for the guesser finding the cluegiver's declared targets |
| `cluegiver_format_reward` | 0.1 | 1.0 if cluegiver output contains a valid `<clue>` block |
| `guesser_format_reward` | 0.1 | 1.0 if guesser output contains a valid `<guesses>` block |

## Metrics

- `assassin_metric`: whether the assassin was hit
- `red_found_metric`: number of red words found
- `shots_hit_metric`: number of declared target words correctly guessed
- `clue_number_metric`: how many words the cluegiver targeted with the clue
