# codenames

PrimeIntellect `verifiers` environment for cooperative Codenames.

The model gives a single clue via the `give_clue` tool. An LLM guesser
(default: `openai/gpt-5.4-nano`) resolves the turn, and the reward reflects how
many RED words the guesser found from that one clue.

## Quickstart

```bash
uv pip install -e ./environments/codenames
prime eval run codenames -m openai/gpt-5.4-nano -n 20 -r 2
```

## Environment Args

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `train_size` | `int` | `800` | Number of generated train boards |
| `eval_size` | `int` | `200` | Number of generated eval boards |
| `seed` | `int` | `0` | Base seed for deterministic board generation |
| `guesser_model` | `str` | `openai/gpt-5.4-nano` | Model used as the simulated guesser |
| `guesser_api_base` | `str \| None` | `None` | API base URL for the guesser (defaults to prime config) |
| `guesser_api_key` | `str \| None` | `None` | API key for the guesser (defaults to prime config) |
| `max_turns` | `int` | `2` | Max turns per rollout (2 needed: clue + tool execution) |

## Reward

- Assassin hit → **-1.0**
- Blue hit → `red_found / 8 - 0.125`
- Otherwise → `red_found / 8`

## Metrics

- `game_reward`: main scalar reward
- `assassin_metric`: whether the assassin was hit
- `red_found_metric`: number of red words found
