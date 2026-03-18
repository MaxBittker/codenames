# codenames

PrimeIntellect `verifiers` environment for cooperative Codenames.

See [environments/codenames](environments/codenames) for the environment package.

## Local usage

```bash
uv pip install -e ./environments/codenames
prime eval run codenames -m openai/gpt-5.4-nano -n 20 -r 2
```

Training config lives at [configs/codenames/rl.toml](configs/codenames/rl.toml).
