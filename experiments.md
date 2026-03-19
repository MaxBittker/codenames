# Codenames RL Experiments

## Results So Far

| Model | Guesser | Status | Notes |
|-------|---------|--------|-------|
| Qwen3-4B-Instruct | self-play (4B) | collapsed step 3 | ModelErrors, death spiral |
| Qwen3-4B-Instruct | gpt-5.4-nano | collapsed step 5 | same collapse pattern |
| Qwen3-4B-Thinking | self-play (4B) | flat after 66 steps | stable but no learning |
| Qwen3-4B-Thinking | gpt-5.4-nano | stopped early | swapped for 30B |
| Qwen3-30B-A3B-Instruct | gpt-5.4-nano | running | stable, reward improving |
| Qwen3-30B-A3B-Thinking | gpt-5.4-nano | running | baseline eval 0.15 |

Key finding: 4B instruct too small to maintain tool-calling under RL. 30B MoE much more stable.

---

## Experiment Ideas

### 1. Remove tool calling — use XML format instead
The wordle environment uses `vf.XMLParser` with `<guess>` tags instead of tool calls. Codenames could do the same with a `<clue>` and `<number>` tag. This would:
- Eliminate ModelError collapse from malformed tool calls (the main 4B failure mode)
- Allow smaller models to train more stably
- Add a format reward component (XMLParser provides `get_format_reward_func()`)
- Simplify the environment — no `StatefulToolEnv`, `update_tool_args`, or `state_token` plumbing

Example model output:
```
<reasoning>ROCKET and ENGINE both relate to propulsion...</reasoning>
<clue>THRUST</clue>
<number>2</number>
```

### 2. Multi-turn clue-giving (multiple rounds per game)
Currently the env is single-clue: give one clue, guesser guesses, game ends. A multi-turn version would let the cluegiver give multiple clues across rounds, seeing which words were revealed after each round. This would:
- Create a richer learning signal (strategic sequencing of clues)
- Reward long-horizon planning (save easy clues for later, do hard combos first)
- Better match the actual Codenames game
- Need `max_turns` increased and `game_over` logic updated to continue until all reds found or assassin hit

### 3. Difficulty curriculum
Start training on easier boards and ramp up:
- Fewer total words (e.g. 9 instead of 25)
- Fewer red words to find (e.g. 3 instead of 8)
- No assassin initially
- Gradually increase board complexity as reward improves
- Could use `BoardConfig` presets or the `difficulty` param in `load_environment()`

### 4. Reward shaping refinements
Current reward: +2/num_red per red found, -2/num_red for blue, -1 for assassin. Ideas:
- Reward for clue number attempted (encourage multi-word clues)
- Bonus for connecting 3+ words with one clue
- Penalty scaling by how "dangerous" the clue was (proximity to assassin/blue words)
- Partial credit for guesses that were close but wrong color

### 5. System prompt optimization with GEPA
Run GEPA on the cluegiver system prompt before RL training to find a better starting prompt. A stronger baseline prompt = better starting policy = more stable RL.
- Config already exists at `configs/gepa/`
- Optimize for clue quality metrics before starting gradient training

### 6. Guesser model ablation
Compare fixed guesser models to understand sensitivity:
- gpt-5-nano (cheap, fast)
- gpt-4.1-mini (stronger)
- qwen3-8b via Prime (mid-tier, no extra API key needed)
- Does a smarter guesser make training easier (clearer signal) or harder (less room for improvement)?

### 7. Self-play with frozen guesser checkpoint
Instead of full self-play (guesser degrades with cluegiver), periodically freeze a checkpoint and use it as the guesser while continuing to train the cluegiver. Combines benefits of self-play (no external API cost) with stability (guesser doesn't degrade).

### 8. Batch size and learning rate sweep
The 4B thinking model trained for 66 steps with no learning on batch_size=256. Potential factors:
- Batch too small for noisy reward signal → try 512, 1024
- Learning rate too high/low
- The 30B runs already use batch_size=2048, so this is partially tested

### 9. Adversarial board generation
Generate boards specifically designed to be challenging:
- Red words that are semantically similar to blue/assassin words
- Boards where obvious clues are traps
- Could create a harder eval set to measure ceiling performance

### 10. Two-player training (cluegiver + guesser)
Train both roles simultaneously but as separate models or with role-specific LoRA adapters. The cluegiver learns to give clues the guesser understands, and the guesser learns to interpret the cluegiver's style. Requires `EnvGroup` or custom multi-agent setup.
