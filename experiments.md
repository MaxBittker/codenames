# Codenames RL Experiments

## Learnings So Far

### Model Size
- **4B Instruct collapses under RL** — tool-calling breaks down after a few steps, producing ModelErrors in a death spiral. XML format didn't fix it either.
- **4B Thinking is stable but flat** — ran 66 steps with no learning. Stable reward but no improvement.
- **30B MoE (Qwen3-30B-A3B-Instruct) is the sweet spot** — stable training, clear reward improvement over 85 steps. Best single-clue game_reward went from 0.13 to 0.39, eval from 0.29 to 0.48.

### Guesser Model
- **gpt-5.4-mini > gpt-5.4-nano** as the fixed guesser. Stronger guesser gives clearer training signal for the cluegiver.
- **gpt-4.1-mini** also works well; used in several experiments with good results.

### Reward Shaping
- **Reward hacking with ambition bonus** — adding clue_ambition_reward (bonus for higher clue numbers) caused the model to claim large numbers without proportional game improvement. Total reward inflated while game_reward stayed flat.
- **Clean reward (game_reward only) is more reliable** — the simple per-card additive scoring (red found vs blue/assassin penalties) produces steady improvement without gaming.


### Board Simplification
- Removed civilians from the default board config (was 25 words, now 16: 8 red, 7 blue, 1 assassin). Reduces noise and makes every wrong guess meaningfully bad (blue or assassin), which should sharpen the reward signal.

### Multi-Turn
- Multi-turn clue-giving (multiple rounds per game) showed high initial reward (~0.9) but with volatile training dynamics. The longer sequence lengths (~1760 tokens) require smaller batch sizes.

---

## Current Best Configuration

- **Model**: Qwen3-30B-A3B-Instruct-2507
- **Guesser**: gpt-5.4-mini (fixed, external)
- **Board**: 16 words (8 red, 7 blue, 0 civilian, 1 assassin)
- **Reward**: game_reward only (no shaping), weight=1.0
- **Batch size**: 2048, rollouts_per_example=16
- **Max tokens**: 512
- **Clue rules**: single word, max 15 letters, case-normalized

---

## Experiment Ideas

### Call Its Shot (Predicted Targets)
Have the cluegiver predict which specific red words its clue is intended for (e.g. via an additional tool parameter `targets: list[str]`). These predictions are NOT passed to the guesser — only the clue word and number are. Award bonus reward when the guesser's correct guesses match the cluegiver's predicted targets. This:
- Encourages intentional, targeted clue-giving over vague associations
- Creates a richer reward signal (did the model connect the words it meant to?)
- Doesn't change the guesser's information — purely a cluegiver self-assessment
- Could surface interesting behaviors: does the model learn to predict which words are "guessable"?

### Multi-Turn Refinement
The multi-turn experiment showed promise but had volatile dynamics. Ideas to stabilize:
- Reduce max_turns (e.g. 4 rounds instead of 8) to keep sequences shorter
- Per-round intermediate rewards instead of only end-of-game scoring
- Progressive difficulty: start single-turn, then increase rounds as the model improves

### Self-Play Variants
- **Frozen self-play**: use a checkpoint of the training model as guesser (no external API cost)
- **Co-training**: train both cluegiver and guesser, but this requires EnvGroup or multi-agent setup
- The 30B model as its own guesser could develop coordinated clue-giving strategies

### Difficulty Curriculum
Start on easier boards (fewer words, fewer blues) and progressively increase difficulty as reward improves. The easy preset (7 words: 3 red, 2 blue, 1 civilian, 1 assassin) provides a faster learning signal.

### Adversarial Boards
Generate boards where red words are semantically close to blue/assassin words. Forces the model to find more creative, discriminating clues rather than relying on obvious category matches.
