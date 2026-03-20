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
- Multi-turn clue-giving (multiple rounds per game) showed strong reward improvement: game_reward 0.75 → 1.50 in 13 steps, assassin rate 31% → 6%, finding 7.4/8 reds.
- **But the model learned "pitter-patter" clues** — giving one safe single-word clue per red word (~8 clues per game) instead of connecting multiple words. Without efficiency pressure in the reward, there's no incentive to take the risk of multi-word clues.
- Multi-turn needs an efficiency component (e.g. penalty per clue given, or bonus for fewer rounds) to incentivize ambitious clue-giving.

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

### Multi-Turn with Efficiency Reward
The multi-turn experiment needs efficiency pressure to prevent pitter-patter (one clue per word). Options:
- **Cap max clue rounds** (e.g. 4 rounds for 8 reds) — forces multi-word clues by construction
- **Per-clue penalty** — small negative reward per clue given, so fewer rounds = higher reward
- **Efficiency bonus** — reward `reds_found / clues_given` ratio
- Key tension: efficiency shaping previously caused reward hacking (ambition bonus). Need a formulation that can't be gamed — capping rounds is the most robust since it's structural, not reward-based

### Self-Play Variants
- **Frozen self-play**: use a checkpoint of the training model as guesser (no external API cost)
- **Co-training**: train both cluegiver and guesser, but this requires EnvGroup or multi-agent setup
- The 30B model as its own guesser could develop coordinated clue-giving strategies

### Difficulty Curriculum
Start on easier boards (fewer words, fewer blues) and progressively increase difficulty as reward improves. The easy preset (7 words: 3 red, 2 blue, 1 civilian, 1 assassin) provides a faster learning signal.

### Adversarial Boards
Generate boards where red words are semantically close to blue/assassin words. Forces the model to find more creative, discriminating clues rather than relying on obvious category matches.

### Smaller Model with XML Format + Prompted Reasoning
Try a smaller model (sub-30B) using XML-based structured output instead of tool calling, with prompt adjustments to make it work. Tool calling broke down on 4B previously, but XML format with a well-tuned prompt might be more robust for smaller models. Also try approximating thinking-mode behavior by prompting the instruct model to reason through its strategy (e.g. consider word associations, assess risk of blue/assassin overlap) before giving its clue. This could get some of the benefits of thinking models without needing the thinking variant.
