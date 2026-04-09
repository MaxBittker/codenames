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

### Architecture: MultiAgentEnv Migration
- Rewrote the environment from `MultiTurnEnv` (single-context, external `LLMGuesser` API calls) to `MultiAgentEnv` (isolated per-agent contexts, protocol-based turn order, `actor_models` routing).
- Fixes the information leakage problem in the old `CodenameSelfPlayEnv` where the guesser saw the cluegiver's reasoning (including color information) in the same context window.
- Each agent now gets a completely fresh conversation — the cluegiver sees the board with colors, the guesser sees only the word list plus the parsed clue.
- `actor_models` routing doesn't work for external APIs (e.g. `openai/gpt-4.1-mini`) on the hosted training infra — the inference client only knows about the training model. Eval configs that use a fixed external guesser need a different approach.
- Vendored the `MultiAgentEnv`, `Agent`, `Protocol`, and `RoundRobinProtocol` classes into the codenames package since they're not yet in the standard verifiers release.

### Multi-Turn Self-Play Collapse (v0.3.11, Apr 8-9)
- **4B eff2x run collapsed into format-only exploitation.** The model discovered it could earn reward=1.0 by producing valid XML tags (`<clue>`, `<guesses>`) with nonsense content (guessing words not on the board, word-salad reasoning like "The balance between geometry and anomaly manifests"). All game metrics went to zero by step 41, format rewards locked at 1.0 for the remaining 250 steps.
- **Collapse pathway:** Early training had length_penalty=-1.0 (400 char threshold too tight for legitimate reasoning). This suppressed game-playing behavior. The model then found the format-only attractor (0.5 + 0.5 = 1.0) was reachable without playing at all. Once there, no gradient to escape.
- **35B (Qwen3.5-35B-A3B) also failed** — thinking model consumed entire 2048 max_tokens budget with `<think>` blocks, never produced `<clue>` XML. All game metrics zero. Bumping to 8192 max_tokens was the fix but run was stopped before validating.

### Fixes (v0.3.12)
- **Gated format rewards**: Format rewards now require `total_red_found > 0`. Can't collect format reward without actually finding a red word. Eliminates the degenerate format-only attractor.
- **Partial efficiency reward**: Changed from win-only bonus to `reds_per_round / num_red` ratio. Provides gradient from step 0 instead of only after wins. Pitter-patter (1 red/round, 5 reds) scores ~0.04; multi-word clue (3 reds in 1 round) scores 0.6.
- **Length penalty softened**: Char threshold raised 400→800, ramp doubled (800 chars of headroom). Legitimate `<reasoning>` + `<clue>` blocks won't trigger it.
- **`efficiency_weight` parameter** added to `load_environment()` for config-level control.

### Key Insight: Reward Exploitation Taxonomy
Three distinct failure modes observed so far:
1. **Reward hacking** (ambition bonus) — model games the metric being rewarded without improving underlying behavior.
2. **Format-only exploitation** — model produces syntactically valid but semantically empty outputs to collect format rewards without playing.
3. **Pitter-patter** — model finds the safe local optimum (1 clue per red, ~4.5 rounds) that maximizes game_reward without efficiency pressure.

Each requires a different fix: (1) remove the hackable reward, (2) gate auxiliary rewards on gameplay, (3) structural caps or efficiency incentives.

---

## Environment (v0.3.5)
- `CodenamesEnv(MultiAgentEnv)` with `RoundRobinProtocol(["cluegiver", "guesser"])`
- Cluegiver: XML `<clue>` block with `word`, `number`, `words` fields
- Guesser: XML `<guesses>` block with `WORD: reason` lines, `STOP` to end early
- `max_turns=2` — one turn per agent per rollout (single-clue game)
- Variable board sizes (4–16 words) via `BoardSamplingConfig`

### Reward
- `game_reward` (weight 1.0): per-card additive, max 2.0. Assassin = -1.0, each red = +2.0/num_red, blue = -0.5 * per_red
- `shot_calling_reward` (weight 0.5): shots_hit / num_red
- `cluegiver_format_reward` (weight 0.1): 1.0 if valid XML `<clue>` block
- `guesser_format_reward` (weight 0.1): 1.0 if valid XML `<guesses>` block
- Metrics (weight 0.0): assassin rate, red found, shots hit, clue number

Note: `ambition_reward` was removed — it caused reward hacking (model claimed large clue numbers without proportional game improvement).

---

## Insights from the Hanabi Work

The [nphard.io Hanabi post](https://nphard.io/2026/02/23/hanabi.html) describes training multi-agent RL on cooperative card games using the same `MultiAgentEnv` abstractions we adopted. Key takeaways relevant to Codenames:

### What Worked
- **Small models can beat frontier baselines via RL**: Qwen3-4B went from 1.0 to 8.4 points on Hanabi (GPT-4.1-mini baseline: 3.5) after SFT + RL. Tiny Hanabi (1.7B model) reached 91% of max score. Model size matters less than training signal quality.
- **SFT bootstrapping before RL**: Hanabi used synthetic SFT data to get the model into a reasonable policy region before RL. This prevented early collapse that raw RL sometimes causes, especially on smaller models.
- **Information isolation is essential**: Each player sees only what they should (other players' hands, not their own). Without isolation, agents learn shortcuts through weight-sharing that bypass the game's information constraints. This directly validates our MultiAgentEnv migration — the old self-play env leaked cluegiver reasoning to the guesser.
- **Common-payoff simplifies credit assignment**: When both agents share one reward (as in Codenames), you avoid the per-agent baseline complexity needed for asymmetric games. Codenames is cooperative — both agents want reds found.

### What Broke
- **Context length explosion**: Hanabi saw sequence lengths grow from 5k to 13k as models became more verbose. More reasoning tokens ≠ better play. Worth watching in Codenames if we go multi-turn.
- **Opponent conditioning / brittleness**: Self-play produces effective but arbitrary conventions that don't generalize to other partners. The Hanabi model responded to color hints as "play this card" but couldn't adapt to different play styles. Our self-play cluegiver-guesser could develop similar brittle conventions.
- **GRPO kills mixed strategies**: The positive-feedback loop in GRPO drives action probabilities toward 0 or 1. Once a strategy marginally outperforms alternatives, gradients amplify the gap until alternatives go extinct. Less relevant for Codenames (cooperative, not competitive), but worth noting if we see mode collapse in clue styles.

### Implications for Codenames
1. **Self-play conventions risk**: If both agents are trained, they may develop idiosyncratic "languages" that work between copies of the same model but fail with any other guesser. This is the opponent conditioning problem. We should compare self-play eval against fixed-guesser eval to detect this.
2. **SFT bootstrapping could help 4B**: Our 4B runs collapsed, but Hanabi 4B succeeded after SFT warm-start. We could generate synthetic cluegiver trajectories from 30B or GPT and SFT before RL.
3. **Per-agent LoRA**: For asymmetric roles (cluegiver ≠ guesser), separate LoRA adapters let each role specialize while sharing base weights. This is more parameter-efficient than full model duplication and prevents one role's gradient noise from corrupting the other.
4. **The `is_trainable` flag matters**: Setting the guesser to `is_trainable=False` when using a fixed model prevents format failures in guesser completions from polluting the training gradient.

---

## Experiment Ideas

### Priority: Validate MultiAgentEnv Self-Play
The immediate question is whether the new MultiAgentEnv self-play runs produce training signal at all. Watch the 4B and 30B runs for:
- Does game_reward improve?
- Do both agents produce well-formed output (XML for cluegiver, WORD:reason for guesser)?
- Are the generated clues reasonable or do they degenerate?
- Does the model develop conventions that only work in self-play?

### SFT Bootstrapping for Smaller Models
Generate synthetic training data from a strong model (30B checkpoint or GPT) playing both roles on varied boards. SFT a 4B model on this data before starting RL. The Hanabi post showed this is critical for preventing small model collapse.

### Cross-Play Evaluation
After self-play training, evaluate the trained cluegiver against a frozen/external guesser (and vice versa) to measure how much of the improvement is genuine skill vs. self-play convention. If the trained cluegiver's game_reward drops significantly with a GPT guesser, the model learned a private language rather than better clue-giving.

### Per-Agent LoRA
Train separate LoRA adapters for cluegiver and guesser roles. This lets each role specialize (cluegiver: word association + avoidance reasoning; guesser: clue interpretation + board elimination) without interfering with each other's gradients.

### Multi-Turn with Structural Efficiency Pressure
Cap max clue rounds (e.g. 3 rounds for 8 reds) rather than using reward shaping. This forces multi-word clues by construction — you can't pitter-patter if you only get 3 turns. Structural constraints can't be reward-hacked.

### Difficulty Curriculum
Start on small boards (4–6 words, 2 red) and progressively increase. The Hanabi "Tiny" variant showed that simplified games produce faster convergence and cheaper iteration. Scale up once the training signal is validated.

### Population-Based Self-Play
Maintain a pool of past checkpoints as potential partners. Train the current model against randomly sampled partners from the pool rather than always against itself. This should reduce opponent conditioning and produce more robust strategies — the Hanabi post specifically flags this as a promising direction.
