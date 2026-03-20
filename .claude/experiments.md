# Codenames XML Experiments Log

## Session: 2026-03-20

### Environment v0.3.0
- Switched from StatefulToolEnv (tool calling) to MultiTurnEnv (XML parsing)
- GEPA-optimized system prompt for qwen3-30b-i
- Parser: XMLParser(fields=["reasoning", "clue"])
- Rewards: game_reward (1.0), shot_calling (0.5), format_reward (0.1)

### All Evals (20 examples x 3 rollouts, GEPA prompt unless noted)

#### With gpt-5-mini guesser

| Model | reward | game_reward | shot_calling | assassin% | red_found | pass@1 | output_tokens |
|-------|--------|-------------|-------------|-----------|-----------|--------|--------------|
| gpt-5-mini (GEPA) | **1.664** | **1.175** | **0.778** | 11.7% | **2.55** | **85.0%** | ~2662 |
| gpt-5-mini (old prompt) | 1.598 | 1.114 | 0.767 | 13.3% | 2.53 | 83.3% | ~1627 |
| gpt-4.1-mini (GEPA) | 1.475 | 1.041 | 0.667 | **10.0%** | 2.28 | 85.0% | **239** |
| qwen3-8b (GEPA) | 1.177 | 0.756 | 0.643 | 23.3% | 2.30 | 70.0% | ~2589 |
| qwen3-30b-i (GEPA) | 1.097 | 0.700 | 0.594 | 25.0% | 2.40 | 71.7% | ~2771 |
| qwen3-30b-i (old prompt) | 0.959 | 0.628 | 0.463 | 23.3% | 2.05 | 56.7% | ~3241 |

#### With gpt-4.1-mini guesser (new baselines for RL v2 runs)

| Model | reward | game_reward | shot_calling | assassin% | red_found | pass@1 | output_tokens |
|-------|--------|-------------|-------------|-----------|-----------|--------|--------------|
| gpt-5-mini | **1.702** | **1.216** | **0.772** | **8.3%** | **2.57** | **90.0%** | ~2451 |
| gpt-4.1-mini | 1.572 | 1.106 | 0.733 | 6.7% | 2.60 | 93.3% | **~219** |
| qwen3-8b | 1.167 | 0.788 | 0.559 | 16.7% | 2.18 | 73.3% | ~2811 |
| qwen3-30b-i | 1.083 | 0.734 | 0.498 | 18.3% | 2.08 | 68.3% | ~3224 |

### Key Findings

#### 1. GEPA prompt helps all models — gpt-5-mini is now best at 1.664
- gpt-5-mini on GEPA prompt: **1.664** (up from 1.598 on old prompt, +4%)
- The improvement is across all dimensions: game_reward +5%, shot_calling +1.4%, assassin rate 13.3%→11.7%
- GEPA prompt generalizes well across model families

#### 2. Token efficiency: gpt-4.1-mini is the efficiency king
- gpt-4.1-mini uses only **239 tokens** avg output vs 2500-3200 for all other models
- Despite 10x fewer output tokens, it has the lowest assassin rate (10%) and tied for best pass@1 (85%)
- The verbose reasoning from Qwen/gpt-5-mini models doesn't translate to better gameplay
- **RL implication**: Models that learn concise, focused reasoning may outperform verbose ones

#### 3. qwen3-8b slightly outperforms qwen3-30b-i
- qwen3-8b reward **1.177** vs qwen3-30b-i **1.097** despite being ~4x smaller
- Both have ~23-25% assassin rate — the main bottleneck
- Model size isn't the bottleneck for this task — reasoning quality and safety checking are

#### 4. Assassin avoidance is the #1 improvement lever
- Best: gpt-4.1-mini at 10%, worst: qwen3-30b-i at 25%
- Each assassin hit costs -1.0 reward, devastating to averages
- Reducing Qwen assassin rate from 25%→10% would boost reward by ~0.4-0.5
- This is where RL training should have the biggest impact

#### 5. Shot-calling accuracy tracks game quality
- gpt-5-mini: 0.778, gpt-4.1-mini: 0.667, Qwen models: 0.46-0.64
- Theory of mind (predicting what guesser will pick) is a learnable skill
- Shot-calling reward weight (0.5) seems well-calibrated

#### 6. The guesser creates a ceiling
- The gpt-5-mini guesser makes questionable choices (e.g., BANK for "ocean")
- This means even perfect cluegiver play can't achieve 100% — there's inherent variance from the guesser
- The cluegiver must learn to give clues the guesser won't misinterpret

#### 7. Guesser choice affects assassin rate significantly
- Switching guesser from gpt-5-mini → gpt-4.1-mini reduced assassin rates:
  - qwen3-30b-i: 25.0% → 18.3%
  - qwen3-8b: 23.3% → 16.7%
- gpt-4.1-mini guesser is more conservative — lower assassin rate but also slightly fewer reds found
- Shot-calling accuracy drops with gpt-4.1-mini guesser (cluegiver's predictions less accurate for new guesser)
- For RL training, gpt-4.1-mini guesser is preferred: lower variance, cheaper, deterministic (temp=0)

#### 8. gpt-5-mini + gpt-4.1-mini guesser is the new best combo
- reward=1.702, assassin=8.3%, pass@1=90% — new best across all configurations
- The gpt-4.1-mini guesser universally improves results across all cluegiver models

### GEPA Results
- Budget: 200 calls, 6 candidates explored
- Best score: 1.021 (validation set, 25 examples)
- Key improvements in GEPA prompt:
  - Explicit safety checks against BLUE/ASSASSIN words
  - Validation checklist before responding
  - Strict format enforcement
  - Substring/morphological variant rules with examples
- Edited out: overly conservative "prefer safe 1-target clue" line

### RL Runs v1 (STOPPED — critical guesser bug)
- 30B: l56pi84zxc6yf1pino3y6b3i (qwen3-30b-i-xml-v1, stopped at step 0)
- 4B: lh1olhocia0mwclw6seyu8k4 (qwen3-4b-i-xml-v1, stopped at step 0)
- **Root cause**: gpt-5-mini guesser doesn't support temperature=0.0 (reasoning model)
  - Error: "Unsupported value: 'temperature' does not support 0.0 with this model"
  - Every valid clue hit guesser error → game_reward=0.0 across all training samples
  - RL had no signal to learn from — only format_reward provided gradient
- **Secondary issue**: max_tokens=1024 too low
  - 30B: 68% truncation rate (models ramble in <reasoning> block at temp=1.0)
  - 4B: 49% truncation rate
  - Truncated outputs → incomplete XML → low format_reward

### RL Runs v2/v3 (ACTIVE — guesser bug fixed)
- **Environment v0.3.1**: Fixed guesser temperature (configurable, defaults to 0.0)
- **Guesser changed**: gpt-5-mini → gpt-4.1-mini (supports temp=0, cheaper, deterministic)
- **max_tokens**: 1024 → 2048 (reduces truncation)

#### 30B v2: s6s36zuqb501ogiiw2lz25vg (RUNNING, batch=2048, compute L)

| Step | reward | game | format | assassin | trunc | decode_len | eval |
|------|--------|------|--------|----------|-------|------------|------|
| 0 | 0.486 | 0.315 | 0.640 | 5.7% | 60.2% | 1496 | 1.004 |
| 1 | 0.395 | 0.234 | 0.625 | 8.6% | 54.0% | 1381 | |
| 2 | 0.529 | 0.341 | 0.651 | 6.4% | 52.1% | 1364 | |
| 3 | 0.496 | 0.308 | 0.675 | 8.8% | 53.5% | 1380 | |

- Truncation dropped ~7% from step 0→3 but plateauing around 53%
- Format reward slowly improving (0.640 → 0.675)
- Game reward oscillating, no clear improvement yet
- decode_len decreased from 1496→1380 (model getting slightly more concise)
- 53% truncation still very high — model needs more steps or prompt changes

#### 4B v2: gbqv6oowphhix2rgsdxussuc (STOPPED at step 0, batch=2048)
- Step 0 reward=0.787, eval=1.160, then run stopped unexpectedly during step 1

#### 4B v3: uf9rmo4lh91b4myp8bvp5md8 (RUNNING, batch=256, compute M)

| Step | reward | game | format | assassin | trunc | decode_len | eval |
|------|--------|------|--------|----------|-------|------------|------|
| 0 | 0.918 | 0.645 | 0.818 | 6.2% | 30.9% | 1282 | 1.119 |
| 1 | 0.733 | 0.463 | 0.866 | 12.5% | 23.0% | 1167 | |
| 2 | 0.686 | 0.472 | 0.648 | 3.9% | 19.5% | 912 | |

- **Truncation rapidly decreasing**: 30.9% → 23.0% → 19.5% — model learning to be concise!
- **Decode length shrinking fast**: 1282 → 1167 → 912 (29% reduction in 3 steps)
- **Format reward dropped at step 2** (0.648) — model may be cutting output too aggressively
- Smaller batch_size (256 vs 2048) gives faster steps, more gradient updates
- Need to monitor: is format drop a blip or a trend?

### RL Training Key Insights

1. **Guesser fix was critical** — v1 runs had 0 game_reward everywhere due to temperature bug
2. **Models learn conciseness through RL** — both 4B and 30B show decreasing truncation and decode_len
3. **4B learns faster than 30B** — smaller model adapts more quickly, lower truncation baseline
4. **30B truncation is the bottleneck** — 53% truncation means half of training signal is wasted
5. **Batch size matters** — 4B with batch=256 progresses faster than with batch=2048

### Environment v0.3.2 — Clue-only format reward

**Changes from v0.3.1:**
- Parser: `XMLParser(fields=["clue"])` — only requires `<clue>` block (was `["reasoning", "clue"]`)
- System prompt: `<reasoning>` made optional ("SHOULD include" vs "MUST contain"), limited to 2-4 sentences
- Validation checklist simplified (removed "mention relevant checks in reasoning")
- Rationale: reduce truncation by making reasoning optional; sharper format signal for RL

**Impact on format_reward:**
- With old parser (2 fields): truncated reasoning-only outputs got ~0.4 partial credit
- With new parser (1 field): truncated outputs without `<clue>` get ~0.0 — stronger gradient
- Properly formatted outputs (with both tags): consistently score 0.800 (not 1.0 — XMLParser penalizes extra content)
- The 0.800 vs 1.0 gap is a minor concern — may slightly confuse RL signal

**v0.3.2 baselines (gpt-4.1-mini guesser, 20x3):**

| Model | reward | game_reward | shot_calling | assassin% | red_found | pass@1 | output_tokens |
|-------|--------|-------------|-------------|-----------|-----------|--------|--------------|
| gpt-4.1-mini | 1.368 | 0.976 | 0.625 | 13.3% | 2.30 | 78.3% | ~126 |
| qwen3-30b-i | 1.061 | 0.742 | 0.478 | 20.0% | 2.22 | 61.7% | ~1205 |

**Comparison v0.3.1 → v0.3.2:**

| Model | reward Δ | assassin Δ | output_tokens Δ | verdict |
|-------|----------|-----------|-----------------|---------|
| gpt-4.1-mini | 1.572 → 1.368 (-13%) | 6.7% → 13.3% | 219 → 126 (-42%) | worse quality, more concise |
| qwen3-30b-i | 1.083 → 1.061 (-2%) | 18.3% → 20.0% | 3224 → 1205 (-63%) | ~same quality, **much** more concise |

- gpt-4.1-mini: shorter reasoning hurts safety checking → higher assassin rate
- qwen3-30b-i: nearly identical quality despite 63% token reduction — big win for RL
- The v0.3.2 prompt trades gpt-4.1-mini eval quality for RL trainability (reduced truncation)
- For RL with Qwen models, v0.3.2 is clearly the better choice

### RL Runs v2/v3 (v0.3.1 env)

#### 30B v2: s6s36zuqb501ogiiw2lz25vg (STOPPED at step 3, batch=2048, compute L)
- Stopped to free slot for v4 (clue-only) run

| Step | reward | game | format | assassin | trunc | decode_len | eval |
|------|--------|------|--------|----------|-------|------------|------|
| 0 | 0.486 | 0.315 | 0.640 | 5.7% | 60.2% | 1496 | 1.004 |
| 1 | 0.395 | 0.234 | 0.625 | 8.6% | 54.0% | 1381 | |
| 2 | 0.529 | 0.341 | 0.651 | 6.4% | 52.1% | 1364 | |
| 3 | 0.496 | 0.308 | 0.675 | 8.8% | 53.5% | 1380 | |

- Truncation plateaued at ~53% — fundamentally limited by verbose reasoning at temp=1.0
- Game reward oscillating 0.23-0.34, no clear upward trend
- Decision: stop and relaunch with v0.3.2 env + max_tokens=4096

#### 4B v2: gbqv6oowphhix2rgsdxussuc (STOPPED at step 0, batch=2048)
- Step 0 reward=0.787, eval=1.160, then run stopped unexpectedly during step 1

#### 4B v3: uf9rmo4lh91b4myp8bvp5md8 (COLLAPSED at step 3, batch=256, compute M, env v0.3.1)

| Step | reward | game | format | shot | assassin | trunc | decode_len | eval |
|------|--------|------|--------|------|----------|-------|------------|------|
| 0 | 0.918 | 0.645 | 0.818 | 0.384 | 6.2% | 30.9% | 1282 | 1.119 |
| 1 | 0.733 | 0.463 | 0.866 | 0.367 | 12.5% | 23.0% | 1167 | |
| 2 | 0.686 | 0.472 | 0.648 | 0.298 | 3.9% | 19.5% | 912 | |
| 3 | COLLAPSED — 0/256 valid rollouts, retrying (attempt 2/5 also failed) | | | |

- Truncation rapidly decreased: 30.9% → 19.5% — but too aggressive
- **Model collapsed at step 3**: ALL rollouts produce no trajectory steps
- Signs were visible at step 2: format_reward crashed 0.866 → 0.648, decode_len dropped 29%
- The model learned to be concise too fast → now producing gibberish/empty output
- **Root cause hypothesis**: batch_size=256 gives noisy gradients; 4B model is too small for aggressive RL updates
- No checkpoints exist (checkpoint interval=50, collapse at step 3)

### RL Runs v4 (v0.3.2 env — clue-only format)

#### 30B v4: h2edinbl61phmz18zbs1cn5a (RUNNING, batch=2048, compute L, env v0.3.2)
- max_tokens=4096, guesser=gpt-4.1-mini
- **Step 0 eval: Avg@4=0.966, Truncation=1.2%** — truncation problem SOLVED!
- Completion length: 1235 avg (was 3224 in v0.3.1 eval, 1380 decode in v2 training)
- Max completion length: 64628 — some outlier rollouts, but 98.8% fit within 4096 tokens
- Eval reward 0.966 is close to v2's 1.004 despite prompt changes

| Step | reward | game | format | shot | assassin | trunc | decode_len | red_found | eval |
|------|--------|------|--------|------|----------|-------|------------|-----------|------|
| 0 | 0.644 | 0.433 | 0.689 | 0.284 | 13.0% | 18.7% | 1153 | 1.70 | 0.966 |
| 1 | 0.708 | 0.465 | 0.701 | 0.346 | 16.7% | 16.6% | 1048 | 1.89 | |
| 2 | 0.766 | 0.510 | 0.731 | 0.366 | 13.4% | 12.0% | 872 | 1.85 | |
| 3 | 0.774 | 0.515 | 0.735 | 0.371 | 14.4% | 11.8% | 828 | 1.99 | |
| 4 | 0.934 | 0.644 | 0.784 | 0.423 | 12.8% | 4.2% | 366 | 2.31 | |
| 5 | **1.103** | 0.758 | 0.808 | 0.529 | 11.5% | 0.3% | 150 | 2.48 | |
| 6 | 1.037 | 0.708 | 0.889 | 0.480 | 11.1% | 0.4% | 115 | 2.27 | |
| 7 | 1.092 | 0.742 | 0.939 | 0.512 | 12.4% | 0.2% | 70 | 2.41 | |
| 8 | 1.153 | 0.778 | 0.983 | 0.555 | **7.0%** | 0.2% | 58 | 2.30 | |
| 9 | 1.141 | 0.747 | **0.996** | 0.590 | 11.2% | **0.0%** | **49** | 2.51 | |
| 10 | **1.298** | 0.884 | 0.994 | 0.630 | 6.7% | 0.0% | 81 | 2.68 | |
| 11 | **1.317** | 0.872 | 0.998 | 0.691 | **3.9%** | 0.0% | 93 | 2.43 | |
| 12 | 1.287 | 0.815 | 0.999 | 0.743 | 5.8% | 0.0% | 65 | 2.24 | |
| 13 | **1.421** | **0.938** | **1.000** | 0.765 | **3.0%** | 0.0% | 34 | 2.47 | |
| 14 | 1.425 | 0.930 | 1.000 | 0.791 | 3.4% | 0.0% | 26 | 2.34 | |
| 15 | 1.391 | 0.897 | 1.000 | 0.789 | 3.9% | 0.0% | 26 | 2.40 | |
| 16 | **1.451** | **0.946** | 1.000 | **0.810** | **3.0%** | 0.0% | **25** | **2.55** | |
| 17 | 1.340 | 0.852 | 1.000 | 0.781 | 2.6% | 0.0% | 25 | 2.15 | |
| 18 | 1.472 | 0.954 | 1.000 | 0.837 | 3.7% | 0.0% | 25 | 2.41 | |
| 19 | **1.477** | 0.950 | 1.000 | 0.855 | **1.6%** | 0.0% | 25 | 2.23 | |
| 20 | 1.420 | 0.891 | 1.000 | **0.858** | 2.2% | 0.0% | 25 | 2.37 | |
| 21 | 1.399 | 0.847 | 1.000 | **0.905** | 3.3% | 0.0% | 25 | — | |
| 22 | **1.534** | **0.973** | 1.000 | **0.921** | **0.5%** | 0.0% | 25 | — | |

**30B v4 full analysis (steps 0→22) — outstanding RL learning:**
- **reward**: 0.644 → **1.534** (+138%) — new peak at step 22
- **game_reward**: 0.433 → **0.973** (+125%) — surpasses gpt-4.1-mini eval baseline (0.976)!
- **format_reward**: 0.689 → **1.000** — perfect format since step 13
- **shot_calling**: 0.284 → **0.921** (+224%) — model very accurately predicts guesser behavior
- **assassin**: 13.0% → **0.5%** (step 22 low, -96%) — nearly zero assassin hits
- **truncation**: 18.7% → **0.0%** — completely eliminated by step 9
- **decode_len**: 1153 → **25** — 98% reduction, stable at ~25 tokens since step 14
- **red_found**: 1.70 → 2.55 (step 16), oscillating 2.1-2.5
- **Continued improvement at steps 19-22**: reward 1.477→1.534, shot_calling 0.855→0.921, assassin 1.6%→0.5%

**Four phases of RL learning:**
1. **Conciseness phase (steps 0-4)**: Model learns to output shorter responses → decode_len 1153→366, truncation 18.7%→4.2%
2. **Format optimization (steps 5-9)**: Model learns `<clue>`-first format → format 0.808→0.996, decode_len 150→49
3. **Gameplay improvement (steps 8-12)**: With format mastered, model optimizes game strategy → game 0.778→0.884, assassin 7.0%→3.9%, shot_calling 0.555→0.743
4. **Strategy mastery (steps 13-14)**: All components peak simultaneously → game 0.938, assassin 3.0%, shot_calling 0.791, format 1.000. Decode_len compresses to 26 tokens (98% reduction from start).

**Decode length trajectory**: 1153→366→150→49→81→93→65→34→**26**. After initial compression, brief rebound at steps 10-12 (model experimenting with more ambitious clues), then final compression to ultra-minimal output.

**Strategy shift at step 11-14**: Model became more conservative AND more accurate — assassin dropped from 13%→3%, shot_calling doubled from 0.28→0.79. The model learned theory of mind: predicting which words the guesser will and won't pick, then crafting clues accordingly.

**Key milestones:**
- Step 5: first reward > 1.0 (1.103)
- Step 8: assassin drops below 7% for first time (7.0%)
- Step 10: best red_found (2.68)
- Step 13: best game_reward (0.938), best assassin (3.0%), format reaches 1.000
- Step 14: reward 1.425, shot_calling 0.791, decode=26 tokens
- Step 16: reward 1.451, game 0.946, shot 0.810, assassin 3.0%
- Step 17: assassin reaches 2.6% (new low)
- Step 18: reward 1.472, game 0.954, shot 0.837
- Step 19: best reward (1.477), best assassin (1.6%!)
- Step 20: shot_calling 0.858
- Step 21: shot_calling **0.905** — new milestone, first > 0.9
- Step 22: **reward 1.534** (new peak), **game 0.973**, **shot 0.921**, **assassin 0.5%** — all metrics at new highs
- Next eval at step 25 (~3 steps away) will confirm whether training gains generalize to held-out data

**v4 vs v2 step 0 comparison:**
| Metric | v2 (env 0.3.1, 2048 tok) | v4 (env 0.3.2, 4096 tok) | improvement |
|--------|--------------------------|--------------------------|-------------|
| reward | 0.486 | 0.644 | **+33%** |
| game_reward | 0.315 | 0.433 | **+37%** |
| truncation | 60.2% | 18.7% | **-69%** |

**v4 step 4 vs v2 step 3 comparison (same model, different env):**
| Metric | v2 step 3 (best) | v4 step 4 (latest) | improvement |
|--------|------------------|-------------------|-------------|
| reward | 0.496 | **0.934** | **+88%** |
| game_reward | 0.308 | **0.644** | **+109%** |
| truncation | 53.5% | **4.2%** | **-92%** |
| format_reward | 0.675 | **0.784** | **+16%** |

The v0.3.2 env + 4096 tokens is a massive improvement. The 30B model is now able to learn effectively — reward nearly doubled from v2's best, truncation dropped from 53% to 4%, and game_reward more than doubled.

#### 4B v4: rxfdbx3n6b8swibwth3eycol (STALLED at step 2 → STOPPED, batch=1024, compute M, env v0.3.2)
- Auto-launched after 4B v3 collapsed at step 3
- **Key changes from v3**: batch_size 256 → 1024 (less noisy gradients), env v0.3.2 (clue-only format)
- **Step 0 eval: Avg@4=1.087** — excellent baseline
- **Truncation essentially solved: 3.1%** (was 30.9% with v0.3.1!)
- **STALLED**: Step 2 started at 11:07 but no output for 65+ minutes — infrastructure issue, not model collapse
- Stopped and relaunched as v5

| Step | reward | game | format | shot | assassin | trunc | decode_len | eval |
|------|--------|------|--------|------|----------|-------|------------|------|
| 0 | 1.095 | 0.759 | 0.762 | 0.520 | 15.0% | 3.1% | 527 | 1.087 |
| 1 | 0.960 | 0.683 | 0.755 | 0.403 | 13.1% | 4.0% | 602 | |

#### 4B v5: i7p2s1v0fzxmhjwtrwlikmom (STOPPED — stalled at step 1, batch=1024, compute M, env v0.3.2)
- Relaunch of stalled v4 with identical config
- **Step 0 eval: Avg@4=1.055, Truncation=0.0%** — clean start
- Stalled at step 1 after 30+ minutes with no output. Stopped and relaunched as v6.

| Step | reward | game | format | shot | assassin | trunc | decode_len | red_found | eval |
|------|--------|------|--------|------|----------|-------|------------|-----------|------|
| 0 | 0.931 | 0.639 | 0.755 | 0.433 | 9.2% | 4.1% | 583 | 2.17 | 1.055 |

#### 4B v6: xokx9cdfzkxrxvg92vrmyzfg (STOPPED — stalled at step 6, batch=1024, compute M, env v0.3.2)
- Third relaunch attempt. Completed 6 steps (best of all 4B runs) before auto-stopping.
- Even when running, the 4B model was learning the WRONG things (getting more verbose, not less).

| Step | reward | game | format | shot | assassin | trunc | decode_len | red_found | eval |
|------|--------|------|--------|------|----------|-------|------------|-----------|------|
| 0 | **1.039** | **0.753** | 0.767 | 0.418 | **7.2%** | **3.2%** | **560** | 2.46 | |
| 1 | 1.007 | 0.699 | 0.759 | 0.464 | 13.6% | 4.0% | 570 | 2.47 | |
| 2 | 0.846 | 0.549 | 0.767 | 0.443 | 16.0% | 3.5% | 672 | 2.39 | |
| 3 | 0.989 | 0.676 | 0.743 | 0.477 | 14.7% | 9.0% | 828 | 2.41 | |
| 4 | 0.924 | 0.610 | 0.748 | 0.478 | 17.1% | 8.4% | 873 | 2.28 | |
| 5 | 0.927 | 0.635 | 0.735 | 0.436 | 12.3% | 11.0% | 1001 | 2.17 | |

**4B v6 trend — concerning divergence from 30B learning pattern:**
- **Verbosity increasing**: decode_len 560→1001 (+79%) — model getting MORE verbose, not less
- **Truncation increasing**: 3.2%→11.0% — more outputs hitting token limit
- **Assassin volatile**: 7.2%→17.1%→12.3% — not improving, oscillating
- **Format reward declining**: 0.767→0.735, getting worse (30B had 0.689→0.808 by step 5)
- **Reward declining**: 1.039→0.927 (-11%) overall downward trend

**Why 4B ≠ 30B learning trajectory:**
- 30B learned conciseness rapidly (decode 1153→366 by step 4, 150 by step 5)
- 4B is getting MORE verbose (560→873 by step 4) — going the wrong direction
- Hypothesis: 4B model lacks capacity to simultaneously learn format + gameplay
- 30B may have reached a critical mass of understanding that 4B hasn't
- The 4B model may need more steps, or fundamentally different training approach
- **Key question**: will 4B have a breakthrough similar to 30B's step 4-5? Or will it continue diverging?
- **Decision**: stopped v6 (M compute diverging) and launched v7 on L compute to test infrastructure hypothesis

#### 4B v7: mn2wt8te3841w7hhujmyuv53 (RUNNING, batch=1024, compute L, env v0.3.2)
- Switched from M → L compute to fix recurring stalls
- **L compute fixed the infrastructure stalls** — 12 steps completed smoothly (checkpoint pauses ~30s vs 500s for 30B)
- 4B learning pattern is fundamentally different from 30B

| Step | reward | game | format | shot | assassin | trunc | decode_len | red_found |
|------|--------|------|--------|------|----------|-------|------------|-----------|
| 0 | 0.940 | 0.638 | 0.765 | 0.451 | 12.3% | 2.9% | 573 | 2.25 |
| 1 | 0.908 | 0.615 | 0.754 | 0.434 | 12.7% | 4.0% | 582 | 2.25 |
| 2 | 0.873 | 0.589 | 0.758 | 0.418 | 14.9% | 2.5% | 536 | 2.28 |
| 3 | 1.007 | 0.689 | 0.754 | 0.486 | 12.3% | 4.7% | 602 | 2.09 |
| 4 | 0.877 | 0.596 | 0.752 | 0.410 | 14.0% | 5.5% | 645 | 2.30 |
| 5 | 1.073 | 0.746 | 0.779 | 0.497 | 14.0% | 2.8% | 542 | 2.67 |
| 6 | 1.064 | 0.723 | 0.794 | 0.524 | 13.7% | 0.9% | 382 | 2.33 |
| 7 | 1.072 | 0.723 | 0.793 | 0.539 | 12.6% | 1.1% | 369 | 2.30 |
| 8 | 1.042 | 0.713 | 0.790 | 0.501 | 15.9% | 1.8% | 357 | 2.64 |
| 9 | 1.025 | 0.684 | 0.781 | 0.525 | 19.0% | 3.1% | 466 | 2.70 |
| 10 | 0.986 | 0.675 | 0.781 | 0.466 | 12.6% | 3.1% | 558 | 2.41 |
| 11 | **1.164** | **0.816** | 0.768 | **0.541** | **10.8%** | 5.5% | 641 | 2.50 |
| 12 | 0.978 | 0.653 | 0.766 | 0.503 | 15.3% | 10.6% | 844 | — |
| 13 | 1.048 | 0.752 | 0.762 | 0.453 | 7.6% | 17.0% | 1008 | — |
| 14 | 0.947 | 0.659 | 0.756 | 0.438 | 9.0% | 18.6% | 1077 | — |
| ... | *verbosity peaked at step 14, then partially recovered* | | | | | | | |
| 21 | 1.192 | 0.834 | — | 0.562 | 6.5% | 6.6% | 802 | — |
| 22 | 1.087 | 0.754 | — | 0.513 | 9.2% | 5.4% | 747 | — |
| 23 | 1.170 | 0.814 | — | 0.556 | 11.2% | 4.5% | 762 | — |
| 24 | 1.185 | — | — | — | — | — | — | — |
| 25 | **1.258** | — | — | — | — | — | — | — | **1.216** |
| 26 | **1.301** | 0.915 | 0.798 | 0.611 | 5.3% | — | 506 | — |
| 27 | 1.255 | 0.860 | 0.797 | 0.631 | 10.1% | — | 475 | — |
| 28 | 1.185 | 0.795 | 0.797 | 0.620 | 11.6% | — | 485 | — |
| 29 | 1.149 | 0.791 | 0.799 | 0.555 | 11.1% | — | 441 | — |
| 30 | 1.211 | 0.834 | 0.799 | 0.595 | 9.6% | — | 404 | — |
| 31 | 1.283 | 0.877 | 0.800 | 0.654 | 9.1% | — | 377 | — |
| 32 | 1.184 | 0.784 | 0.799 | 0.640 | 10.1% | — | 381 | — |
| 33 | 1.232 | 0.831 | 0.800 | 0.642 | 9.5% | — | 363 | — |
| 34 | 1.172 | 0.784 | 0.800 | 0.616 | 10.8% | — | 376 | — |
| 35 | 1.258 | 0.860 | 0.800 | 0.634 | 8.8% | — | 390 | — |
| 36 | 1.194 | 0.810 | 0.799 | 0.609 | 11.4% | — | 398 | — |
| 37 | 1.215 | 0.833 | 0.800 | 0.603 | 10.7% | — | 433 | — |
| 38 | **1.301** | 0.896 | 0.799 | 0.650 | 4.1% | — | 490 | — |
| 39 | 1.292 | 0.885 | 0.799 | 0.655 | 5.0% | — | 516 | — |
| 40 | 1.247 | 0.837 | 0.799 | 0.660 | 6.4% | — | 519 | — |
| 41 | 1.316 | 0.880 | — | 0.712 | 6.2% | — | — | — |
| 42 | **1.377** | **0.948** | — | 0.698 | 4.4% | — | — | — |
| 43 | 1.281 | 0.879 | — | 0.643 | 4.5% | — | — | — |
| 44 | 1.223 | 0.778 | — | 0.729 | 7.1% | — | — | — |

**Step 25 eval: Avg@4 = 1.216** (up from 1.102 at step 0, **+10.3%**). Completion length 470 (down 21% from 597). 0% truncation. **RL learning confirmed — generalizes to eval!**

**4B v7 analysis — five-phase trajectory:**
- **Phase 1 (steps 0-7)**: Initial learning, decode oscillates 357-641, reward 0.87-1.16
- **Phase 2 (steps 8-14)**: Verbosity explosion, decode 357→1077, truncation 1.8%→18.6%
- **Phase 3 (steps 15-20)**: Recovery, decode ~750-800, truncation 4-7%, reward ~1.0-1.2
- **Phase 4 (steps 21-26)**: Breakthrough — reward climbing 1.19→1.30, decode shrinking
- **Phase 5 (steps 27-44)**: Plateau/oscillation — reward oscillates 1.15-1.38, format stuck at 0.800, decode 360-520
- **Infrastructure**: L compute stable — 44+ steps without stalling (M compute stalled at 1-2 steps)
- **Eval confirms learning**: Step 0→25 eval improved +10.3% (1.102→1.216)
- **Format reward plateaued at 0.800** — the 4B model cannot learn to produce perfect XML like 30B
- **Assassin still volatile**: oscillating 4-11%, not consistently improving (30B at 0.5%)
- **Shot-calling slowly improving**: 0.451→0.729, but much noisier than 30B (0.921)
- 4B learning is slower and noisier than 30B, reaching a plateau around reward ~1.25 while 30B continues climbing to 1.53

**4B vs 30B at step 11 comparison:**
| Metric | 30B step 11 | 4B step 11 | gap |
|--------|-------------|------------|-----|
| reward | 1.317 | 1.164 | 4B is 12% lower |
| game | 0.872 | 0.816 | 4B is 6% lower |
| format | 0.998 | 0.768 | 4B is 23% lower! |
| decode | 93 | 641 | 4B outputs 7x more tokens |
| assassin | 3.9% | 10.8% | 4B 2.8x more assassin hits |
| shot | 0.691 | 0.541 | 4B is 22% lower |

**Key insight**: The 4B model learns game quality (game_reward improving) through a different path than the 30B. Rather than the 30B's dramatic conciseness breakthrough (decode 1153→25 by step 14), the 4B maintains verbose output (~750 tokens) but improves clue quality within that format. The 4B learning is real (eval +10.3% at step 25) but slower and less dramatic than the 30B (+129% training reward). The 30B's conciseness breakthrough is a capacity-dependent phenomenon.

**4B v3 vs v4 step 0 comparison:**
| Metric | v3 (env 0.3.1, bs=256) | v4 (env 0.3.2, bs=1024) | improvement |
|--------|----------------------|----------------------|-------------|
| reward | 0.918 | **1.095** | +19% |
| game_reward | 0.645 | **0.759** | +18% |
| truncation | 30.9% | **3.1%** | **-90%** |
| decode_len | 1282 | **527** | -59% |
| format_reward | 0.818 | 0.762 | -7% (lower ceiling of 0.8) |

- v0.3.2 env essentially eliminates truncation for the 4B model
- Much shorter outputs (527 vs 1282 tokens) — model outputs clue more directly
- Higher game_reward because valid rollouts → game can be played → reward signal

### Rollout Sample Analysis (from step 0 samples)

**Main failure mode: truncation in `<reasoning>` before `<clue>` is output**
- At temp=1.0, models are much more verbose than at temp=0
- The `<reasoning>` block acts as a token sink — models ramble about board analysis
- Truncated outputs get format_reward=0.4 (v0.3.1) or 0.0 (v0.3.2) and game_reward=0.0
- This is the single biggest source of wasted training signal

**Successful rollout pattern:**
- Model outputs concise reasoning + valid `<clue>` block within token limit
- Guesser correctly identifies 1-3 RED words
- Reward range: 0.83 to 2.6 for successful games

**Assassin hit pattern:**
- Model gives reasonable clue (e.g., "city" for LONDON, CAPITAL, SUB)
- Guesser picks unintended word (e.g., LIMOUSINE for "city") → Assassin
- Reward: -0.9 (game=-1.0 + format=1.0×0.1 = -0.9)

### Step 10 Rollout Samples (30B v4, post-RL)

**Model output style**: Clean `<clue>` blocks only — no reasoning. Model targets 2 words per clue typically.

**Multi-round games**: The game runs for multiple rounds. Best games find 4-6 red words over several clue-guess rounds.

**Top-scoring examples:**
- reward=1.93: "tool 2" for HOOK, BRUSH → guesser picks both correctly → 4/6 reds found
- reward=1.93: "time 2" for WATCH, WAKE → guesser picks LEAD first (blue) but eventually finds 6/6 reds
- reward=1.40: "lens 2" for MICROSCOPE, FILM → both correct → 2/5 reds found

**Failure modes at step 10:**
1. **Invalid clue** (reward=0.10): Model gives a clue that matches a board word (e.g., "deck" when DECK is on the board). Environment rejects with "Invalid clue: Clue cannot exactly match a board word."
2. **Guesser misinterpretation** (reward=0.35): Model gives "country" for AMERICA, WELL but guesser picks EMBASSY (blue). The guesser makes reasonable-but-wrong associations.
3. **Format fine**: format_reward=1.0 on almost all samples. Model has fully learned the XML format.

**Key observation**: Shot-calling is still weak even in high-reward games. The model doesn't accurately predict *which* words the guesser will pick, but it gives good enough clues that the right words get picked anyway. Shot-calling improvement (0.284→0.691) comes from the model learning which clue-word pairs are unambiguous.

### Major Findings (Session Summary)

1. **30B RL training is a clear success**: reward 0.644→1.451 (+125% over 16 steps), approaching gpt-4.1-mini eval performance
2. **4B RL training diverges**: all runs (v3-v7) show increasing verbosity and declining format reward. The 4B model does not learn output compression.
3. **Model size is the differentiator**: The 30B model's conciseness breakthrough (step 4-5) is a capacity-dependent phenomenon. 4B lacks capacity to learn format+game simultaneously.
4. **env v0.3.2 solved truncation for both models** at step 0, but only 30B maintained low truncation through training
5. **Assassin avoidance is learnable**: 30B went from 13%→2.6% assassin rate. This is the biggest contributor to reward improvement.
6. **Shot-calling (theory of mind) improves dramatically through RL**: 30B went from 0.284→0.810. The model learns to predict guesser behavior.
7. **M compute is unreliable for 4B**: v4/v5/v6 all stalled. L compute fixes this but doesn't fix the learning divergence.

### Potential Next Steps
- [x] All v0.3.0-v0.3.2 env improvements and evaluations
- [x] Monitor 30B v4 — reward 0.644→1.451, assassin 13%→2.6%, format→1.000
- [x] Monitor 4B runs v3-v7 — all diverge or stall, fundamental learning issue
- [x] Analyze rollout samples — model produces clean clues, main failures: board-word matches + guesser misinterpretation
- [ ] **HIGH PRIORITY**: Wait for step 25 eval (30B v4) — confirm training gains transfer to held-out eval
- [ ] Consider for 4B: increase format_reward weight from 0.1 to 0.3-0.5 to force format learning
- [ ] Consider for 4B: add explicit length penalty reward to discourage verbosity
- [ ] Consider for 4B: reduce max_tokens to 1024 to force conciseness via hard constraint
- [ ] Evaluate trained 30B checkpoint against gpt-4.1-mini baseline on same eval set
- [ ] Try: qwen3-8b RL run (better baseline than 30B, but may not be available for RL training)
