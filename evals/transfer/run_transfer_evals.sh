#!/bin/bash
# Transfer evaluation suite for Codenames RL models.
#
# Compares a base model against a merged LoRA adapter across:
#   1. Codenames (in-domain)
#   2. IFBench (instruction following — retention)
#   3. AIME 2025 (math — retention)
#   4. Group Guessing Game (cooperative coordination — transfer)
#   5. Hanabi (cooperative card game — transfer)
#
# Prerequisites:
#   - vLLM inference server running on localhost:8000
#   - prime-rl installed with: uv sync --extra flash-attn --extra envs --locked --no-dev
#   - Environments installed:
#       uv pip install -e /path/to/codenames-env
#       uv run prime env install shashwat23/ifbench
#       uv run prime env install primeintellect/aime2025
#       uv run prime env install nph4rd/hanabi
#       cp evals/transfer/group_guessing_game.py .venv/lib/python3.12/site-packages/
#
# Usage:
#   # Start inference server first:
#   CUDA_VISIBLE_DEVICES=0 uv run inference --model.name <model> --model.max-model-len 8192
#
#   # For hanabi, add tool calling:
#   CUDA_VISIBLE_DEVICES=0 uv run inference --model.name <model> --model.max-model-len 8192 --model.tool-call-parser auto
#
#   # Then run evals:
#   bash evals/transfer/run_transfer_evals.sh <model-name>

set -e
export PATH="$HOME/.local/bin:$PATH"

MODEL=${1:?"Usage: $0 <model-name-or-path>"}
BASE_URL=${2:-"http://localhost:8000/v1"}
API_KEY_VAR="VLLM_API_KEY"
export VLLM_API_KEY=dummy

cd /workspace/prime-rl

echo "=== Transfer Eval Suite ==="
echo "Model: $MODEL"
echo "Server: $BASE_URL"
echo ""

# 1. Codenames (self-play)
echo "--- Codenames (20 examples x 4 rollouts) ---"
uv run vf-eval codenames \
  --model "$MODEL" --api-base-url "$BASE_URL" --api-key-var "$API_KEY_VAR" \
  --num-examples 20 --rollouts-per-example 4 --max-tokens 4096 \
  --env-args '{"train_size": 100000, "eval_size": 500, "self_play": true}' \
  --disable-env-server --abbreviated-summary 2>&1 | grep -E "^reward|^game_reward|^shot_calling|^assassin"
echo ""

# 2. IFBench
echo "--- IFBench (100 examples x 10 rollouts) ---"
uv run vf-eval ifbench \
  --model "$MODEL" --api-base-url "$BASE_URL" --api-key-var "$API_KEY_VAR" \
  --num-examples 100 --rollouts-per-example 10 --max-tokens 4096 \
  --disable-env-server --abbreviated-summary 2>&1 | grep "^reward"
echo ""

# 3. AIME 2025
echo "--- AIME 2025 (30 examples x 10 rollouts) ---"
uv run vf-eval aime2025 \
  --model "$MODEL" --api-base-url "$BASE_URL" --api-key-var "$API_KEY_VAR" \
  --num-examples 30 --rollouts-per-example 10 --max-tokens 4096 \
  --disable-env-server --abbreviated-summary 2>&1 | grep -E "^reward|^correct"
echo ""

# 4. Group Guessing Game (200 games, 10 agents, 50 rounds)
echo "--- Group Guessing Game (200 games x 1 rollout) ---"
uv run vf-eval group_guessing_game \
  --model "$MODEL" --api-base-url "$BASE_URL" --api-key-var "$API_KEY_VAR" \
  --num-examples 200 --rollouts-per-example 1 --max-tokens 256 \
  --env-args '{"num_agents": 10, "max_rounds": 50}' \
  --disable-env-server --abbreviated-summary 2>&1 | grep -E "^reward|^game_reward|^rounds_used"
echo ""

# 5. Hanabi (requires tool-call-parser enabled on server)
echo "--- Hanabi (10 examples x 4 rollouts, 2-player) ---"
uv run vf-eval hanabi \
  --model "$MODEL" --api-base-url "$BASE_URL" --api-key-var "$API_KEY_VAR" \
  --num-examples 10 --rollouts-per-example 4 --max-tokens 4096 \
  --env-args '{"num_players": 2}' \
  --disable-env-server --abbreviated-summary 2>&1 | grep -E "^reward|^points"
echo ""

echo "=== Done ==="
