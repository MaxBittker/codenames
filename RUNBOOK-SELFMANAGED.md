# Self-Managed Codenames RL Training on Prime On-Demand Cloud

Run codenames RL training on a Prime Intellect on-demand GPU pod using prime-rl directly, rather than hosted training (`prime rl run`).

## Prerequisites

- Prime Intellect account with API key
- `prime` CLI installed locally: `uv tool install prime && prime login`
- `WANDB_API_KEY` (optional, for logging)
- No `OPENAI_API_KEY` needed — self-play mode uses the training model as both cluegiver and guesser

## GPU Requirements

| Config | Model | GPUs | Cost |
|--------|-------|------|------|
| `rl-4b-selfmanaged.toml` | Qwen3-4B-Instruct | 2x H100 | ~$4/hr |
| `rl-30b-selfmanaged.toml` | Qwen3-30B-A3B-MoE | 8x H100 | ~$22/hr |

The 4B config uses split configs (separate inference/orchestrator/trainer TOMLs) since it runs each component manually on dedicated GPUs. The 30B config uses the `rl` entrypoint with a `[deployment]` section.

## Step 1: Provision a Pod

```bash
prime availability list --gpu-type H100_80GB --gpu-count 2
prime pods create --id <availability_id>
# Name it, accept defaults, select any image (ubuntu_22_cuda_12 works)
```

## Step 2: Set Up the Pod

SSH in and run:

```bash
# Install uv + Python 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

# Clone and install prime-rl
sudo mkdir -p /workspace && sudo chown $(whoami):$(whoami) /workspace
cd /workspace
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
uv sync --extra flash-attn --extra envs --locked --no-dev
```

## Step 3: Install the Codenames Environment

Copy the environment from your local machine:

```bash
# From your local machine:
scp -r environments/codenames <user>@<pod-ip>:/workspace/codenames-env

# On the pod:
cd /workspace/prime-rl
uv pip install -e /workspace/codenames-env
```

Or if the env is published to the Hub and your API key has access:

```bash
uv run prime config set-api-key <your-key>
uv run prime env install maxbittker/codenames
```

## Step 4: Set Up Wandb (Optional)

```bash
uv run wandb login <your-wandb-key>
```

## Step 5: Create Split Configs

For 2x H100, use three separate config files (one per component). This lets you assign GPU 0 to inference and GPU 1 to training via `CUDA_VISIBLE_DEVICES`.

**`/workspace/infer.toml`** — Inference server config:
```toml
[model]
name = "Qwen/Qwen3-4B-Instruct-2507"
max_model_len = 8192
```

**`/workspace/orch.toml`** — Orchestrator config:
```toml
max_steps = 200
batch_size = 2048
rollouts_per_example = 16
seq_len = 2048

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[sampling]
max_tokens = 4096

[[env]]
id = "codenames"

[env.args]
train_size = 100000
eval_size = 500
self_play = true

[eval]
interval = 25
num_examples = 20
rollouts_per_example = 4

[[eval.env]]
id = "codenames"

[eval.env.args]
train_size = 100000
eval_size = 500
self_play = true

[wandb]
project = "codenames"
name = "qwen3-4b-i-selfmanaged-selfplay"
```

**`/workspace/train.toml`** — Trainer config:
```toml
max_steps = 200

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"
seq_len = 2048

[optim]
lr = 3e-6

[ckpt]
interval = 50

[wandb]
project = "codenames"
name = "qwen3-4b-i-selfmanaged-selfplay"
```

### Important: Eval Env

You **must** specify `[[eval.env]]` in the orchestrator config. If omitted, prime-rl defaults to evaluating `reverse-text` instead of your environment.

## Step 6: Create the Launch Script

**`/workspace/run.sh`**:
```bash
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/prime-rl

rm -f /workspace/inference.log /workspace/orchestrator.log /workspace/trainer.log

# 1. Inference server on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run inference @ /workspace/infer.toml >> /workspace/inference.log 2>&1 &
INFER_PID=$!

echo "Waiting for inference server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Inference server ready after ${i}s"
        break
    fi
    sleep 1
done

# 2. Orchestrator (CPU only)
CUDA_VISIBLE_DEVICES="" uv run orchestrator @ /workspace/orch.toml >> /workspace/orchestrator.log 2>&1 &
ORCH_PID=$!
sleep 5

# 3. Trainer on GPU 1
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run trainer @ /workspace/train.toml >> /workspace/trainer.log 2>&1 &
TRAIN_PID=$!

echo "All components launched."
wait $TRAIN_PID
kill $INFER_PID $ORCH_PID 2>/dev/null
```

```bash
chmod +x /workspace/run.sh
```

## Step 7: Launch Training

```bash
nohup /workspace/run.sh > /workspace/run.log 2>&1 &
```

Monitor:
```bash
tail -f /workspace/run.log           # startup progress
tail -f /workspace/orchestrator.log  # rollout generation + rewards
tail -f /workspace/trainer.log       # gradient steps + loss
```

## Step 8: Evaluate a LoRA Adapter

If you have a LoRA adapter from hosted training (downloaded as a zip), you can evaluate it locally by merging into the base model:

```python
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

# Load config
with open("adapter_config.json") as f:
    cfg = json.load(f)
scaling = cfg["lora_alpha"] / cfg["r"]

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16
)
state_dict = model.state_dict()

# Load and apply adapter weights
adapter = load_file("adapter_model.safetensors")
for key, tensor in adapter.items():
    if "lora_A" not in key:
        continue
    base_key = key.replace(".lora_A.weight", ".weight")
    b_key = key.replace("lora_A", "lora_B")
    A = tensor.float()
    B = adapter[b_key].float()
    state_dict[base_key] = (state_dict[base_key].float() + (B @ A) * scaling).to(torch.bfloat16)

model.load_state_dict(state_dict)
model.save_pretrained("/workspace/merged-model")
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507").save_pretrained("/workspace/merged-model")
```

> **Note:** Do NOT use `peft.PeftModel.from_pretrained()` for adapters downloaded from Prime's hosted training. The adapter key format (`model.layers.X...`) doesn't match peft's expected format (`base_model.model.model.layers.X...`), resulting in a silent no-op merge. Use the manual merge above instead.

Then serve and eval:
```bash
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name /workspace/merged-model --model.max-model-len 8192 &
# wait for health check...
VLLM_API_KEY=dummy uv run vf-eval codenames \
  --model /workspace/merged-model \
  --api-base-url http://localhost:8000/v1 \
  --api-key-var VLLM_API_KEY \
  --num-examples 20 --rollouts-per-example 4 --max-tokens 4096 \
  --env-args '{"train_size": 100000, "eval_size": 500, "self_play": true}' \
  --disable-env-server
```

## Verified Results

Eval rewards (20 examples x 4 rollouts) match hosted training:

| | Hosted | Self-managed |
|---|---|---|
| Base model (step 0) | 1.22 | 1.38 |
| LoRA adapter (step 150) | 1.72 | 1.83 |

Differences are within expected sampling variance for 20 examples.

## Troubleshooting

**OOM on trainer**: Single-GPU training doesn't work for 4B models — you need dedicated GPUs for inference and training. Use 2x H100 minimum.

**Eval shows wrong env**: If you see "Evaluated reverse-text" instead of "Evaluated codenames", you're missing the `[[eval.env]]` section in orch.toml.

**LoRA merge produces worse-than-base results**: You're likely using peft which silently fails on Prime's adapter format. Use the manual merge script above.

**Port 29500 in use**: Run `sudo fuser -k 29500/tcp` before restarting.
