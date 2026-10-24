#!/usr/bin/env bash
set -euo pipefail
source /workspace/env-vllm/bin/activate
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec vllm serve Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.85 \
  --enforce-eager --language-model-only \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
