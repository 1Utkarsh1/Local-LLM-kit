#!/usr/bin/env bash
set -euo pipefail
source /workspace/env-sglang/bin/activate
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
exec python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --host 127.0.0.1 --port 30000 --dtype bfloat16 \
  --context-length 8192 --max-running-requests 4 --mem-fraction-static 0.82 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
