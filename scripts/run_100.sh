#!/usr/bin/env bash
set -Eeuo pipefail

# Optional environment file; no recorded container path is required.
if [[ -n "${BENCH_ENV:-}" ]]; then
    source "$BENCH_ENV"
fi
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VLLM_OMNI_ROOT:?Set VLLM_OMNI_ROOT to the prepared source checkout}"
: "${BENCH_PYTHON:?Set BENCH_PYTHON to the prepared environment interpreter}"
: "${VLLM_BIN:?Set VLLM_BIN to the prepared vllm executable}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_ROOT/results/splitter-en-de-50-reproduction}"
GPU_INDEX="${GPU_INDEX:-0}"
PORT="${PORT:-8091}"

# The controller is for a fresh run. Resume the Python runner directly instead.
mkdir "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
cd "$EXPERIMENT_ROOT"
exec > >(tee -a "$OUTPUT_DIR/controller.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT_DIR/controller.exit"' EXIT
export PYTHONUNBUFFERED=1
export HIP_VISIBLE_DEVICES="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export MIOPEN_FIND_MODE=FAST
export PYTHONPATH="$EXPERIMENT_ROOT/harness${PYTHONPATH:+:$PYTHONPATH}"

"$BENCH_PYTHON" scripts/validate_wikipedia.py
"$BENCH_PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
config = OmniVoiceConfig.from_pretrained('k2-fsa/OmniVoice', revision='c5fdb5ccb189668d56333f77ba2629f4cd7535f4')
values = {name: getattr(config, name) for name in ('frame_rate', 'audio_chunk_duration', 'audio_chunk_threshold')}
assert values == {'frame_rate': 25, 'audio_chunk_duration': 15.0, 'audio_chunk_threshold': 30.0}, values
Path(sys.argv[1], 'runtime-chunk-config.json').write_text(json.dumps(values, indent=2) + '\n')
print('Verified runtime chunk settings:', values, flush=True)
PY
"$BENCH_PYTHON" scripts/analyze_splitter_boundaries.py \
  --ours variants/splitter-en-de-50/ours.py \
  --theirs variants/splitter-en-de-50/theirs.py \
  --duration-estimator "$VLLM_OMNI_ROOT/vllm_omni/model_executor/models/omnivoice/duration.py" \
  --frame-rate 25 --output "$OUTPUT_DIR/boundaries.json"
"$BENCH_PYTHON" scripts/run_splitter_comparison.py \
  --vllm-omni-root "$VLLM_OMNI_ROOT" --vllm-bin "$VLLM_BIN" \
  --gpu-index "$GPU_INDEX" --port "$PORT" --output-dir "$OUTPUT_DIR"
"$BENCH_PYTHON" scripts/evaluate_splitter_comparison.py \
  --records "$OUTPUT_DIR/ours/generation.jsonl" "$OUTPUT_DIR/theirs/generation.jsonl" \
  --output-dir "$OUTPUT_DIR/evaluation"
