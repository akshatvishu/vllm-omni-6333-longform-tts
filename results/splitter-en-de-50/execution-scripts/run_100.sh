#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/benchmark.env
cd "$EXPERIMENT_ROOT"
mkdir -p results/splitter-en-de-50
exec >> results/splitter-en-de-50/controller.log 2>&1
trap 'code=$?; printf "%s\n" "$code" > results/splitter-en-de-50/controller.exit' EXIT
export PYTHONUNBUFFERED=1
# Keep both import locations identical; the checkout owns the benchmarks package.
cp harness/benchmarks/tts/omnivoice_longform/evaluate.py "$VLLM_OMNI_ROOT/benchmarks/tts/omnivoice_longform/evaluate.py"
"$BENCH_PYTHON" scripts/validate_wikipedia.py
"$BENCH_PYTHON" - <<'PY'
import json
from pathlib import Path
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig
config = OmniVoiceConfig.from_pretrained('k2-fsa/OmniVoice', revision='c5fdb5ccb189668d56333f77ba2629f4cd7535f4')
values = {name: getattr(config, name) for name in ('frame_rate', 'audio_chunk_duration', 'audio_chunk_threshold')}
assert values == {'frame_rate': 25, 'audio_chunk_duration': 15.0, 'audio_chunk_threshold': 30.0}, values
Path('results/splitter-en-de-50/runtime-chunk-config.json').write_text(json.dumps(values, indent=2))
print('Verified runtime chunk settings:', values, flush=True)
PY
"$BENCH_PYTHON" scripts/analyze_splitter_boundaries.py \
  --ours variants/splitter-en-de-50/ours.py \
  --theirs variants/splitter-en-de-50/theirs.py \
  --duration-estimator "$VLLM_OMNI_ROOT/vllm_omni/model_executor/models/omnivoice/duration.py" \
  --frame-rate 25 --output results/splitter-en-de-50/boundaries.json
"$BENCH_PYTHON" scripts/run_splitter_comparison.py \
  --vllm-omni-root "$VLLM_OMNI_ROOT" --vllm-bin "$VLLM_BIN" --gpu-index 0
"$BENCH_PYTHON" scripts/evaluate_splitter_comparison.py \
  --records results/splitter-en-de-50/ours/generation.jsonl results/splitter-en-de-50/theirs/generation.jsonl \
  --output-dir results/splitter-en-de-50/evaluation
