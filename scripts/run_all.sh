#!/usr/bin/env bash
set -euo pipefail
export EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VLLM_OMNI_ROOT:?Set VLLM_OMNI_ROOT to the prepared checkout}"
export VLLM_OMNI_ROOT="$(cd "$VLLM_OMNI_ROOT" && pwd)"
file="$VLLM_OMNI_ROOT/vllm_omni/diffusion/models/omnivoice/chunking.py"
cmp "$file" "$EXPERIMENT_ROOT/variants/original.py"
trap 'cp "$EXPERIMENT_ROOT/variants/original.py" "$file"' EXIT
export SEEDS="42 123 7 0 1 2 3 4 5 6"
export PYTHONPATH="$VLLM_OMNI_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
for variant in original decimal_only full; do
    target="$EXPERIMENT_ROOT/results/$variant"
    if [[ -e "$target" ]]; then
        echo "Refusing to overwrite $target" >&2
        exit 1
    fi
    cp "$EXPERIMENT_ROOT/variants/$variant.py" "$file"
    OUTPUT_DIR="$target" bash "$EXPERIMENT_ROOT/scripts/run_variant.sh" --manifest "$EXPERIMENT_ROOT/prompts.json" --diagnostics
done
