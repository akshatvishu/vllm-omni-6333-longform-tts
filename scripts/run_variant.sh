#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

SCRIPT_DIR="${VLLM_OMNI_ROOT:?Set VLLM_OMNI_ROOT}/benchmarks/tts/omnivoice_longform"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_COMMAND="${BENCH_PYTHON:-python}"
VLLM_COMMAND="${VLLM_BIN:-vllm}"
MODEL="${MODEL:-k2-fsa/OmniVoice}"
MODEL_REVISION="${MODEL_REVISION:-c5fdb5ccb189668d56333f77ba2629f4cd7535f4}"
PORT="${PORT:-8091}"
GPU_INDEX="${GPU_INDEX:-0}"
WHISPER_DTYPE="${WHISPER_DTYPE:-float32}"
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-large-v3-turbo}"
WHISPER_REVISION="${WHISPER_REVISION:-41f01f3fe87f28c78e2fbf8b568835947dd65ed9}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/$(date +%Y%m%d-%H%M%S)}"
SELECTION="$SCRIPT_DIR/selection.toml"
read -r -a SEED_VALUES <<< "${SEEDS:-42}"
CONCURRENCY_VALUES=(1)
PREPARE_DATASET_ARGS=()
INPUT_MANIFEST=""
DIAGNOSTICS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            INPUT_MANIFEST="${2:?--manifest requires a path}"
            shift 2
            ;;
        --diagnostics)
            DIAGNOSTICS=1
            shift
            ;;
        --small)
            PREPARE_DATASET_ARGS=(--source-examples 10)
            shift
            ;;
        --concurrency)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--concurrency requires a positive integer" >&2
                exit 2
            fi
            for concurrency in "${CONCURRENCY_VALUES[@]}"; do
                if [[ "$concurrency" == "$2" ]]; then
                    echo "Duplicate concurrency: $2" >&2
                    exit 2
                fi
            done
            CONCURRENCY_VALUES+=("$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--small | --manifest FILE] [--diagnostics] [--concurrency N]..."
            exit 0
            ;;
        *)
            echo "Usage: $0 [--small | --manifest FILE] [--diagnostics] [--concurrency N]..." >&2
            exit 2
            ;;
    esac
done

if ! BENCH_PYTHON="$(command -v "$PYTHON_COMMAND")"; then
    echo "Python executable not found: $PYTHON_COMMAND" >&2
    exit 1
fi
if ! VLLM_BIN="$(command -v "$VLLM_COMMAND")"; then
    echo "vLLM executable not found: $VLLM_COMMAND" >&2
    exit 1
fi
if (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    echo "Port $PORT is already in use" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/reference" "$OUTPUT_DIR/vllm-omni" "$OUTPUT_DIR/evaluation"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
exec > >(tee -a "$OUTPUT_DIR/runner.log") 2>&1
MANIFEST="$OUTPUT_DIR/prompts.json"
export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

cd "$REPO_ROOT"

echo "Saving results to: $OUTPUT_DIR"

if [[ -n "$INPUT_MANIFEST" ]]; then
    "$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.prepare_dataset \
        --manifest "$INPUT_MANIFEST" --output "$MANIFEST"
else
    "$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.prepare_dataset \
        --selection "$SELECTION" --output "$MANIFEST" "${PREPARE_DATASET_ARGS[@]}"
fi

DIAGNOSTIC_ARGS=()
if [[ "$DIAGNOSTICS" == 1 ]]; then
    DIAGNOSTIC_ARGS=(--diagnostics)
fi

"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.metadata \
    --output "$OUTPUT_DIR/run_metadata.json" \
    --repo-root "$REPO_ROOT" \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --gpu-index "$GPU_INDEX" \
    --whisper-model "$WHISPER_MODEL" \
    --whisper-revision "$WHISPER_REVISION" \
    --whisper-dtype "$WHISPER_DTYPE" \
    --seeds "${SEED_VALUES[@]}" \
    --concurrencies "${CONCURRENCY_VALUES[@]}" \
    "${DIAGNOSTIC_ARGS[@]}"

SERVER_PID=""
cleanup_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID"
        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID"
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

echo "Starting vLLM-Omni server"
if [[ "$DIAGNOSTICS" == 1 ]]; then
    export OMNIVOICE_DIAGNOSTICS_DIR="$OUTPUT_DIR/vllm-omni/chunks"
    export PYTHONPATH="$SCRIPT_DIR/vllm_omni/hooks:$REPO_ROOT:${PYTHONPATH:-}"
    export VLLM_LOGGING_LEVEL=DEBUG
else
    unset OMNIVOICE_DIAGNOSTICS_DIR
    export VLLM_LOGGING_LEVEL=INFO
fi
"$VLLM_BIN" serve "$MODEL" \
    --revision "$MODEL_REVISION" \
    --omni \
    --deploy-config vllm_omni/deploy/omnivoice.yaml \
    --host 127.0.0.1 \
    --port "$PORT" \
    --log-stats \
    --trust-remote-code \
    >"$OUTPUT_DIR/vllm-omni/server.log" 2>&1 &
SERVER_PID=$!

SERVER_READY=0
for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM server exited before becoming ready" >&2
        tail -n 100 "$OUTPUT_DIR/vllm-omni/server.log" >&2
        exit 1
    fi
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
        SERVER_READY=1
        break
    fi
    sleep 5
done
if [[ "$SERVER_READY" -ne 1 ]]; then
    echo "vLLM server did not become ready within 15 minutes" >&2
    exit 1
fi

echo "Running vLLM-Omni serving benchmark"
"$BENCH_PYTHON" -m benchmarks.tts.omnivoice_longform.vllm_omni.benchmark \
    --api-base "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR/vllm-omni" \
    --seeds "${SEED_VALUES[@]}" \
    --concurrencies "${CONCURRENCY_VALUES[@]}" \
    "${DIAGNOSTIC_ARGS[@]}" >"$OUTPUT_DIR/vllm-omni/client.log" 2>&1

cleanup_server
SERVER_PID=""
trap - EXIT

echo "Running Whisper transcription and scoring"
"$BENCH_PYTHON" "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT}/scripts/evaluate_single.py" \
    --records \
        "$OUTPUT_DIR/vllm-omni/generation.jsonl" \
    --output-dir "$OUTPUT_DIR/evaluation" \
    --whisper-model "$WHISPER_MODEL" \
    --model-revision "$WHISPER_REVISION" \
    --device cuda:0 \
    --dtype "$WHISPER_DTYPE" >"$OUTPUT_DIR/evaluation/console.log" 2>&1

echo "Benchmark results: $OUTPUT_DIR"
echo "Quality summary: $OUTPUT_DIR/evaluation/summary.md"
echo "Serving summary: $OUTPUT_DIR/vllm-omni/serving_summary.json"
