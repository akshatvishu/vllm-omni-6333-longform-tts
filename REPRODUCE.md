# Reproduce

These scripts require a working ROCm vLLM-Omni environment and one available GPU. The recorded runs used the `vllm/vllm-openai-rocm:v0.31.0` base image. Exact observed package versions are in [environment.json](docs/data/environment.json); the image tag alone does not lock every dependency.

The baseline is vLLM-Omni `197d4b025bda6c1203e7e76289baf0ebbbb9a29d`. To reconstruct it without relying on an unpublished branch, apply the included patch to upstream `65699fe94c355e235f1acabddd866bce0bba1c02`:

```bash
git clone https://github.com/akshatvishu/vllm-omni-6333-longform-tts.git
export EXPERIMENT_ROOT="$PWD/vllm-omni-6333-longform-tts"
git clone https://github.com/vllm-project/vllm-omni.git vllm-omni-experiment
export VLLM_OMNI_ROOT="$PWD/vllm-omni-experiment"
git -C "$VLLM_OMNI_ROOT" checkout 65699fe94c355e235f1acabddd866bce0bba1c02
git -C "$VLLM_OMNI_ROOT" apply "$EXPERIMENT_ROOT/patches/6409-baseline.patch"
cp -R "$EXPERIMENT_ROOT/harness/benchmarks/." "$VLLM_OMNI_ROOT/benchmarks/"
```

Use the interpreter belonging to your prepared environment. Install this checkout editable and the harness dependencies there, retaining ROCm torch and a compatible torchaudio. `--no-deps` below assumes model dependencies are already installed; it is not a complete environment setup.

```bash
export BENCH_PYTHON=/path/to/environment/bin/python
export VLLM_BIN=/path/to/environment/bin/vllm
"$BENCH_PYTHON" -m pip install --no-deps -e "$VLLM_OMNI_ROOT"
"$BENCH_PYTHON" -m pip install --no-deps omnivoice==0.2.1 jiwer==4.0.0 pydub==0.25.1
GPU_INDEX=0 bash "$EXPERIMENT_ROOT/scripts/run_all.sh"
```

The runner temporarily replaces only `chunking.py` and restores the baseline on exit. It runs the three variants sequentially, using seeds `42 123 7 0 1 2 3 4 5 6`, and writes to `results/` in this repository. It will refuse to overwrite an existing result directory. GPU visibility is restricted to `GPU_INDEX`.

Generation uses model revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`, default 30-second chunk threshold and 15-second target, and configured float32. Attention still uses BF16. Whisper revision is `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`; the evaluator requires ROCm GPU execution and checks model and input placement.

Coverage is `1 - (substitutions + deletions) / reference words`. WER also counts insertions. Numeric spellings are not normalized to spoken forms. Each variant changes chunk lengths and random-number consumption, so matching seeds does not match later chunks' random draws. Diagnostic audio capture adds synchronization and file writes; recorded latency is not a performance benchmark.

The page includes 14 separately transcribed chunks from the earlier three-seed experiment. Those are supporting diagnostics, not extra samples in the ten-seed table. Their expected text, transcripts, and audio are in `docs/data/chunk-transcripts.json` and `docs/audio/`.
