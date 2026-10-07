# Reproduce the 100-output splitter comparison

The latest experiment is `splitter-en-de-50`. It contains 50 frozen Wikipedia passages, with 25 English and 25 German passages. Each passage was generated once with each frozen splitter, seed 42, automatic voice, concurrency 1, and no reference audio. The earlier three-variant diagnostic experiment remains available through `scripts/run_all.sh` and is separate from this comparison.

The portable scripts under `scripts/` include path and output protection changes made after the run. The original execution scripts are retained under `results/splitter-en-de-50/execution-scripts/`. The archived generation runner matches the `runner_sha256` in the recorded configuration. New runs record their own runner hash.

## Prepare the source and environment

Use a prepared ROCm vLLM environment and one available GPU. The recorded base image was `vllm/vllm-openai-rocm:v0.31.0`, digest `sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f`. The installed runtime included vLLM `0.31.0+rocm723`, Transformers `5.14.1`, OmniVoice `0.2.1`, JiWER `4.0.0`, and datasets `5.0.1`. The initialized experiment image was local to the experiment host. The base image and [package pins](requirements-reproduction.txt) do not fully lock the initialized environment. The [experiment plan](EXPERIMENT_PLAN.md) records the prepared environment, and [the recorded configuration](results/splitter-en-de-50/config.json) records the command and source hashes. `docs/data/environment.json` describes the earlier diagnostic experiment.

Use a dedicated source checkout so temporary splitter replacement does not affect another server or unrelated local changes. Set these paths to your own directories. The scripts do not require `/workspace` or an environment file at a fixed location.

```bash
git clone https://github.com/akshatvishu/vllm-omni-6333-longform-tts.git
export EXPERIMENT_ROOT="$PWD/vllm-omni-6333-longform-tts"
git clone https://github.com/akshatvishu/vllm-omni.git vllm-omni-experiment
export VLLM_OMNI_ROOT="$PWD/vllm-omni-experiment"
git -C "$VLLM_OMNI_ROOT" checkout 2ece3987427d24a7825973b94b7d6db662b262e2
cp "$EXPERIMENT_ROOT/variants/splitter-en-de-50/ours.py" \
  "$VLLM_OMNI_ROOT/vllm_omni/diffusion/models/omnivoice/chunking.py"
cp -R "$EXPERIMENT_ROOT/harness/benchmarks/." "$VLLM_OMNI_ROOT/benchmarks/"
export BENCH_PYTHON=/path/to/prepared/environment/bin/python
export VLLM_BIN=/path/to/prepared/environment/bin/vllm
export VLLM_OMNI_TARGET_DEVICE=rocm
"$BENCH_PYTHON" -m pip install --no-deps -e "$VLLM_OMNI_ROOT"
"$BENCH_PYTHON" -m pip install --no-deps -r "$EXPERIMENT_ROOT/requirements-reproduction.txt"
```

The `--no-deps` commands assume the model and harness dependencies are already installed. They preserve the prepared ROCm torch and compatible torchaudio. The scripts also import httpx, numpy, soundfile, tqdm, pyarrow, and huggingface_hub. Install missing dependencies in the prepared environment, retaining the ROCm packages. Save the resulting package list with `"$BENCH_PYTHON" -m pip freeze` for any new run.

The latest run uses base `2ece3987427d24a7825973b94b7d6db662b262e2` with the frozen `ours.py` replacement. Do not apply `patches/6409-baseline.patch` to this base. That patch reconstructs the earlier diagnostic baseline.

| Input | Pinned revision or SHA256 |
| --- | --- |
| OmniVoice model | `k2-fsa/OmniVoice`, revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` |
| Whisper model | `openai/whisper-large-v3-turbo`, revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` |
| Wikipedia source | `wikimedia/wikipedia`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configurations `20231101.en` and `20231101.de` |
| Frozen prompt manifest | `0024b112ef13d33196e89f7b0576ed399000d4a02d11c81814e65a7c51446d48` |
| Frozen `ours.py` | `84e07ebf944de2a71e133e0643ade789eb1fafb1c9660c5a00c21fa7fa64aab4` |
| Frozen `theirs.py` | `6c8b6f578bfbda6f590a4070cde72092634a8080927ba9fdf7470acb22fa35aa` |
| Base deployment YAML | `3ce36bd562d575ded0e2abaa3497e61c1c22e6d76928f9b46d90c86481be7674` |
| Base duration estimator | `0075cc62910766859fd3a7fec8711123852c8bfd1947e77e00e6cb8ceabe3b2c` |

Use the supplied prompts unchanged. You do not need to download Wikipedia again. The [dataset provenance](data/splitter-en-de-50/README.md) records the selection, exact article spans, and attribution.

## Check the inputs without generating audio

```bash
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT/harness:$VLLM_OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$BENCH_PYTHON" scripts/validate_wikipedia.py
"$BENCH_PYTHON" scripts/analyze_splitter_boundaries.py \
  --ours variants/splitter-en-de-50/ours.py \
  --theirs variants/splitter-en-de-50/theirs.py \
  --duration-estimator "$VLLM_OMNI_ROOT/vllm_omni/model_executor/models/omnivoice/duration.py" \
  --frame-rate 25 --output results/reproduction-boundaries.json
```

The boundary script refuses to overwrite its output. The recorded configuration used frame rate 25, a 15-second chunk target, and a 30-second trigger threshold. The recorded CPU checks found 23 prompts with different chunk text, nine with different chunk counts, 566 total chunks for ours, and 568 for theirs. All content and character-bound checks passed. Character bounds do not establish strict audio duration bounds or correct sentence boundaries.

## Generate and score a fresh run

Inspect GPU memory allocation and running containers before using a shared host, and again immediately before starting the workload. Use a GPU only when allocated memory is 0%. Use only your own or explicitly assigned container. Remove diagnostic capture hooks from `PYTHONPATH`.

```bash
cd "$EXPERIMENT_ROOT"
GPU_INDEX=0 PORT=8091 \
  OUTPUT_DIR="$EXPERIMENT_ROOT/results/splitter-en-de-50-reproduction" \
  bash scripts/run_100.sh
```

`VLLM_OMNI_ROOT`, `BENCH_PYTHON`, and `VLLM_BIN` are required. `BENCH_ENV` may name an optional shell environment file. `GPU_INDEX`, `PORT`, and `OUTPUT_DIR` are configurable. The controller requires a new output directory and defaults to `results/splitter-en-de-50-reproduction`, preserving recorded evidence. Generation and evaluation both use the selected visible GPU, mapped to `cuda:0` in the ROCm Python API. The runtime uses eager mode and `MIOPEN_FIND_MODE=FAST`.

The runner starts four clean servers in this order: ours for English, theirs for English, theirs for German, and ours for German. It replaces only `chunking.py` and restores its original bytes on normal exit or a handled interruption. The first request after each server start is flagged and excluded from the serving timing summary, leaving 48 timing observations per variant. All 100 outputs remain in the quality evaluation. There are no warm-up, smoke, retry, or fallback audio generations.

The durable `attempts.jsonl` ledger reserves each request before sending it. After an interruption, resume the runner directly with the same paths and arguments. Reserved attempts are never retried, including requests whose outcome is unknown. A changed source or configuration requires a new output directory. Do not rerun the fresh controller to resume. After generation finishes, run the evaluator separately if needed:

```bash
"$BENCH_PYTHON" scripts/run_splitter_comparison.py \
  --vllm-omni-root "$VLLM_OMNI_ROOT" --vllm-bin "$VLLM_BIN" \
  --gpu-index 0 --port 8091 \
  --output-dir "$EXPERIMENT_ROOT/results/splitter-en-de-50-reproduction"
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  "$BENCH_PYTHON" scripts/evaluate_splitter_comparison.py \
  --records results/splitter-en-de-50-reproduction/ours/generation.jsonl \
    results/splitter-en-de-50-reproduction/theirs/generation.jsonl \
  --output-dir results/splitter-en-de-50-reproduction/evaluation
```

Use the same GPU index and port as the interrupted run. If a process is killed without cleanup, inspect its server processes and restore `chunking.py` from the run's `original-chunking.py` before starting other work. The lock prevents concurrent runners using the same checkout; it cannot protect unrelated servers using that checkout.

## Score the published outputs after download

The recorded generation rows retain their original absolute audio paths. Restore the published lossless FLAC outputs to WAV files, then use `--audio-root` to map those paths to your local results. The restoration script verifies the audio against the publication manifest. Use a fresh scoring directory to preserve the recorded evaluation:

```bash
cd "$EXPERIMENT_ROOT"
"$BENCH_PYTHON" scripts/prepare_publication.py --restore-wav
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  "$BENCH_PYTHON" scripts/evaluate_splitter_comparison.py \
  --records results/splitter-en-de-50/ours/generation.jsonl \
    results/splitter-en-de-50/theirs/generation.jsonl \
  --audio-root results/splitter-en-de-50 \
  --output-dir results/splitter-en-de-50-rescoring
```

Whisper scoring requires ROCm GPU execution and checks model and input placement. It uses whole-output sequential transcription with float32, an explicit language per prompt, no reference text prompt, and the deterministic settings saved in `scoring-config.json`. An unchanged scoring configuration can resume from its saved evaluation checkpoint.

Coverage is `hits / reference words`, equivalent to `1 - (substitutions + deletions) / reference words`. WER also counts insertions. Numeric spellings are not normalized to spoken forms. Matching seeds do not align later random draws when splitters change chunk lengths. ASR scores require listening before attributing a mismatch to TTS. Native-speaker judgments and independent boundary annotations remain pending. Timing observations are exploratory and do not establish a repeatable speed advantage.

## Validate and rebuild the public report on CPU

A model installation is not needed to verify the published results or rebuild the site. Use Python 3.12 in an environment of your choice, then run these commands from this repository:

```bash
"$BENCH_PYTHON" -m pip install -r requirements-publication.txt
PYTHONPATH=harness "$BENCH_PYTHON" scripts/validate_wikipedia.py
"$BENCH_PYTHON" scripts/prepare_publication.py --verify
"$BENCH_PYTHON" scripts/summarize_splitter_results.py
"$BENCH_PYTHON" scripts/build_latest_site.py
"$BENCH_PYTHON" scripts/validate_publication.py
"$BENCH_PYTHON" -m http.server 8000 --directory docs
```

Open `http://localhost:8000` to browse the report and audio. The summary script recomputes the paired comparisons from saved transcripts, boundaries, and audio diagnostics. It compares them against the original audit. The publication check also verifies the copied evidence and local page links. The GitHub Pages workflow runs these checks before deployment.

The original anonymous listening sheet is retained at `results/splitter-en-de-50/audio-review/listening.html`. The WAV restoration command also restores its audio links. Open the sheet locally or serve the repository root to use it. The public main page identifies the variants and is therefore not a blind listening test.
