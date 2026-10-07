# English and German splitter comparison

Status: all 100 primary generations completed successfully, with 50 per splitter, seed `42`, and auto voice. All 100 outputs were scored successfully with whole-output sequential Whisper. Results and checksums are saved in [the run report](results/splitter-en-de-50/REPORT.md). The anonymous listening page is prepared; native-speaker judgments and independent boundary annotation remain pending. No smoke, warm-up, repetition, retry, or fallback audio generations were performed.

## Purpose and revision

Compare our sentence-first splitter with the exact splitter supplied by `lazariv` in [PR #6409, comment 6023328036](https://github.com/vllm-project/vllm-omni/pull/6409#issuecomment-6023328036). Both prefer sentences and newlines over clause punctuation. The comparison concerns their treatment of abbreviations, initials, numbers, and resulting audio boundaries.

The revised protocol replaces diagnostic chunk capture and per-chunk transcription with clean generation and whole-output sequential transcription. It removes extra smoke and warm-up generations, repeated timing passes, and automatic diagnostic audio fallbacks. It retains all 50 frozen prompts, including any with identical chunk lists. Timing observations are exploratory. Silence detections are candidate join locations without validated per-join measurements. Listening and linguistic annotation remain pending until qualified people provide them.

Sources for the protocol are the [reviewer's proposed splitter and experiment discussion](https://github.com/vllm-project/vllm-omni/pull/6409#issuecomment-6023328036), the [pinned Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa), and the [Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo). The request limit, absence of extra generations, and explicit scoring settings below are the revised user-authorized protocol. The older published experiment remains separate.

## Workspace and prepared environment

- Publication repository: <https://github.com/akshatvishu/vllm-omni-6333-longform-tts>.
- Local experiment checkout: `/home/aja/vllm-omni/work/issues/6333-omnivoice-longform/publication/vllm-omni-6333-longform-tts`.
- Canonical source checkout: `/home/aja/vllm-omni`.
- Existing diagnostic material: `/home/aja/vllm-omni/work/issues/6333-omnivoice-longform/upstream-review`.
- Raw outputs: `results/splitter-en-de-50/<run-id>/` inside the experiment checkout.
- New comparison: `docs/splitter-en-de-50/`, preserving the earlier comparison and assets.

The prepared environment was recorded as follows. Verify the actual environment and record any changes before execution.

| Item | Prepared value |
| --- | --- |
| SSH | `root@129.212.176.155`, local key `~/.ssh/amd_devcloud_ed25519` |
| Assigned container | `omnivoice-splitter-bench`, GPU visibility restricted to device 0 |
| Base image | `vllm/vllm-openai-rocm:v0.31.0` |
| Base digest | `sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f` |
| Initialized image | `vllm-omni-splitter-bench:2ece39874` |
| Initialized image ID | `sha256:176d4e6c057dbe67ac3c92559e2a0eb2f4aef626c8b25ab9974af548b53d42ca` |
| Host mount | `/opt/omnivoice-splitter-bench` mounted at `/workspace` |
| Source | `/workspace/vllm-omni`, branch `benchmark/splitter-en-de-50`, base `2ece3987427d24a7825973b94b7d6db662b262e2`, with the scoped sentence-first patch |
| Experiment | `/workspace/experiment`, prepared from `8ff3a90ed5430e5a44c087ea4af1eda1c8d21265` |
| Splitters | `/workspace/experiment/variants/splitter-en-de-50/` |
| Python | Existing `/usr/bin/python3` in the container, explicitly requested by the user |
| Packages | vLLM `0.31.0+rocm723`, Transformers `5.14.1`, OmniVoice `0.2.1`, JiWER `4.0.0`, datasets `5.0.1` |
| Environment and install logs | `/workspace/benchmark.env` and `/workspace/logs/` |

The original ROCm torch, torchaudio, and torchvision were retained. FFmpeg, espeak-ng, and jq were installed. The source and experiment files are in the persistent mount, outside the image snapshot. At setup completion no weights had been downloaded and no container inference or tests had run.

Preserve the canonical source branch and unrelated edits. Run variant replacement only in the task's remote source directory. Record the experiment commit, pending changes, source base, exact patch, and file hashes used for execution. For noninteractive container commands, explicitly source `/workspace/benchmark.env`.

Inspect GPU memory allocations and running containers on connection and immediately before a GPU workload. Use only confirmed free GPUs with 0% allocated memory and restrict visibility with `HIP_VISIBLE_DEVICES`. Use the assigned container; do not modify other containers or stop other workloads. Record GPU, driver, ROCm, Python, PyTorch, model, tokenizer, runtime, and evaluation package versions. Check disk space and device placement. Download final outputs and logs to the local experiment directory and verify their checksums.

## Variants and frozen dataset

| Variant | Definition |
| --- | --- |
| `ours` | Frozen local `chunking.py` with the sentence-first and newline changes. |
| `theirs` | The exact supplied diff applied to PR commit `2ece39874`, reconstructed in `upstream-review/lazariv-proposed-chunking.py`. |

Record SHA256 hashes for both frozen files under `variants/splitter-en-de-50/`. Confirm the reconstruction against the saved comment. Run the same compatible runtime for both, changing only the splitter. The prior PR splitter is a CPU reference, not a third audio variant.

Use the frozen `data/splitter-en-de-50/prompts.json` unchanged. Its [README](data/splitter-en-de-50/README.md) and provenance describe the pinned source, exact spans, exclusions, bounded sampling method, and hashes.

| Group | English | German | Total |
| --- | ---: | ---: | ---: |
| Random eligible prose | 15 | 15 | 30 |
| Targeted boundary cases | 10 | 10 | 20 |
| Total | 25 | 25 | 50 |

Selection used seed `42` and original Wikipedia prose with 200–400 whitespace words per passage. The harness also stores a normalized scoring word count. Targeted feature matches are text selection rules, not gold boundary labels. Wikipedia is not a sentence-boundary benchmark or a representative sample of every TTS workload.

Keep all 50 prompts even if the splitters produce identical chunks. Runtime preflight verified frame rate 25, target duration 15 seconds and threshold 30 seconds. With the deployed duration estimator, 23 prompts have different chunk text and 27 have identical chunks; nine differ in chunk count. Total chunks are 566 for ours and 568 for theirs. All 100 variant/prompt content and character-bound checks passed. See `results/splitter-en-de-50/boundaries.json`. Do not replace prompts or add a disagreement-enriched audio set after inspecting results.

## Existing CPU evidence and required metadata

The earlier saved `upstream-review/current-vs-proposed-splitters.json`, produced by `compare_current_splitters.py`, records nine inputs at eight synthetic budgets. The revised splitter and proposed patch differed in 13 of 72 pairs. They agreed at the tested budgets on sentence preference, newline lists, decimal lists, and the supplied German 311-word input. Differences involved abbreviations, dates, sentence-ending numbers, and initialisms. Earlier records report 9,816 content and budget checks and 60 CPU unit tests. These are historical CPU checks, not evidence of audio quality or current actual-budget disagreement counts.

For each frozen prompt, derive the exact budget with the selected runtime's duration estimator and auto-voice configuration. Record the estimator inputs and outputs, estimated total duration, activation decision, actual character budget, ordered chunk text and lengths, and source boundary offsets for both variants. Use matching inputs for both variants. Do not substitute a fixed synthetic character limit for the runtime calculation.

The selected protocol uses `audio_chunk_threshold=30` and `audio_chunk_duration=15`. The duration target does not establish the length of a generated chunk. Verify the chosen runtime's calculation before labeling CPU metadata as exact. If the required estimator input is unavailable, report that limitation instead of fabricating a budget or boundary list.

Record whether chunk lists match and the boundary differences. Verify content order, nonempty chunks, and budget compliance while preserving raw whitespace differences for inspection. Budget fallback cases include oversized sentences, lines, or tokens and absent punctuation. These checks require no extra TTS generation.

Independent text annotation must classify boundaries as `sentence`, `acceptable phrase`, `inappropriate`, or `budget-forced`, with context and reasons. Keep annotations separate from splitter output and automated feature matches. Annotation remains pending; neither a changed boundary nor a regex match establishes an inappropriate split.

## Generation protocol

Schedule exactly 50 requests per splitter, one per frozen prompt, for 100 total primary generation attempts. Use seed `42`, auto voice without uploaded or inline reference, concurrency 1, and identical model revision, dtype, attention backend, inference steps, guidance, language handling, and deployment settings. Record the request and all effective settings.

Use clean generation without diagnostic capture hooks, chunk audio export, or instrumentation that adds chunk synchronization or copies. Save the full raw response WAV and response metadata. Start the client timer immediately before sending the request and stop it after the full response body has arrived, before writing audio to disk. Record the clock method and response success or error. If available, retain a response header reporting peak reserved GPU memory, with its exact header, unit, and measurement scope. Do not call reserved memory allocated memory, and leave the field unavailable if it is not provided.

Record server start and stop times, request order, and variant order. Never replace a splitter under a running server. Flag the first generation request after every server start as startup affected, even if the server reports ready. Health checks must not synthesize audio. There are no separate smoke or warm-up generations and no repetitions.

Matching seeds does not ensure matching later random draws when chunk lengths differ. In auto-voice mode, changing the first chunk can also change the reference used for later chunks. The comparison therefore measures the whole auto-voice output of each splitter.

Retain every scheduled case, including HTTP, generation, decoding, and scoring failures. Do not automatically retry failed generations or fill missing outputs with new audio. A successful run yields 100 outputs; a run with failures must report fewer successful outputs against the same 100 scheduled attempts. Resume only genuinely unsubmitted cases after checking the request ledger, hashes, and settings. A request with uncertain completion must not be silently resubmitted.

## Whole-output transcription

Score each complete WAV with [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo). Record the loaded model revision; the earlier evaluator used `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Record the OmniVoice model revision too; the earlier generation pin was `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`. Verify availability and compatibility before reuse.

Preserve the raw WAV for listening. Create a separate mono 16 kHz scoring copy and transcribe the complete output with sequential long-form decoding. Use the known language, `task="transcribe"`, `num_beams=1`, `temperature=0`, `do_sample=False`, and `condition_on_prev_tokens=False`. Supply no expected-text prompt. Save raw transcripts, effective decoding settings, dtype, durations, and scoring errors. Keep the same dtype and decoder for both variants.

Do not independently transcribe generated chunks or fixed overlapping 24-second windows. Do not truncate long audio to a short-form window. Confirm complete audio consumption from evaluator metadata. Sequential long-form decoding may process internal segments, but the scored unit is the full output paired with the full prompt.

## Evaluation and limitations

### Text correctness

Compute whole-output WER as `(S + D + I) / N` and coverage as `1 - (S + D) / N`, where `N` is the normalized reference word count and `S`, `D`, and `I` are substitutions, deletions, and insertions. Retain the existing NFKC, lowercase, and punctuation-tokenization policy identically for both variants, preserving umlauts. Save normalized text and counts alongside raw transcripts.

Inspect numbers and abbreviation expansions as possible normalization mismatches. Do not silently repair transcripts. A high WER or apparent omission does not alone establish missing synthesized speech; review the original audio before attributing it to TTS. Do not generate fallback audio automatically. The reviewer's LCS and umlaut-folding scores are not directly interchangeable with these JiWER scores.

### Listening and boundary diagnostics

Prepare blind A/B materials with randomized labels and order for complete outputs. Qualified English and German listeners may record naturalness preference, inappropriate pauses or intonation, audible losses or repetitions, and notes. Keep native-speaker listening pending until actual judgments are collected. Do not invent scores or infer preference from WER. The prompt is the unit of comparison; multiple joins within it are not independent votes.

Without direct chunk capture, zero-sample runs in the full WAV can identify only candidate joins. Even agreement between their count and the CPU split count does not prove their locations or correspondence. Do not publish per-join pause durations, dropout rates, or chunk-opening accuracy as validated measurements without independent ground truth. Keep candidate locations explicitly labeled and ground-truth validation pending. Do not automatically generate instrumented fallback audio to resolve uncertainty.

Whole-output quiet intervals may be reported as descriptive audio diagnostics with their exact detector settings. They do not establish voice activity, linguistic appropriateness, or naturalness. Independent boundary annotation and listening are separate pending assessments.

### Exploratory timing

Report each request's single client end-to-end observation and real-time factor, calculated as latency divided by output duration. Retain the startup-affected flag and present first requests separately when discussing timing. Include optional peak reserved memory only with a consistent recorded scope.

The run has no warm-up or repeated observations. Do not report repeated-run medians, performance significance, or stable speed superiority. Timing differences may include startup, request order, server, and host effects. Do not add a separate timing pass under this authorization.

UTMOS, speaker similarity (SIM), time to first packet (TTFP), and streaming underrun measurements are excluded. The run supplies no calibrated naturalness predictor, fixed voice reference, or streaming measurement protocol for those metrics.

## Reporting

Report language and sampling group separately, including their intersections where useful. For each group, show scheduled attempts, successful generations, scoring successes and failures, summed reference words, summed S/D/I, pooled WER and coverage from summed counts, and paired per-prompt differences for successfully scored pairs. State how many pairs are missing and why. Do not silently remove failed cases or average chunk percentages.

Report actual-budget identical and different chunk counts only after metadata verification. Show text differences with annotation status. Publish listening wins, ties, and losses only when collected; otherwise mark them pending. Keep all raw counts and single-request timings available. The balanced aggregate describes the selected 50 prompts and one seed. It cannot establish general superiority or equivalence, and the targeted group cannot estimate ordinary-prose error prevalence.

## Preparation and execution checks

1. Verify the frozen manifest, 50 unique texts, agreed language/group counts, disjoint article IDs, and checksums without changing the dataset.
2. Record exact splitter sources, runtime patch, model and dataset revisions, environment versions, request settings, and decoding settings.
3. Prepare exact CPU budget and boundary metadata and preserve unresolved annotation status.
4. Verify the request ledger enforces 100 primary attempts and prevents smoke, warm-up, repeat, retry, and fallback generations. Verify hooks are disabled and the timer stops before file writing.
5. Inspect the assigned environment and free GPU memory, then start each selected server with its frozen splitter. Mark every first request after server startup.
6. Generate the scheduled requests, preserving failures and raw WAVs. Transcribe the full outputs sequentially and save counts, transcripts, and errors.
7. Download final artifacts, verify checksums, and prepare grouped reporting and blind listening materials with pending judgments explicit.

Local Python commands use `/home/aja/vllm-omni/.venv/bin/python`. Remote commands use the user-authorized container interpreter. Preparation checks must not add generation requests. Document executed checks and failures from logs; do not label unexecuted checks as passing.

## Publication artifacts

Retain the frozen dataset and attribution, exact splitters, source/runtime patches, environment records, CPU metadata, request ledger, request/response metadata, raw WAVs, scoring copies, transcripts, scoring configuration, grouped counts, timing observations, failures, and checksum manifest. Keep annotation and listening status explicit. No diagnostic chunk audio is expected from this clean run.

Prepare the comparison under `docs/splitter-en-de-50/`, preserving prior results. Keep original WAVs locally. If browser previews are converted, record the format and any quantization; do not describe float-to-PCM16 conversion as lossless. Retain Wikipedia attribution and review asset size and links before publication. Exclude credentials, weights, and machine secrets.

Preparing artifacts does not itself authorize a commit, push, or PR. Follow the user's explicit publication instructions and repository workflow for external writes. Report final local paths, actual attempt and success counts, failures, pending judgments, and any published commit or URL only after verifying them.
