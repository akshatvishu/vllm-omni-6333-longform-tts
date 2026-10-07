# First 100-output splitter comparison

The run completed 100 of 100 generation attempts and 100 of 100 ASR evaluations successfully. Chunk text differs on 23 prompts, so only those pairs measure changed splitting. The other 27 pairs are a control with identical chunk text. The automated scores do not establish a quality winner, and no listening judgments have been collected.

Each of the 50 frozen prompts was generated once per splitter with seed 42 and auto voice. No repetitions, warm-up generations, retries, or fallback generations were performed. English ran ours first, while German ran theirs first. Language and execution order are therefore confounded. Generation used clean servers without capture hooks; scoring ran afterward.

## Automated correctness

WER below pools error counts over all 25 prompts in each language. Lower is better. These are ASR-based scores, including number spelling and abbreviation mismatches, and do not establish naturalness or a statistically reliable winner.

| Language | Our WER | Their WER | Our coverage | Their coverage |
| --- | ---: | ---: | ---: | ---: |
| English | 5.19% | 5.39% | 95.79% | 95.56% |
| German | 7.58% | 7.25% | 93.08% | 93.35% |

[Random/targeted breakdown and error counts](evaluation/comparison.md), [raw transcripts and scores](evaluation/evaluated.jsonl), and [paired differences](evaluation/paired-differences.json) retain the detailed evidence. Whisper large-v3-turbo used pinned revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, float32, known language, whole-output sequential decoding, and identical deterministic settings. Input sample counts and the generation defaults are saved with the evaluation.

The comparison on changed prompts is below. Each exact two-sided sign test excludes WER ties and tests equal probability of either splitter having lower WER. The tests use prompt-level WER, rather than the pooled error counts above.

| Changed prompts | Pairs | Our WER lower | Their WER lower | Ties | Exact sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 23 | 6 | 11 | 6 | 0.332306 |
| English | 7 | 4 | 2 | 1 | 0.687500 |
| German | 16 | 2 | 9 | 5 | 0.065430 |

All 27 pairs with identical chunk text have equal WER. The language results above are exploratory and use one seed. Their p-values are not adjusted for multiple tests. No significant difference at a 0.05 threshold is established by these tests, and a nonsignificant result does not establish equivalence.

## Exploratory performance

The following values use 48 single observations per variant. The first request after each of four server starts is excluded from this timing summary but retained in all quality scores. The runner recorded the exclusion rule before executing requests, and the saved execution script matches the hash in the run configuration. The exclusion was not selected after inspecting timings. There are no repeated per-prompt timings and no claim of a repeatable speed difference.

| Metric | Ours | Theirs |
| --- | ---: | ---: |
| Mean client end-to-end latency | 6.264 s | 6.294 s |
| Median across prompt latencies | 6.035 s | 5.970 s |
| Mean end-to-end RTF | 0.05349 | 0.05375 |
| Mean reported peak reserved memory | 4.983 GiB | 4.979 GiB |

The client timer stops after receiving the complete response, before decoding or saving the WAV. RTF uses the final audio duration, including inserted join silence. Memory is the server-reported allocator reserve, not incremental request memory. [Serving summary](serving_summary.json) includes the raw summary fields and exclusion policy.

Paired latency differences below are theirs minus ours, so a positive value means ours finished sooner. Both requests must pass the first-request exclusion rule to enter the timing comparison.

| Chunk text | Warm pairs | Median difference | Sample standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identical | 26 | 0.001136 s | 0.140766 s | -0.552969 s | 0.087746 s |
| Different | 22 | 0.232706 s | 0.600569 s | -1.114234 s | 1.227518 s |

The identical-text pairs expose timing variation despite equal chunk inputs. Among changed-text pairs, the Pearson correlation between chunk-count difference and latency difference is 0.759285. The association is descriptive, and repeated timings with balanced order are needed before claiming a speed improvement. Only four prompts have a changed first chunk, so the run provides little evidence about effects that depend on the first chunk.

## Boundaries and listening

The deployed CPU estimator and runtime defaults give different chunk text on 23 prompts and identical text on 27. Nine prompts differ in chunk count. Total chunks are 566 for ours and 568 for theirs. All 100 content-preservation and character-bound checks passed. There are 86 boundaries exclusive to one splitter, comprising 42 for ours and 44 for theirs. No independent sentence or phrase labels have been supplied for those boundaries, so their counts do not establish boundary correctness. [Boundary metadata](boundaries.json) records the exact budgets and text.

The changed first chunks are `wiki_en_targeted_08`, `wiki_de_random_14`, `wiki_de_targeted_05`, and `wiki_de_targeted_08`. Audio duration differs on 17 pairs. The sum of paired duration differences is zero within floating-point rounding, which does not establish equal prosody.

For the 27 identical-split prompts, all paired WAV lengths match. Two pairs are byte-identical. The median mean absolute PCM sample difference among these 27 pairs is approximately 9.67e-9 on the normalized amplitude scale. These checks do not constitute a listening judgment. [Sample comparison](identical-split-audio-comparison.json) records the measurements.

All 100 files have the expected number of internal zero-run candidates. Excluding the measured zero runs, the inferred chunk durations differ from the CPU predictions by at most 4.004167 model frames at 25 Hz, or approximately 0.160167 seconds. The median maximum error per file is 1.023958 frames. Maximum errors are at most four frames in 98 files, and the predicted total duration matches exactly in four files. Counts and duration agreement corroborate the candidate joins, but no runtime chunk capture supplies ground truth for their positions. The saved diagnostics retain their original `UNVALIDATED` labels. [Audio diagnostics](audio-review/audio-diagnostics.json) retains the candidates and native-rate 20 ms RMS quiet intervals at minus 60 dBFS.

After restoring the WAVs using [the reproduction instructions](../../REPRODUCE.md), open the [anonymous A/B listening page](audio-review/listening.html) locally to judge phrasing and pauses. Choices start blank, and the page can export judgments. Keep the separate answer key closed while listening. Native-speaker judgments and independent sentence/phrase annotations remain pending.

The [CPU summary script](../../scripts/summarize_splitter_results.py) recomputes the paired subsets, exact sign tests, timing statistics, exclusive boundaries, first-chunk changes, duration checks, and execution-script hash from saved files. It checks every field against the unchanged [review audit](review-claims-audit.json) and writes a separate [recomputed summary](recomputed-summary.json). Run it from the publication directory with `"$BENCH_PYTHON" scripts/summarize_splitter_results.py`. It uses only the Python standard library and does not generate audio, run ASR, or infer listening labels.

## Integrity and execution

The host rebooted during setup deployment, before any benchmark request. The assigned container and workspace survived. Only the task-owned container was restarted, and the complete run then finished with controller exit code 0. The runner restored the original source hash `84e07ebf944de2a71e133e0643ade789eb1fafb1c9660c5a00c21fa7fa64aab4`.

All 129 files in the remote checksum manifest were downloaded and verified locally. [Transfer verification](transfer-verification.json), [generation audit](local-generation-audit.json), [configuration](config.json), and [request ledger](attempts.jsonl) are published with all 100 outputs as lossless FLAC. The WAV restoration script verifies the original WAV hashes. Local review reports added after transfer are not part of the remote checksum manifest.
