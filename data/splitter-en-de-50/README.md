# Frozen English and German splitter comparison

The frozen dataset contains 50 passages. The first paired run completed 100 successful generations and 100 successful ASR evaluations, with 50 per splitter, seed `42`, and auto voice. No extra smoke, warm-up, repeated, retry, or fallback audio generations were performed. See [the run report](../../results/splitter-en-de-50/REPORT.md). Native-speaker listening and independent boundary annotation remain pending.

| Language | Random eligible passages | Targeted passages | Words per passage | Median words |
| --- | ---: | ---: | ---: | ---: |
| English | 15 | 10 | 220–318 | 288 |
| German | 15 | 10 | 201–329 | 257 |

Counts here use whitespace words. The harness also records its normalized word count for scoring.

## Source and selection

The source is [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa), revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configurations `20231101.en` and `20231101.de`. Retain the article URLs and attribution in `provenance.json` when redistributing. The dataset card specifies CC BY-SA 3.0 and GFDL; this source text is not covered by any separate code license in this repository.

Selection uses seed 42 to hash-rank shards and article IDs. It scans the first 5,000 rows of one selected shard per language. There were 971 eligible English and 1,008 eligible German articles after exclusions. This is a bounded sample, not a uniform sample of all Wikipedia.

Each passage preserves an exact source span of consecutive prose paragraphs, including punctuation and newlines. Eligible passages have 200–400 whitespace words. The script chooses the passage closest to 300 words per article. Two English articles were excluded for damaged source endings; their IDs and reasons are recorded in the manifest and preparation script. No splitter output or audio score was used for selection.

The targeted passages include dotted abbreviations, spaced initials or abbreviations, and periods after short numbers. English also targets single initials and common abbreviations. German also targets dates and German abbreviations. Feature labels are regular-expression matches, not gold sentence-boundary labels. A short-number match in German may be an ordinal or date. Report random and targeted results separately.

## Files and checks

- `prompts.json` is the harness input. Both variants must use this same file.
- `provenance.json` records article URLs, IDs, offsets, features, selection details, and review status.
- `source-articles.jsonl` preserves the selected source articles for offline extraction checks.
- `SHA256SUMS` freezes these three files.

The agent inspected passage starts and endings. Offline validation passed for the strict harness schema, all 50 unique texts, language/group counts, source hashes, exact whole-paragraph spans, word limits, feature matches, and frozen file hashes. This is not native-speaker annotation or a factual review of the articles. Source wording, grammar, and names are preserved.

From the experiment repository, validate without downloading anything:

```bash
PYTHONPATH=harness "$BENCH_PYTHON" scripts/validate_wikipedia.py
```

To reproduce the selection into a new directory, run `scripts/prepare_wikipedia.py --output <new-directory>` with the same interpreter and `PYTHONPATH`. The script downloads the pinned shards and refuses to overwrite an existing output. Regeneration leaves review status pending; review metadata and checksums are finalized separately.

The container copy is at `/workspace/experiment/data/splitter-en-de-50` on `129.212.176.155`, in `omnivoice-splitter-bench`. The revised [experiment plan](../../EXPERIMENT_PLAN.md) specifies clean generation without diagnostic chunk capture, preservation of the full raw WAV, and whole-output sequential long-form Whisper large-v3-turbo scoring from a separate mono 16 kHz copy. Decoding uses `num_beams=1`, `temperature=0`, `do_sample=False`, and `condition_on_prev_tokens=False`, with no expected-text prompt.

Keep every frozen prompt even when the actual-budget chunk lists match. Derive and record the exact runtime budget and text splits before reporting agreement counts. Runtime preflight found 23 prompts with different chunk text and 27 with identical chunks, with all content and character-bound checks passing. Zero-sample runs are candidate audio joins only, even when their count matches the number of expected joins; per-join measurements remain unvalidated without independent ground truth.

Independent text annotation must classify boundaries as sentence, acceptable phrase, inappropriate, or budget-forced. Native-speaker listening and boundary annotation remain pending. Report whole-output WER, coverage, S/D/I counts, failures, and paired prompt differences by language and sampling group. Client end-to-end timing ends after receiving the full response and before writing audio. Flag the first request after every server start as startup affected and describe timings as single exploratory observations. UTMOS, SIM, TTFP, and streaming underrun metrics are excluded.

## Protocol revision

The revised protocol removes the earlier per-chunk ASR requirement and extra smoke or warm-up generations. It retains the unchanged frozen dataset and follows the [splitter comparison discussion](https://github.com/vllm-project/vllm-omni/pull/6409#issuecomment-6023328036), the pinned dataset source above, and the [Whisper model documentation](https://huggingface.co/openai/whisper-large-v3-turbo), with the explicit request budget and scoring settings recorded in the experiment plan. The completed run is linked above. Human judgments remain pending.
