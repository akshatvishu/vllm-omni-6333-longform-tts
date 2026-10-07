# vLLM-Omni #6333 long-form TTS experiments

[Listen and inspect the latest comparison](https://akshatvishu.github.io/vllm-omni-6333-longform-tts/). The repository contains the prompts, splitter source, raw transcripts, scores, lossless audio, and scripts for [issue #6333](https://github.com/vllm-project/vllm-omni/issues/6333) and [PR #6409](https://github.com/vllm-project/vllm-omni/pull/6409).

## Latest experiment

We generated each of 50 Wikipedia passages once with each splitter, for 100 outputs on one AMD MI300X. The dataset has 25 English and 25 German passages, with random and targeted punctuation groups. Both variants used seed 42 and auto voice. There were no repeated generations or warm-up requests.

The splitters produced different chunk text on 23 prompts. Those prompts are the main comparison; the other 27 are controls with identical splits.

| Differing prompts | Theirs lower WER | Ours lower WER | Ties | Exact two-sided sign test |
| --- | ---: | ---: | ---: | ---: |
| All 23 | 11 | 6 | 6 | 0.332 |
| German 16 | 9 | 2 | 5 | 0.065 |
| English 7 | 2 | 4 | 1 | 0.688 |

WER is word error rate from Whisper transcription. These results do not establish a quality winner. There is only one generation seed, and listening judgments remain pending. Timing differences depend on chunk count; neither splitter has an established overall speed advantage. See the [full report](results/splitter-en-de-50/REPORT.md) for paired timing, controls, exclusions, and limitations.

"Ours" is the sentence-first revision now included in PR commit `71b8cafa4d62afad25ce92cee2714623f16b7e0d`. The recorded runtime used base `2ece3987427d24a7825973b94b7d6db662b262e2` with that exact splitter file. "Theirs" is the reconstructed proposal in [lazariv's comment](https://github.com/vllm-project/vllm-omni/pull/6409#issuecomment-6023328036). Both files are frozen in [variants/splitter-en-de-50](variants/splitter-en-de-50/); their hashes are recorded in the run configuration. This experiment does not compare against the original unmodified PR splitter.

## Reproduce and inspect

- [Reproduction instructions](REPRODUCE.md) cover CPU inspection, restoring WAVs, rescoring, and a fresh 100-output run.
- [Frozen dataset](data/splitter-en-de-50/) includes source attribution, exact text spans, selection code, and checksums.
- [Results](results/splitter-en-de-50/) include all 100 transcripts and scores, generation records, boundary metadata, execution logs, and pinned configuration.
- [Audio](docs/audio/splitter-en-de-50/) contains all 100 outputs as lossless FLAC. [The publication manifest](results/splitter-en-de-50/published-audio.json) records original WAV hashes, FLAC hashes, and decoded PCM hashes.
- [The earlier 60-output experiment](https://akshatvishu.github.io/vllm-omni-6333-longform-tts/previous-experiment.html) remains available. Its synthetic prompts and three variants are separate from the latest comparison.

The Pages workflow validates the frozen dataset and all audio, rebuilds the latest page, and deploys `docs/`. It does not run model inference. No weights or reference voices are included. Wikipedia passages retain their source licensing and attribution; see [NOTICE](NOTICE) and the dataset README.
