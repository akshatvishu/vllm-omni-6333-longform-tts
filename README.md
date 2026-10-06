# vLLM-Omni #6333: long-form TTS

Audio comparisons and reproduction scripts for [issue #6333](https://github.com/vllm-project/vllm-omni/issues/6333) and [PR #6409](https://github.com/vllm-project/vllm-omni/pull/6409).

**[Listen to the comparison](https://akshatvishu.github.io/vllm-omni-6333-longform-tts/)**

This experiment compares the original PR splitter, numeric-separator protection only, and a broader sentence-first splitter. It contains 60 outputs: two synthetic English prompts × ten matched seeds × three variants, generated on one MI300X. Whisper large-v3-turbo scored every output on GPU. Fourteen affected chunks from an earlier run were also transcribed separately.

| Splitter | Decimal prompt coverage | Comma prompt coverage |
|---|---:|---:|
| Original | 96.67% | 99.80% |
| Numeric protection only | 97.27% | 99.80% |
| Sentence-first candidate | 96.59% | 100.00% |

Numeric protection prevents splitting inside `0.26`. These ASR scores do not establish an audio-quality winner or equivalence. Paired sign tests on decimal coverage give p=0.289 and p=0.344 against the original. No human listening assessment was performed. This is not the commenter's H100 MIG setup and does not reproduce their German inputs.

- [Reproduce the runs](REPRODUCE.md)
- [Prompts](prompts.json), [scores and metadata](docs/data/), [splitter variants](variants/)
- Complete outputs are lossless FLAC, verified sample-for-sample against the generated PCM16 WAV files. Diagnostic chunks retain their original floating-point WAV format.

The narrow numeric fix was retained for its smaller behavior change. Sentence-first splitting remains a separate candidate. No model weights or reference voices are included.
