# Splitter comparison: first 100 generations

One generation per prompt and variant, seed 42, auto voice. Whole-output sequential Whisper turbo.

WER and coverage are pooled from word counts. ASR mismatches require listening before attribution to TTS.

| Variant | Language | Group | Scored/total | Generation/ASR failures | WER | Coverage | S / D / I |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ours | English | random | 15/15 | 0/0 | 4.76% | 96.09% | 141 / 21 / 35 |
| ours | English | targeted | 10/10 | 0/0 | 5.80% | 95.36% | 119 / 17 / 34 |
| ours | German | random | 15/15 | 0/0 | 7.55% | 93.04% | 194 / 78 / 23 |
| ours | German | targeted | 10/10 | 0/0 | 7.63% | 93.14% | 145 / 43 / 21 |
| theirs | English | random | 15/15 | 0/0 | 4.78% | 96.04% | 142 / 22 / 34 |
| theirs | English | targeted | 10/10 | 0/0 | 6.24% | 94.89% | 117 / 33 / 33 |
| theirs | German | random | 15/15 | 0/0 | 7.09% | 93.45% | 187 / 69 / 21 |
| theirs | German | targeted | 10/10 | 0/0 | 7.48% | 93.21% | 151 / 35 / 19 |

Timings are single observations per prompt, including flagged first requests after server startup. They do not establish a repeatable speed advantage. RTF includes inserted join silence in audio duration.

Peak memory is the server-reported peak reserved allocator memory when available.

Native-speaker blind listening and independent boundary annotations remain pending. No naturalness winner is inferred from WER. Zero-run locations, if provided, are unvalidated candidates.
