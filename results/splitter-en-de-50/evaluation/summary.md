# OmniVoice long form benchmark

Coverage is `1 - (substitutions + deletions) / reference words`. Insertions are included in WER.
Numbers are not normalized to spoken forms. Decimal-case coverage is secondary to boundaries and pauses.

| Backend | Mode | Bucket | Words min/mean/max | Scored/total | Failures | Coverage mean | Coverage stddev | WER mean | Latency mean (s) | RTF mean | Peak reserved GPU memory mean (GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ours | default | random | 199/268.1/330 | 30/30 | 0 | 94.65% | 3.73% | 0.0611 | 6.41 | 0.056 | 4.95 |
| ours | default | targeted | 232/283.6/330 | 20/20 | 0 | 94.32% | 4.43% | 0.0667 | 6.13 | 0.050 | 5.02 |
| theirs | default | random | 199/268.1/330 | 30/30 | 0 | 94.85% | 3.53% | 0.0587 | 6.40 | 0.056 | 4.95 |
| theirs | default | targeted | 232/283.6/330 | 20/20 | 0 | 94.09% | 4.23% | 0.0684 | 6.23 | 0.051 | 5.02 |
