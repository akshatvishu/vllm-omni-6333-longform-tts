# Frozen splitter implementations

`ours.py` is the sentence-first splitter used for the 100-output run. Its bytes match `vllm_omni/diffusion/models/omnivoice/chunking.py` in PR commit `71b8cafa4d62afad25ce92cee2714623f16b7e0d`. The experiment ran those bytes on source base `2ece3987427d24a7825973b94b7d6db662b262e2` before that follow-up commit was published.

`theirs.py` reconstructs the splitter proposal in [lazariv's comment](https://github.com/vllm-project/vllm-omni/pull/6409#issuecomment-6023328036). It includes the additional abbreviation, initialism, and short-number period rules. It is the complete combined proposal, not an ablation of individual rules.

| File | SHA256 |
| --- | --- |
| ours.py | `84e07ebf944de2a71e133e0643ade789eb1fafb1c9660c5a00c21fa7fa64aab4` |
| theirs.py | `6c8b6f578bfbda6f590a4070cde72092634a8080927ba9fdf7470acb22fa35aa` |

Keep these files unchanged when reproducing the recorded comparison. The original PR splitter is a different baseline and is not one of these two variants. The earlier 60-output experiment uses the files in the parent directory.
