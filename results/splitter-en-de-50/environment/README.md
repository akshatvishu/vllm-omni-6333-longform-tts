# Recorded environment

`setup-environment.json` and `python-packages.txt` are snapshots from container preparation, before generation. The false run flags describe that snapshot time; the experiment subsequently completed with `controller.exit` equal to zero. The package listing includes image-local wheel paths and is evidence, not a portable requirements file.

The run used one AMD MI300X with base image `vllm/vllm-openai-rocm:v0.31.0`. The recorded image identifier from setup-status.txt is `sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f`.

Use the repository reproduction instructions to prepare a compatible stack. Different hardware, dependencies, or kernels may change generated samples and timings even with a fixed seed.
