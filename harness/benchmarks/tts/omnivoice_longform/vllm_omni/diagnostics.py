# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker instrumentation for benchmark runs, including step execution.

Start the server with this directory on PYTHONPATH and set
OMNIVOICE_DIAGNOSTICS_DIR to an absolute output directory. Timings include
synchronization and file capture overhead and are not throughput measurements.
"""

import contextvars
import functools
import json
import os
import re
import time
from pathlib import Path

import soundfile as sf
import torch

_CURRENT = contextvars.ContextVar("omnivoice_diagnostic", default=None)
_EXECUTION = contextvars.ContextVar("omnivoice_execution", default="unknown")


def install():
    destination = os.environ.get("OMNIVOICE_DIAGNOSTICS_DIR")
    if not destination:
        return
    from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import OmniVoicePipeline

    if getattr(OmniVoicePipeline, "_benchmark_diagnostics", False):
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    def emit(event, **values):
        with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "time_ns": time.time_ns(), **values}) + "\n")

    def sync(pipeline):
        if torch.device(pipeline.device).type == "cuda":
            torch.accelerator.synchronize(pipeline.device)

    original_request = OmniVoicePipeline._prepare_request_input
    original_chunk = OmniVoicePipeline._prepare_chunk_input
    original_decode = OmniVoicePipeline._decode_chunk

    @functools.wraps(original_request)
    def request(self, prompt, extra):
        case_id = extra.get("diagnostic_case_id")
        if not case_id:
            return original_request(self, prompt, extra)
        data = {
            "diagnostic_case_id": case_id,
            "submitted_seed": extra.get("diagnostic_seed"),
            "execution_mode": _EXECUTION.get(),
            "voice_name": prompt.get("voice_name") if isinstance(prompt, dict) else None,
        }
        token = _CURRENT.set(data)
        sync(self)
        started = time.perf_counter()
        try:
            prepared = original_request(self, prompt, extra)
            sync(self)
            if hasattr(prepared, "target_len"):
                chunks = prepared.chunks
                text = (
                    (prompt.get("input") or prompt.get("text") or prompt.get("prompt"))
                    if isinstance(prompt, dict)
                    else str(prompt)
                )
                # Request preparation already encoded the reference. Use that same
                # tensor to record the estimate used for the split decision.
                ref_tokens = data.pop("ref_tokens", None)
                ref_text = data.get("ref_text")
                estimate = self._estimate_target_length(text, ref_text, ref_tokens)
                emit(
                    "request_prepared",
                    **data,
                    full_text=text,
                    full_estimated_duration_s=estimate / self.config.frame_rate,
                    chunk_texts=chunks.texts if chunks else [text],
                    actual_chunk_count=len(chunks.texts) if chunks else 1,
                    preparation_s=time.perf_counter() - started,
                    chunk_duration=extra.get("audio_chunk_duration", self.config.audio_chunk_duration),
                    chunk_threshold=extra.get("audio_chunk_threshold", self.config.audio_chunk_threshold),
                )
            else:
                emit("preparation_error", **data, error=str(getattr(prepared, "error", prepared)))
            return prepared
        finally:
            _CURRENT.reset(token)

    @functools.wraps(original_chunk)
    def chunk(self, text, lang, instruct, ref_text, ref_audio_tokens, seed, chunks=None):
        data = _CURRENT.get() or (getattr(chunks, "_benchmark_diagnostic", None) if chunks else None)
        result = original_chunk(self, text, lang, instruct, ref_text, ref_audio_tokens, seed, chunks)
        if data is not None:
            data.update(language=lang, ref_text=ref_text)
            if _CURRENT.get() is data:
                data["ref_tokens"] = ref_audio_tokens
            metadata = {key: value for key, value in data.items() if key != "ref_tokens"}
            result._benchmark_diagnostic = metadata
            if chunks is not None:
                chunks._benchmark_diagnostic = metadata
            sync(self)
            result._benchmark_started = time.perf_counter()
            emit(
                "chunk_prepared",
                **metadata,
                chunk_index=chunks.index if chunks else 0,
                text=text,
                target_frames=result.target_len,
                estimated_duration_s=result.target_len / self.config.frame_rate,
                reference_frames=ref_audio_tokens.shape[-1] if ref_audio_tokens is not None else 0,
            )
        return result

    @functools.wraps(original_decode)
    def decode(self, prepared, tokens):
        data = getattr(prepared, "_benchmark_diagnostic", None)
        if data is None:
            return original_decode(self, prepared, tokens)
        index = prepared.chunks.index if prepared.chunks else 0
        sync(self)
        decode_started = time.perf_counter()
        generation_s = decode_started - prepared._benchmark_started
        captured = []

        def capture(module, args, audio):
            sync(self)
            captured.append((audio.detach().cpu().float().reshape(-1).numpy(), time.perf_counter()))

        hook = self.decoder.register_forward_hook(capture)
        try:
            result = original_decode(self, prepared, tokens)
        finally:
            hook.remove()
        if captured:
            audio, decoded_at = captured[0]
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(data["diagnostic_case_id"]))
            path = root / f"{name}-pid{os.getpid()}-chunk{index:03d}.wav"
            sf.write(path, audio, self.sample_rate, subtype="FLOAT")
            emit(
                "chunk_decoded",
                **data,
                chunk_index=index,
                audio_path=str(path),
                audio_duration_s=len(audio) / self.sample_rate,
                generation_s=generation_s,
                decode_and_copy_s=decoded_at - decode_started,
            )
        return result

    def execution_wrapper(original, mode):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            token = _EXECUTION.set(mode)
            try:
                return original(*args, **kwargs)
            finally:
                _EXECUTION.reset(token)

        return wrapped

    OmniVoicePipeline.forward = execution_wrapper(OmniVoicePipeline.forward, "full_request")
    OmniVoicePipeline.prepare_encode = execution_wrapper(OmniVoicePipeline.prepare_encode, "step")
    OmniVoicePipeline._prepare_request_input = request
    OmniVoicePipeline._prepare_chunk_input = chunk
    OmniVoicePipeline._decode_chunk = decode
    OmniVoicePipeline._benchmark_diagnostics = True
    emit("instrumentation_installed")
