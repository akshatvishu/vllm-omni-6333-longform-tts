# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

from benchmarks.tts.omnivoice_longform.common import (
    DEFAULT_SEEDS,
    build_generation_cases,
    case_asdict,
    chunking_args,
    load_prompt_manifest,
    read_jsonl,
    representative_warmup_cases,
    write_json,
    write_jsonl,
)
from benchmarks.tts.omnivoice_longform.reference.diagnostics import capture_generation, save_capture


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synchronize() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def _reset_peak_memory() -> dict[str, float]:
    if not torch.accelerator.is_available():
        return {}
    torch.accelerator.reset_peak_memory_stats()
    gib = 1024**3
    return {
        "allocated_before_gib": torch.accelerator.memory_allocated() / gib,
        "reserved_before_gib": torch.accelerator.memory_reserved() / gib,
    }


def _peak_memory() -> dict[str, float]:
    if not torch.accelerator.is_available():
        return {}
    gib = 1024**3
    return {
        "peak_allocated_gib": torch.accelerator.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.accelerator.max_memory_reserved() / gib,
    }


def _generation_config(mode: str) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(**chunking_args(mode))


def _request_args(case):
    result = {"text": case.text, "language": case.language, "generation_config": _generation_config(case.mode)}
    if case.ref_audio:
        result["ref_audio"] = case.ref_audio
        if case.ref_text is not None:
            result["ref_text"] = case.ref_text
    return result


def _gpu_evidence(model):
    device = next(model.parameters()).device
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError(f"Expected GPU model; got {device}")
    return {
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "model_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
    }


def _log(output_dir, event, **fields):
    with (output_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")


def _warm_asr(model, cases, args, output_dir):
    refs = sorted({case.ref_audio for case in cases if case.ref_audio and not case.ref_text})
    if not refs:
        return {"required": False}
    asr_path = Path(args.asr_model)
    if not asr_path.exists():
        if not args.asr_model_revision:
            raise ValueError("ASR requires a pinned --asr-model-revision or local snapshot path")
        asr_path = Path(snapshot_download(args.asr_model, revision=args.asr_model_revision))
    started = time.perf_counter()
    model.load_asr_model(model_name=str(asr_path), device="cuda:0")
    evidence = _gpu_evidence(model._asr_pipe.model)
    evidence["input_devices"] = []

    def verify_inputs(module, inputs, kwargs):
        tensors = [item for item in (*inputs, *kwargs.values()) if isinstance(item, torch.Tensor)]
        if not tensors or any(item.device.type != "cuda" for item in tensors):
            raise RuntimeError("Whisper encoder inputs are not all on GPU")
        evidence["input_devices"] = sorted({str(item.device) for item in tensors})

    # Keep this verification active for both warmup and measured auto-transcription.
    model._asr_pipe.model.get_encoder().register_forward_pre_hook(verify_inputs, with_kwargs=True)
    transcripts = {ref: model.transcribe(ref) for ref in refs}
    _synchronize()
    evidence.update(
        {
            "required": True,
            "snapshot": str(asr_path.resolve()),
            "cold_load_and_warmup_s": time.perf_counter() - started,
            "warmup_transcripts": transcripts,
            "measurement": "ASR load excluded; repeated warm transcription included",
        }
    )
    _log(output_dir, "asr_warmup", **evidence)
    return evidence


def prepare_reference(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path(snapshot_download(args.model, revision=args.model_revision))
    model = OmniVoice.from_pretrained(
        model_path, device_map=args.device, dtype=getattr(torch, args.dtype), asr_device="cuda:0"
    )
    evidence = _gpu_evidence(model)
    text = (
        "The morning light falls across the quiet room. I will read this passage clearly, "
        "at a steady pace, with a natural and relaxed voice."
    )
    _seed_everything(args.seeds[0])
    audio = np.asarray(model.generate(text=text, language="English", duration=8.0)[0]).squeeze()
    if audio.size / model.sampling_rate >= 10:
        raise RuntimeError("Generated reference exceeds the 10 second limit")
    sf.write(output_dir / "reference.wav", audio, model.sampling_rate, subtype="FLOAT")
    write_json(
        output_dir / "reference.json",
        {
            "text": text,
            "duration_s": audio.size / model.sampling_rate,
            "seed": args.seeds[0],
            "model_revision": args.model_revision,
            **evidence,
        },
    )


def run(args: argparse.Namespace) -> None:
    if args.prepare_reference:
        prepare_reference(args)
        return
    if args.manifest is None:
        raise ValueError("--manifest is required for benchmark generation")
    _, prompts = load_prompt_manifest(args.manifest)
    cases = build_generation_cases(prompts, args.seeds)
    warmup_cases = representative_warmup_cases(cases, concurrency=1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generation.jsonl"
    existing_rows = read_jsonl(generation_path) if generation_path.exists() else []
    if any(row.get("diagnostic_run", False) != args.diagnostics for row in existing_rows):
        raise ValueError("Timing and diagnostic runs require separate output directories")
    rows_by_case_id = {row["case_id"]: row for row in existing_rows}
    if len(rows_by_case_id) != len(existing_rows):
        raise ValueError(f"duplicate cases in checkpoint: {generation_path}")
    expected_case_ids = {case.case_id for case in cases}
    unexpected_case_ids = rows_by_case_id.keys() - expected_case_ids
    if unexpected_case_ids:
        raise ValueError(f"unexpected cases in checkpoint: {sorted(unexpected_case_ids)}")

    pending_cases = [case for case in cases if case.case_id not in rows_by_case_id]
    model = None
    gpu = {}
    asr = {}
    if pending_cases:
        dtype = getattr(torch, args.dtype)
        model_path = Path(args.model)
        if not model_path.exists():
            model_path = Path(snapshot_download(args.model, revision=args.model_revision))
        load_start = time.perf_counter()
        model = OmniVoice.from_pretrained(
            model_path,
            device_map=args.device,
            dtype=dtype,
            asr_device="cuda:0",
        )
        init_time_s = time.perf_counter() - load_start
        gpu = _gpu_evidence(model)
        asr = _warm_asr(model, pending_cases, args, output_dir)
        write_json(output_dir / "environment.json", {**gpu, "asr": asr})

        for index, case in enumerate(warmup_cases, start=1):
            print(f"[warmup {index}/{len(warmup_cases)}] {case.mode} {case.bucket}")
            _seed_everything(case.seed)
            try:
                model.generate(**_request_args(case))
            except Exception:
                _log(output_dir, "warmup_error", case_id=case.case_id, traceback=traceback.format_exc())
    else:
        init_time_s = existing_rows[0]["init_time_s"]
        environment_path = output_dir / "environment.json"
        if environment_path.exists():
            environment = json.loads(environment_path.read_text())
            asr = environment.pop("asr", {})
            gpu = environment

    for index, case in enumerate(cases, start=1):
        if case.case_id in rows_by_case_id:
            print(f"[{index}/{len(cases)}] {case.case_id}: restored from checkpoint")
            continue

        _seed_everything(case.seed)
        started = time.perf_counter()
        capture = None
        _log(output_dir, "request_start", case_id=case.case_id, request=case_asdict(case))
        try:
            _synchronize()
            memory = _reset_peak_memory()
            started = time.perf_counter()
            with capture_generation(model, args.diagnostics) as capture:
                audio = model.generate(**_request_args(case))[0]
            _synchronize()
            latency_s = time.perf_counter() - started
            memory.update(_peak_memory())

            audio = np.asarray(audio, dtype=np.float32).squeeze()
            if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"invalid audio returned for {case.case_id}")

            audio_path = output_dir / f"{case.case_id}.wav"
            sf.write(audio_path, audio, model.sampling_rate, subtype="FLOAT")
            duration_s = audio.size / model.sampling_rate
            row = {
                **case_asdict(case),
                "backend": "reference",
                "status": "success",
                "order_index": index - 1,
                "audio_path": str(audio_path.resolve()),
                "sample_rate": model.sampling_rate,
                "audio_duration_s": duration_s,
                "latency_s": latency_s,
                "rtf": latency_s / duration_s,
                "init_time_s": init_time_s,
                **memory,
            }
            print(
                f"[{index}/{len(cases)}] {case.case_id}: {latency_s:.2f}s, "
                f"{duration_s:.2f}s audio, RTF {row['rtf']:.3f}"
            )
        except Exception as error:
            row = {
                **case_asdict(case),
                "backend": "reference",
                "status": "error",
                "order_index": index - 1,
                "audio_path": None,
                "sample_rate": model.sampling_rate,
                "audio_duration_s": None,
                "latency_s": time.perf_counter() - started,
                "rtf": None,
                "init_time_s": init_time_s,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            print(f"[{index}/{len(cases)}] {case.case_id}: failed: {type(error).__name__}: {error}")

        if capture is not None:
            try:
                row["diagnostics"] = save_capture(capture, output_dir, case.case_id, model.sampling_rate)
            except Exception as error:
                row.update(
                    status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
                )
        row["diagnostic_run"] = args.diagnostics
        _log(output_dir, "request_end", **row)
        rows_by_case_id[case.case_id] = row
        write_jsonl(
            generation_path,
            [rows_by_case_id[item.case_id] for item in cases if item.case_id in rows_by_case_id],
        )

    rows = [rows_by_case_id[case.case_id] for case in cases]
    successful_requests = sum(row["status"] == "success" for row in rows)
    summary = {
        "backend": "reference",
        "diagnostic_run": args.diagnostics,
        "gpu": gpu,
        "asr": asr,
        "model": args.model,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "device": args.device,
        "init_time_s": init_time_s,
        "warmup_requests": len(warmup_cases),
        "measured_requests": len(rows),
        "successful_requests": successful_requests,
        "failed_requests": len(rows) - successful_requests,
        "batch_size": 1,
        "concurrency": 1,
        "seeds": args.seeds,
        "case_order": [case.case_id for case in cases],
    }
    write_json(output_dir / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the reference OmniVoice implementation")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--model-revision")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--prepare-reference", action="store_true")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--asr-model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
