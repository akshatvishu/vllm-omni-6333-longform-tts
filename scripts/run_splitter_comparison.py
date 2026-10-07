#!/usr/bin/env python3
"""Run exactly one attempt per prompt and variant, without warmup generation.

Run inside the assigned experiment container after checking GPU availability.
The durable attempt journal prevents retries after interruption, including when
it is unknown whether the server received the interrupted request.
"""

# ruff: noqa: E402
# Import the experiment harness after selecting its package path.
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "harness"))

import httpx

import benchmarks

# Prefer this published harness even when the checkout has a regular package.
benchmarks.__path__ = [str(EXPERIMENT_ROOT / "harness/benchmarks")]

from benchmarks.tts.omnivoice_longform.common import (
    build_generation_cases,
    case_asdict,
    latency_summary,
    load_prompt_manifest,
    read_jsonl,
    write_immutable_json,
    write_json,
    write_jsonl,
)
from benchmarks.tts.omnivoice_longform.vllm_omni.benchmark import _generate_case_record

BLOCKS = (("ours", 0, 25), ("theirs", 0, 25), ("theirs", 25, 50), ("ours", 25, 50))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_port_free(port: int) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def stop_server(server: subprocess.Popen) -> None:
    # This process group was created by this runner; never signal other servers.
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(30):
        server.poll()
        try:
            os.killpg(server.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    else:
        os.killpg(server.pid, signal.SIGKILL)
    server.wait()


async def wait_ready(server: subprocess.Popen, port: int) -> None:
    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("server exited before readiness; inspect server log")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
    raise RuntimeError("server readiness exceeded 15 minutes")


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def restore_records(journal: Path) -> dict:
    records = {}
    for event in read_jsonl(journal) if journal.exists() else []:
        key = (event["variant"], event["case_id"])
        if event["event"] == "attempt":
            if key in records:
                raise ValueError(f"duplicate attempt in journal: {key}")
            records[key] = {
                **event,
                "status": "interrupted",
                "error": "Attempt reserved; result unknown. Never retried.",
            }
        elif event["event"] == "result":
            if key not in records or records[key]["status"] != "interrupted":
                raise ValueError(f"result without unique attempt: {key}")
            records[key] = event
        else:
            raise ValueError(f"unknown journal event: {event['event']}")
    return records


def save_results(output: Path, records: dict) -> None:
    summaries = {}
    for variant in ("ours", "theirs"):
        rows = sorted(
            (row for (name, _), row in records.items() if name == variant), key=lambda row: row["order_index"]
        )
        write_jsonl(output / variant / "generation.jsonl", rows)
        speed = [row for row in rows if row["status"] == "success" and not row["cold_start_affected"]]
        summaries[variant] = {
            "attempts": len(rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "cold_start_excluded": sum(row["cold_start_affected"] for row in rows),
            "speed_samples": len(speed),
            "latency_s": latency_summary([row["latency_s"] for row in speed]),
            "rtf": latency_summary([row["rtf"] for row in speed]),
            "peak_reserved_gib": latency_summary(
                [row["peak_reserved_gib"] for row in speed if row.get("peak_reserved_gib") is not None]
            ),
        }
    write_json(
        output / "serving_summary.json",
        {
            "variants": summaries,
            "attempts_reserved": len(records),
            "planned_attempts": 100,
            "warmup_generations": 0,
            "speed_policy": "Exclude the first attempted request after every server start.",
            "timing": "Client POST through complete response body; excludes WAV decoding and saving.",
            "memory": "X-Peak-Memory-MB / 1024; server-reported peak reserved GiB.",
        },
    )


async def run(args: argparse.Namespace) -> None:
    _, prompts = load_prompt_manifest(args.manifest)
    cases = build_generation_cases(prompts, [42])
    if len(prompts) != 50 or len(cases) != 50:
        raise ValueError("expected exactly 50 prompts, one mode and seed per prompt")
    if Counter(case.language for case in cases) != {"English": 25, "German": 25}:
        raise ValueError("expected 25 English and 25 German prompts")
    if any(case.mode != "default" or case.reference_mode != "none" or case.ref_text for case in cases):
        raise ValueError("expected default chunking, automatic voice, and no reference inputs")
    root = args.vllm_omni_root.resolve()
    target = root / "vllm_omni/diffusion/models/omnivoice/chunking.py"
    sources = {name: args.variants_dir / f"{name}.py" for name in ("ours", "theirs")}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.vllm_bin.resolve()),
        "serve",
        args.model,
        "--revision",
        args.model_revision,
        "--omni",
        "--deploy-config",
        "vllm_omni/deploy/omnivoice.yaml",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--log-stats",
        "--trust-remote-code",
    ]
    config = {
        "manifest_sha256": sha256(args.manifest),
        "sources": {name: sha256(path) for name, path in sources.items()},
        "deploy_config_sha256": sha256(root / "vllm_omni/deploy/omnivoice.yaml"),
        "runner_sha256": sha256(Path(__file__)),
        "client_sha256": sha256(EXPERIMENT_ROOT / "harness/benchmarks/tts/omnivoice_longform/vllm_omni/benchmark.py"),
        "command": command,
        "blocks": BLOCKS,
        "seed": 42,
        "gpu_index": args.gpu_index,
        "timeout": args.timeout,
        "warmup_generations": 0,
        "diagnostics": False,
        "python": sys.executable,
        "repo_root": str(root),
    }
    # JSON round trip makes tuples comparable on resumed runs.
    write_immutable_json(output / "config.json", json.loads(json.dumps(config)))
    write_immutable_json(output / "prompts.json", json.loads(args.manifest.read_text()))
    journal = output / "attempts.jsonl"
    records = restore_records(journal)
    expected = {(variant, case.case_id) for variant in sources for case in cases}
    if records.keys() - expected:
        raise ValueError("journal contains unexpected cases")
    save_results(output, records)
    env = os.environ.copy()
    if any("hooks" in part for part in env.get("PYTHONPATH", "").split(os.pathsep)):
        raise ValueError("remove capture hooks from PYTHONPATH before running")
    env.pop("OMNIVOICE_DIAGNOSTICS_DIR", None)
    env.update(
        HIP_VISIBLE_DEVICES=args.gpu_index,
        CUDA_VISIBLE_DEVICES=args.gpu_index,
        VLLM_LOGGING_LEVEL="INFO",
        MIOPEN_FIND_MODE="FAST",
    )
    backup_path = output / "original-chunking.py"
    if not backup_path.exists():
        backup_path.write_bytes(target.read_bytes())
    backup = backup_path.read_bytes()
    server = None
    try:
        for block_index, (variant, start, end) in enumerate(BLOCKS):
            pending = [
                (index, cases[index]) for index in range(start, end) if (variant, cases[index].case_id) not in records
            ]
            if not pending:
                continue
            assert_port_free(args.port)
            target.write_bytes(sources[variant].read_bytes())
            # Python timestamp-based caches can otherwise retain a same-size variant.
            for cached in (target.parent / "__pycache__").glob("chunking.*.pyc"):
                cached.unlink()
            session = f"block{block_index}-{variant}-{uuid.uuid4().hex[:12]}"
            with (output / f"{session}.server.log").open("w") as log:
                server = subprocess.Popen(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                append_event(
                    output / "server_starts.jsonl",
                    {
                        "session": session,
                        "pid": server.pid,
                        "variant": variant,
                        "source_sha256": sha256(target),
                        "unix_time": time.time(),
                    },
                )
                await wait_ready(server, args.port)
                async with httpx.AsyncClient(
                    timeout=args.timeout, trust_env=False, headers={"Authorization": "Bearer EMPTY"}
                ) as client:
                    for ordinal, (index, case) in enumerate(pending):
                        if server.poll() is not None:
                            raise RuntimeError("server exited; no further requests issued")
                        metadata = {
                            **case_asdict(case),
                            "variant": variant,
                            "order_index": index,
                            "block_index": block_index,
                            "server_session": session,
                            "cold_start_affected": ordinal == 0,
                            "concurrency": 1,
                        }
                        attempt = {**metadata, "event": "attempt", "unix_time": time.time()}
                        append_event(journal, attempt)
                        row = await _generate_case_record(
                            client,
                            f"http://127.0.0.1:{args.port}/v1/audio/speech",
                            args.model,
                            case,
                            asyncio.Semaphore(1),
                            output / variant,
                            index,
                        )
                        row = {**row, **metadata, "event": "result"}
                        append_event(journal, row)
                        records[(variant, case.case_id)] = row
                        save_results(output, records)
                        print(f"{len(records)}/100 {variant} {case.case_id}: {row['status']}", flush=True)
                        if row["status"] != "success":
                            raise RuntimeError(
                                "request failed; recorded without retry. Resume to attempt remaining cases"
                            )
                stop_server(server)
                server = None
    finally:
        if server is not None:
            stop_server(server)
        target.write_bytes(backup)
        for cached in (target.parent / "__pycache__").glob("chunking.*.pyc"):
            cached.unlink()
    print(f"Finished: {len(records)}/100 attempts reserved; inspect serving_summary.json for failures/interruption.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-omni-root", type=Path, required=True)
    parser.add_argument("--vllm-bin", type=Path, required=True)
    parser.add_argument("--variants-dir", type=Path, default=EXPERIMENT_ROOT / "variants/splitter-en-de-50")
    parser.add_argument("--manifest", type=Path, default=EXPERIMENT_ROOT / "data/splitter-en-de-50/prompts.json")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results/splitter-en-de-50-reproduction")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--model-revision", default="c5fdb5ccb189668d56333f77ba2629f4cd7535f4")
    return parser.parse_args()


def interrupted(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Also protects source replacement against another runner using this checkout.
    with (args.vllm_omni_root / ".splitter-comparison.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, interrupted)
        asyncio.run(run(args))
