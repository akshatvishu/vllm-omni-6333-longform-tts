"""CPU preflight of exact splitter sources using OmniVoice's no-reference budget.

Supply the duration.py used by the runtime and its model frame rate. This script
does not load model weights or establish that a supplied source matches a remote
runtime. Source hashes make that comparison auditable. Sentence annotations are
left pending; boundary differences are not correctness or audio-quality scores.
"""

import argparse
import ast
import hashlib
import json
import math
import re
import runpy
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_metadata(path):
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes())}


def load_splitter(path):
    """Load trusted local splitter code without importing torch or audio joins."""
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and not (isinstance(node, ast.FunctionDef) and node.name == "join_audio_chunks")
    ]
    namespace = {"re": re}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["split_text_into_chunks"]


def normalize(text):
    return "".join(character for character in text if not character.isspace())


def describe_chunks(text, chunks, budget, estimate_frames):
    normalized = normalize(text)
    original_offsets = [index for index, character in enumerate(text) if not character.isspace()]
    preserved = normalize("".join(chunks)) == normalized
    bounded = bool(chunks) and all(0 < len(chunk) <= budget for chunk in chunks)
    rows = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        end = cursor + len(normalize(chunk))
        row = {
            "index": index,
            "text": chunk,
            "text_sha256": sha256(chunk.encode()),
            "characters": len(chunk),
            "estimated_frames": estimate_frames(chunk),
        }
        # A failed preservation check must never produce misleading coordinates.
        if preserved and end > cursor:
            row.update(
                normalized_start=cursor,
                normalized_end=end,
                original_start=original_offsets[cursor],
                original_end=original_offsets[end - 1] + 1,
            )
        rows.append(row)
        cursor = end
    boundaries = []
    if preserved:
        for left, right in zip(rows, rows[1:]):
            if "normalized_end" not in left or "normalized_start" not in right:
                continue
            boundaries.append(
                {
                    "normalized_offset": left["normalized_end"],
                    "original_left_end": left["original_end"],
                    "original_right_start": right["original_start"],
                    "left_context": text[max(0, left["original_end"] - 80) : left["original_end"]],
                    "right_context": text[right["original_start"] : right["original_start"] + 80],
                    "annotation": "pending",
                }
            )
    return {
        "chunk_count": len(chunks),
        "content_preserved_ignoring_whitespace": preserved,
        "character_bounds_valid": bounded,
        "chunks": rows,
        "boundaries": boundaries,
    }


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--theirs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/splitter-en-de-50/prompts.json",
    )
    parser.add_argument("--duration-estimator", type=Path, required=True, help="Trusted runtime duration.py source")
    parser.add_argument("--frame-rate", type=positive_float, required=True, help="Frame rate from runtime model config")
    parser.add_argument("--chunk-duration", type=positive_float, default=15.0)
    parser.add_argument("--chunk-threshold", type=positive_float, default=30.0)
    parser.add_argument("--expected-prompts", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    manifest = json.loads(args.manifest.read_text())
    prompts = manifest["prompts"]
    if len(prompts) != args.expected_prompts:
        parser.error(f"Expected {args.expected_prompts} prompts, found {len(prompts)}")
    if len({prompt["prompt_id"] for prompt in prompts}) != len(prompts):
        parser.error("Prompt IDs must be unique")
    # run_name is not __main__, so duration.py's example program does not run.
    estimator = runpy.run_path(str(args.duration_estimator))["RuleDurationEstimator"]()

    def estimate_frames(text):
        return max(1, int(estimator.estimate_duration(text, "Nice to meet you.", 25)))

    splitters = {"ours": load_splitter(args.ours), "theirs": load_splitter(args.theirs)}
    records = []
    for prompt in prompts:
        if prompt.get("reference_mode") != "none":
            parser.error(f"{prompt['prompt_id']}: only reference_mode=none is supported")
        text = prompt["text"]
        text_hash = sha256(text.encode())
        if text_hash != prompt.get("text_sha256"):
            parser.error(f"{prompt['prompt_id']}: frozen text hash mismatch")
        target_frames = estimate_frames(text)
        triggered = target_frames > args.chunk_threshold * args.frame_rate
        frame_budget = args.chunk_duration * args.frame_rate
        budget = len(text) if frame_budget >= target_frames else max(1, int(frame_budget * len(text) / target_frames))
        variants = {}
        for name, split in splitters.items():
            chunks = split(text, budget) if triggered else [text]
            variants[name] = describe_chunks(text, chunks, budget if triggered else len(text), estimate_frames)
        ours, theirs = variants["ours"], variants["theirs"]
        ours_offsets = {boundary["normalized_offset"] for boundary in ours["boundaries"]}
        theirs_offsets = {boundary["normalized_offset"] for boundary in theirs["boundaries"]}
        records.append(
            {
                "prompt_id": prompt["prompt_id"],
                "language": prompt.get("language"),
                "bucket": prompt.get("bucket"),
                "text_sha256": text_hash,
                "characters": len(text),
                "estimated_target_frames": target_frames,
                "chunking_triggered": triggered,
                "character_budget": budget if triggered else None,
                "variants": variants,
                "comparison": {
                    "same_chunks": [row["text"] for row in ours["chunks"]] == [row["text"] for row in theirs["chunks"]],
                    "chunk_count_difference_theirs_minus_ours": theirs["chunk_count"] - ours["chunk_count"],
                    "shared_boundaries": sorted(ours_offsets & theirs_offsets),
                    "ours_only_boundaries": sorted(ours_offsets - theirs_offsets),
                    "theirs_only_boundaries": sorted(theirs_offsets - ours_offsets),
                },
            }
        )
    valid = all(
        variant["content_preserved_ignoring_whitespace"] and variant["character_bounds_valid"]
        for record in records
        for variant in record["variants"].values()
    )
    summary = {
        "prompt_count": len(records),
        "chunking_triggered_count": sum(record["chunking_triggered"] for record in records),
        "different_chunk_text_count": sum(not record["comparison"]["same_chunks"] for record in records),
        "different_chunk_count_count": sum(
            record["comparison"]["chunk_count_difference_theirs_minus_ours"] != 0 for record in records
        ),
        "total_chunks": {
            name: sum(record["variants"][name]["chunk_count"] for record in records) for name in splitters
        },
        "all_content_and_character_bounds_valid": valid,
        "sentence_annotation": "pending",
    }
    report = {
        "metadata": {
            "manifest": source_metadata(args.manifest),
            "splitter_sources": {"ours": source_metadata(args.ours), "theirs": source_metadata(args.theirs)},
            "duration_estimator": source_metadata(args.duration_estimator),
            "frame_rate": args.frame_rate,
            "chunk_duration_seconds": args.chunk_duration,
            "chunk_threshold_seconds": args.chunk_threshold,
            "reference_text": "Nice to meet you.",
            "reference_frames": 25,
            "normalization": "Remove all str.isspace() characters, preserving every other Unicode code point.",
            "offsets": "Zero-based Unicode code-point offsets; end offsets are exclusive.",
            "scope": (
                "CPU prediction for supplied sources/config; "
                "verify hashes and settings against runtime before audio runs."
            ),
            "limits": (
                "Character bounds are not strict duration bounds. "
                "Boundary differences do not establish sentence correctness."
            ),
        },
        "summary": summary,
        "prompts": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not valid:
        raise SystemExit("Content preservation or character bound validation failed; inspect report")


if __name__ == "__main__":
    main()
