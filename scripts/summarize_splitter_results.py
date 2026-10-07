"""Recompute paired splitter statistics from saved evidence, using only the CPU.

The original review audit is read-only. No listening or sentence labels are
inferred from text boundaries, ASR scores, or zero runs.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sign_test(rows):
    deltas = [row["wer_theirs_minus_ours"] for row in rows]
    theirs = sum(value < 0 for value in deltas)
    ours = sum(value > 0 for value in deltas)
    n = theirs + ours
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(theirs, ours) + 1)) / 2**n)
    return {"n": len(rows), "theirs_lower": theirs, "ours_lower": ours, "ties": len(rows) - n, "sign_p": p}


def latency(rows, boundaries):
    values = [row["latency_s_theirs_minus_ours"] for row in rows]
    counts = [boundaries[row["prompt_id"]]["comparison"]["chunk_count_difference_theirs_minus_ours"] for row in rows]
    correlation = None
    if len(set(counts)) > 1 and len(set(values)) > 1:
        correlation = statistics.correlation(counts, values)
    return {
        "n": len(rows),
        "median": statistics.median(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "chunk_count_r": correlation,
    }


def compare(expected, actual, path="audit"):
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError(f"{path}: keys differ")
        for key in expected:
            compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{path}: {expected!r} != {actual!r}")
    elif expected != actual:
        raise ValueError(f"{path}: {expected!r} != {actual!r}")


def summarize(root):
    boundary_data = load(root / "boundaries.json")
    boundaries = {row["prompt_id"]: row for row in boundary_data["prompts"]}
    pairs = load(root / "evaluation/paired-differences.json")
    pair_ids = [row["prompt_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != set(boundaries):
        raise ValueError("Paired and boundary prompt IDs do not match uniquely")
    evaluated = [json.loads(line) for line in (root / "evaluation/evaluated.jsonl").read_text().splitlines()]
    records = {(row["prompt_id"], row["variant"]): row for row in evaluated}
    if len(records) != 2 * len(pairs) or len(records) != len(evaluated):
        raise ValueError("Expected exactly one evaluation per prompt and variant")
    for pair in pairs:
        ours = records[pair["prompt_id"], "ours"]
        theirs = records[pair["prompt_id"], "theirs"]
        for metric in ("wer", "coverage", "latency_s", "audio_duration_s", "rtf", "peak_reserved_gib"):
            compare(pair[f"{metric}_theirs_minus_ours"], theirs[metric] - ours[metric], metric)
        compare(
            pair["cold_start_affected"],
            {v: records[pair["prompt_id"], v]["cold_start_affected"] for v in ("ours", "theirs")},
            "cold_start_affected",
        )
    different = []
    first_changed = []
    exclusive = {"ours": 0, "theirs": 0}
    for prompt in boundaries.values():
        variants = prompt["variants"]
        same = [chunk["text"] for chunk in variants["ours"]["chunks"]] == [
            chunk["text"] for chunk in variants["theirs"]["chunks"]
        ]
        compare(prompt["comparison"]["same_chunks"], same, "same_chunks")
        if not same:
            different.append(prompt["prompt_id"])
        if variants["ours"]["chunks"][0]["text"] != variants["theirs"]["chunks"][0]["text"]:
            first_changed.append(prompt["prompt_id"])
        offsets = {v: {chunk["normalized_end"] for chunk in variants[v]["chunks"][:-1]} for v in ("ours", "theirs")}
        for v, other in (("ours", "theirs"), ("theirs", "ours")):
            exclusive[v] += len(offsets[v] - offsets[other])
    changed_pairs = [row for row in pairs if row["prompt_id"] in different]
    same_pairs = [row for row in pairs if row["prompt_id"] not in different]
    warm = [row for row in pairs if not any(row["cold_start_affected"].values())]
    audit = {"all": sign_test(changed_pairs)}
    for language in ("German", "English"):
        audit[language] = sign_test([row for row in changed_pairs if row["language"] == language])
    audit.update(
        same_split_equal_wer=sum(row["wer_theirs_minus_ours"] == 0 for row in same_pairs),
        duration_different=sum(row["audio_duration_s_theirs_minus_ours"] != 0 for row in pairs),
        duration_delta_sum=sum(row["audio_duration_s_theirs_minus_ours"] for row in pairs),
        first_chunk_changed=first_changed,
        exclusive_boundaries=exclusive,
        latency_same=latency([row for row in warm if row["prompt_id"] not in different], boundaries),
        latency_different=latency([row for row in warm if row["prompt_id"] in different], boundaries),
    )
    diagnostics = load(root / "audio-review/audio-diagnostics.json")["records"]
    diagnostic_ids = [(row["prompt_id"], row["variant"]) for row in diagnostics]
    if len(diagnostic_ids) != len(set(diagnostic_ids)) or set(diagnostic_ids) != set(records):
        raise ValueError("Audio diagnostics do not cover every evaluated output uniquely")
    file_errors = []
    exact_totals = 0
    count_matches = 0
    frame_rate = boundary_data["metadata"]["frame_rate"]
    for row in diagnostics:
        chunks = boundaries[row["prompt_id"]]["variants"][row["variant"]]["chunks"]
        candidates = [c for c in row["zero_run_candidates"] if not c["touches_audio_edge"]]
        if len(candidates) != len(chunks) - 1:
            raise ValueError(f"{row['audio_path']}: candidate count mismatch")
        count_matches += 1
        # Exclude measured zero-run extents. Fades can extend them into speech.
        starts = [0] + [c["end_frame"] for c in candidates]
        ends = [c["start_frame"] for c in candidates] + [row["frames"]]
        errors = [
            abs((end - start) / row["sample_rate"] * frame_rate - chunk["estimated_frames"])
            for start, end, chunk in zip(starts, ends, chunks)
        ]
        file_errors.append(max(errors))
        predicted_samples = sum(chunk["estimated_frames"] for chunk in chunks) / frame_rate * row["sample_rate"]
        predicted_samples += (len(chunks) - 1) * row["minimum_zero_run_frames"]
        exact_totals += predicted_samples == row["frames"]
    audit["join_length_check"] = {
        "max_abs_frame_error": max(file_errors),
        "median_file_max": statistics.median(file_errors),
        "files_within4": sum(error <= 4 for error in file_errors),
        "exact_total_predicted_files": exact_totals,
        "method": "Zero-run extents excluded; frames at 25Hz",
    }
    audit["runner_hash_matches_prerun_config"] = (
        hashlib.sha256((root / "execution-scripts/run_splitter_comparison.py").read_bytes()).hexdigest()
        == load(root / "config.json")["runner_sha256"]
    )
    compare(load(root / "review-claims-audit.json"), audit)
    return {
        "audit_validation": "PASS",
        "recomputed_audit": audit,
        "different_prompt_ids": different,
        "same_split_prompt_count": len(same_pairs),
        "internal_candidate_count_matches_expected_files": count_matches,
        "listening_judgments": "pending",
        "boundary_annotations": "pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--results", type=Path, default=project / "results/splitter-en-de-50")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "recomputed-summary.json"
    evidence_paths = (
        "boundaries.json",
        "evaluation/paired-differences.json",
        "evaluation/evaluated.jsonl",
        "audio-review/audio-diagnostics.json",
        "config.json",
        "review-claims-audit.json",
        "execution-scripts/run_splitter_comparison.py",
    )
    if output.resolve() in {(args.results / path).resolve() for path in evidence_paths}:
        parser.error("Original evidence files must remain unchanged")
    summary = summarize(args.results)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: saved recomputed summary to {output}")


if __name__ == "__main__":
    main()
