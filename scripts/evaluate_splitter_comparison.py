"""Score clean full-output generations and report paired results by language/group."""

# ruff: noqa: E402
# Select the published harness before importing its modules.
import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "harness"))

import benchmarks

benchmarks.__path__ = [str(EXPERIMENT_ROOT / "harness/benchmarks")]

from benchmarks.tts.omnivoice_longform.common import read_jsonl, write_json, write_jsonl

from benchmarks.tts.omnivoice_longform import evaluate


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["language"], row["bucket"])].append(row)
    result = []
    for (variant, language, group), records in sorted(groups.items()):
        scored = [r for r in records if r["evaluation_status"] == "success"]
        counts = {
            k: sum(r[k] for r in scored)
            for k in ("reference_words", "hits", "substitutions", "deletions", "insertions")
        }
        n = counts["reference_words"]
        generated = [r for r in records if r.get("status") == "success"]
        result.append(
            {
                "variant": variant,
                "language": language,
                "group": group,
                "samples": len(records),
                "scored": len(scored),
                **counts,
                "generation_failures": len(records) - len(generated),
                "asr_failures": sum(r["evaluation_status"] == "transcription_error" for r in records),
                "pooled_wer": sum(counts[k] for k in ("substitutions", "deletions", "insertions")) / n if n else None,
                "pooled_coverage": counts["hits"] / n if n else None,
                "latency_mean_s": statistics.fmean(r["latency_s"] for r in generated) if generated else None,
                "rtf_mean": statistics.fmean(r["rtf"] for r in generated) if generated else None,
            }
        )
    return result


def paired(rows):
    prompts = defaultdict(dict)
    for row in rows:
        prompts[row["prompt_id"]][row["backend"]] = row
    results = []
    for prompt_id, variants in sorted(prompts.items()):
        ours, theirs = variants["ours"], variants["theirs"]
        row = {"prompt_id": prompt_id, "language": ours["language"], "group": ours["bucket"]}
        row["cold_start_affected"] = {
            "ours": ours.get("cold_start_affected", False),
            "theirs": theirs.get("cold_start_affected", False),
        }
        for name in ("wer", "coverage", "latency_s", "rtf", "audio_duration_s", "peak_reserved_gib"):
            a, b = ours.get(name), theirs.get(name)
            row[f"{name}_theirs_minus_ours"] = b - a if a is not None and b is not None else None
        row["listening_preference"] = "pending"
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, help="Relocate WAVs to ROOT/<variant>/<saved WAV basename>")
    parser.add_argument("--whisper-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--model-revision", default="41f01f3fe87f28c78e2fbf8b568835947dd65ed9")
    args = parser.parse_args()
    rows = [r for path in args.records for r in read_jsonl(path)]
    for row in rows:
        row["backend"] = row["variant"]
        if args.audio_root is not None and row.get("audio_path"):
            row["audio_path"] = str((args.audio_root / row["variant"] / Path(row["audio_path"]).name).resolve())
        if row.get("status") == "success" and not Path(row.get("audio_path") or "").is_file():
            raise ValueError(f"Missing WAV for {row['case_id']}; use --audio-root after moving results")
    if len(rows) != 100 or {r["backend"] for r in rows} != {"ours", "theirs"}:
        raise ValueError("Expected exactly 100 generation records for ours/theirs")
    evaluate._validate_backend_cases(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scoring_input = args.output_dir / "scoring-input.jsonl"
    signature = {
        "records_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(evaluate.__file__).read_bytes()).hexdigest(),
        "model": args.whisper_model,
        "revision": args.model_revision,
        "mode": "whole-output sequential long-form",
        "dtype": "float32",
        "do_sample": False,
        "num_beams": 1,
        "temperature": 0.0,
        "condition_on_prev_tokens": False,
        "return_timestamps": True,
        "language": "explicit per prompt",
        "reference_text_prompt": False,
    }
    signature_path = args.output_dir / "scoring-config.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Scoring configuration changed; use a new output directory")
    write_json(signature_path, signature)
    write_jsonl(scoring_input, rows)
    evaluate.run(
        argparse.Namespace(
            records=[scoring_input],
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            model_revision=args.model_revision,
            device="cuda:0",
            dtype="float32",
        )
    )
    evaluated = read_jsonl(args.output_dir / "evaluated.jsonl")
    summary = summarize(evaluated)
    write_json(args.output_dir / "grouped-summary.json", summary)
    write_json(args.output_dir / "paired-differences.json", paired(evaluated))
    lines = [
        "# Splitter comparison: first 100 generations",
        "",
        "One generation per prompt and variant, seed 42, auto voice. Whole-output sequential Whisper turbo.",
        "",
        "WER and coverage are pooled from word counts. ASR mismatches require listening before attribution to TTS.",
        "",
        "| Variant | Language | Group | Scored/total | Generation/ASR failures | WER | Coverage | S / D / I |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        wer = f"{r['pooled_wer']:.2%}" if r["pooled_wer"] is not None else "N/A"
        coverage = f"{r['pooled_coverage']:.2%}" if r["pooled_coverage"] is not None else "N/A"
        lines.append(
            f"| {r['variant']} | {r['language']} | {r['group']} | {r['scored']}/{r['samples']} | "
            f"{r['generation_failures']}/{r['asr_failures']} | {wer} | {coverage} | "
            f"{r['substitutions']} / {r['deletions']} / {r['insertions']} |"
        )
    lines.extend(
        [
            "",
            "Timings are single observations per prompt, including flagged first requests after server startup. "
            "They do not establish a repeatable speed advantage. RTF includes inserted join silence in audio duration.",
            "",
            "Peak memory is the server-reported peak reserved allocator memory when available.",
            "",
            "Native-speaker blind listening and independent boundary annotations remain pending. "
            "No naturalness winner is inferred from WER. Zero-run locations, if provided, are unvalidated candidates.",
        ]
    )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("Saved grouped summary and paired differences", flush=True)


if __name__ == "__main__":
    main()
