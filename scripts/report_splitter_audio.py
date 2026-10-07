"""CPU audio diagnostics and a listening form. No joins or judgments are inferred."""

import argparse
import hashlib
import html
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def interval(start, end, rate):
    return {
        "start_frame": start,
        "end_frame": end,
        "start_seconds": start / rate,
        "end_seconds": end / rate,
        "duration_seconds": (end - start) / rate,
    }


def analyze_samples(samples, rate, expected_chunk_count):
    if samples.ndim != 2 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty, finite audio shaped [frames, channels]")
    # The inspected joiner inserts int(0.3 * sample_rate) // 3 zero frames.
    minimum_zero = max(1, int(0.3 * rate) // 3)
    window = max(1, int(0.02 * rate))
    starts = np.arange(0, len(samples), window)
    lengths = np.minimum(window, len(samples) - starts)
    squared = np.mean(samples * samples, axis=1)
    rms = np.sqrt(np.add.reduceat(squared, starts) / lengths)
    quiet_runs = [(start * window, min(end * window, len(samples))) for start, end in runs(rms < 0.001)]
    quiet_intervals = [interval(start, end, rate) for start, end in quiet_runs]
    candidates = []
    for start, end in runs(np.all(samples == 0, axis=1)):
        if end - start < minimum_zero:
            continue
        overlapping = [(left, right) for left, right in quiet_runs if left < end and right > start]
        quiet_start, quiet_end = max(overlapping, key=lambda span: min(span[1], end) - max(span[0], start))
        candidates.append(
            {
                **interval(start, end, rate),
                "join_status": "UNVALIDATED",
                "touches_audio_edge": start == 0 or end == len(samples),
                "overlapping_quiet_interval": interval(quiet_start, quiet_end, rate),
            }
        )
    expected_joins = max(0, expected_chunk_count - 1)
    internal = [candidate for candidate in candidates if not candidate["touches_audio_edge"]]
    return {
        "sample_rate": rate,
        "frames": len(samples),
        "channels": samples.shape[1],
        "duration_seconds": len(samples) / rate,
        "minimum_zero_run_frames": minimum_zero,
        "expected_cpu_chunk_count": expected_chunk_count,
        "expected_cpu_join_count": expected_joins,
        "zero_run_candidate_count": len(candidates),
        "internal_zero_run_candidate_count": len(internal),
        "internal_candidate_count_matches_expected": len(internal) == expected_joins,
        "join_validation": "UNVALIDATED",
        "zero_run_candidates": candidates,
        "quiet_intervals": quiet_intervals,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def listening_page(items, manifest_hash):
    sections = []
    for item in items:
        number = item["number"]
        players = "".join(
            f'<p>{label}: <audio controls preload="none" src="audio/{number:02d}-{label}.wav"></audio></p>'
            if item["available"][label]
            else f"<p>{label}: Audio unavailable.</p>"
            for label in ("A", "B")
        )
        choices = " ".join(
            f'<label><input type="radio" name="choice-{number}" value="{value}">{label}</label>'
            for value, label in (("A", "A"), ("B", "B"), ("tie", "Tie"), ("unable", "Unable to judge"))
        )
        sections.append(
            f'<fieldset data-prompt="{html.escape(item["prompt_id"], quote=True)}">'
            f"<legend>{number}. {html.escape(item['language'])}</legend>"
            f'<p class="text">{html.escape(item["text"])}</p>{players}'
            f"<p>Which recording has more natural pauses and phrasing? {choices}</p>"
            '<label>Notes <textarea rows="2"></textarea></label></fieldset>'
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paired audio listening</title>
<style>body{max-width:1000px;margin:2em auto;padding:0 1em;font:17px system-ui;line-height:1.5}
fieldset{margin:2em 0;padding:1em}.text{white-space:pre-wrap}audio{width:90%;vertical-align:middle}
textarea{width:95%}button{padding:.7em}label{margin-right:.8em}</style>
<h1>Paired audio listening</h1>
<p>Listen to both recordings with the same playback settings. Compare pauses and phrasing against the text.
Use Unable to judge when you cannot make a fair comparison. All choices start blank.</p>
<p>A and B order varies by prompt. Keep the answer key closed during listening.
Anonymous links conceal the variant in this page; filesystem inspection can reveal it.</p>
<p>Export your answers before closing this page. Answers are not saved automatically.</p>
<label>Listener ID <input id="listener"></label><button id="export">Export answers as JSON</button>
"""
        + "\n".join(sections)
        + """
<script>
document.querySelector('#export').onclick = () => {
  const judgments = [...document.querySelectorAll('fieldset')].map(field => ({
    prompt_id: field.dataset.prompt,
    choice: field.querySelector('input:checked')?.value ?? null,
    notes: field.querySelector('textarea').value
  }));
  const result = {manifest_sha256: MANIFEST_HASH, listener_id: document.querySelector('#listener').value,
    exported_at: new Date().toISOString(), judgments};
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'listening-judgments.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
</script></html>
""".replace("MANIFEST_HASH", json.dumps(manifest_hash))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, help="Defaults to RUN_DIR/boundaries.json")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN_DIR/audio-review; must not exist")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    boundary_path = args.boundaries or root / "boundaries.json"
    output = args.output_dir or root / "audio-review"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    manifest_path = root / "prompts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    boundaries = json.loads(boundary_path.read_text())
    if boundaries["metadata"]["manifest"]["sha256"] != manifest_hash:
        # The driver may reformat the immutable manifest when copying it.
        expected = {row["prompt_id"]: row["text_sha256"] for row in boundaries["prompts"]}
        actual = {row["prompt_id"]: hashlib.sha256(row["text"].encode()).hexdigest() for row in manifest["prompts"]}
        if expected != actual:
            parser.error("CPU boundaries and run manifest have different prompts")
    expected_by_id = {row["prompt_id"]: row for row in boundaries["prompts"]}
    records = {}
    for variant in ("ours", "theirs"):
        path = root / variant / "generation.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (variant, row["prompt_id"])
            if key in records:
                parser.error(f"Duplicate generation record: {key}")
            records[key] = row
    rng = random.Random(42)
    diagnostics, key_rows, page_items = [], [], []
    (output / "audio").mkdir(parents=True)
    for number, prompt in enumerate(manifest["prompts"], 1):
        prompt_id = prompt["prompt_id"]
        order = ["ours", "theirs"]
        rng.shuffle(order)
        available = {}
        labels = {}
        for label, variant in zip(("A", "B"), order):
            row = records.get((variant, prompt_id))
            labels[label] = variant
            available[label] = False
            if row is None or row.get("status") != "success":
                diagnostics.append({"prompt_id": prompt_id, "variant": variant, "status": "audio_unavailable"})
                continue
            # Driver stores remote absolute paths. Rebase only its WAV basename.
            audio = root / variant / Path(row["audio_path"]).name
            if not audio.is_file():
                diagnostics.append({"prompt_id": prompt_id, "variant": variant, "status": "wav_missing"})
                continue
            samples, rate = sf.read(audio, dtype="float64", always_2d=True)
            cpu = expected_by_id[prompt_id]["variants"][variant]
            diagnostics.append(
                {
                    "prompt_id": prompt_id,
                    "variant": variant,
                    "status": "analyzed",
                    "audio_path": str(audio.relative_to(root)),
                    "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                    "expected_cpu_chunks": cpu["chunks"],
                    **analyze_samples(samples, rate, cpu["chunk_count"]),
                }
            )
            link = output / "audio" / f"{number:02d}-{label}.wav"
            link.symlink_to(os.path.relpath(audio, link.parent))
            available[label] = True
        key_rows.append({"number": number, "prompt_id": prompt_id, "labels": labels})
        page_items.append({"number": number, **prompt, "available": available})
    report = {
        "manifest_sha256": manifest_hash,
        "boundaries_sha256": hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
        "join_validation": "UNVALIDATED",
        "method": {
            "zero_runs": "Every channel is exactly zero at native WAV sample rate; minimum int(0.3*rate)//3 frames.",
            "quiet_intervals": (
                "RMS <0.001 (-60 dBFS) in nonoverlapping native-rate 20 ms windows, averaging squared "
                "samples across channels. Final partial window uses its actual length. No resampling."
            ),
            "candidate_local_quiet_duration": (
                "Duration of the quiet interval with greatest overlap with each zero-run candidate. Window "
                "boundaries can truncate or extend the span; not a validated join pause."
            ),
            "limits": (
                "Natural silence can be exactly zero. Fades can extend zero runs. Matching expected join "
                "counts does not validate alignment. No sentence correctness labels or listening judgments "
                "are supplied."
            ),
        },
        "records": diagnostics,
    }
    write_json(output / "audio-diagnostics.json", report)
    write_json(output / "listening-key.json", {"seed": 42, "manifest_sha256": manifest_hash, "mapping": key_rows})
    (output / "listening.html").write_text(listening_page(page_items, manifest_hash))
    print(
        json.dumps(
            {
                "prompts": len(page_items),
                "analyzed": sum(row["status"] == "analyzed" for row in diagnostics),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
