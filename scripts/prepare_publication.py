#!/usr/bin/env python3
"""Publish lossless PCM16 audio, verify it, or restore the recorded WAV files."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/splitter-en-de-50"
AUDIO = ROOT / "docs/audio/splitter-en-de-50"
MANIFEST = RESULTS / "published-audio.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm_hash(samples):
    return hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--restore-wav", action="store_true")
    args = parser.parse_args()
    if args.verify or args.restore_wav:
        entries = json.loads(MANIFEST.read_text())
        if len(entries) != 100:
            raise ValueError("Expected 100 published outputs")
        for entry in entries:
            flac = ROOT / entry["flac"]
            if sha256(flac) != entry["flac_sha256"]:
                raise ValueError(f"FLAC checksum mismatch: {flac}")
            samples, rate = sf.read(flac, dtype="int16", always_2d=True)
            if (rate, samples.shape[0], samples.shape[1], pcm_hash(samples)) != (
                entry["sample_rate"],
                entry["frames"],
                entry["channels"],
                entry["pcm_sha256"],
            ):
                raise ValueError(f"PCM mismatch: {flac}")
            if args.restore_wav:
                wav = ROOT / entry["wav"]
                if not wav.exists():
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(wav, samples, rate, subtype="PCM_16")
                if sha256(wav) != entry["wav_sha256"]:
                    raise ValueError(f"Original WAV checksum mismatch: {wav}")
        if args.restore_wav:
            key = json.loads((RESULTS / "audio-review/listening-key.json").read_text())
            links = RESULTS / "audio-review/audio"
            links.mkdir(exist_ok=True)
            for row in key["mapping"]:
                for label, variant in row["labels"].items():
                    target = RESULTS / variant / (row["prompt_id"] + "_default_seed42.wav")
                    link = links / f"{row['number']:02d}-{label}.wav"
                    if not link.exists():
                        link.symlink_to(os.path.relpath(target, links))
        print(f"Verified {len(entries)} lossless files" + (" and restored WAVs" if args.restore_wav else ""))
        return

    entries = []
    for variant in ("ours", "theirs"):
        wavs = sorted((RESULTS / variant).glob("*.wav"))
        if len(wavs) != 50:
            raise ValueError(f"Expected 50 WAVs for {variant}, found {len(wavs)}")
        (AUDIO / variant).mkdir(parents=True, exist_ok=True)
        for wav in wavs:
            if sf.info(wav).subtype != "PCM_16":
                raise ValueError(f"Refusing lossy conversion of {wav}")
            samples, rate = sf.read(wav, dtype="int16", always_2d=True)
            flac = AUDIO / variant / (wav.stem + ".flac")
            sf.write(flac, samples, rate, format="FLAC", subtype="PCM_16")
            decoded, decoded_rate = sf.read(flac, dtype="int16", always_2d=True)
            if decoded_rate != rate or not np.array_equal(samples, decoded):
                raise ValueError(f"Lossless check failed: {flac}")
            entries.append(
                {
                    "wav": str(wav.relative_to(ROOT)),
                    "flac": str(flac.relative_to(ROOT)),
                    "wav_sha256": sha256(wav),
                    "flac_sha256": sha256(flac),
                    "pcm_sha256": pcm_hash(samples),
                    "frames": len(samples),
                    "channels": samples.shape[1],
                    "sample_rate": rate,
                }
            )
    MANIFEST.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Published {len(entries)} sample-exact lossless FLAC files")


if __name__ == "__main__":
    main()
