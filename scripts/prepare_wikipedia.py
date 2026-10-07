"""Freeze paired English/German prompts without rewriting Wikipedia text."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from benchmarks.tts.omnivoice_longform.common import load_prompt_manifest, word_count
from huggingface_hub import HfApi, hf_hub_download

DATASET = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
LANGUAGES = {"en": "English", "de": "German"}
SEED = 42
# Exclude damaged source passages before ranking, without consulting either splitter.
REVIEW_EXCLUSIONS = {
    "en:55681184": "Selected paragraph ends with an unclosed quotation in the source.",
    "en:55677350": "Selected paragraph ends with an incomplete sentence in the source.",
}
FEATURES = {
    "dotted_initialism": r"(?<!\w)(?:[A-Za-z]\.){2,}",
    "spaced_initials": r"(?<!\w)[A-Za-z]\.\s+[A-Za-z]\.",
    "single_initial": r"(?<!\w)[A-Z]\.\s+[A-Z][a-z]",
    "short_number_period": r"(?<![\w.])\d{1,2}\.\s+[A-ZÄÖÜ]",
    "decimal": r"\d[.,]\d",
    "english_abbreviation": r"\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|Co|Corp|Inc|Ltd)\.",
    "german_abbreviation": r"\b(?:bzw|ca|usw|vgl|Nr|Anm|Kap|Abb|ggf|Jh|Jhd|geb|gest|Str)\.",
    "german_date": (
        r"\b\d{1,2}\.\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|"
        r"September|Oktober|November|Dezember)\b"
    ),
}
TARGETS = {
    "en": ["dotted_initialism", "spaced_initials", "single_initial", "short_number_period", "english_abbreviation"] * 2,
    "de": ["german_abbreviation", "spaced_initials", "german_date", "short_number_period", "dotted_initialism"] * 2,
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rank(value):
    return digest(f"{SEED}:{value}")


def features(text):
    return {
        name: [m.group() for m in re.finditer(pattern, text)]
        for name, pattern in FEATURES.items()
        if re.search(pattern, text)
    }


def passage(article):
    """Choose whole consecutive prose paragraphs, never splitter-derived spans."""
    lines = list(re.finditer(r"[^\n]+", article))
    candidates = []
    for i, line in enumerate(lines):
        for end_line in lines[i:]:
            paragraph = end_line.group().strip()
            if len(paragraph.split()) < 25 or not re.match(r'[A-ZÄÖÜ“"„]', paragraph):
                break
            if paragraph.startswith(("Category:", "Kategorie:", "http")):
                break
            start, end = line.start(), end_line.end()
            text = article[start:end]
            count = len(text.split())
            if count > 400:
                break
            if count >= 200 and re.search(r'[.!?]["”’»)]*$', text):
                candidates.append((abs(count - 300), start, end))
    if not candidates:
        return None
    _, start, end = min(candidates)
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/splitter-en-de-50"))
    parser.add_argument("--cache", type=Path, default=Path("results/dataset-preparation/cache"))
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=REVISION, token=False)
    prompts, provenance, source_articles, selections = [], [], [], {}
    for lang, language in LANGUAGES.items():
        config = f"20231101.{lang}"
        shards = [f for f in files if f.startswith(config + "/") and f.endswith(".parquet")]
        shard = min(shards, key=rank)
        print(f"Downloading {lang}: {shard}", flush=True)
        path = hf_hub_download(
            DATASET, shard, repo_type="dataset", revision=REVISION, cache_dir=args.cache, token=False
        )
        candidates, excluded = [], Counter()
        read_count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["id", "url", "title", "text"]):
            for row in batch.to_pylist():
                if read_count >= args.rows:
                    break
                index = read_count
                read_count += 1
                if re.search(r"\(disambiguation\)|\(Begriffsklärung\)", row["title"], re.I):
                    excluded["disambiguation_title"] += 1
                    continue
                if f"{lang}:{row['id']}" in REVIEW_EXCLUSIONS:
                    excluded["source_quality_review"] += 1
                    continue
                span = passage(row["text"])
                if span is None:
                    excluded["no_200_400_word_contiguous_prose_passage"] += 1
                    continue
                start, end = span
                excerpt = row["text"][start:end]
                candidates.append(
                    {
                        "row": row,
                        "row_index": index,
                        "start": start,
                        "end": end,
                        "text": excerpt,
                        "features": features(excerpt),
                    }
                )
            if read_count >= args.rows:
                break
        candidates.sort(key=lambda c: rank(f"{lang}:{c['row']['id']}"))
        if len(candidates) < 25:
            raise RuntimeError(f"Only {len(candidates)} eligible candidates for {lang}")
        selected = [(c, "random", None) for c in candidates[:15]]
        used = {c["row"]["id"] for c, _, _ in selected}
        for target in TARGETS[lang]:
            chosen = next((c for c in candidates if c["row"]["id"] not in used and target in c["features"]), None)
            if chosen is None:
                raise RuntimeError(f"Not enough {lang} candidates for {target}; expand --rows before freezing")
            selected.append((chosen, "targeted", target))
            used.add(chosen["row"]["id"])
        selections[lang] = {
            "config": config,
            "shard": shard,
            "shard_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "rows_scanned": read_count,
            "eligible_articles": len(candidates),
            "exclusions": dict(excluded),
            "eligible_feature_counts": dict(Counter(f for c in candidates for f in c["features"])),
        }
        counts = Counter()
        for candidate, group, target in selected:
            counts[group] += 1
            row = candidate["row"]
            text = candidate["text"]
            prompt_id = f"wiki_{lang}_{group}_{counts[group]:02d}"
            source_id = f"{config}:{row['id']}"
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "bucket": group,
                    "source_id": source_id,
                    "category": group,
                    "word_count": word_count(text),
                    "text": text,
                    "text_sha256": digest(text),
                    "language": language,
                    "reference_mode": "none",
                    "modes": ["default"],
                }
            )
            provenance.append(
                {
                    "prompt_id": prompt_id,
                    "source_id": source_id,
                    "article_id": row["id"],
                    "title": row["title"],
                    "url": row["url"],
                    "language": language,
                    "group": group,
                    "shard": shard,
                    "row_index": candidate["row_index"],
                    "start_char": candidate["start"],
                    "end_char": candidate["end"],
                    "article_sha256": digest(row["text"]),
                    "text_sha256": digest(text),
                    "whitespace_word_count": len(text.split()),
                    "features": candidate["features"],
                    "selection_feature": target,
                    "boundary_review": "pending",
                }
            )
            source_articles.append({"source_id": source_id, **row})
        print(f"{lang}: selected {dict(counts)} from {len(candidates)} eligible articles", flush=True)
    source = {
        "kind": "diagnostic",
        "dataset": DATASET,
        "revision": REVISION,
        "selection_seed": SEED,
        "review_exclusions": REVIEW_EXCLUSIONS,
        "sampling": (
            "One hash-selected shard per language; first N rows; "
            "hash-ranked eligible articles. Not uniform over Wikipedia."
        ),
        "passage_policy": (
            "One exact 200–400-word span of consecutive whole prose paragraphs per article, closest to 300 words."
        ),
        "selection": selections,
        "primary_generation_seed": 42,
        "license": "CC BY-SA 3.0 and GFDL as specified by the dataset card; retain source URLs and attribution.",
    }
    args.output.mkdir(parents=True)
    for filename, data in (
        ("prompts.json", {"source": source, "prompts": prompts}),
        ("provenance.json", {"source": source, "prompts": provenance}),
    ):
        (args.output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (args.output / "source-articles.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_articles)
    )
    load_prompt_manifest(args.output / "prompts.json")
    print(f"Wrote {len(prompts)} prompts to {args.output}. Boundary review is pending.")


if __name__ == "__main__":
    main()
