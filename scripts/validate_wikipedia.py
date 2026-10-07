"""Check frozen prompts against their saved source articles without network access."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmarks.tts.omnivoice_longform.common import load_prompt_manifest
from prepare_wikipedia import features


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1] / "data/splitter-en-de-50"
    load_prompt_manifest(root / "prompts.json")
    manifest = json.loads((root / "prompts.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    articles = [json.loads(line) for line in (root / "source-articles.jsonl").read_text().splitlines()]
    prompts = manifest["prompts"]
    metadata = {row["prompt_id"]: row for row in provenance["prompts"]}
    sources = {row["source_id"]: row for row in articles}
    assert len(prompts) == len(metadata) == len(sources) == 50
    assert len({row["text_sha256"] for row in prompts}) == 50
    assert manifest["source"] == provenance["source"]
    assert Counter((p["language"], p["bucket"]) for p in prompts) == {
        ("English", "random"): 15,
        ("English", "targeted"): 10,
        ("German", "random"): 15,
        ("German", "targeted"): 10,
    }
    for prompt in prompts:
        meta = metadata[prompt["prompt_id"]]
        article = sources[prompt["source_id"]]
        text = prompt["text"]
        start, end = meta["start_char"], meta["end_char"]
        assert article["text"][start:end] == text
        assert start == 0 or article["text"][start - 1] == "\n"
        assert end == len(article["text"]) or article["text"][end] == "\n"
        assert digest(article["text"]) == meta["article_sha256"]
        assert digest(text) == prompt["text_sha256"] == meta["text_sha256"]
        assert 200 <= len(text.split()) == meta["whitespace_word_count"] <= 400
        assert features(text) == meta["features"]
        assert meta["selection_feature"] is None or meta["selection_feature"] in features(text)
        assert prompt["modes"] == ["default"] and prompt["reference_mode"] == "none"
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    print("PASS: 50 unique prompts; language/group counts, exact source spans, features, schema and frozen hashes.")


if __name__ == "__main__":
    main()
