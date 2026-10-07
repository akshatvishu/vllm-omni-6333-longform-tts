"""Check published links, source hashes, and recorded results without a GPU."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from summarize_splitter_results import summarize

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.audio = 0

    def handle_starttag(self, tag, attrs):
        if tag == "audio":
            self.audio += 1
        self.links.extend(value for name, value in attrs if name in {"href", "src"} and value)


def main():
    page = ROOT / "docs/index.html"
    parser = Links()
    parser.feed(page.read_text())
    if parser.audio != 100:
        raise ValueError(f"Expected 100 players, found {parser.audio}")
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc or not url.path:
            continue
        if not (page.parent / unquote(url.path)).exists():
            raise ValueError(f"Broken local link: {link}")
    results = ROOT / "results/splitter-en-de-50"
    config = json.loads((results / "config.json").read_text())
    for variant, expected in config["sources"].items():
        path = ROOT / f"variants/splitter-en-de-50/{variant}.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed frozen splitter: {variant}")
    for copy in (ROOT / "docs/data/splitter-en-de-50").rglob("*"):
        if copy.is_file():
            original = results / copy.relative_to(ROOT / "docs/data/splitter-en-de-50")
            if copy.read_bytes() != original.read_bytes():
                raise ValueError(f"Stale published evidence: {copy}")
    summarize(results)
    print("PASS: 100 players, local links, frozen splitters, published evidence, and recomputed audit")


if __name__ == "__main__":
    main()
