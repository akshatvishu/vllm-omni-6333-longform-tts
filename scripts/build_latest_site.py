#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the static English/German splitter report from saved evidence."""

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/splitter-en-de-50"
DOCS = ROOT / "docs"
DATA = DOCS / "data/splitter-en-de-50"


def read(name):
    return json.loads((RESULTS / name).read_text())


def escape(value):
    return html.escape(str(value), quote=True).replace("\n", "&#10;").replace("\r", "&#13;")


def pct(value):
    return f"{value * 100:.2f}%"


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    index = DOCS / "index.html"
    previous = DOCS / "previous-experiment.html"
    if not previous.exists():
        if not index.exists() or "60 complete outputs" not in index.read_text():
            raise RuntimeError("The previous 60-output report must be preserved first.")
        shutil.copyfile(index, previous)

    evidence = [
        "all-metrics.json",
        "review-claims-audit.json",
        "boundaries.json",
        "prompts.json",
        "config.json",
        "runtime-chunk-config.json",
        "identical-split-audio-comparison.json",
        "local-generation-audit.json",
        "transfer-verification.json",
        "serving_summary.json",
        "evaluation/paired-differences.json",
        "evaluation/scoring-config.json",
        "evaluation/evaluated.jsonl",
        "audio-review/audio-diagnostics.json",
    ]
    for name in evidence:
        destination = DATA / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RESULTS / name, destination)

    metrics = read("all-metrics.json")
    audit = read("review-claims-audit.json")
    boundaries = read("boundaries.json")
    pairs = {p["prompt_id"]: p for p in read("evaluation/paired-differences.json")}
    records = [json.loads(line) for line in (RESULTS / "evaluation/evaluated.jsonl").read_text().splitlines()]
    by_output = {(r["prompt_id"], r["backend"]): r for r in records}
    prompts = read("prompts.json")["prompts"]
    assert len(prompts) == 50 and len(records) == 100
    by_boundary = {b["prompt_id"]: b for b in boundaries["prompts"]}
    first_changed = set(audit["first_chunk_changed"])
    assert len(first_changed) == 4

    quality_rows = []
    for language, count in [("all", 50), ("English", 25), ("German", 25)]:
        ours = metrics["ours"]["quality"][language]
        theirs = metrics["theirs"]["quality"][language]
        quality_rows.append(
            [
                language.capitalize(),
                count,
                f"{ours['reference_words']:,}",
                pct(ours["wer"]),
                pct(theirs["wer"]),
                f"{100 * (theirs['wer'] - ours['wer']):+.3f} pp",
                pct(ours["coverage"]),
                pct(theirs["coverage"]),
            ]
        )
    quality_table = table(
        [
            "Language",
            "Pairs",
            "Words per splitter",
            "Our WER ↓",
            "Their WER ↓",
            "Their WER minus ours",
            "Our coverage ↑",
            "Their coverage ↑",
        ],
        quality_rows,
    )

    sign_table = table(
        ["Different splits", "Pairs", "Ours lower WER", "Theirs lower WER", "Ties", "Exact two-sided p"],
        [
            [
                language.capitalize(),
                audit[language]["n"],
                audit[language]["ours_lower"],
                audit[language]["theirs_lower"],
                audit[language]["ties"],
                f"{audit[language]['sign_p']:.4f}",
            ]
            for language in ["all", "English", "German"]
        ],
    )
    group_rows = []
    groups = {(r["language"], r["group"], r["variant"]): r for r in metrics["language_group_quality"]}
    for language in ["English", "German"]:
        for group in ["random", "targeted"]:
            ours, theirs = (groups[language, group, variant] for variant in ["ours", "theirs"])
            group_rows.append(
                [
                    language,
                    group,
                    ours["samples"],
                    pct(ours["pooled_wer"]),
                    pct(theirs["pooled_wer"]),
                    pct(ours["pooled_coverage"]),
                    pct(theirs["pooled_coverage"]),
                ]
            )
    group_table = table(
        ["Language", "Selection", "Pairs", "Our WER", "Their WER", "Our coverage", "Their coverage"], group_rows
    )
    performance = metrics["performance_excluding_startup"]["variants"]
    speed_table = table(
        ["Metric", "Ours", "Theirs"],
        [
            ["Mean client latency", *[f"{performance[v]['latency_s']['mean']:.3f} s" for v in ["ours", "theirs"]]],
            ["Median client latency", *[f"{performance[v]['latency_s']['p50']:.3f} s" for v in ["ours", "theirs"]]],
            ["Mean real-time factor", *[f"{performance[v]['rtf']['mean']:.5f}" for v in ["ours", "theirs"]]],
            [
                "Mean reported peak reserved memory",
                *[f"{performance[v]['peak_reserved_gib']['mean']:.3f} GiB" for v in ["ours", "theirs"]],
            ],
            ["Total audio duration (all 50)", *[f"{metrics[v]['audio']['total_s']:.2f} s" for v in ["ours", "theirs"]]],
            ["Total chunks (all 50)", *[metrics[v]["chunks"]["total"] for v in ["ours", "theirs"]]],
            [
                "Quiet fraction (all 50)",
                *[pct(metrics[v]["quiet"]["duration_weighted_fraction"]) for v in ["ours", "theirs"]],
            ],
        ],
    )

    paired_latency_table = table(
        [
            "Paired latency (theirs minus ours)",
            "Pairs",
            "Median difference",
            "Sample standard deviation",
            "Chunk-count difference correlation",
        ],
        [
            [
                label,
                audit[key]["n"],
                f"{audit[key]['median']:+.3f} s",
                f"{audit[key]['sample_sd']:.3f} s",
                "Not defined (no chunk-count variation)"
                if audit[key]["chunk_count_r"] is None
                else f"r = {audit[key]['chunk_count_r']:.3f}",
            ]
            for label, key in [
                ("Same splits (A/A controls)", "latency_same"),
                ("Different splits", "latency_different"),
            ]
        ],
    )
    options, cases = [], []
    for prompt in prompts:
        pid = prompt["prompt_id"]
        boundary = by_boundary[pid]
        different = not boundary["comparison"]["same_chunks"]
        flags = "Different splits" if different else "Same splits (A/A control)"
        if pid in first_changed:
            flags += ", first chunk changed"
        options.append(
            f'<option value="{escape(pid)}" data-language="{escape(prompt["language"])}" data-different="{int(different)}" data-first="{int(pid in first_changed)}">{escape(pid)} | {escape(flags)}</option>'
        )
        cards = []
        for variant, label in [("ours", "Ours"), ("theirs", "Theirs")]:
            record = by_output[pid, variant]
            chunks = boundary["variants"][variant]["chunks"]
            stem = Path(record["audio_path"]).stem
            chunks_html = "".join(
                f'<li><span class="chunk-meta">Chunk {c["index"] + 1}, {c["characters"]} characters, predicted {c["estimated_frames"] / 25:.2f} s</span><p>{escape(c["text"])}</p></li>'
                for c in chunks
            )
            cards.append(f'''<article class="audio-card {variant}"><h3>{label}</h3>
<audio controls preload="none" aria-label="{label} audio for {escape(pid)}" src="audio/splitter-en-de-50/{variant}/{escape(stem)}.flac"></audio>
<p class="scores">WER {pct(record["wer"])}, coverage {pct(record["coverage"])}<br>{record["audio_duration_s"]:.2f} s of audio, {len(chunks)} chunks</p>
<p class="muted">Substitutions {record["substitutions"]}, deletions {record["deletions"]}, insertions {record["insertions"]}</p>
<details><summary>Whisper transcript</summary><p class="text" lang="{"de" if prompt["language"] == "German" else "en"}">{escape(record["transcript"])}</p></details>
<details><summary>Predicted chunk text and boundaries</summary><ol class="chunks">{chunks_html}</ol></details></article>''')
        pair = pairs[pid]
        cases.append(f'''<section class="case" id="{escape(pid)}" data-language="{escape(prompt["language"])}" data-different="{int(different)}" data-first="{int(pid in first_changed)}">
<div class="case-heading"><h3>{escape(pid)}</h3><span class="badge {"changed" if different else ""}">{escape(flags)}</span></div>
<p class="muted">{escape(prompt["language"])}, {escape(prompt["bucket"])} selection, seed 42. Listening judgment pending.</p>
{'<p class="notice">The first chunk differs. Auto voice is used, so a voice change may confound a listening comparison of pauses and phrasing.</p>' if pid in first_changed else ""}
<details><summary>Full input text ({prompt["word_count"]} words)</summary><p class="text" lang="{"de" if prompt["language"] == "German" else "en"}">{escape(prompt["text"])}</p><p class="muted">Frozen source ID {escape(prompt["source_id"])}</p></details>
<p>Theirs minus ours. WER {pair["wer_theirs_minus_ours"] * 100:+.3f} percentage points, coverage {pair["coverage_theirs_minus_ours"] * 100:+.3f} percentage points, duration {pair["audio_duration_s_theirs_minus_ours"]:+.2f} s.</p>
<div class="audio-grid">{"".join(cards)}</div></section>''')

    links = "".join(
        f'<li><a href="data/splitter-en-de-50/{escape(name)}" download>{escape(name)}</a></li>' for name in evidence
    )
    page = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="OmniVoice English and German long-form TTS splitter comparison. 50 paired prompts, audio, transcripts, boundary evidence, and limitations.">
<title>OmniVoice splitter comparison | vLLM-Omni #6333</title>
<style>
:root{color-scheme:light;--ink:#172b3a;--muted:#526576;--line:#d8e2e7;--blue:#165eac;--teal:#087c71}[hidden]{display:none!important}*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1180px;padding:36px 24px 60px;margin:auto}a{color:var(--blue);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:10px 24px;font-size:.9rem}header{padding:46px 0 28px}.eyebrow{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);font-weight:700}h1{font-size:clamp(2rem,5vw,3.3rem);line-height:1.15;letter-spacing:-.04em;max-width:900px;margin:12px 0 22px}h2{font-size:1.55rem;line-height:1.3;margin:0 0 16px}h3{font-size:1.12rem;margin:0 0 12px}p{margin:12px 0}.lead{font-size:1.15rem;max-width:960px}.muted{color:var(--muted);font-size:.93rem}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.stat,.panel{background:white;border:1px solid var(--line);border-radius:12px}.stat{padding:18px}.stat strong{display:block;font-size:2rem;line-height:1.2}.stat span{color:var(--muted);font-size:.9rem}.panel{padding:26px;margin:22px 0}.notice{border-left:4px solid #bf8527;background:#fff8e9;padding:14px 18px}.conclusion{border-left:4px solid var(--teal);background:#eef8f6;padding:16px 20px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:.93rem}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{font-size:.82rem;color:var(--muted);background:#f8fafb}tbody tr:last-child td{border:0}.controls{display:grid;grid-template-columns:1fr 1fr 2fr;gap:16px;background:#eff4f7;padding:18px;border-radius:8px;margin:20px 0}label{font-size:.88rem;display:block}select,button{font:inherit;max-width:100%;border:1px solid #bbcbd5;border-radius:6px;background:white;padding:9px 10px;color:var(--ink)}select{display:block;width:100%;margin-top:6px}button{cursor:pointer}button:hover{background:#e8f0f5}button:disabled{opacity:.4;cursor:default}.pager{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.case{border-top:1px solid var(--line);margin-top:24px;padding-top:24px}.case[hidden]{display:none}.case-heading{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.badge{font-size:.8rem;background:#edf1f4;border-radius:20px;padding:4px 10px}.badge.changed{background:#e5f4f1;color:#075d53}.audio-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:18px}.audio-card{border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:8px;padding:20px;min-width:0}.audio-card.theirs{border-top-color:var(--teal)}audio{width:100%;margin:8px 0}.scores{font-weight:600}details{border-top:1px solid var(--line);margin:12px 0;padding-top:12px}summary{cursor:pointer;font-weight:600}.text{white-space:pre-wrap;overflow-wrap:anywhere}.chunks{padding-left:24px;font-size:.92rem}.chunks li{margin-bottom:18px}.chunk-meta{color:var(--muted);font-size:.82rem}.downloads{columns:2;padding-left:20px;font-size:.9rem}.downloads li{padding:4px;overflow-wrap:anywhere}footer{font-size:.9rem;color:var(--muted);padding-top:14px}code{font-size:.9em;overflow-wrap:anywhere}@media(max-width:760px){main{padding:20px 14px}.panel{padding:18px}.stats{grid-template-columns:1fr 1fr}.controls,.audio-grid{grid-template-columns:1fr}.downloads{columns:1}header{padding-top:28px}}
</style></head><body><main>
<nav aria-label="Report links"><a href="https://github.com/akshatvishu/vllm-omni-6333-longform-tts">Reproduction repository</a><a href="https://github.com/vllm-project/vllm-omni/issues/6333">Issue #6333</a><a href="https://github.com/vllm-project/vllm-omni/pull/6409">PR #6409</a><a href="previous-experiment.html">Previous 60-output experiment</a></nav>
<header><p class="eyebrow">OmniVoice / saved experiment evidence</p><h1>English and German<br>long-form TTS splitter comparison</h1>
<p class="lead">Both splitters completed every prompt. The pooled word error rates are close, with opposite directions in English and German. The paired sign tests do not establish a reliable quality winner.</p>
<div class="stats"><div class="stat"><strong>50</strong><span>paired Wikipedia prompts</span></div><div class="stat"><strong>100</strong><span>successful audio outputs</span></div><div class="stat"><strong>23</strong><span>pairs with different splits</span></div><div class="stat"><strong>27</strong><span>pairs with identical splits</span></div></div>
<p class="muted">25 English and 25 German prompts. Each language has 15 random and 10 targeted excerpts. One generation per splitter per prompt, seed 42, auto voice, one MI300X. Human listening and independent sentence annotations remain pending.</p></header>
<section class="panel"><h2>Compared implementations</h2>
<p>Ours is the frozen sentence-first splitter now included in <a href="https://github.com/vllm-project/vllm-omni/pull/6409/commits/71b8cafa4d62afad25ce92cee2714623f16b7e0d">PR commit <code>71b8cafa</code></a>. The recorded run used base <code>2ece3987427d24a7825973b94b7d6db662b262e2</code> with that exact splitter file. Theirs is the reconstructed proposal from <a href="https://github.com/vllm-project/vllm-omni/pull/6409#issuecomment-6023328036">lazariv's reviewer comment</a>. Both prefer sentence and newline boundaries over clause punctuation. Their implementation adds German abbreviation and initialism rules. The original unmodified PR splitter was not an audio variant in this run.</p>
<details><summary>Exact frozen source files and SHA256 hashes</summary><p><a href="https://github.com/akshatvishu/vllm-omni-6333-longform-tts/blob/main/variants/splitter-en-de-50/ours.py">Ours</a><br><code>84e07ebf944de2a71e133e0643ade789eb1fafb1c9660c5a00c21fa7fa64aab4</code></p><p><a href="https://github.com/akshatvishu/vllm-omni-6333-longform-tts/blob/main/variants/splitter-en-de-50/theirs.py">Theirs</a><br><code>6c8b6f578bfbda6f590a4070cde72092634a8080927ba9fdf7470acb22fa35aa</code></p></details></section>
<section class="panel" id="quality"><h2>Automated word accuracy</h2>
<p>Whisper large-v3-turbo was used to transcribe all 100 complete outputs. Word error rate (WER) counts substitutions, deletions, and insertions divided by reference words. Coverage is the share of reference words matched correctly, excluding insertions. The table pools word counts across prompts.</p>
"""
        + quality_table
        + """
<p class="conclusion">Their overall WER is lower by 0.058 percentage points. Their English WER is higher by 0.198 points, and their German WER is lower by 0.331 points. Automated transcription errors include number spelling and abbreviation mismatches, so the scores do not directly measure voice quality or naturalness.</p>
<details><summary>Random and targeted selection breakdown</summary>"""
        + group_table
        + """</details></section>
<section class="panel"><h2>Paired comparison of different splits</h2>
<p>Only 23 prompts have different chunk text. The remaining 27 pairs have identical WER and serve as A/A controls for the splitter comparison. A two-sided exact sign test uses only non-tied pairs, under an equal chance of either splitter having lower WER.</p>
"""
        + sign_table
        + """
<p class="muted">All-prompt counts are 6 lower WER for ours, 11 lower WER for theirs, and 33 ties. Excluding ties gives the same p values as the different-split subset. None of the three tests is below 0.05, and failure to reject does not establish equivalence. The language tests are exploratory and have no multiple-testing correction.</p></section>
<section class="panel"><h2>Generation time and audio measurements</h2>
<p>Latency and memory summaries use 48 observations per splitter after excluding the first request after each of four server starts. Quality, duration, chunk, and quiet measurements include all 50 outputs per splitter.</p>
"""
        + speed_table
        + """
<p class="muted">Client latency ends after receipt of the complete response, before WAV decoding and saving. Real-time factor is client latency divided by final audio duration, including inserted silence. Memory is the server-reported allocator reserve. Quiet audio uses 20 ms RMS windows below minus 60 dBFS and includes natural pauses. Quiet intervals and internal zero runs are not verified join labels.</p>
<p>Mean latencies differ by about 0.030 seconds. Each prompt has only one timing observation, and there were no warm-up generations. English ran ours first, while German ran theirs first. The run does not establish a repeatable speed difference.</p></section>
<section class="panel"><h2>Paired latency and chunk count</h2>
<p>Paired differences retain only prompts for which neither splitter's request was the first after a server start. Positive latency differences mean theirs took longer. Same-split pairs provide an A/A timing comparison under the same protocol.</p>
"""
        + paired_latency_table
        + """
<p>Among different-split pairs, the latency difference and chunk-count difference have Pearson correlation <code>r = 0.759</code>. Extra chunks are associated with longer generation time in these observations, but the association does not isolate a causal splitter overhead. The timing variation and single observations limit any speed conclusion.</p></section>
<section class="panel"><h2>Boundary and voice checks</h2>
<p>The saved CPU boundary predictions use the runtime settings of 15 seconds per chunk, a 30-second chunking threshold, and 25 frames per second. All 100 content-preservation and character-bound checks passed. Nine prompts differ in chunk count, and there are 42 boundaries exclusive to ours and 44 exclusive to theirs.</p>
<p>Independent labels for whether each boundary is a correct sentence or phrase break are pending. Character bounds do not impose strict duration bounds. All 100 files have the expected number of internal zero-run candidates. After excluding those zero-run spans, the largest segment duration mismatch in 98 files is within four predicted frames (0.16 seconds). The maximum across all files is 4.004 frames. The counts and durations corroborate the CPU predictions, but runtime boundary positions were not captured as ground truth.</p>
<p>The first chunk differs for <code>wiki_en_targeted_08</code>, <code>wiki_de_random_14</code>, <code>wiki_de_targeted_05</code>, and <code>wiki_de_targeted_08</code>. Each pair is marked below because a different first chunk may also change the auto voice.</p>
<p class="muted">For the 27 identical-split pairs, every WAV length matches and two WAV pairs are byte-identical. Their median mean absolute PCM difference is approximately 9.67e-9 on the normalized amplitude scale. Equal splits do not imply byte-identical audio, and waveform checks are not listening judgments.</p></section>
<section class="panel" id="listen"><h2>Listen to every paired output</h2>
<p>Choose a pair to compare its audio, full prompt, transcripts, and predicted chunk text. Labels identify the implementations, so listening on this page is not blind. No listening preference has been filled in.</p>
<noscript><p class="notice">JavaScript is disabled. All 50 paired examples are displayed below.</p></noscript>
<div class="controls" id="controls" hidden><label>Language<select id="language"><option value="all">All languages</option><option>English</option><option>German</option></select></label><label>Pair group<select id="group"><option value="all">All 50 pairs</option><option value="different">Different splits (23)</option><option value="same">Same splits (27 A/A controls)</option><option value="first">First chunk changed (4)</option></select></label><label>Prompt<select id="prompt">"""
        + "".join(options)
        + """</select></label></div>
<div class="pager" id="pager" hidden><button id="prev" type="button">Previous pair</button><button id="next" type="button">Next pair</button><span id="position" class="muted" aria-live="polite"></span><a id="permalink" href="#listen">Link to selected pair</a></div>
"""
        + "".join(cases)
        + """</section>
<section class="panel"><h2>Evidence and limits</h2>
<p>The prompts, transcripts, boundary predictions, aggregate metrics, and audit files below are copied from the saved run. The model revision is <code>c5fdb5ccb189668d56333f77ba2629f4cd7535f4</code>. Scoring used Whisper revision <code>41f01f3fe87f28c78e2fbf8b568835947dd65ed9</code>, float32, known language, deterministic whole-output sequential decoding, and no reference-text prompt.</p>
<p>One seed and one observation per prompt do not measure variation across seeds, voices, repeated runs, or hardware. Language is linked to execution order in this run. Native-speaker listening, sentence annotation, and captured runtime boundary ground truth are pending. Neither the scores nor the boundary differences establish a naturalness advantage.</p>
<ul class="downloads">"""
        + links
        + """</ul>
<p><a href="https://github.com/akshatvishu/vllm-omni-6333-longform-tts/tree/main/results/splitter-en-de-50">Full saved run and WAV restoration instructions</a> | <a href="https://github.com/akshatvishu/vllm-omni-6333-longform-tts/tree/main/variants/splitter-en-de-50">Exact splitter sources</a></p></section>
<footer>The previous synthetic English experiment is preserved separately. Its 60 outputs are not included in any count or metric on this page.</footer>
</main><script>
const language=document.getElementById('language'),group=document.getElementById('group'),prompt=document.getElementById('prompt');
const originalOptions=Array.from(prompt.options).map(o=>o.cloneNode(true));
const cases=Array.from(document.querySelectorAll('.case'));
const previous=document.getElementById('prev'),next=document.getElementById('next');
function showPair(){
  cases.forEach(c=>{c.hidden=c.id!==prompt.value;if(c.hidden)c.querySelectorAll('audio').forEach(a=>a.pause());});
  const i=prompt.selectedIndex;
  previous.disabled=i<=0;next.disabled=i<0||i>=prompt.options.length-1;
  document.getElementById('position').textContent=prompt.options.length?`${i+1} of ${prompt.options.length} matching pairs`:'No pairs match these filters.';
  document.getElementById('permalink').href=prompt.value?'#'+prompt.value:'#listen';
}
function filter(){
  const selected=prompt.value;
  prompt.replaceChildren(...originalOptions.filter(o=>(language.value==='all'||o.dataset.language===language.value)&&(group.value==='all'||(group.value==='different'&&o.dataset.different==='1')||(group.value==='same'&&o.dataset.different==='0')||(group.value==='first'&&o.dataset.first==='1'))).map(o=>o.cloneNode(true)));
  if(Array.from(prompt.options).some(o=>o.value===selected))prompt.value=selected;
  showPair();
}
language.addEventListener('change',filter);group.addEventListener('change',filter);prompt.addEventListener('change',showPair);
previous.addEventListener('click',()=>{if(prompt.selectedIndex>0){prompt.selectedIndex--;showPair();}});
next.addEventListener('click',()=>{if(prompt.selectedIndex<prompt.options.length-1){prompt.selectedIndex++;showPair();}});
function selectHash(){const id=location.hash.slice(1);if(originalOptions.some(o=>o.value===id)){language.value='all';group.value='all';filter();prompt.value=id;showPair();}}
document.getElementById('controls').hidden=false;document.getElementById('pager').hidden=false;
filter();selectHash();window.addEventListener('hashchange',selectHash);
</script></body></html>
"""
    )
    index.write_text(page)
    print(f"Built {index}: {len(prompts)} pairs, {len(records)} audio players, {len(evidence)} evidence files.")


if __name__ == "__main__":
    main()
