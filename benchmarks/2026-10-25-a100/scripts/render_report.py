"""Render an evidence page from the saved JSON and test logs; no invented results."""

import html
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def suite(name):
    text = (RESULTS / name).read_text()
    matches = re.findall(r"\b(\d+) passed(?:, (\d+) skipped)?", text)
    assert matches, name
    passed, skipped = matches[-1]
    return f"{passed} passed" + (f", {skipped} skipped" if skipped else "")


def main():
    assert "261 subtests passed" in (RESULTS / "final-suite.log").read_text()
    bench_rows = []
    for backend, label in [
        ("vllm", "vLLM · BF16"),
        ("sglang", "SGLang · BF16"),
        ("ollama", "Ollama · Q4_K_M"),
    ]:
        report = load(f"qwen-{backend}-benchmark.json")
        s = report["summary"]
        assert len(report["sequential"]) == 10
        assert all(r["usage"]["completion_tokens"] == 64 for r in report["sequential"])
        assert (
            statistics.median(r["seconds"] for r in report["sequential"])
            == s["median_latency_seconds"]
        )
        bench_rows.append(
            f'<tr><td>{label}</td><td>{s["median_latency_seconds"]:.3f} s</td>'
            f'<td>{s["median_ttft_seconds"]:.3f} s</td><td>{s["median_output_tokens_per_second"]:.1f}</td>'
            f'<td>{s["concurrent_output_tokens_per_second"]:.1f}</td></tr>'
        )
    checks = []
    for backend in ["vllm", "sglang", "ollama"]:
        report = load(f"qwen-{backend}-probes.json")
        assert all(r["passed"] for r in report["results"])
        checks.append(
            f'<li><b>{backend}</b>: {len(report["results"])} / {len(report["results"])} live compatibility checks</li>'
        )
    for label, file in [
        ("Transformers · TinyLlama 1.1B", "transformers-live-final.log"),
        ("llama.cpp · Qwen GGUF", "llamacpp-live.log"),
        ("AirLLM · TinyLlama 1.1B", "airllm-live.log"),
    ]:
        checks.append(f"<li><b>{label}</b>: {suite(file)}</li>")
    vision = load("qwen-ollama-vision.json")
    assert vision["passed"]
    css = """*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#15273b;font:16px/1.5 system-ui,sans-serif}main{max-width:1080px;margin:28px auto;padding:28px 36px;background:white;border:1px solid #d9e3eb;border-radius:14px}h1{font-size:31px;line-height:1.15;margin:6px 0 12px}h2{font-size:20px;margin:24px 0 8px}.eyebrow{color:#2563eb;font-size:13px;font-weight:700;letter-spacing:.06em}.meta{color:#506174;font-size:14px}.badge{display:inline-block;background:#e6f7ee;color:#17663a;padding:6px 12px;border-radius:8px;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;border-bottom:1px solid #e1e8ef;padding:11px 9px}th{background:#f1f5f9;font-size:13px}ul{padding-left:21px;columns:2;column-gap:30px}li{margin-bottom:6px;break-inside:avoid}.note{font-size:13px;color:#566779}.limits{background:#f5f7fa;padding:12px 16px;border-left:3px solid #7c92a8;font-size:14px}a{color:#1d4ed8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#101e2f;color:#e2eaf4;padding:16px;border-radius:10px;font-size:13px}"""
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local LLM Kit — recorded GPU validation</title><style>{css}</style><main>
    <div class="eyebrow">LOCAL LLM KIT · RECORDED VALIDATION · 25 OCTOBER 2026 (SYDNEY)</div>
    <h1>Real model checks on one A100 80 GB</h1>
    <p class="meta">NVIDIA A100-SXM4-80GB · Python 3.12.3 · Qwen3.8-27B / TinyLlama 1.1B<br>Source: v0.3.1 + recorded fixes. This page is rendered from the linked JSON and logs.</p>
    <div class="badge">Final suite: {suite('final-suite.log')} · 261 subtests passed</div>
    <h2>Warm loopback HTTP measurements</h2>
    <table><thead><tr><th>Engine / model precision</th><th>Median latency</th><th>Median first content</th><th>Output tokens/s</th><th>4 concurrent tokens/s</th></tr></thead><tbody>{''.join(bench_rows)}</tbody></table>
    <p class="note">Per engine: 2 warmups, 10 sequential requests × 64 output tokens, 5 streaming requests, 4 concurrent requests. Temperature 0, seed 7, 8,192 context; thinking disabled. Output tokens/s includes prompt processing and HTTP time. Different quantization and engine settings prevent a fair engine ranking.</p>
    <h2>Live inference evidence</h2><ul>{''.join(checks)}<li><b>Ollama discovery suite</b>: {suite('ollama-live-discovery-after.log')}</li><li><b>Ollama inline vision</b>: solid-red image identified correctly</li></ul>
    <h2>Bugs reproduced and fixed</h2><p>Streaming code indentation after newlines; native backend locality identity; engine-reported token counts replaced by estimates. Before-and-after logs and source patch are included.</p>
    <div class="limits">Scope: one rented Linux GPU and short workloads. No cold-start, quality or competitor benchmark. Flint, Apple FM/MLX and other runtimes were not run. Model download required internet; inference used local weights and loopback, without an egress-isolation test.</div>
    <p class="note"><a href="README.md">Method and reproduction commands</a> · <a href="results/environment.json">Environment</a> · <a href="SHA256SUMS">Artifact checksums</a> · <a href="test-evidence.html">Captured log excerpts</a></p>
    </main></html>"""
    (ROOT / "report.html").write_text(page)
    snippets = []
    for label, file in [
        ("Final suite", "final-suite.log"),
        ("Actual Transformers CUDA run", "transformers-live-final.log"),
        ("Ollama discovery and inference", "ollama-live-discovery-after.log"),
    ]:
        lines = (RESULTS / file).read_text().splitlines()
        summary = "\n".join(
            [
                line
                for line in lines
                if re.search(r"\b\d+ passed", line) or line.startswith("SKIPPED")
            ][-5:]
        )
        snippets.append(
            f'<h2>{label}</h2><p class="note">Saved stdout: <a href="results/{file}">{file}</a></p><pre>{html.escape(summary)}</pre>'
        )
    reply = vision["response"]["choices"][0]["message"]["content"]
    (ROOT / "test-evidence.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Saved validation log excerpts</title><style>{css}</style><main><div class="eyebrow">CAPTURED REPORT · SAVED STDOUT EXCERPTS</div><h1>Evidence from the saved run</h1>{"".join(snippets)}<h2>Real inline-image request</h2><p><img width="64" height="64" src="images/vision-input-red.png" alt="Solid-red PNG sent to the model"> Model response: <b>{html.escape(reply)}</b></p><p class="note">Synthetic color check only. Raw response: <a href="results/qwen-ollama-vision.json">qwen-ollama-vision.json</a></p></main></html>'
    )


if __name__ == "__main__":
    main()
