# Recorded A100 validation — 25 October 2026

These are saved results from one rented **NVIDIA A100-SXM4-80GB** on Runpod Secure Cloud, using Python 3.12.3. The run date is 25 October in Sydney; raw timestamps use 24 October UTC. This bundle contains actual responses, stdout, source hashes, package versions and reproduction scripts. The screenshots capture HTML reports generated from these files.

**Tested source:** tag `v0.3.1`, commit `df9b95526b029601cdd0775950a72fd9485e7c8e`, plus [source-fixes.patch](source-fixes.patch). The package version string remains `0.3.1`; the additional fixes are recorded under [Unreleased](../../CHANGELOG.md#unreleased). These results must not be attributed to the unchanged PyPI 0.3.1 wheel.

[Open the report source](report.html) · [Captured report](images/validation-report.jpg) · [Captured log excerpts](images/test-evidence.jpg) · [Environment](results/environment.json) · [Artifact SHA-256 checksums](SHA256SUMS)

## Compatibility results

| Actual execution path | Model / precision | Passing live checks | Evidence |
|---|---|---:|---|
| vLLM 0.32.0, HTTP | Qwen3.8-27B, BF16, language model only | 10 | [JSON](results/qwen-vllm-probes.json), [stdout](results/qwen-vllm-probes.log) |
| SGLang 0.5.22, HTTP | Qwen3.8-27B, BF16, full model loaded | 10 + 1 image check | [JSON](results/qwen-sglang-probes.json), [vision](results/qwen-sglang-vision.json) |
| Ollama 0.40.10, native HTTP | Qwen3.8:27b, Q4_K_M | 7 + 1 image check | [JSON](results/qwen-ollama-probes.json), [vision](results/qwen-ollama-vision.json) |
| llama-cpp-python 0.3.36, in process | Same Qwen GGUF, all layers offloaded | 4 | [GPU offload and test stdout](results/llamacpp-live.log), [response](results/llamacpp-response.json) |
| Transformers 5.20.2 / torch 2.8.0+cu128, in process | TinyLlama 1.1B, CUDA BF16 | 13 | [Final stdout](results/transformers-live-final.log), [HTTP responses](results/native-http.json) |
| AirLLM 4.0.0, in process | TinyLlama 1.1B, CUDA, layer offload | 3 | [Stdout](results/airllm-live.log), [response](results/airllm-response.json) |

The ten vLLM/SGLang checks cover model listing, chat, greedy streaming equality, JSON Schema output, unexecuted forced tool calls, reasoning metadata, text completion, oversized context rejection in both modes, and four concurrent requests. Ollama covers locality/capability metadata, chat, streaming equality, JSON Schema, unexecuted tool calls, reasoning metadata and completion. Ollama's tool selection remains model-dependent.

The native and Ollama probes exercised `require_local=True`. The generic OpenAI model metadata from vLLM/SGLang does not attest locality, so those probes used the default guard setting; their execution on this host was observed independently through the server processes and CUDA logs. Unknown locality still fails closed when the guard is requested.

Native Transformers checks cover actual token boundaries, exact context fit, output reservation, worker-error propagation, streamed indentation, Unicode, public chat and three HTTP routes. Its health endpoint answered while model generation was unfinished; [the raw measurement](results/native-http.json) is one observation, not a service latency guarantee. Native llama.cpp checks completion/locality, streaming equality and context overflow in both modes. AirLLM checks completion, buffered streaming and context overflow; it is not an AirLLM throughput or large-model capacity benchmark.

The image checks send a generated [64 × 64 solid-red PNG](images/vision-input-red.png) through the public chat API and check that the model identifies red. They establish one inline-image path, not general visual understanding or OCR accuracy.

Separately, the existing [live Ollama discovery suite](results/ollama-live-discovery-after.log) passed **13 tests and 2 subtests**, with **2 skipped** because the server listed no cloud-routed models. This run made no cloud-model inference requests.

## Warm HTTP measurements

All requests go through `LLM.chat()` over loopback. Each engine ran alone on the GPU: two excluded warmups, **10 sequential measured requests**, **5 streaming requests**, then **4 concurrent requests**. Every measured non-streaming request produced **64 engine-reported output tokens**. Sampling was temperature 0, seed 7, top-p 1, top-k 40 and repetition penalty 1. Thinking was disabled; context was configured to 8,192. The full prompt, settings, per-request counts, outputs and timings are in the JSON files.

| Engine / precision | Median request latency | Observed latency range, n=10 | Median first visible streamed content, n=5 | Median output tokens/s | Aggregate output tokens/s, 4 concurrent |
|---|---:|---:|---:|---:|---:|
| vLLM / BF16 | 4.023 s | 4.014–4.049 s | 0.171 s | 15.9 | 59.3 |
| SGLang / BF16 | 3.181 s | 3.174–3.196 s | 0.135 s | 20.1 | 78.8 |
| Ollama / Q4_K_M | 1.552 s | 1.493–1.897 s | 0.265 s | 41.3 | 41.3 |

[Benchmark script](scripts/benchmark_http.py) · [vLLM raw samples](results/qwen-vllm-benchmark.json) · [SGLang raw samples](results/qwen-sglang-benchmark.json) · [Ollama raw samples](results/qwen-ollama-benchmark.json)

**Interpretation:** output tokens/s is output count divided by complete request wall time, including prompt processing and HTTP. It is not isolated decode throughput. Streaming latency starts before the request and ends at the first non-empty content delta. The concurrency figure divides all four output counts by total wall time. Repeated prompts and default engine caching make this a warm workload. vLLM used eager execution with vision disabled; SGLang disabled CUDA graphs and loaded the full model; Ollama used different quantization and its own scheduling. These are distinct configurations, so the table is not a fair engine ranking. No client-overhead comparison, model-quality evaluation, cold-start result or competitor benchmark was performed.

## Bugs found and verified

1. **Streaming lost code indentation.** Real TinyLlama generation returned identical token IDs in both paths, yet the upstream newline flush caused streamed decoding to lose an indentation space. The adapter keeps a bounded newline anchor. [Original live failure](results/transformers-live.log), [token-level diagnosis](results/stream-diagnosis.json), [seven failing regression subtests before the fix](results/indentation-regression-before.log), [passing live run](results/transformers-live-final.log).
2. **Native locality identity was missing.** Transformers and llama.cpp inherited the backend name `base`, causing `require_local=True` to reject these in-process engines. Both now identify themselves correctly. [Failing regression](results/native-locality-before.log), [passing regressions](results/regressions-after.log), and actual native inference with the guard enabled.
3. **Reported token counts were replaced by estimates.** `LLM.chat()` and `LLM.complete()` discarded upstream counts. They now preserve valid counts; Ollama maps its native counters, and tool loops sum each generation round. [Before](results/usage-before.log), [after](results/usage-after.log). The initial benchmark with incorrect estimated counters was discarded; only corrected 64-token measurements appear above. Streaming usage is still estimated and was not used to calculate tokens/s.

The [original opt-in Ollama test failure](results/ollama-live-discovery-before.log) also exposed a test assumption: its 16/24-token budgets could be consumed by a thinking model before visible output. The text smoke tests now explicitly pass `think=False`; independent reasoning probes pass `think=True`. This was a test-budget correction, not an inference crash.

## Suite and provenance

- The unchanged v0.3.1 [baseline suite](results/release-suite.log) passed **484 tests and 245 subtests**, with **16 skipped**.
- The patched [final suite](results/final-suite.log) passed **490 tests and 261 subtests**, with **16 skipped**. Official OpenAI and MCP SDK interoperability were enabled. These are mostly unit, fixture and SDK tests; **490 is not a count of GPU model inference tests**.
- Strict Sphinx, Ruff, Black, wheel/sdist builds, Twine checks and a fresh-wheel light-core chat smoke test also passed. [The local build record](results/local-package-check.json) includes artifact hashes and confirms that packaged Python sources match the checkout; these validation builds were not uploaded to PyPI.
- [Source hashes](results/tested-source-sha256.json) identify the tested Python sources and project metadata. [Package manifests](results/packages-base.json) record resolved versions; separate manifests cover [vLLM](results/packages-vllm.json), [SGLang](results/packages-sglang.json), [llama.cpp](results/packages-llamacpp.json) and [AirLLM](results/packages-airllm.json).
- Qwen Hugging Face revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. TinyLlama revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`.
- Qwen GGUF SHA-256: `f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d`, 16,810,714,464 bytes. The Ollama vision projector is a separate file. Weights are not included in this repository.
- Driver: 595.91.07. Container: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. The driver reports CUDA 13.2 support; individual environments use different CUDA runtimes, as their package manifests show.

Logs retain test output and responses. ANSI terminal controls and the local macOS checkout path are removed where present. Credentials, SSH keys, account records and model weights are excluded. Screenshots are captures of generated reports, not screenshots of a terminal or CI badge.

## Reproduce

These scripts record the disposable Linux test layout under `/workspace`; they do not provision a GPU or start billing. Install the recorded versions from the manifests using the upstream installation guides, prepare model weights, and run one engine at a time. The baseline source plus patch recreates the tested client code:

```bash
git checkout df9b95526b029601cdd0775950a72fd9485e7c8e
git apply /path/to/source-fixes.patch
```

The run used separate environments named `/workspace/env-base`, `env-vllm`, `env-sglang`, `env-llamacpp` and `env-airllm`, with the repository installed editable and `HF_HOME=/workspace/hf-cache`. Copy the scripts to `/workspace`, create `/workspace/artifacts`, and use the pinned model revisions above. `env-base` needs the development, server and Transformers extras plus the recorded OpenAI/MCP SDKs. The live test scripts require pytest; they are outside the default `tests/` collection to avoid accidental model downloads.

```bash
LOCAL_LLM_KIT_OPENAI_INTEROP=1 LOCAL_LLM_KIT_MCP_INTEROP=1 \
  /workspace/env-base/bin/python -m pytest
HF_HOME=/workspace/hf-cache /workspace/env-base/bin/python -m pytest -s \
  /workspace/test_transformers_live.py
```

Start the appropriate server with [start_vllm.sh](scripts/start_vllm.sh) or [start_sglang.sh](scripts/start_sglang.sh), wait for its health endpoint, then run the compatibility probes and benchmark. For SGLang, change the backend to `sglang` and the port to `30000`:

```bash
/workspace/env-base/bin/python /workspace/probe_server.py \
  --backend vllm --url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.8-27B --report /workspace/artifacts/qwen-vllm-probes.json
/workspace/env-base/bin/python /workspace/benchmark_http.py \
  --backend vllm --url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.8-27B --report /workspace/artifacts/qwen-vllm-benchmark.json
```

For Ollama, run the official 0.40.10 binary bound to loopback with `OLLAMA_NO_CLOUD=1`, pull `qwen3.8:27b` and check its digest. Run [probe_ollama.py](scripts/probe_ollama.py), then the benchmark with `--backend ollama --url http://127.0.0.1:11434 --model qwen3.8:27b`, and [probe_vision.py](scripts/probe_vision.py) with `--backend ollama`. The existing suite is opt-in:

```bash
LOCAL_LLM_KIT_LIVE=1 /workspace/env-base/bin/python -m pytest -s tests/test_live_servers.py
```

[The llama.cpp script](scripts/test_llamacpp_live.py) reads the verified GGUF from the Ollama blob store and requests `n_gpu_layers=-1`. The CUDA 13 wheel required its shared runtime on this CUDA 12.8 base image; the run reused the installed vLLM environment's runtime libraries:

```bash
LD_LIBRARY_PATH=/workspace/env-vllm/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  /workspace/env-llamacpp/bin/python -m pytest -s /workspace/test_llamacpp_live.py
HF_HOME=/workspace/hf-cache /workspace/env-airllm/bin/python -m pytest -s \
  /workspace/test_airllm_live.py
```

Check your wheel's runtime requirements rather than assuming this library path applies to another image. See the official [vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), [SGLang Qwen recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), [Ollama Linux guide](https://docs.ollama.com/linux), [llama-cpp-python install guide](https://github.com/abetlen/llama-cpp-python), [AirLLM repository](https://github.com/lyogavin/airllm), [Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B) and [TinyLlama checkpoint](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).

To regenerate the report after verifying the files, run `python scripts/render_report.py` from this bundle. On Linux, `sha256sum -c SHA256SUMS` verifies the recorded artifact hashes; on macOS, use `shasum -a 256 -c SHA256SUMS`.

## Limits

Flint, Apple Foundation Models, MLX, LM Studio, LocalAI, Jan, GPT4All, KoboldCpp and the other server presets were not exercised on this rented Linux GPU. No embedding, retrieval-quality, long soak, cancellation, tool side-effect isolation or adversarial security benchmark was run. Only one synthetic image was checked per vision-capable engine. The tested native Transformers context was 2,048 tokens; Qwen servers used 8,192, below the model's advertised maximum. Model downloads used the internet; inference used local weights and loopback endpoints without a firewall/egress-isolation test. This is evidence for the recorded combinations, not certification of every backend or hardware configuration.
