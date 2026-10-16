# Local LLM server discovery/probe — precision research

**Purpose:** ground a "local LLM server discovery/probe" feature for `local-llm-kit` in verified primary sources.
**Method:** every claim below was read from source code, official docs, release notes or upstream PRs fetched directly (raw.githubusercontent.com, `gh` GitHub API). No endpoint or response shape is invented; anything not directly verified is marked **UNVERIFIED** and listed in §4.
**Ledger:** `ledger_probe.jsonl` — 232 rows, keys exactly `url, project, type, date_or_version, what_examined, finding, relevance, confidence, limitation`.

---

## 1. Master comparison table

`—` = the server has no such endpoint (verified absent in the code read). `?` = not verified. "CORS default" describes whether a **browser** on an arbitrary origin can reach the server at all.

| Server | Default host:port | Configurable via | Health / liveness endpoint | Model-listing endpoint | Auth required to list? | Context / capability metadata | CORS default (browser reach) |
|---|---|---|---|---|---|---|---|
| **Ollama** | `127.0.0.1:11434` | `OLLAMA_HOST` | `GET /` → **plain text** `Ollama is running`; `GET /api/version` → `{"version":…}`; `GET /api/status` → `{"cloud":{…}}` | `GET /api/tags` → `{"models":[…]}` **and** `GET /v1/models` → `{"object":"list","data":[{"id","object","created","owned_by"}]}` | No | `POST /api/show` → `model_info{"<arch>.context_length"}`, `capabilities[]`; `GET /api/ps` for loaded | Allowlist: `localhost`/`127.0.0.1`/`0.0.0.0` + `app://`,`file://`,`tauri://`,`vscode-webview://` (+`OLLAMA_ORIGINS`). Foreign origin → 403 |
| **llama.cpp `llama-server`** | `127.0.0.1:9931` ⚠️ **was 8080 until 2026-10-09** | `--host`/`--port`, `LLAMA_ARG_HOST`/`LLAMA_ARG_PORT` | `GET /health` → `200 {"status":"ok"}` or **`503`** `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`; `/v1/health` alias; **public, no API key** | `GET /v1/models` (always exactly 1 element) and `GET /models` (router) | No | `GET /props` → `n_ctx`, `default_generation_settings`, `chat_template_caps`, `modalities`, `build_info`; `/v1/models` → `data[].meta.n_ctx_train`, `architecture.input_modalities/output_modalities` | `*` by default; **silently narrowed to `localhost` if `--tools`/`--agent`/`--mcp-*` is used** |
| **LM Studio** `/v1` + native | `localhost:1234` | Server Port (UI), `lms server start` | — **no health endpoint documented** | `GET /v1/models`; native `GET /api/v1/models`; legacy `GET /api/v0/models` | No (auth off by default) | `GET /api/v1/models` → `max_context_length`, `loaded_instances[].config.context_length`, `capabilities{vision,trained_for_tool_use,reasoning}`. `GET /api/v0/models` → `max_context_length`,`arch`,`type`,`state`,`quantization` | **Off** — explicit "Enable CORS" switch. Browser probe fails unless enabled |
| **vLLM** | `0.0.0.0:8000` (host default `None` → all interfaces) | `--host`/`--port`, `--uds`; `VLLM_API_KEY` for auth | `GET /health` → **200 with EMPTY body**, 503 if engine dead (status-code only, no JSON) | `GET /v1/models` → `{"object":"list","data":[{id,object,created,owned_by:"vllm",root,parent,max_model_len,permission[]}]}` | No | `max_model_len` on the standard `/v1/models` entry — rare and convenient | **Disabled by default** (`allowed_origins=[]`, docstring: "Cross-origin browser access is disabled by default"). Extra health port 9256 in multi-port DP mode |
| **SGLang** | `127.0.0.1:30000` | `--host`/`--port` | `GET /ready` → 200/**503**, **empty body**; `GET /health` and `/health_generate` → 200 normally. ⚠️ `/health` performs a **real 1-token generation** if `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` is set | `GET /v1/models` | No by default (`--api-key` optional) | `GET /model_info` → `model_path`,`served_model_name`,`model_type`,`architectures`,`has_image_understanding`,`has_audio_understanding` — **but NO context length** (`hf_config` is commented out). `GET /server_info` → resolved startup config + `launch_command` | Not verified; CORS config not read (mark UNVERIFIED) |
| **LocalAI** | `0.0.0.0:8080` (default address literal `":8080"`) | `--address`, `LOCALAI_ADDRESS`/`ADDRESS` | `GET /healthz` → **204 No Content** (empty); `GET /readyz` → 204, or 503 `{"status":"starting","reason":"startup preload in progress"}`. **Not `/health`** | `GET /v1/models` **and** `GET /models` | No by default | `GET /v1/models/capabilities` (**LocalAI-specific**, "additive superset of /v1/models") | Default = Echo's permissive CORS. With `LOCALAI_CORS=true` you **must** also set `LOCALAI_CORS_ALLOW_ORIGINS` or no CORS policy is registered at all |
| **Jan** | `127.0.0.1:1337` (⚠️ **port 0 = random** on Android/iOS) | Server Host/Port/API Prefix (UI); Trusted Hosts list | — none documented (absent from its own OpenAPI spec) | `GET /v1/models` | No (API key empty by default) | `GET /models/{id}` likely (spec lists `/models/{id}`) — **field names UNVERIFIED** | Trusted Hosts setting exists; CORS behaviour UNVERIFIED |
| **GPT4All** Local API Server | `127.0.0.1:<networkPort setting>` — **port number UNVERIFIED** | Qt setting (GUI) | — **NO HEALTH ENDPOINT** (verified: full route block read) | `GET /v1/models`, `GET /v1/models/<arg>` | No | — none | **Permissive**: `Access-Control-Allow-Origin: *` |
| **KoboldCpp** | **`0.0.0.0:5001`** (`--host` defaults to `""` = "all routable interfaces") | `--host`, `--port` or bare positional port | — no health endpoint; cheapest: `GET /api/v1/model` → `{"result":"<name>"}`, `GET /api/v1/info/version` → `{"result":"1.2.5"}` | `GET /api/v1/model`, `GET /v1/models` (=`/models`) → `{"object":"list","data":[{id,object,created,owned_by:"koboldcpp",status:{value:"loaded"},permission:[],root}]}` | No | `GET /api/v1/config/max_context_length` → `{"value":N}`; `/api/v1/config/max_length`; `GET /api/extra/true_max_context_length` (in-code "do not advertise") | Not verified (UNVERIFIED) |
| **text-generation-webui** (oobabooga) | `127.0.0.1:5000` for API (`--api-port`); API **off unless `--api`** | `--api`, `--api-port`, `--api-key`, `--listen`/`--listen-host`/`--listen-port` | `GET /v1/internal/health` → `{"status":"ok"}` (⚠️ nonstandard path, behind `--api-key` if set) | `GET /v1/models` → `{"object":"list","data":[]}` ⚠️ **EMPTY until a model is loaded** | No (`--api-key` empty ⇒ no-op) | — none on `/v1/models` (`created` hardcoded `0`, `owned_by:"user"`). Native `/v1/internal/model/info` gated by api key | **Localhost-only regex** unless `--listen`/`--public-api`: `https?://(localhost\|127\.0\.0\.1)(:\d+)?` |
| **MLX-LM** | `127.0.0.1:8080` | `--host`, `--port`, `--allowed-origins` | `GET /health` → `{"status":"ok"}` 200 / `{"status":"unavailable"}` **503** | `GET /v1/models` (also matches `/v1/models/<repo_id>`) | No (`--api-key` flag does not exist) | — none | `*` (default) |
| **apfel** | `127.0.0.1:11434` ⚠️ **collides with Ollama** | `--port`/`--host`, `APFEL_PORT`/`APFEL_HOST`; `--token`/`APFEL_TOKEN` | `GET /health` → rich JSON incl. `status, model, version, active_requests, context_window, capabilities, model_available, prewarmed, supported_languages` | `GET /v1/models` → `{"object":"list","data":[{id,object,created,owned_by:"apple",context_window,capabilities,supported_parameters,unsupported_parameters,notes}]}` | No (token off by default) | **Yes — both `/health` and `/v1/models` report `context_window` + capabilities** | **Origin check ON** by default (localhost only) → foreign origin **403**; plus a **Host-header allowlist** (DNS-rebinding defense) |
| **Apple `fm serve`** | `127.0.0.1:**1976**` ⚠️ **not 11434** | `--host`, `--port`, `--socket` (unix socket, mutually exclusive) | `GET /health` → `{"status":"fm serve is running","models":[{"name":"system","available":true}]}` ⚠️ nonstandard status string | `GET /v1/models` (single entry, `id` = `system`, `owned_by` = `Apple`) | **No auth at all** | — none (`/v1/models` full shape UNVERIFIED) | No Origin/CORS handling observed; no auth ⇒ nothing to bypass |
| **llama-cpp-python** | `localhost:8000` (literal hostname, may resolve to `::1`) | ServerSettings (pydantic ⇒ env/.env), CLI | — **no health endpoint** | `GET /v1/models` → `{"object":"list","data":[{id,object:"model",owned_by:"me",permissions:[]}]}` (⚠️ `permissions` **plural**, and **no `created`**) | No | — none | Not verified (UNVERIFIED) |
| **TGI** | container port **80**, published as `8080` in all official examples | docker `-p` mapping | `/health` exists per convention — **UNVERIFIED from primary source** | `/v1/models` — **UNVERIFIED** | `--api-key` optional | `/info` — **UNVERIFIED** | Not verified (UNVERIFIED) |
| **RamaLama** | `127.0.0.1:<first free port in 8080–8180>` ⚠️ **dynamic** | `--host`, `--port`/`-p` | inherits llama.cpp's `/health` | inherits llama.cpp | No (default no auth) | inherits llama.cpp `/props` | Not verified (UNVERIFIED) |
| **Docker Model Runner** | `localhost:12434` (also `172.17.0.1:12434` from containers, `model-runner.docker.internal:12434`) | Docker Desktop toggle: `docker desktop enable model-runner --tcp <port>` | — not documented | **`GET /engines/v1/models`** ⚠️ `/engines` prefix | Not verified | `GET /api/show` (Ollama-compat) — shape UNVERIFIED | Not verified (UNVERIFIED) |

---

## 2. The three biggest divergence risks for a naive probe

### 2.1 llama.cpp's default port moved 8080 → 9931 in October 2026
Any probe table written before then — including the one in the task brief — is now wrong.

- `common/common.h`: `int32_t port = 9931;  // server listens on this network port` and `std::vector<std::string> hostnames = {"127.0.0.1"};`
- PR **#30159** "server: use port 9931 by default", merged **2026-10-09**: *"For docker, we still use 8080 inside the container to avoid breaking any tools using our base image"*.
- PR **#26508** "server: add notice for upcoming default port change 8080 --> 9931", merged **2026-08-03**: *"transitioning llama-server from an on-demand command to a daemon process"*, *"Port gonna change from 8080 to 9931 (leetspeak of GGML)"*.
- **Consequence:** probe **both** 9931 and 8080, and treat a llama.cpp hit on 8080 as a legacy/daemon-container instance. Meanwhile **MLX-LM still defaults to 8080** — so port 8080 is now ambiguous between the two.

### 2.2 `apfel` and `Ollama` share port 11434; `fm serve` uses 1976
`Sources/CLI/CLIArguments.swift`: `public var serverPort: Int = 11434; public var serverHost: String = "127.0.0.1"`. apfel's own README calls it *"Drop-in local http://localhost:11434/v1 backend for OpenAI SDKs"* — deliberately the same port as Ollama.

**So port 11434 alone proves nothing.** The discriminator must be the payload:
- Ollama: `GET /` returns plain text `Ollama is running`; `GET /api/version` returns `{"version":…}`.
- apfel: `GET /health` returns a JSON object containing `"prewarmed"` and `"context_window_measured"`; `GET /v1/models` ids include `apple-foundationmodel`.
- Ollama has **no** `GET /health`. apfel **does**.

`fm serve` is on **1976**, so it does not collide — but a probe list that assumes "all Apple backends are on 11434" will miss it entirely, and vice versa.

### 2.3 "Does `/v1/models` answering mean it's usable?" — no, and 200-with-empty-data is the trap
- **oobabooga**: `modules/api/models.py` — `if shared.model_name and shared.model_name != 'None': data = [model_info_dict(...)] else: data = []` then `return {"object": "list", "data": data}`. HTTP **200** with `data: []`. A probe that requires a non-empty list reports a healthy server as dead.
- **llama.cpp**: `/v1/models` "always has one single element" but `meta` "can be `null` (for example, while the model is still loading)". So `id`/`meta` must be read defensively.
- **MLX-LM**: `/v1/models` does **not** describe the served model at all; it runs `scan_cache_dir()` over the **Hugging Face cache** and returns repos containing `config.json` + `model.safetensors.index.json` + `tokenizer_config.json`. It requires an index file, so single-shard models never appear.
- **KoboldCpp**: `/api/v1/model` returns HTTP 200 with `{"result": "koboldcpp/protected-model"}` when admin protection is on and the probe sent no credentials — a **bogus model id**, not an error.

### 2.4 (runner-up) Browser probes and non-browser probes have different truth values
Four servers were verified to treat the two differently:
| Server | Default browser posture | Non-browser probe |
|---|---|---|
| apfel | origin check **ON**, localhost only → **403** for foreign origin; plus Host-header allowlist | always works (no Origin header sent) |
| vLLM | `allowed_origins=[]` → *"Cross-origin browser access is disabled by default"* | works |
| text-generation-webui | `allow_origin_regex = https?://(localhost\|127\.0\.0\.1)(:\d+)?` unless `--listen` | works |
| Ollama | origin allowlist + `allowedHostsMiddleware` → **403** | works; but a custom **Host** header from a loopback-bound server also 403s |
| GPT4All | `Access-Control-Allow-Origin: *` | works |
| MLX-LM / KoboldCpp / llama.cpp | `*` (llama.cpp narrows to `localhost` with `--tools`/`--agent`) | works |

**Conclusion:** discovery must run **outside** the browser (or via a same-origin proxy). A failed browser probe is not evidence the server is down.

### 2.5 (runner-up) Endpoint-shape sniffing does not identify a server
Overlap verified from source:
- **Docker Model Runner** answers OpenAI (`/engines/v1/*`), Anthropic (`/v1/messages`), **and** Ollama (`/api/tags`, `/api/chat`, `/api/generate`, `/api/show`) on one port.
- **KoboldCpp** answers `/v1/*`, `/api/generate`, `/api/chat`, `/v1/messages`, `/messages`, `/sdapi/v1/*`, `/api/extra/*`.
- **LocalAI** registers `RegisterOllamaRoutes` **and** `RegisterOllamaRootEndpoint`, plus `/v1/*`.
- **SGLang** ships `srt/entrypoints/ollama/protocol.py`.
- **llama.cpp** answers `/v1/systemone`; **LM Studio** answers `/v1/messages`.

**So: `/api/tags` returning 200 does NOT mean "Ollama".** Identity must come from a discriminating payload (e.g. Ollama's plain-text `GET /`, version string format, `owned_by` values, or a `/health` body shape).

### 2.6 (runner-up) Auth defaults differ wildly, and `--api-key` does not always mean "everything is protected"
| Server | Default | Trap |
|---|---|---|
| Ollama | none | — |
| apfel | none (`--token` optional) | binding `0.0.0.0` with no token = *"zero authentication"* (source comment); tooling only warns |
| vLLM | none | `--api-key` authenticates **only** `/v1`, `/v2`, `/inference`; `/health`, `/metrics`, `/invocations` stay open |
| llama.cpp | none | `/health` is explicitly public even with `--api-key`; env var is `LLAMA_API_KEY`, **not** `LLAMA_ARG_API_KEY` |
| LocalAI | none by default | **refuses to start** on a public bind without auth; `/metrics` is behind admin middleware |
| oobabooga | none | `--api-key` and `--admin-key` are **two different keys** guarding different lists |
| Jan | none | empty key ⇒ auth disabled; *"not recommended when binding to 0.0.0.0"* |
| `fm serve` | **none, ever** | *"the Authorization header is neither required nor checked"* |

---

## 3. Existing discovery/auto-detection: what actually exists (source-read)

**Bottom line: I found no library that implements multi-port local-server discovery.** Every client surveyed hardcodes a default and provides an env var to override it. But two real pieces of discovery prior art exist, and they solve *different* problems.

### 3.1 They all hardcode, none probe
| Library | Mechanism | Exact default | Timeout |
|---|---|---|---|
| **LiteLLM** | `OllamaModelInfo.get_api_base()` → `api_base or get_secret_str("OLLAMA_API_BASE") or "http://localhost:11434"` | one URL | `httpx.get(...)` with **no explicit timeout** ⇒ httpx's 5s default |
| **Continue** | `apiBase: "http://localhost:11434/"` in `core/llm/llms/Ollama.ts` | one URL | not read |
| **Open WebUI** | `OLLAMA_API_BASE_URL` default `http://localhost:11434/api`; `OLLAMA_BASE_URL` default `''` then derived | one URL + suffix stripping | not read |
| **lmstudio-python** | `default_api_ports = (41343, 52993, 16141, 39414, 22931)` | **five high ephemeral ports — not 1234** | n/a |
| **aider** | no `11434` in executable code paths matched; only docs | one URL | n/a |

**Techniques worth stealing from these libraries:**
1. **Suffix normalisation.** LiteLLM's `get_server_api_base()` strips any of `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings`, `/api/show`, `/api/tags`; Open WebUI strips a trailing `/api` or `/`. Let the user paste any URL.
2. **Architecture-agnostic context lookup.** LiteLLM's `_get_max_tokens()` iterates `model_info.items()` and returns the value of **any key containing `context_length`** — correct, because the prefix varies (`llama.context_length`, `qwen2.context_length`, …).
3. **Shape tolerance.** LiteLLM's `get_models()` accepts both `{"models":[...]}` and a bare list, and reads `entry.get("name") or entry.get("model")`. Ollama's own docs use `name` **and** `model` in the same object, so both are live.
4. **Idempotent endpoint appending.** `get_complete_url()` refuses to double-append `/api/chat`.

**Anti-patterns observed (do not copy):**
- LiteLLM **silently falls back to a static model list** when `/api/tags` fails — a network failure becomes indistinguishable from a real answer.
- Continue **swallows** `/api/show` failures (`console.warn` is commented out) and then assumes `DEFAULT_OLLAMA_CONTEXT_LENGTH = 8192` — a hardcoded guess pinned to an old Ollama commit, while Ollama's current docs describe VRAM-tiered defaults (4k / 32k / 256k).
- Continue's fix for `localhost` vs `127.0.0.1` is a **string replace** applied only under `process.env.IS_BINARY`, with the comment *"localhost was causing fetch failed in pkg binary only for this Ollama endpoint"* — a targeted workaround, not a general solution, and it will also rewrite the substring inside a user hostname.

### 3.2 The two genuine discovery implementations found
**(a) `apfel-server-kit` (Swift) — `PortScanner.swift`** — closest thing to the feature you are building:
```swift
/// Check whether TCP ports on `127.0.0.1` are available, and pick the first
/// free one from a configured range.
/// Uses the same socket + SO_REUSEADDR + bind() pattern that apfel-quick,
/// apfel-chat, and apfel-clip all use. No `lsof`, no network connect probe.
```
- `isAvailable(_:)` → `socket(AF_INET, SOCK_STREAM)` + `SO_REUSEADDR` + `bind()` to `127.0.0.1:port`; success ⇔ `result == 0`.
- `firstAvailable(in range: ClosedRange<Int>) -> Int?` → first bindable port in range.
- ⚠️ **It solves the opposite problem**: finding a **free** port to *start* a server on, not finding a **live** server. It explicitly avoids connect-probing. But it tells you the ecosystem convention is *"spawn on whatever is free"* — which is why a discovery feature cannot assume 11434 is permanently Ollama *or* permanently apfel.
- The kit also bundles `ApfelBinaryFinder.swift` (locate the executable) alongside `PortScanner.swift` — **discovery needs both a process/binary locator and a port probe**.
- ⚠️ **UNVERIFIED:** I read `PortScanner.swift` and the file list, not `ApfelBinaryFinder.swift` or `ApfelClient.swift`. Its binary search paths and its client-side readiness check are unexamined.

**(b) LocalAI probing itself** — `core/cli/run.go`, `waitForServerReady()`:
```go
host, port, err := net.SplitHostPort(address)
if err != nil { address = "127.0.0.1:" + port }
conn, err := net.DialTimeout("tcp", address, 500*time.Millisecond)
```
- A **500 ms TCP-dial** readiness loop, and — crucially — it **normalises an unspecified host to 127.0.0.1 before connecting**, because you cannot dial `0.0.0.0`. Independent confirmation that the `0.0.0.0`-vs-`127.0.0.1` mismatch is a real problem that production code hits.
- vLLM does the same class of thing pre-bind: `find_process_using_port(port)` and logs *"port %s is used by process %s launched with command:"*.

### 3.3 Why nobody discovers — the ecosystem ships *starters* instead
Continue ships **`core/util/start_ollama.sh`** and a dedicated **`gui/src/components/OnboardingCard/components/OllamaStatus.tsx`** onboarding component. The prevailing UX pattern is "start the server for the user", not "find a running server". LiteLLM's `setup_wizard.py` matched a `11434` search for the same reason.
⚠️ **UNVERIFIED:** I read the file *list* (via GitHub code search), not the contents of `start_ollama.sh` or `OllamaStatus.tsx`. Their detection methods are unexamined.

---

## 4. UNVERIFIED / COULD NOT CONFIRM

Honest list of everything I could not establish. **Do not encode any of these as fact.**

### 4.1 Servers not covered at all
| Item | Status |
|---|---|
| **AnythingLLM** | **NOT COVERED.** No primary source read. Port, endpoints, auth, CORS all unknown. |
| **Msty** | **NOT COVERED.** No primary source read. |
| **ExLlamaV2 server** | **NOT COVERED.** My guessed source path (`examples/openai_server.py` in `turboderp-org/exllamav2`) returned **404** — that proves only that my path was wrong, *not* that no server exists. `examples/chat.py` does exist (HTTP 200). |
| **`gpt4all-bindings` Python `gpt4all` package** | Not examined as a discovery client. |
| **`llm` (simonw) plugin ecosystem** | **Not searched.** No evidence gathered either way about whether any local-server auto-detection plugin exists. |
| **`ollama-python` / `ollama-js` official clients** | **Not read.** Their default-URL behaviour is unexamined (LiteLLM/Continue/Open WebUI were used as proxies for ecosystem behaviour). |

### 4.2 Specific facts I could not verify
| Claim | Why unverified |
|---|---|
| **GPT4All's default port number** | `server.cpp` takes it from `MySettings::globalInstance()->networkPort()`. I did not read `mysettings.cpp`. **No number for GPT4All is stated in this report.** |
| **LM Studio: is there any health endpoint?** | LM Studio is closed-source. Absence from the docs and from the v1 endpoint table is not proof. I found **no** documented health/ready endpoint, but cannot assert none exists. |
| **LM Studio: default state of the "Enable CORS" switch** | `settings.md` lists the switch and describes it, but does **not** state its default. I cannot say whether browser probes work out of the box. |
| **LM Studio: `/api/v1/models` under the OpenAI-compat `/v1/models`** | `3_openai-compat/models.md` is a **stub** — it gives method, path and a curl example, but **no response body**. The exact JSON of LM Studio's `/v1/models` is unknown. |
| **LM Studio: `state` values other than `not-loaded`** | Only `"not-loaded"` appears in the docs example. The full enum (e.g. `"loaded"`) is unverified. |
| **lmstudio-python `default_api_ports` semantics** | I read the whole file (`default_api_ports = (41343, 52993, 16141, 39414, 22931)`) but **not its consumer**. How those five ports are used — fallback scan list? lms-cli ports? — is unknown. |
| **vLLM `GET /ping`** | Only mentioned in `disable_access_log_for_endpoints` help text (`"Example: \"/health,/metrics,/ping\""`). I did **not** confirm a `/ping` route is registered. |
| **vLLM port 9256 aggregated health endpoint** | Known only from a docstring: *"HTTP port for aggregated health endpoints in multi-port external LB mode."* Behaviour and response shape unknown. |
| **SGLang CORS default** | Not read. Browser reachability of SGLang is unknown. |
| **SGLang: exact final `/health` response body** | The success path returns a `Response`; I read the status logic but not the body bytes. Assume status-code-only, but unproven. |
| **SGLang's Ollama-compat route prefix** | `srt/entrypoints/ollama/protocol.py` exists, but I did not confirm the routes are served at `/api/*` or when they are enabled. |
| **KoboldCpp CORS default** | Not read. Browser reachability unknown. |
| **KoboldCpp `get_capabilities()` field names** | The `/api/extra/version` body is `json.dumps(get_capabilities())`; I did not read that function, so the capability field names are unknown. |
| **KoboldCpp `is_local` branch semantics** | `is_local = client_ip in ('127.0.0.1', '::1', 'localhost')` exists; the branches that consume it were not read. |
| **oobabooga `ModelInfoResponse` fields** | `/v1/internal/model/info` is typed `response_model=ModelInfoResponse`, but I did not read the model — so whether it carries a context length is unknown. |
| **oobabooga API bind host (as distinct from CORS)** | I read the CORS regex (localhost-only by default) but did not trace the API server's own bind address. |
| **Ollama: `/api/show` required body field** | Ollama's own docs POST `{"model": "llava"}`; **LiteLLM and Continue both POST `{"name": model}`**. I did not resolve which Ollama actually requires, or whether both work. |
| **Ollama: `allowedHost()` hostname allowlist** | `allowedHostsMiddleware` calls an `allowedHost(host)` helper I did not read, so the exact hostnames accepted (e.g. `<hostname>.local`) are unknown. |
| **Ollama `/api/status` `CloudStatus` fields** | Only the outer envelope `{"cloud": {…}}` is confirmed from `api/types.go`. |
| **llama.cpp: route table** | I confirmed `/health`, `/props`, `/metrics`, `/v1/models`, `/models` and the CLI defaults from the README and `arg.cpp`, but did **not** line-read `tools/server/server.cpp` for the authoritative route list. |
| **TGI: anything beyond the quickstart** | The launcher doc I tried returned 404. `/health`, `/info`, `/v1/models`, `/metrics`, the native `--port` default, and whether `/health` is unauthenticated are all **unverified**. Its port behaviour is only known as a docker `-p 8080:80` convention. |
| **RamaLama: full port set under `--api llama-stack`** | Known to publish multiple ports (model server, RAG proxy, 8321) but not enumerated. |
| **Docker Model Runner: `/engines/v1/models` response body** | The docker docs link to the OpenAI API reference instead of showing the body. Exact field names unknown. |
| **Docker Model Runner: auth and `/api/show` shape** | Auth requirements and whether `/api/show` returns Ollama's `model_info` shape are unverified. |
| **Docker Model Runner: a health endpoint** | None documented; unverified whether one exists. |
| **`fm serve`: full `/v1/models` body** | Only `id` = `system` and `owned_by` = `Apple` are given. The complete field set is unknown. |
| **`fm serve`: the exact `/health` `status` string across versions** | `"fm serve is running"` is a measured observation on one build (`FoundationModels 2.0.68`). Brittle by construction. |
| **`fm serve`: Apple primary documentation** | **None found.** All `fm serve` findings come from third-party measurement (`github.com/llostinthesauce/fm-guide`, corroborated by `github.com/1duo/apple-fm-serve` and a byteiota article). The guide's author explicitly states it was produced by *"running the shipped `fm` tool and its HTTP server"* and that nothing was decompiled. **Confidence: medium, not high.** |
| **Apple's licensing gate for `fm`** | `sudo fm license` appears in a third-party guide. The failure mode when unlicensed is unknown. |
| **apfel: when `context_window`/`capabilities` were added to `/v1/models`** | The fields are present in source now; I did not map them to a release in `CHANGELOG.md`, so version-gating is unknown. |
| **apfel: `/health` authentication exemption condition** | Source contains an expression involving `token != nil && !publicHealth && !isLoopbackHost(host)`; read out of context, so the precise condition is unknown. |
| **apfel-server-kit: `ApfelBinaryFinder.swift` search paths and `ApfelClient.swift` readiness check** | Not read. |
| **apfel: `ModelCapability` name strings** | `Sources/Core/ModelCapability.swift` was fetched but not line-read; the exact capability strings are unknown. |
| **Continue: `OllamaStatus.tsx` and `start_ollama.sh` detection method** | Read the file list only, not the contents. |
| **Aider's discovery behaviour** | A `11434` code search matched only docs/blog/assets. Aider could build the URL from a variable elsewhere. **No evidence of port probing, but not proven.** |
| **Open WebUI's docker-compose port wiring** | `docker-compose.yaml` and `docker-compose.api.yaml` located but not read. |
| **Any issue-tracker discussion of a standardised discovery mechanism** | I did **not** search the Ollama / llama.cpp / vLLM issue trackers. If such a proposal exists, it is not in this ledger. |

### 4.3 Method limitations worth flagging
- **Branch, not release.** Almost all source was read from `main`/`master`/`concedo` (KoboldCpp), not a pinned tag. Line numbers will drift.
- **No live servers were probed.** Every response body here is quoted from source or docs, never from a running instance. The only two response bodies measured *as observed at runtime* are `fm serve`'s (third-party) — everything else is code-derived.
- **Closed-source servers** (LM Studio, Jan's bundled llama.cpp, Msty, AnythingLLM) can only be characterised by docs, which are frequently stubs or silent on the exact points that matter.
- **Ledger granularity.** The 232 rows count *items examined* (a file, a doc page, or a major section of one), not 232 distinct URLs. There are ~100 distinct URLs. Rows that record a **negative** result ("no health endpoint") or an explicit **coverage gap** are included deliberately and marked as such.

---

## 5. Practical recommendations for the probe feature

1. **Probe by payload, not by port.** Port 11434 is legitimately Ollama *or* apfel; 9931 is new llama.cpp; 8080 is old llama.cpp *or* MLX-LM *or* RamaLama; 1976 is `fm serve`; 1337 Jan; 30000 SGLang; 12434 DMR; 5001 KoboldCpp; 5000 oobabooga; 1234 LM Studio; 8000 vLLM *or* llama-cpp-python. Prefer a discriminating response over a port guess.
2. **Always dial IPv4 loopback explicitly.** Defaults include the bare hostname `localhost` (llama-cpp-python), `0.0.0.0` (vLLM, LocalAI, KoboldCpp), and empty-string-means-all (KoboldCpp). Normalise to `127.0.0.1` before connecting — the same fix LocalAI's `waitForServerReady()` and Continue's binary build both apply. Consider also trying `::1`.
3. **Accept three health-endpoint dialects, and never require JSON.** HTTP-204/empty (`LocalAI /healthz`, `/readyz`; **vLLM `/health`**), status-code-only 503-while-loading (`llama.cpp`), and JSON with wildly different status strings (`ok`, `ok`, `fm serve is running`, `unavailable`). Also accept **plain text** (`Ollama`'s `GET /` → `Ollama is running`).
4. **Handle 503 as "exists but not ready" — it is the most common non-200.** `llama.cpp` (loading), `SGLang` (starting), `MLX-LM` (`unavailable`), `LocalAI /readyz` (preloading). `503` should mean *discovered*, not *absent*.
5. **Treat `200` + empty `data` as a live server.** oobabooga does this by design. `200` + error body also occurs (GPT4All registers POST handlers that return structured errors on `/v1/models`).
6. **Probe from outside the browser**, or proxy server-side. vLLM, apfel and oobabooga all block arbitrary browser origins by default; a failed browser probe proves nothing. If a browser client is a requirement, document which servers need `Enable CORS` / `--cors-origins` / `LOCALAI_CORS` / `--allowed-origins` / `--listen`.
7. **Do not send a Host header other than the one you dialled.** Loopback-bound Ollama (`allowedHostsMiddleware`) and apfel (`isAllowedHostHeader`) both reject alien Host headers with 403.
8. **Read capabilities defensively.** MLX-LM's `/v1/models` describes the HF cache, not the served model. llama.cpp's `meta` may be `null` mid-load and `output_modalities` may be `["text"]` "for compatibility only". SGLang's `/model_info` carries **no context length** (`hf_config` is commented out). Ollama's effective context is VRAM-tiered (4k/32k/256k) — prefer `POST /api/show`'s `model_info["*context_length*"]` over any constant.
9. **For dynamic-port servers you must read the port, not scan it.** RamaLama picks the first free port in 8080–8180; Jan binds port 0 on mobile. Log-scraping works for MLX-LM and LocalAI (plain `logging.info`) but **fails for `fm serve`**, whose banner and access log are TTY-gated — `fm serve > serve.log 2>&1` captures nothing.
10. **Never call mutation endpoints.** `POST /props` (llama.cpp), `POST /api/show` with side effects, `GET /api/extra/shutdown` (KoboldCpp), `/v1/internal/model/unload` (oobabooga), `/v1/models/load|unload` (LM Studio), `/models/{id}/start|stop` (Jan). Discovery must be read-only.
11. **Advertise identity, not just liveness.** Suggested markers, each verified above: Ollama → `GET /` plain text or `GET /api/version`; apfel → `GET /health` containing `prewarmed` + `context_window_measured`; `fm serve` → `/health` `status == "fm serve is running"`; MLX-LM → `/health` body `{"status":"unavailable"}` on 503; KoboldCpp → `/api/v1/info/version` `{"result":"1.2.5"}`; vLLM → `/v1/models` entries with `owned_by == "vllm"`; llama-cpp-python → `owned_by == "me"` and a plural `permissions` key; llama.cpp → `owned_by == "llamacpp"`; KoboldCpp → `owned_by == "koboldcpp"`; LM Studio → `owned_by`/`publisher` + `compatibility_type`.
