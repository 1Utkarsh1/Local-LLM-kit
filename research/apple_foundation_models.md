# Apple Intelligence / Foundation Models as a backend for `local-llm-kit`

**Research date:** October 2026
**Scope:** How should a lightweight OpenAI-compatible Python client/adapter (Ollama, llama.cpp, vLLM, LM Studio, LocalAI, transformers, AirLLM) add Apple's on-device Apple Intelligence / Foundation Models as an *optional* backend?

**Bottom line:** There are exactly three viable integration paths, and they are not equivalent:

| Path | Mechanism | Dep weight | Min OS | Extra install |
|---|---|---|---|---|
| **A. `apple-fm-sdk` (Apple official)** | In-process Python bindings over a Swift/C shim | Medium (`pip install apple-fm-sdk`) | macOS 26.0 | **Full Xcode 26.0+** — package is **sdist-only on PyPI, no wheels** |
| **B. `apfel --serve` (3rd party, MIT)** | Local **OpenAI-compatible HTTP** server | Zero (subprocess/`brew`) | macOS 26, Apple Silicon | `brew install apfel` — **no Xcode needed** |
| **C. `fm serve` (Apple system tool, macOS 27)** | Local **OpenAI-compatible HTTP** server | Zero (ships with OS) | macOS 27 | `sudo fm license` (admin) |

**Recommendation:** make **B (`apfel --serve`) the primary/default path** because it is architecturally identical to every other backend `local-llm-kit` already supports (base URL + model id, zero new Python deps, no Xcode). Add **C (`fm serve`) as an auto-probed second HTTP option** for macOS 27 users (free, no install — but tool calling is broken on the current build). Add **A (`apple-fm-sdk`) as an opt-in extra** (`pip install local-llm-kit[apple]`) for users who already have Xcode 26 and want real in-process tool calling + `@generable` structured output — **never as a default or hard dependency**, because the sdist build requires a full Xcode toolchain and will fail outright on Linux/CI/Intel Macs.

---

## 1. Apple's official SDK: `apple/python-apple-fm-sdk`

**URL:** https://github.com/apple/python-apple-fm-sdk
**Docs site:** https://apple.github.io/python-apple-fm-sdk/
**PyPI:** https://pypi.org/project/apple-fm-sdk/

### 1.1 Identity and licensing

- Package name: `apple-fm-sdk`, import name `apple_fm_sdk`, distribution `apple-fm-sdk`.
- Latest version: **0.2.1**, uploaded **2026-06-29**. Release history on PyPI: `0.1.0` (2026-03-08), `0.1.1` (2026-03-08), `0.2.0` (2026-06-08), `0.2.1` (2026-06-29).
- License: **Apache-2.0** (`license_expression: Apache-2.0` on PyPI; full Apache 2.0 text in `LICENSE.md`). Source headers read `Copyright (C) 2026 Apple Inc. All Rights Reserved.` Apache-2.0 is permissive and redistribution-friendly **for the SDK itself**.
- Development status classifier: `Development Status :: 3 - Alpha`. `Operating System :: MacOS` **only**.
- README states verbatim: *"This project is not yet taking contributions. Stay tuned!"*
- **Version-skew bug worth knowing:** `pyproject.toml` declares `version = "0.2.1"` but `src/apple_fm_sdk/__init__.py` still hardcodes `__version__ = "0.1.0"`.

### 1.2 Requirements (verbatim from README)

> ## Requirements
> - macOS 26.0+
> - Download [Xcode 26.0+](https://developer.apple.com/xcode/) and agree to the [Xcode and Apple SDKs agreement](https://www.apple.com/legal/sla/docs/xcode.pdf) in the Xcode app.
> - Python 3.10+
> - Apple Intelligence turned on for [a compatible Mac](https://support.apple.com/en-us/121115)

`pyproject.toml`: `requires-python = ">=3.10"`, `dependencies = ["build", "setuptools>=75.3.2"]`.

### 1.3 ⚠️ The packaging detail that decides the integration path

**`apple-fm-sdk` is published to PyPI as an sdist ONLY — there are no wheels.** Verified from the PyPI JSON API: every release (`0.1.0`, `0.1.1`, `0.2.0`, `0.2.1`) has exactly one file, `apple_fm_sdk-<v>.tar.gz`, `packagetype: "sdist"`, `python_version: "source"`.

Consequences, all verified in-tree:

- `bin/build-distribution.sh` only runs `python3 -m build --sdist`. Apple's own release script never produces a wheel.
- `MANIFEST.in` contains `prune src/apple_fm_sdk/lib` and `global-exclude _ctypes_bindings.py` — i.e. compiled binaries and generated ctypes bindings are deliberately excluded from the sdist.
- Therefore `pip install apple-fm-sdk` **compiles Swift from source at install time**. `build_backend.py` (a custom PEP 517 backend, `build-backend = "build_backend"`, `backend-path = ["."]`) runs `swift build -c release` and `shutil.copytree(build_dir, libraryDirectory)`, then emits `_ctypes_bindings.py`.
- Hard preflight gates in `build_backend.py` that will *fail the install*:
  - macOS `platform.mac_ver()` must be `>= 26.0` → else `SwiftToolingError("macOS version X found, but version 26.0 or higher is required. This package requires macOS 26.0+ to build the Swift bindings.")`
  - `swift` must be on `PATH`.
  - `xcode-select -p` must **not** contain `CommandLineTools`: `"The active developer directory is set to Command Line Tools (...), but a full Xcode installation is required. Please install Xcode. Then open Xcode at least once to accept the license agreement and install the Swift SDKs."`
  - `xcodebuild -version` must report `>= 26.0`.
- The C bindings are conditional on SDK version: `_macos_sdk_major_version()` parses `xcrun --sdk macosx --show-sdk-version`; comment in source: *"`Attachment` (image support) only exists in the macOS 27+ SDK"* — `if sdk_major is not None and sdk_major >= 27: extra_swift_args = [...]`.
- `SystemLanguageModel.variant` docstring confirms the runtime consequence: *"`None` if the current OS doesn't support it or the Xcode version used to build this package doesn't include macOS 27 SDKs... Check that `xcrun --sdk macosx --show-sdk-version` reports 27 or later (switch Xcode with `sudo xcode-select --switch <path>` if not), then reinstall the package so the bindings are rebuilt."*

`pyproject.toml` `package-data` does declare `apple_fm_sdk = ["lib/*.a", "lib/*.dylib"]`, and `build_backend.py` rewrites library search dirs to "runtime-relative paths" — so a *wheel built locally* would run without Xcode. But since no wheels are published, end users always pay the build cost.

### 1.4 Exact API surface

Top-level `__init__.py` `__all__` (verbatim, complete):

```
SystemLanguageModel, LanguageModelSession, Attachment, ImageAttachment,
PromptComponent, Prompt, PromptError, ImagePromptError, Transcript,
SystemLanguageModelUseCase, SystemLanguageModelGuardrails,
SystemLanguageModelUnavailableReason, SystemLanguageModelVariant, Tool,
FoundationModelsError, GenerationError, ExceededContextWindowSizeError,
AssetsUnavailableError, GuardrailViolationError, UnsupportedGuideError,
UnsupportedLanguageOrLocaleError, InvalidGenerationSchemaError,
DecodingFailureError, RateLimitedError, ConcurrentRequestsError, RefusalError,
ToolCallError, GenerationErrorCode,
generable, guide, GenerationSchema, GeneratedContent, GenerationGuide,
GuideType, GenerationOptions, SamplingMode, SamplingModeType, GenerationID,
ConvertibleFromGeneratedContent, ConvertibleToGeneratedContent, Generable
```

**⚠️ Naming correction for the task brief:** there is **no `UnsupportedLanguageError`**. The real name is **`UnsupportedLanguageOrLocaleError`** (maps to C status code 5, `UNSUPPORTED_LANGUAGE_OR_LOCALE`).

**Sync vs async: async-only.** There is no synchronous API. `LanguageModelSession.respond` is `async def`; `stream_response` is an async generator. Both examples and README wrap everything in `asyncio.run(main())`. `c_helpers.py` bridges C callbacks into `loop.create_future()` / `loop = asyncio.get_running_loop()`.

#### `SystemLanguageModel`

```python
import apple_fm_sdk as fm

model = fm.SystemLanguageModel()          # default: GENERAL + DEFAULT guardrails
# or:
model = fm.SystemLanguageModel(
    use_case=fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
    guardrails=fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS,
)

is_available, reason = model.is_available()   # -> tuple[bool, Optional[SystemLanguageModelUnavailableReason]]
budget = model.context_size                   # -> int, property
v = model.variant                             # -> Optional[SystemLanguageModelVariant]; .display_name e.g. "AFM 3 Core"
await model.token_count("Hello, world!")      # -> int  (async)
await model.token_count(instructions="You are a helpful assistant.")
await model.token_count([MyTool()])           # list[Tool]
await model.token_count(Cat.generation_schema())
await model.token_count(session.transcript)
```

Note: Python uses a **constructor**, not Swift's `SystemLanguageModel.default`. Availability is a **method returning a tuple**, not Swift's `availability` property / `.unavailable(.deviceNotEligible)` enum pattern.

Enums:
- `SystemLanguageModelUseCase`: `GENERAL = 0`, `CONTENT_TAGGING = 1`
- `SystemLanguageModelGuardrails`: `DEFAULT = 0`, `PERMISSIVE_CONTENT_TRANSFORMATIONS = 1`
- `SystemLanguageModelUnavailableReason`: `APPLE_INTELLIGENCE_NOT_ENABLED = 0`, `DEVICE_NOT_ELIGIBLE = 1`, `MODEL_NOT_READY = 2`, `UNKNOWN = 0xFF`
- `SystemLanguageModelVariant`: frozen dataclass with `display_name: str`

#### `LanguageModelSession`

```python
session = fm.LanguageModelSession(
    instructions="You are a helpful assistant.",   # optional, maps to Swift Instructions
    model=model,                                   # optional SystemLanguageModel
    tools=[CalculatorTool(), WeatherTool()],      # optional list[Tool]
)
session.transcript                                 # Transcript, auto-updated after each request
session.is_responding                               # bool property
fm.LanguageModelSession.from_transcript(transcript, model=None)   # classmethod

# Basic
response: str = await session.respond("What is the capital of France?")

# Guided generation via decorator
cat: Cat = await session.respond("Generate an adorable rescue cat", generating=Cat)

# Guided generation via explicit schema
content = await session.respond("...", schema=fm.GenerationSchema(...))

# Guided generation via raw JSON Schema dict
content = await session.respond("...", json_schema={"type": "object", "properties": {...}})

# Streaming
async for chunk in session.stream_response("Tell me a story"):
    print(chunk, end="", flush=True)
```

`respond()` full signature:

```python
async def respond(
    self,
    prompt: Prompt,
    generating: Optional[Union[Type[Generable], Generable]] = None,
    *,
    schema: Optional[GenerationSchema] = None,
    json_schema: Optional[dict] = None,
    options: Optional[GenerationOptions] = None,
) -> Union[str, Any, GeneratedContent]:
```

`stream_response()` signature and a **critical behavioural difference from OpenAI**:

```python
async def stream_response(self, prompt: Prompt, options: Optional[GenerationOptions] = None) -> AsyncIterator
```

> *"Yields complete text snapshots (not deltas) as generation progresses"* … *"Each yielded value represents the complete response text generated so far, rather than the delta from the previous chunk."* … *"Does not support guided generation (text responses only)."* … *"The session transcript is updated only after streaming completes."*

So a `local-llm-kit` streaming adapter must **diff successive snapshots** to synthesise `delta` chunks. This is the single most likely integration bug.

Concurrency, from the class docstring verbatim:

> *"The session is thread-safe for sequential requests but does not support concurrent requests. If a request is in progress, attempting another request will wait for the first to complete."* … *"Sessions use an internal lock to prevent concurrent requests. If you need to handle multiple requests simultaneously, create multiple session instances."*

(`self._request_lock = asyncio.Lock()` in `__init__`; `asyncio.CancelledError` mid-stream triggers `lib.FMTaskCancel(task)`.)

#### Structured output: `@generable` and `guide`

```python
import apple_fm_sdk as fm

@fm.generable                        # bare-decorator form works (overloads allow @fm.generable, @fm.generable(), @fm.generable("description"))
class Cat:
    name: str
    age: int = fm.guide("Age in years", range=(0, 20))

cat = await session.respond("Generate an adorable rescue cat", generating=Cat)
```

Also exported: `GenerationGuide`, `GuideType`, `GeneratedContent`, `GenerationID`, `ConvertibleFromGeneratedContent`, `ConvertibleToGeneratedContent`, `GenerationSchema`. `generable_utils.generable` raises `GenerableDecoratorError` (a subclass of `InvalidGenerationSchemaError`) for malformed types.

#### Tools

```python
class CalculatorTool(fm.Tool):
    name = "calculator"
    description = "Performs basic arithmetic operations"

    @property
    def arguments_schema(self) -> fm.GenerationSchema: ...

    async def call(self, args) -> ...:   # args is GeneratedContent; access via args.value
        ...
```

`Tool` is `class Tool(_ManagedObject, ABC)`; subclasses must set `name: str` and `description: str` and implement the `arguments_schema` property plus an async `call()`. Tools are registered at **session construction**, not per-request.

#### `GenerationOptions` — note the missing `top_p`

```python
@dataclass
class GenerationOptions:
    sampling: Optional[SamplingMode] = None
    temperature: Optional[float] = None
    maximum_response_tokens: Optional[int] = None
```

That is the **complete** set of fields. `fm.SamplingMode.greedy()` or `fm.SamplingMode.random(top=50, probability_threshold=..., seed=42)` (fields: `mode_type`, `top`, `probability_threshold`, `seed`).

⚠️ **There is no `top_p` and no `max_tokens` parameter.** The `LanguageModelSession.__init__` docstring is *wrong* when it says *"Use this to customize temperature, top_p, and other generation parameters."* A `local-llm-kit` adapter must map OpenAI `top_p` → unsupported (drop or error), `max_tokens` → `maximum_response_tokens`, and `top_k` → `SamplingMode.random(top=...)`.

### 1.5 Error types (exact, from `src/apple_fm_sdk/errors.py`)

Hierarchy:

```
Exception
└── FoundationModelsError
    ├── GenerationError
    │   ├── ExceededContextWindowSizeError
    │   ├── AssetsUnavailableError
    │   ├── GuardrailViolationError
    │   ├── UnsupportedGuideError
    │   ├── UnsupportedLanguageOrLocaleError
    │   ├── DecodingFailureError
    │   ├── RateLimitedError
    │   ├── ConcurrentRequestsError
    │   └── RefusalError            # has .explanation_entries
    ├── InvalidGenerationSchemaError
    └── ToolCallError               # has .tool_name, .underlying_error
```

Wait — precisely: `InvalidGenerationSchemaError(FoundationModelsError)` and `ToolCallError(FoundationModelsError)` are siblings of `GenerationError`, **not** children. `ExceededContextWindowSizeError` etc. are children of `GenerationError`. `RefusalError.__init__(message, debug_description=None, explanation_entries=None)`.

`GenerationErrorCode(IntEnum)` maps 1:1 to C status codes:

| Code | Value | Exception |
|---|---|---|
| `SUCCESS` | 0 | — |
| `EXCEEDED_CONTEXT_WINDOW_SIZE` | 1 | `ExceededContextWindowSizeError` |
| `ASSETS_UNAVAILABLE` | 2 | `AssetsUnavailableError` |
| `GUARDRAIL_VIOLATION` | 3 | `GuardrailViolationError` |
| `UNSUPPORTED_GUIDE` | 4 | `UnsupportedGuideError` |
| `UNSUPPORTED_LANGUAGE_OR_LOCALE` | 5 | `UnsupportedLanguageOrLocaleError` |
| `DECODING_FAILURE` | 6 | `DecodingFailureError` |
| `RATE_LIMITED` | 7 | `RateLimitedError` |
| `CONCURRENT_REQUESTS` | 8 | `ConcurrentRequestsError` |
| `REFUSAL` | 9 | `RefusalError` |
| `INVALID_SCHEMA` | 10 | `InvalidGenerationSchemaError` |
| `UNKNOWN_ERROR` | 255 | generic `GenerationError("Unknown generation error (status: ...)")` |

`_status_code_to_exception()` builds messages as `f"{message}: {debug_description}"`, e.g. `"Context window size exceeded: ..."`. Human-readable messages: `"Context window size exceeded"`, `"Required assets are unavailable"`, `"Guardrail violation occurred"`, `"Unsupported guide used"`, `"Unsupported language or locale"`, `"Failed to decode response"`, `"Request was rate limited"`, `"Too many concurrent requests"`, `"Model refused to generate content"`, `"Invalid generation schema provided"`.

Suggested `local-llm-kit` mapping: `RateLimitedError`/`ConcurrentRequestsError` → retryable (429/503 equivalent); `ExceededContextWindowSizeError` → 400 context-length; `GuardrailViolationError`/`RefusalError` → content-filter finish reason; `AssetsUnavailableError` → "model not downloaded / Apple Intelligence off" (503); `UnsupportedLanguageOrLocaleError` → 400; `ToolCallError` → surface as an errored tool result, not a 5xx.

### 1.6 README code (verbatim)

```python
import apple_fm_sdk as fm
import asyncio

async def main():
    # Get the default system foundation model
    model = fm.SystemLanguageModel()

    # Check if the model is available
    is_available, reason = model.is_available()
    if is_available:
        # Create a session
        session = fm.LanguageModelSession()

        # Generate a response
        response = await session.respond("Hello, how are you?")
        print(f"Model response: {response}")
    else:
        print(f"Foundation Models not available: {reason}")

# Run async function
asyncio.run(main())
```

---

## 2. Apple's Foundation Models framework (Swift)

**URL:** https://developer.apple.com/documentation/foundationmodels
**Machine-readable source used:** `https://developer.apple.com/tutorials/data/documentation/foundationmodels/<page>.json`

### 2.1 Platform availability

`metadata.platforms` for the framework: iOS 26.0, iPadOS 26.0, Mac Catalyst 26.0, **macOS 26.0**, visionOS 26.0, watchOS 27.0.

`SystemLanguageModel.Availability` — macOS/iOS 26.0+. Cases: `.available`, `.unavailable(_:)` with `UnavailableReason` (confirmed cases from the Apple code sample: `.deviceNotEligible`, `.modelNotReady`, plus a catch-all `other`).

```swift
private var model = SystemLanguageModel.default
switch model.availability {
case .available:                    // Show your intelligence UI.
case .unavailable(.deviceNotEligible):  // Show an alternative UI.
case .unavailable(.modelNotReady):      // downloading, or other system reasons.
case .unavailable(let other):           // unknown reason.
}
```

Also `SystemLanguageModel.isAvailable: Bool`, `SystemLanguageModel.supportedLanguages: Set<Locale.Language>`, `SystemLanguageModel.supportsLocale(_:) -> Bool`, `SystemLanguageModel.contextSize: Int`, `SystemLanguageModel.tokenCount(for:)`, `SystemLanguageModel.variant` (`SystemLanguageModel.Variant`).

**Model versioning (important for prompt stability):** the `SystemLanguageModel` page states verbatim that there are *"3 model versions that align with: iOS, iPadOS, macOS, and visionOS 26.0 - 26.3; iOS, iPadOS, macOS, visionOS 26.4; iOS, iPadOS, macOS, and visionOS 27.0."* Apple's updates page adds: *"Because the model changes when a person updates to ... macOS 27 ... test your prompts with the new model to verify your app's behavior."*

### 2.2 Context window — **4096 tokens**

From https://developer.apple.com/documentation/foundationmodels/managing-the-context-window, verbatim:

> *"Apple's on-device foundation model has a context window of 4096 tokens per session, with a token representing each word, or partial word."*

> *"The size of the context window matters, because every session interaction with the model consumes input and output tokens from this window. This includes all prompts, instructions, tool definitions and their input and output, generable type schemas, and all of the model's responses."*

> *"In Latin alphabet languages such as English, a token typically represents three to four characters. For multibyte languages such as Chinese, Japanese, Korean, and Vietnamese a token typically represents one character."*

Apple's Feb-2026 update entry confirms the same number for the 26.4 model: *"Use the `#Playground` macro in Xcode to view an estimate of the usage of 4,096 tokens in the available context window."*

Guidance Apple gives that matters to a client library:
- *"Provide no more than three to five tools per request."*
- *"Reduce prompts to no more than three paragraphs in length."*
- *"Only use `maximumResponseTokens` to prevent verbose responses. Limiting tokens can cause the model to generate incomplete or grammatically incorrect responses."*
- On overflow: *"the session can no longer process additional requests and throws an `contextSizeExceeded` error. When this happens, trim the session history or create a new session to get a fresh context window."*
- **A session that has overflowed is dead** — the framework does not auto-trim. A client library must detect `ExceededContextWindowSizeError` and rebuild the session (via `LanguageModelSession(transcript:)` / Python `from_transcript`) rather than retrying.

**Third-party report of a larger window on macOS 27 (NOT in Apple's docs I read):** `apfel`'s README states *"4096 tokens on macOS 26 and on M1/M2 Macs, 8192 tokens on macOS 27 with an M3 or newer chip and 12 GB+ (Apple's AFM 3 Core Advanced model)"*, read at runtime from `SystemLanguageModel.contextSize`. `fm-guide` independently measured *"8,192 tokens, of which 8,132 can be prompt"* on macOS 27. Treat 4096 as the guaranteed/documented floor; read `context_size` at runtime rather than hardcoding.

### 2.3 Error types (Swift side)

Two generations of error types, which is a real portability trap:

**`LanguageModelError`** — **macOS/iOS/visionOS 27.0+**. Cases (from topic sections):
- `contextSizeExceeded(_:)` / `ContextSizeExceeded`
- `rateLimited(_:)` / `RateLimited`
- `refusal(_:)` / `Refusal`
- `timeout(_:)` / `Timeout`
- `guardrailViolation(_:)` / `GuardrailViolation`
- `unsupportedCapability(_:)` / `UnsupportedCapability`
- `unsupportedTranscriptContent(_:)` / `UnsupportedTranscriptContent`
- `unsupportedGenerationGuide(_:)` / `UnsupportedGenerationGuide`
- `unsupportedLanguageOrLocale(_:)` / `UnsupportedLanguageOrLocale`

**`SystemLanguageModel.Error`** — macOS 27.0+, case `assetsUnavailable(_:)` / `AssetsUnavailable`.

**Deprecated predecessor on macOS 26:** `LanguageModelSession.GenerationError`. Apple's own migration snippet:

```swift
do {
    let response = try await session.respond(to: prompt)
} catch {
    if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *),
        let error = error as? LanguageModelError
    {
        // Handle the error.
    } else if let error = error as? LanguageModelSession.GenerationError {
        // Handle the deprecated error.
    }
}
```

The Python SDK's flat `GenerationErrorCode` IntEnum (section 1.5) is a **stable simplification over both Swift generations** — that is an argument in favour of the Python SDK over hand-rolling against the Swift surface.

### 2.4 Private Cloud Compute — a different product

**`PrivateCloudComputeLanguageModel`** — macOS/iOS/iPadOS/visionOS/watchOS **27.0+**.
Apple's abstract: *"A variant of Apple Foundation Models that runs on Private Cloud Compute to provide enhanced capabilities while maintaining privacy guarantees."*
Overview: *"To use the server-based model that powers Apple Intelligence, you change a single line of code that you apply when creating a [session]."* … *"To develop with PCC you must meet certain eligibility requirements. To learn more and request access to the managed entitlement, see [com.apple.developer.private-cloud-compute]."*

**This is not "local" inference.** It requires the `com.apple.developer.private-cloud-compute` managed entitlement (Apple approval), it sends the request off-device to Apple's servers, and it is macOS 27+. For a library whose value proposition is *local* LLM backends, PCC should either be excluded or offered as a clearly-labelled separate backend. It is **not** in the Python SDK's `__all__` — the Python SDK wraps `SystemLanguageModel` only.

### 2.5 `@Generable` / `Tool` macros (Swift)

- `@Generable(description:)` macro on a struct → the framework guarantees the model produces instances of that type. Apple: *"the framework converts each type's structure into a JSON schema and sends that to the model"*, and *"Any `@Guide` descriptions you add also consume tokens."*
- `@Guide(description:)` and `.minimumCount/.maximumCount`, `.minimum/.maximum` refinements.
- `Tool` protocol: `let name`, `let description`, `@Generable struct Arguments`, `func call(arguments:) async throws -> String`.
- macOS 27 adds `ToolCallingMode`, dynamic profiles (`DynamicProfile`), `LanguageModel` / `LanguageModelExecutor` protocols for **custom providers** — this is the officially sanctioned hook for plugging in other models (including MLX).

---

## 3. `Arthur-Ficial/apfel` — the strongest HTTP-bridge candidate

**URL:** https://github.com/Arthur-Ficial/apfel · **Version:** 1.16.0 · **License: MIT** · **Stars: ~6,476** · Created 2026-03-24, last push 2026-10-14 (actively maintained; releases v1.15.0/v1.16.0 both 2026-10-05).

### 3.1 What it is

> *"Apple Silicon Macs ship a built-in LLM via Apple FoundationModels. `apfel` exposes it as a UNIX tool and a local OpenAI-compatible server. 100% on-device. No API keys, no cloud."*

Two modes (verbatim table from README):

| Mode | Command | What you get |
|------|---------|--------------|
| UNIX tool | `apfel "prompt"` / `echo "text" \| apfel` | Pipe-friendly answers, file attachments, JSON output, exit codes |
| OpenAI-compatible server | `apfel --serve` | Drop-in local `http://localhost:11434/v1` backend for OpenAI SDKs |

It is a **Swift 6.3 native binary**, not a Python package. Architecture per README:

```
CLI (single/stream/chat) ──┐
                           ├─→ FoundationModels.SystemLanguageModel
HTTP Server (/v1/*) ───────┘   (100% on-device, zero network)
                                ContextManager → Transcript API
                                SchemaConverter → native ToolDefinitions
                                TokenCounter → real token counts (SDK 26.4)
```

### 3.2 Install & requirements

```bash
brew install apfel              # recommended; homebrew-core
brew install Arthur-Ficial/tap/apfel
brew services start apfel       # background, like Ollama
```

> *"macOS 26 Tahoe+, Apple Silicon (M1+), [Apple Intelligence enabled](https://support.apple.com/en-us/121115)."*

From `docs/install.md`: *"No build tools needed"* for the Homebrew path. Building from source requires *"Swift 6.3+ with developer tools that include the macOS 27 SDK (Command Line Tools 27 or Xcode 27...) since apfel 1.15.0 the sources use macOS 27 FoundationModels API behind runtime availability checks, so a 26.x SDK no longer compiles them — the binary still runs on macOS 26."* Also note: **Command Line Tools are enough — no full Xcode required**, unlike Apple's Python SDK.

Other install channels: `nix profile install nixpkgs#apfel-llm`, Mint, mise, and a tap that adds 8 companion `apfel-*` commands.

### 3.3 OpenAI-compatible server surface

**Base URL: `http://localhost:11434/v1`** (deliberately collides with Ollama's default port — good for drop-in). Model id: **`apple-foundationmodel`**. `GET /v1/models` returns that single id.

Support matrix (verbatim from README):

| Feature | Status | Notes |
|---|---|---|
| `POST /v1/chat/completions` | Supported | Streaming + non-streaming |
| `POST /v1/responses` | Supported | string/message input, `instructions`, streaming (canonical event sequence), `text.format` incl. `json_schema`, function tools (non-streaming). Stateful features (`previous_response_id`, `store: true`, `background`, `reasoning`, hosted tools) → **honest 501s** |
| `GET /v1/models` | Supported | Returns `apple-foundationmodel` |
| `GET /health` | Supported | availability, context window, languages, `capabilities` / `capabilities_reported` |
| `GET /v1/logs`, `/v1/logs/stats` | Debug only | requires `--debug` |
| Tool calling | Supported | Native `ToolDefinition` + JSON detection |
| `response_format: json_object` | Supported | System-prompt injection; markdown fences stripped |
| `response_format: json_schema` | Supported | Guaranteed schema-conforming via `DynamicGenerationSchema`; **works with `stream: true`** |
| `temperature`, `top_p`, `max_tokens`, `seed` | Supported | Mapped to `GenerationOptions`. `top_p` is nucleus sampling; **`temperature: 0` maps to greedy (deterministic)** |
| `stream: true` | Supported | SSE; usage chunk only with `stream_options: {"include_usage": true}` |
| `usage` | Supported | Real tokenizer counts on macOS 26.4+ |
| `finish_reason` | Supported | `stop`, `tool_calls`, `length` |
| Context strategies | Supported | `x_context_strategy`, `x_context_max_turns`, `x_context_output_reserve` extension fields |
| CORS | Supported | Enable with `--cors` |
| `POST /v1/completions` | **501** | Legacy text completions not supported |
| `POST /v1/embeddings` | **501** | **Embeddings not available on-device** |
| `logprobs=true`, `n>1`, `stop`, `presence_penalty`, `frequency_penalty` | **400** | Rejected explicitly. `n=1` and `logprobs=false` accepted as no-ops |
| Multi-modal (images) | macOS 27 only | `image_url` base64 data URLs; 20 MB cap, longest side downscaled to 4096 px. Remote `http(s)` URLs and file paths → **always 400** |
| `Authorization` header | Supported | Required when `--token` is set |

This is a materially **better-shaped OpenAI surface than Apple's own `fm serve`** (see §5): it honours `max_tokens`, streams, emits proper `tool_calls`, supports `/v1/responses`, and returns explicit 400s/501s instead of silently ignoring parameters.

Concrete Python usage (verbatim):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")
resp = client.chat.completions.create(
    model="apple-foundationmodel",
    messages=[{"role": "user", "content": "What is 1+1?"}],
)
print(resp.choices[0].message.content)
```

Server flags worth knowing: `--serve`, `--permissive` (server-wide `.permissiveContentTransformations` guardrails, *no per-request override*), `--token` / `APFEL_TOKEN`, `--cors`, `--mcp` (local + remote MCP servers), `--debug`. Also `apfel --model-info` prints live availability + context size; `apfel --count-tokens` preflights a prompt.

### 3.4 Documented limitations (verbatim excerpts)

| Constraint | Detail |
|---|---|
| Context window | *"4096 tokens on macOS 26 and on M1/M2 Macs, 8192 tokens on macOS 27 with an M3 or newer chip and 12 GB+"* (input + output combined); read at runtime via `SystemLanguageModel.contextSize` |
| Platform | macOS 26+, Apple Silicon only |
| Model | *"One model (`apple-foundationmodel`, ~3B params on-device), **not configurable**"* |
| Guardrails | *"Apple's safety system may block benign prompts. `--permissive` reduces false positives"* |
| No embeddings | *"Not available on-device"* |
| Speed | *"a few seconds per response"* |
| Vision | macOS 27 only; *"OCR remains the reliable channel for reading print"* |
| Training cutoff | *"Apple has not published a precise cutoff... the model **confabulates a different date each sample**... Treat all model self-reports about its own training as unreliable."* |
| Clock/network | *"The model does not know today's date and has no network/clock access."* |

Overflow behaviour: *"If the prompt itself exceeds the window, generation cannot start and the request fails with `[context overflow]` (HTTP 400 / CLI exit 4)."*

### 3.5 Ecosystem

`apfel` is a tree of 10 sibling repos (`apfel-chat`, `apfel-clip`, `apfel-mcp`, `apfel-gui`, `apfel-tag`, `apfel-server-kit`, …) plus community projects including `apfelclaw`, `fruit-chat`, `local-claude` (Claude Code → apfel via an Anthropic-OpenAI proxy), and a Raycast extension. There is a Python guide at `docs/guides/python.md` in-tree. **This is by far the most mature non-Apple route.**

---

## 4. `simonw/llm-apple-foundation` — reference implementation for an in-process adapter

**URL:** https://github.com/simonw/llm-apple-foundation · **License: Apache-2.0** · Created **2026-07-26**, last push **2026-07-26** (13 stars; single-session burst, not yet iterated).
**⚠️ Not yet on PyPI** — `https://pypi.org/pypi/llm-apple-foundation/json` returns **HTTP 404**, despite the README's PyPI badge and its `llm install llm-apple-foundation` instruction. Treat any install instructions as aspirational; the only working install today is `uv run` from a git checkout.

### 4.1 What it is

An `llm` (Simon Willison's CLI) **plugin** that wraps Apple's official Python SDK: *"Run prompts through the Apple Foundation model on macOS using Apple's official Foundation Models SDK for Python."* It depends on `apple-fm-sdk` directly:

```toml
dependencies = [
    "apple-fm-sdk>=0.2.1",
    "llm>=0.32a3"
]
[project.entry-points.llm]
apple_foundation = "llm_apple_foundation"
```

**This is the single most useful artifact for `local-llm-kit`**: it is a working, complete example of adapting `apple-fm-sdk`'s in-process, async-only, snapshot-streaming API to a generic text-generation client interface — exactly the work `local-llm-kit` would otherwise have to redo.

### 4.2 Requirements (verbatim)

> - macOS 26 or later
> - A Mac that supports Apple Intelligence
> - Apple Intelligence turned on, with its model downloaded
> - Xcode 26 or later, with the Xcode and Apple SDKs agreement accepted
> - Python 3.10 or later

### 4.3 How it wraps the Swift framework — concrete techniques worth copying

Read from `llm_apple_foundation.py`:

- **Lazy SDK import.** `def _load_sdk(): try: import apple_fm_sdk as fm` — the plugin imports cleanly on non-macOS so the whole `llm` installation doesn't break. `local-llm-kit` should do the same.
- **Model construction from options.** Maps `use_case` → `fm.SystemLanguageModelUseCase.GENERAL` / `.CONTENT_TAGGING`, `guardrails` → `fm.SystemLanguageModelGuardrails.DEFAULT` / `.PERMISSIVE_CONTENT_TRANSFORMATIONS`.
- **Options mapping.** `sampling="greedy"` → `fm.SamplingMode.greedy()`; `sampling="random"` → `fm.SamplingMode.random(top=..., probability_threshold=..., seed=...)`; then `fm.GenerationOptions(temperature=..., maximum_response_tokens=options.max_tokens)`.
- **Synthetic tool schemas.** `_tool_schema(fm, tool)` imports the *private-ish* `from apple_fm_sdk.generation_property import Property` and walks the tool's JSON Schema (`$defs`/`definitions`/`$ref`, `enum` → `fm.GenerationGuide.anyOf(...)`, `const` → `fm.GenerationGuide.constant(...)`, `minimum` → range guides). It raises a descriptive `schema_error` for unsupported constructs rather than silently degrading — e.g. *"enum values must be strings"*. **Note this reaches into a module not re-exported by `apple_fm_sdk/__init__.py`, which is a real forward-compatibility risk.**
- **Thread-bridging.** Imports `queue` and `threading`, plus an `_ToolExecutionRequest` dataclass — because `Tool.call()` is async and C callbacks arrive on a different thread.
- **Async API registered separately.** `llm.get_async_model("apple-foundation")` alongside the sync `llm.get_model(...)`, because the SDK has no sync path.

### 4.4 Documented limitations (verbatim)

- *"LLM JSON Schemas are translated to the schema format required by Apple"*
- *"Apple's SDK does not stream guided generation, so schema responses are returned as one complete JSON value."*
- *"Image attachments are not yet supported by this plugin."*
- Options supported: `temperature`, `max_tokens`, `sampling`, `top`, `probability_threshold`, `seed`, `use_case`, `guardrails`.
- Native Apple session transcript is stored in LLM response metadata so conversations survive separate CLI invocations.

---

## 5. Other bridges found

### 5.1 `fm` — Apple's system CLI, and `fm serve` (macOS 27)

Not present in any Apple developer-documentation page I read. Third-party sources **disagree** on it, and this matters:

- `chatforest.com` claims flatly: *"Apple's `fm` CLI (macOS 27) ships `fm chat`, `fm respond`, and `fm schema` — verified against Apple's own WWDC26 session, **not** the OpenAI-compatible server some coverage claimed."*
- `llostinthesauce/fm-guide` claims the opposite and backs it with measurements: *"Tested on: macOS 27.0 (`26A5425a`), `FoundationModels` 2.0.68, Apple silicon, 2026-09-06"*, plus a detailed `serve-api.md` reproducing real SSE frames, usage blocks, and error envelopes.

**I could not verify `fm serve` against an Apple primary source, so treat it as unconfirmed-by-Apple.** Details below are from `fm-guide` (MIT-licensed writing, third-party), and should be re-verified on a real machine before being relied on:

- Requires admin license acceptance first: `sudo fm license` — otherwise even `--help` refuses.
- `fm serve` → `http://127.0.0.1:1976`; also `--host/--port` or `--socket /tmp/fm.sock`.
- Model id is **`system`** (`owned_by: "Apple"`). `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`. **No** `/v1/embeddings`, `/v1/completions`, or `/v1/responses` (404).
- **No authentication at all** — the `Authorization` header is neither required nor checked. Cross-origin requests are 403'd, which is the only thing stopping a web page from driving it.
- `max_tokens` is **accepted and ignored** (measured: `max_tokens: 1` → 75 completion tokens, `finish_reason: "stop"`, never `"length"`).
- **Tool calling is broken on FoundationModels 2.0.68**: tool argument schemas still constrain output, but it arrives as plain `content` with `tool_calls: null` and `finish_reason: "stop"`. `tool_choice: "required"` → HTTP 500. Inbound `role: "tool"` + `tool_call_id` still works.
- `response_format: json_schema` works (plain JSON Schema, no `x-order` needed); `json_object` is refused.
- `stop`, `n > 1`, `reasoning_effort` → 400. `temperature: 0` is deterministic.
- Banner/access log are TTY-gated — `fm serve > serve.log` captures nothing.

**Assessment:** attractive as a zero-install fallback on macOS 27, but its broken tool calling, ignored `max_tokens`, absent auth, and undocumented status make it a poor *primary* target.

### 5.2 `SpaceCorps/open-apple-models` — MIT, very new

*"Agentic tool calling for Apple Foundation Models: a Swift package, CLI and OpenAI-compatible server for on-device game AI, NPC dialogue and decisions on iOS, iPadOS, macOS and visionOS."* Created **2026-09-24**, last push **2026-09-25**, 0 stars, MIT. Its `docs/SERVER.md` positions it explicitly against `fm serve`: *"OpenAppleModelsServer serves Apple's on-device Foundation Model (and any other FoundationModels LanguageModel) over the OpenAI Chat Completions API, with working tool_calls. It replaces `fm serve` ... for agentic clients: `fm serve` accepts tools but never returns tool_calls."* **Too new and unproven to depend on**, but it is the only bridge I found that claims to solve the tool-calling gap on macOS 27 while also supporting *arbitrary* `LanguageModel` conformers (i.e. MLX/Core AI models), not just the system model.

### 5.3 `sohail288/apple-fm-cli` (PyPI `apple-fm-cli`) — ⚠️ licensing risk

PyPI: `apple-fm-cli` **0.3.0** (last release 2026-04-24), `requires-python = ">=3.14"`, deps `fastapi>=0.135.2`, `httpx>=0.28.1`, `uvicorn>=0.42.0`. *"A small command-line interface and local HTTP server that drive Apple's on-device Foundation Models (Apple Intelligence) from Python. The repo bundles a ctypes-based apple_fm_sdk wrapper around the system model plus apple_fm_cli..."*

**License field on PyPI: `Mixed; apple_fm_sdk carries Apple Inc. copyright — see file headers.`** This package **vendors Apple's SDK code** (and its ctypes bindings) into a distribution with an ambiguous license, pins Python 3.14+, and pulls in a FastAPI/uvicorn web stack. For `local-llm-kit`, this is **the wrong dependency** — you'd be re-distributing Apple-copyright code under an unclear license, and dragging in a web framework to talk to a local model.

### 5.4 `muta-dev/apple-fm-openai-bridge` — not viable

*"A local experiment harness for calling `SystemLanguageModel` from Python."* Created **2026-07-12**, last push **2026-07-12** (single day, never touched again), 0 stars, Apache-2.0. Explicitly self-described as an experiment. Do not depend on it.

### 5.5 Apple's own `foundation-models-utilities` (Swift, not Python)

https://github.com/apple/foundation-models-utilities — *"Emerging and experimental utilities for working with LLMs via the Foundation Models framework, such as custom skills, context management helpers, and **a chat completions client that connects to a hosted model of your choice**"*; supports *"Apple platforms and select Linux distributions"*. Cited in Apple's June-2026 update notes alongside `coreai-models` and `mlx-swift-lm` — *"Use `coreai-models` and `mlx-swift-lm` to integrate on-device models with the Foundation Models framework."* **Swift only**; useful as an architectural reference, not consumable from Python.

---

## 6. Verification of the specific claims in the brief

| Claim to verify | Verdict | Evidence |
|---|---|---|
| Minimum macOS = **26.0 Tahoe** | ✅ Confirmed | Apple docs `platforms: macOS 26.0`; SDK README "macOS 26.0+"; `build_backend.py` hard-fails below 26.0; apfel "macOS 26 Tahoe+" |
| Apple Silicon required | ✅ Confirmed (indirectly) | `DEVICE_NOT_ELIGIBLE` = "device does not meet the requirements"; apfel: *"Apple Silicon (M1 or later) is required. This is a hard Apple requirement - there is no workaround."* (Intel Macs → `deviceNotEligible`) |
| Minimum Python = **3.10** | ✅ Confirmed | `requires-python = ">=3.10"` in `pyproject.toml` and PyPI; classifiers list only 3.10/3.11/3.12 |
| SDK requires **Xcode** to build the Swift bridge | ✅ Confirmed, and stronger than expected | README lists "Download Xcode 26.0+"; `build_backend.py` explicitly rejects `CommandLineTools` developer dir and requires `xcodebuild >= 26.0`. **Command Line Tools alone are NOT sufficient for the Python SDK** (they *are* sufficient for `apfel` from source) |
| Must the user download/enable models? | ✅ Confirmed | `MODEL_NOT_READY` reason = "model is still being downloaded or prepared"; apfel: *"~3-4 GB on first enable. Keep your Mac on Wi-Fi and power."* Also: Apple Intelligence must be toggled on in System Settings, and device language + Siri language must both be set to the *same* supported language |
| Context window size | ✅ 4096 official; 8192 reported on macOS 27 (unverified by Apple docs) | See §2.2 |
| Streaming support | ✅ Confirmed, with a caveat | Python: `stream_response()` async generator, but yields **cumulative snapshots, not deltas**, and does **not** support guided generation. Swift/apfel/`fm serve`: true SSE deltas |
| Python SDK supports **async** | ✅ Confirmed — it is **async-only** | `async def respond`, async-generator `stream_response`, `asyncio.Lock` for concurrency; no sync API exists |
| Xcode **run script / framework search path** needed? | ❌ Not needed at runtime | `build_backend.py::_fix_library_search_dirs` rewrites search dirs to "runtime-relative paths" and the built `.dylib`/`.a` is copied into `apple_fm_sdk/lib/`. No run-script phase is required (that pattern applies to iOS app targets, not a Python wheel). **But** since this only matters for wheels — and no wheels are published — the point is moot for `pip install` today |

### 6.1 System model vs MLX models — keep these strictly separate

The task asks to distinguish Apple's **system** model from **MLX-downloaded** models. They are entirely different backends and `local-llm-kit` must expose them as distinct entries:

| | Apple system model (Foundation Models) | MLX / `mlx-community` models |
|---|---|---|
| Provenance | Ships with the OS, enabled via System Settings, ~3-4 GB auto-download | Downloaded from Hugging Face by the user |
| Weight access | **None** — opaque, not a file you can point at | Plain safetensors in a local dir |
| Model choice | **1 model**, *"not configurable"* (apfel); variants are OS-version-determined (`AFM 3 Core`, `AFM 3 Core Advanced`) | Any of thousands |
| Context | 4096 (macOS 26) / 8192 (macOS 27 M3+, per apfel) | Model-defined, often 32k–128k+ |
| API | Foundation Models / `apple-fm-sdk` / `apfel` / `fm serve` | `mlx-lm` (Python, in-process) or `mlx_lm.server` (`/v1/chat/completions`, port 8080) |
| License of weights | Apple's OS terms; no redistribution, no weight download | Per-model on HF (often Apache-2.0/MIT, some Llama/Gemma-style community licenses) |
| Python dep | `apple-fm-sdk` (Xcode required) | `mlx-lm` (pip wheel, Apple Silicon) |

**Apple's macOS-27 `LanguageModel` / `LanguageModelExecutor` protocols now officially let non-Apple models (including `mlx-swift-lm` and `coreai-models`) run *inside* a Foundation Models session** — but that's a Swift-side integration an app developer performs. A Python library cannot do it. So in `local-llm-kit` these stay two separate backend classes. Do **not** market the Apple backend as "any Apple-model backend".

### 6.2 Licensing & redistribution constraints

| Component | License | Redistribution notes |
|---|---|---|
| `apple-fm-sdk` (Apple) | **Apache-2.0** (PyPI `license_expression`) | Permissive: can be an optional dependency. Apache §6 grants **no trademark rights** — do not imply Apple endorsement ("Apple Intelligence", "Foundation Models" are Apple marks). Apache §7/§8 disclaim warranty/liability. Must ship the LICENSE and state changes if modified. |
| Apple Foundation Models framework | Apple SDK terms | The *framework* is not redistributable; it ships with the OS. `build_backend.py` links against the system SDK, which is fine, but you cannot bundle the framework. |
| Apple on-device **model weights** | Apple OS terms | Not separable, not downloadable, **cannot be bundled or redistributed**. Requires an Apple-Intelligence-eligible device with the feature enabled. Geographic restriction: Apple Intelligence is **blocked in mainland China**. |
| Apple Intelligence SDKs agreement | Separate agreement | Must be accepted *in the Xcode app* before the toolchain will build. This is a user-action prerequisite a library cannot automate. |
| `apfel` | **MIT** | Cleanest option for redistribution/recommendation. |
| `llm-apple-foundation` | **Apache-2.0** | Reference-reading only (not on PyPI). |
| `open-apple-models` | **MIT** | Too new. |
| `apple-fm-cli` (PyPI) | **"Mixed; apple_fm_sdk carries Apple Inc. copyright"** | ⚠️ **Avoid** — vendored Apple copyright under an unclear license. |
| Private Cloud Compute | Managed entitlement | Requires Apple approval; not redistributable and not local. |

**Practical guidance:** referencing `apfel` (MIT) via a documented, user-run `brew install` is the only path with no licensing exposure at all. Bundling `apple-fm-sdk` is fine (Apache-2.0) but only as an *optional extra*, never vendored.

---

## 7. Recommendation for `local-llm-kit`

### 7.1 Recommended architecture

```
local-llm-kit
├── backends/openai_compat.py         # existing
│     └── AppleBackend(base_url="http://localhost:11434/v1",
│                      model="apple-foundationmodel")        ← apfel, path B (DEFAULT)
├── backends/apple_fm_serve.py        # AppleBackend(base_url="http://127.0.0.1:1976/v1",
│                                     #              model="system")   ← fm serve, path C
└── backends/apple_native.py          # opt-in extra: local-llm-kit[apple]  ← path A
```

**Auto-detect order at construction:** probe `GET /health` on `127.0.0.1:11434` (apfel) and `127.0.0.1:1976` (`fm serve`) before falling back to the in-process SDK. Never *start* a server yourself without explicit opt-in — apfel docs note `fm serve > serve.log 2>&1` captures nothing because logging is TTY-gated, and both servers bind locally by default.

### 7.2 Why HTTP-bridge-first

1. **Zero new hard dependencies.** `apfel` is a `brew`-installed Swift binary; nothing enters `local-llm-kit`'s dependency tree. `apple-fm-sdk` drags in `build` + `setuptools>=75.3.2` *and* a **from-source Swift compile requiring full Xcode 26**.
2. **It matches the library's existing shape exactly.** Every other backend is `base_url + model + /v1/chat/completions`. apfel's base URL (`:11434/v1`) deliberately mirrors Ollama's, so a user can swap backends by changing one string.
3. **`apfel` is a *better* OpenAI citizen than Apple's own tools.** It honours `max_tokens`, emits real `tool_calls`/`finish_reason`, supports `json_schema` **with streaming**, adds `/v1/responses`, and answers with explicit 400/501s. `fm serve` ignores `max_tokens` and currently emits no `tool_calls` at all; the Python SDK has no `top_p` and streams snapshots-not-deltas.
4. **No Xcode, no admin license step, no Xcode-agreement prerequisite** — just `brew install apfel`, `apfel --serve`, `brew services start apfel`.
5. **`/v1/embeddings` is 501 everywhere** (Apple's model has no embedding head). Detect and raise a clear "Apple backend has no embeddings — use Ollama/llama.cpp for that" rather than surfacing a 501.

### 7.3 When to prefer A (in-process `apple-fm-sdk`)

Provide it as `local-llm-kit[apple]`, gated by a runtime capability check, for users who want:
- **Real tool calling** without a server process (the one clear advantage — `fm serve`'s is broken and apfel's requires the server).
- **Guaranteed structured output** via `@generable` / `GenerationSchema` / `json_schema=` (Apple's version is native constrained decoding; apfel's is too, but via HTTP).
- **Per-token cost accounting** via `model.token_count(...)` and `model.context_size` for a pre-flight context check — nicer than estimating.
- **No port, no daemon, no auth surface.**

Hard rules if you do:
- Import `apple_fm_sdk` **lazily inside a try/except** (copy `_load_sdk()` from `llm-apple-foundation`) so `import local_llm_kit` never breaks on Linux/CI/Intel.
- Detect the platform *first*: `sys.platform == "darwin"`, `platform.mac_ver() >= 26.0`, `platform.machine() == "arm64"`. If not, raise a clear `BackendUnavailable` — do not let `pip install` fail.
- **Never** put `apple-fm-sdk` in `[project.dependencies]`. It will attempt a Swift compile on every unsupported platform and fail hard (`SwiftToolingError`).
- Wrap every call: the SDK is async-only, so provide a `run_until_complete`-style sync facade (or document that this backend is async-first, as `llm-apple-foundation` did by registering a separate async model).
- **Diff the cumulative snapshots** from `stream_response()` into OpenAI-style deltas.
- Create **one `LanguageModelSession` per conversation**, and recreate it on `ExceededContextWindowSizeError` (an overflowed session is permanently dead).
- Serialize concurrent requests per session, or create a pool of sessions.
- Do not reach into `apple_fm_sdk.generation_property` as `llm-apple-foundation` does — that module is not in `__all__` and is not API-stable.
- Version-pin defensively: `apple-fm-sdk>=0.2.1,<0.3` (still `Development Status :: 3 - Alpha`, and `__version__` is already out of sync with the dist version).

### 7.4 Do not

- Do not depend on, vendor, or recommend **`apple-fm-cli`** (unclear "Mixed" license over Apple-copyright code, Python 3.14+, FastAPI/uvicorn weight).
- Do not depend on **`muta-dev/apple-fm-openai-bridge`** (abandoned after one day).
- Do not treat **`fm serve`** or **`open-apple-models`** as primary (undocumented/unverified tool calling; 0-star, 2-day-old repo).
- Do not advertise **Private Cloud Compute** as a local backend — it needs a managed Apple entitlement and leaves the device.
- Do not call an `mlx-community` model an "Apple Foundation Model". Different backend, different weights, different license.
- Do not assume `top_p`, `stop`, `n`, `frequency_penalty`, or `presence_penalty` work. All three paths either reject or silently ignore them; fail loudly client-side.

---

## 8. Unresolved questions / could not verify

1. **Does `fm serve` actually exist in macOS 27?** Sources directly contradict each other. `fm-guide` measures it in detail (FoundationModels 2.0.68, build `26A5425a`); `chatforest.com` asserts the OpenAI-server claim was never in Apple's WWDC26 material. **No Apple developer-documentation page I read mentions a `fm` CLI at all.** Verify on a macOS 27 machine (`man fm`, `fm --help`) before shipping support.
2. **Is the macOS 27 context window really 8192?** Apple's `managing-the-context-window` page still says 4,096 and Apple's Feb-2026 update says 4,096. Two independent third parties (apfel, fm-guide) say 8,192 on macOS 27 with M3+/12 GB. Unresolved — always read `context_size` at runtime.
3. **Which Apple Intelligence languages are supported today?** Apple defers to the "Supported languages" section of the Apple Intelligence marketing page, which I did not fetch. apfel enumerates: English, Danish, Dutch, French, German, Italian, Norwegian, Portuguese, Spanish, Swedish, Turkish, Chinese (Simplified/Traditional), Japanese, Korean, Vietnamese. **Unverified against Apple.**
4. **Exact minimum Apple Silicon / RAM.** All sources say "Apple Silicon, M1 or later" and Apple says "compatible Mac". No primary source states a RAM floor; the 12 GB figure in apfel applies only to the *8192-token M3+ variant* on macOS 27.
5. **Does the on-device model handle image input?** Apple ships `ImageAttachmentContent` / `Attachment` and the Python SDK exposes `ImageAttachment`, but `llm-apple-foundation` explicitly does not support image attachments and the SDK gates `Attachment` compilation behind the macOS 27 SDK. Apple's "Analyzing images with multimodal prompting" page (macOS 27) suggests images arrive with the 27 model. Unverified end-to-end from Python.
6. **Whether any *wheel* for `apple-fm-sdk` is planned.** No wheel exists for any of the four published versions. If Apple ships one, the calculus in §7.2 shifts materially — a wheel would remove the Xcode requirement entirely.
7. **`fm serve` authentication story.** fm-guide reports *no* auth at all and a 403 CSRF guard. If true, `local-llm-kit` should refuse to connect to a non-loopback `fm serve` endpoint.
8. **`apfel`'s `/v1/responses` stateful-feature 501s** mean any client relying on `previous_response_id` will need transcript replay. Not tested.

---

## 9. URL ledger — every URL substantively reviewed

### Primary (Apple)

1. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/README.md — README states requirements verbatim: *"macOS 26.0+ / Download Xcode 26.0+ / Python 3.10+ / Apple Intelligence turned on"*, and `pip install apple-fm-sdk`.
2. https://api.github.com/repos/apple/python-apple-fm-sdk/git/trees/main — full file tree; confirms `foundation-models-c/` Swift package, custom `build_backend.py`, and `docs/source/api/*`.
3. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/pyproject.toml — `version = "0.2.1"`, `requires-python = ">=3.10"`, `license = "Apache-2.0"`, `build-backend = "build_backend"`, `package-data = {apple_fm_sdk = ["lib/*.a", "lib/*.dylib"]}`.
4. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/LICENSE.md — Apache License 2.0, full text.
5. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/__init__.py — the exact `__all__` export list (34 symbols), and `__version__ = "0.1.0"` contradicting pyproject's 0.2.1.
6. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/errors.py — complete exception hierarchy + `GenerationErrorCode` IntEnum mapping C status codes 0-10 and 255 to typed exceptions.
7. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/core.py — `SystemLanguageModel(use_case=..., guardrails=...)`, `is_available() -> tuple[bool, Optional[...]]`, `context_size` property, `variant` property (returns `None` without a macOS 27 SDK), `async token_count(...)`.
8. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/session.py — `LanguageModelSession.__init__(instructions, model, tools)`, async-only `respond(prompt, generating, *, schema, json_schema, options)`, `stream_response()` documented to yield *"complete text snapshots (not deltas)"* and to not support guided generation; `asyncio.Lock` serializes concurrent requests.
9. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/generation_options.py — the **complete** `GenerationOptions` field list is only `sampling`, `temperature`, `maximum_response_tokens` — **there is no `top_p`**, despite the session docstring claiming so.
10. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/tool.py — `class Tool(_ManagedObject, ABC)` with `name`, `description`, `arguments_schema` property, and async `call()`; tools are bound at session construction.
11. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/src/apple_fm_sdk/generable_utils.py — `generable()` has three overloads (`@generable`, `@generable()`, `@generable("description")`) and raises `GenerableDecoratorError(InvalidGenerationSchemaError)`.
12. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/build_backend.py — install-time hard gates: macOS >= 26.0, `swift` on PATH, **`xcode-select -p` must not contain `CommandLineTools`** ("a full Xcode installation is required"), `xcodebuild >= 26.0`; runs `swift build -c release`; `Attachment` compiled only when `xcrun --sdk macosx --show-sdk-version` >= 27.
13. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/MANIFEST.in — `prune src/apple_fm_sdk/lib` and `global-exclude _ctypes_bindings.py` prove the sdist ships no compiled binaries.
14. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/bin/build-distribution.sh — Apple's own release script runs `python3 -m build --sdist` only; no wheel is ever produced.
15. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/docs/source/getting_started.rst — prerequisites repeat Xcode 26.0+ and add the troubleshooting line *"Verify that your Xcode version matches your macOS version exactly."*
16. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/docs/source/streaming.rst — canonical `async for chunk in session.stream_response(...)` example.
17. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/docs/source/api/systemmodel.rst — Sphinx `autoclass` listing confirming `SystemLanguageModelUseCase`, `...Guardrails`, `...UnavailableReason`, `...Variant` are the four public config enums.
18. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/examples/simple_inference.py — minimal `asyncio.run(main())` example with instructions and follow-up turn.
19. https://raw.githubusercontent.com/apple/python-apple-fm-sdk/main/examples/streaming_example.py — streaming example confirming async-generator usage.
20. https://pypi.org/pypi/apple-fm-sdk/json — **sdist-only across all four releases**; latest 0.2.1 uploaded 2026-06-29; `license_expression: Apache-2.0`; `requires_dist: ['build', 'setuptools>=75.3.2']`.
21. https://developer.apple.com/documentation/foundationmodels (and its `.json` payload) — framework platforms iOS/iPadOS/Mac Catalyst/macOS/visionOS **26.0**, watchOS 27.0; topic index enumerates `SystemLanguageModel`, `LanguageModelError`, `PrivateCloudComputeLanguageModel`, `LanguageModel`/`LanguageModelExecutor`, `Tool`, `@Generable`.
22. https://developer.apple.com/documentation/foundationmodels/managing-the-context-window — **"Apple's on-device foundation model has a context window of 4096 tokens per session"**; every prompt/instruction/tool/schema/response draws from the same window; exceeding it throws `contextSizeExceeded` and the session stops responding; recommends 3-5 tools max.
23. https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel — three model versions (26.0-26.3, 26.4, 27.0); availability switch sample with `.deviceNotEligible` / `.modelNotReady`; `contextSize`, `supportedLanguages`, `supportsLocale(_:)`, `tokenCount(for:)`, `variant`.
24. https://developer.apple.com/documentation/foundationmodels/languagemodelerror — macOS 27.0+; nine cases including `contextSizeExceeded`, `rateLimited`, `refusal`, `timeout`, `guardrailViolation`, `unsupportedLanguageOrLocale`; and the `#available` fallback to the deprecated `LanguageModelSession.GenerationError` on macOS 26.
25. https://developer.apple.com/documentation/foundationmodels/privatecloudcomputelanguagemodel — macOS 27.0+; requires the `com.apple.developer.private-cloud-compute` **managed entitlement**; *"runs on Private Cloud Compute"* — a server model, not on-device.
26. https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel/availability-swift.enum — macOS 26.0+; `.available` / `.unavailable(_:)` + `UnavailableReason`.
27. https://developer.apple.com/documentation/updates/foundationmodels — timeline: **March 2026** python-apple-fm-sdk; **February 2026** new 26.4 model + guardrail improvements + `tokenCount`/`contextSize` (with the 4,096-token Xcode figure); **June 2026** macOS 27 model, `LanguageModelError`, `LanguageModel` protocol, PCC, `foundation-models-utilities`, and *"Use `coreai-models` and `mlx-swift-lm` to integrate on-device models with the Foundation Models framework."*

### Primary (third-party projects)

28. https://raw.githubusercontent.com/Arthur-Ficial/apfel/main/README.md — MIT; `brew install apfel`; `apfel --serve` on `http://localhost:11434/v1` with model id `apple-foundationmodel`; full OpenAI support matrix (embeddings/completions → 501, `stop`/`n>1`/penalties → 400); context *"4096 tokens on macOS 26 and on M1/M2 Macs, 8192 tokens on macOS 27 with an M3 or newer chip and 12 GB+"*.
29. https://raw.githubusercontent.com/Arthur-Ficial/apfel/main/docs/install.md — homebrew path needs *"No build tools needed"*; source build needs *"macOS 27 SDK (Command Line Tools 27 or Xcode 27)"* and **CLT suffices** (contrast with Apple's SDK, which demands full Xcode); the three `Model unavailable` causes: Apple Intelligence off / not eligible / still downloading (~3-4 GB); Apple Intelligence blocked in mainland China.
30. https://api.github.com/repos/Arthur-Ficial/apfel — MIT, 6,476 stars, created 2026-03-24, pushed 2026-10-14.
31. https://api.github.com/repos/Arthur-Ficial/apfel/releases — v1.16.0 and v1.15.0 both published 2026-10-05; release notes reference macOS 27 vision input, SDK stamp assertions, and mid-stream overflow handling.
32. https://raw.githubusercontent.com/simonw/llm-apple-foundation/main/README.md — Apache-2.0 plugin over `apple-fm-sdk`; requirements list *"Xcode 26 or later, with the Xcode and Apple SDKs agreement accepted"*; *"Apple's SDK does not stream guided generation, so schema responses are returned as one complete JSON value"*; image attachments unsupported.
33. https://api.github.com/repos/simonw/llm-apple-foundation — created **2026-07-26**, pushed same day, 13 stars, Apache-2.0.
34. https://api.github.com/repos/simonw/llm-apple-foundation/git/trees/main — single-module plugin (`llm_apple_foundation.py`, 17.7 KB) + tests.
35. https://raw.githubusercontent.com/simonw/llm-apple-foundation/main/pyproject.toml — `dependencies = ["apple-fm-sdk>=0.2.1", "llm>=0.32a3"]`; entry point `[project.entry-points.llm] apple_foundation = "llm_apple_foundation"`.
36. https://raw.githubusercontent.com/simonw/llm-apple-foundation/main/llm_apple_foundation.py — `_load_sdk()` lazy import; `_create_system_model()` maps use_case/guardrails; `_create_generation_options()` maps `sampling`/`temperature`/`maximum_response_tokens`; `_tool_schema()` imports the non-exported `apple_fm_sdk.generation_property.Property` and uses `threading` + `queue` to bridge async tool calls.
37. https://pypi.org/pypi/llm-apple-foundation/json — **HTTP 404: not published to PyPI**, contradicting its own README install instructions.
38. https://raw.githubusercontent.com/llostinthesauce/fm-guide/main/README.md — claims `fm serve` is an OpenAI-compatible endpoint on macOS 27; *"Tested on: macOS 27.0 (`26A5425a`), `FoundationModels` 2.0.68, Apple silicon, 2026-09-06"*; requires `sudo fm license` first; context *"8,192 tokens, of which 8,132 can be prompt"*.
39. https://raw.githubusercontent.com/llostinthesauce/fm-guide/main/docs/serve-api.md — `fm serve` binds `127.0.0.1:1976`, model id `system`; only `/health`, `/v1/models`, `/v1/chat/completions`; **no authentication whatsoever**; `max_tokens` accepted and ignored (measured `max_tokens: 1` → 75 completion tokens); **tool calling broken on 2.0.68** (`tool_calls: null`, args delivered as prose); `json_schema` works without `x-order`; cross-origin → 403.
40. https://pypi.org/pypi/apple-fm-cli/json — `apple-fm-cli` 0.3.0 (2026-04-24), `requires-python >=3.14`, deps FastAPI/httpx/uvicorn, **license `"Mixed; apple_fm_sdk carries Apple Inc. copyright"`** — a redistribution red flag.
41. https://api.github.com/repos/muta-dev/apple-fm-openai-bridge — created and last pushed **2026-07-12**, 0 stars, Apache-2.0, self-described *"experiment harness"* — unmaintained.
42. https://api.github.com/repos/SpaceCorps/open-apple-models — MIT, created **2026-09-24**, pushed 2026-09-25, 0 stars; *"Agentic tool calling for Apple Foundation Models ... OpenAI-compatible server"*.
43. https://github.com/SpaceCorps/open-apple-models/blob/main/docs/SERVER.md — *"fm serve accepts tools but never returns tool_calls, and it answers tool_choice ..."*; positions itself as the tool-calling-capable replacement.
44. https://github.com/apple/foundation-models-utilities (via search result) — Apple's own Swift utilities package incl. *"a chat completions client that connects to a hosted model of your choice"*; *"Apple platforms and select Linux distributions"*; Swift-only.
45. https://apple.github.io/python-apple-fm-sdk/ — official hosted docs; reiterates Python 3.10+ and *"macOS 26.0+"*.
46. https://chatforest.com/builders-log/apple-fm-cli-python-sdk-fm-serve-openai-compatible-psotu-wwdc-2026/ — **dissents**: *"Apple's `fm` CLI (macOS 27) ships `fm chat`, `fm respond`, and `fm schema` — verified against Apple's own WWDC26 session, not the OpenAI-compatible server some coverage claimed."* The basis for my "unverified" tag on `fm serve`.
47. https://byteiota.com/apple-fm-cli-local-ai-server/ — supports the `fm serve` claim: *"`fm serve`, which turns your Mac into a local OpenAI-compatible inference server."* Cited only to document the contradiction.

### Reviewed but non-authoritative / corroborating only

48. https://www.iosdev.in/articles/fm-cli-macos-27 — third-party blog repeating the `fm serve` OpenAI-compatibility claim.
49. https://apidog.com/blog/how-to-convert-apple-on-device-llm-to-openai-compatible-api/ — third-party blog on converting Apple on-device LLM to an OpenAI-compatible API.

---

## 10. One-paragraph answer

`local-llm-kit` should treat Apple Intelligence as **two separate optional backends, both gated behind an availability probe, with the HTTP bridge as the default**. The default is `apfel --serve` (MIT, `brew install apfel`, no Xcode, base URL `http://localhost:11434/v1`, model `apple-foundationmodel`, real streaming + real `tool_calls` + honoured `max_tokens`), auto-probed on `127.0.0.1:11434` alongside Apple's own `fm serve` on `127.0.0.1:1976` for macOS 27 users (free and pre-installed, but with broken tool calling and an ignored `max_tokens` on FoundationModels 2.0.68, and no auth). The opt-in `[apple]` extra wraps Apple's official `apple-fm-sdk` (Apache-2.0, latest 0.2.1, macOS 26.0+, Apple Silicon, Python 3.10+, **async-only**, in-process, no HTTP server) — but it must never be a hard dependency, because PyPI ships **sdist only, no wheels**, so `pip install apple-fm-sdk` compiles Swift from source and hard-fails unless the machine has **full Xcode 26.0+** (Command Line Tools are explicitly rejected by `build_backend.py`), macOS 26.0+, and the Apple SDKs agreement accepted in the Xcode app. Whatever path is chosen, the backend must map Apple's hard limits into the client's semantics: a **4096-token session window** (reported as 8192 on macOS 27 with M3+/12 GB by third parties, read at runtime from `context_size` rather than hardcoded), a **dead session after `ExceededContextWindowSizeError`** requiring rebuild, **no embeddings** (501 everywhere), **17 supported languages** (`UnsupportedLanguageOrLocaleError` is the real exception name, not `UnsupportedLanguageError`), a **~3-4 GB first-run model download** plus an Apple-Intelligence toggle the library cannot enable, and **no `top_p`** in the native Python options. Keep the Apple *system* model (~3B, opaque, OS-managed, non-redistributable weights) strictly distinct from `mlx-community` downloaded models — Apple's macOS-27 `LanguageModel`/`LanguageModelExecutor` protocols can bridge the latter, but only from Swift, not from Python.
