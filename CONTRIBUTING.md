# Contributing to Local LLM Kit

Thanks for contributing! This guide gets you from zero to a passing PR in ~10 minutes.

> **v0.2.0 context:** core install stays light (stdlib + `typing-extensions`, Python 3.9+).
> Heavy deps (`torch`/`transformers`/`llama-cpp-python`/`fastapi`/`uvicorn`, etc.)
> are **optional extras, lazily imported** — never import them at module top-level in core paths.
> Keep the legacy API working: `LLM` / `chat()` / `complete()` / `add_function()`.

## 1. Code of Conduct

Be kind and professional. We follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
(see `CODE_OF_CONDUCT.md`). By participating you agree to it.

## 2. Ways to contribute

- Report bugs (use the Bug template — include repro + versions).
- Request a backend / model / endpoint (use the Backend Request template).
- Fix a `good first issue` / `help wanted` issue.
- Add docs or examples (`docs/`, `examples/`).
- Add a backend, CLI command, or OpenAI-compatible endpoint + tests.

## 3. Ground rules

- Backwards compat: don't break `LLM/chat/complete/add_function` signatures or response shapes.
- Light core: no new hard dependency in `install_requires` without maintainer approval.
- Lazy imports: `import torch`, `import transformers`, `import llama_cpp`, `import fastapi`, etc.
  **inside** functions/methods, with a helpful error telling the user which extra to install.
- Python 3.9 compatible: no `X | Y` annotations at runtime, no `match` where avoidable,
  no 3.10+-only stdlib.
- Style: `ruff` clean. Tests: `pytest` green.

## 4. Dev setup (venv + editable install)

```bash
git clone https://github.com/1Utkarsh1/Local-LLM-kit.git
cd Local-LLM-kit
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

python -m pip install -U pip
pip install -e ".[dev]"      # core + pytest + ruff (+ docs tooling)
```

### Backend-specific extras

```bash
pip install -e ".[transformers]"  # HF transformers backend (torch, transformers, accelerate)
pip install -e ".[llamacpp]"      # llama.cpp backend (llama-cpp-python, GGUF)
pip install -e ".[ollama]"        # Ollama convenience client
pip install -e ".[server]"        # `local-llm-kit serve` (fastapi, uvicorn)
pip install -e ".[all]"           # everything for full local testing
```

> Only install what you need. Core tests must pass with **no extras installed**
> (backends requiring extras are skipped, not failed, when the extra is missing).

## 5. Checks to run before you push

```bash
pytest -q                      # full suite; extras-missing backends should skip
ruff check . && ruff format --check .
python verify_install.py       # smoke-check packaged install
```

If you changed an OpenAI-compatible surface (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, tool-calling JSON), add/extend a test that asserts the wire shape.

## 6. Branch, commit, PR

```bash
git checkout -b feat/<short-name>   # or fix/<short-name>, docs/<short-name>
```

- Keep PRs small and focused (one feature/fix per PR).
- Conventional commits preferred: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.
- Update `CHANGELOG.md` under `[Unreleased]` (Added / Changed / Fixed).
- PR description must state: what, why, how tested (`pytest -q` output), extras needed,
  and any compat impact on `LLM/chat/complete/add_function`.

## 7. How to add a backend (with `EchoBackend` example)

All backends live in `local_llm_kit/backends/` and subclass `BaseBackend`
(`local_llm_kit/backends/base.py`): implement `generate`, `generate_stream`,
`get_context_window`, `count_tokens` (plus `chat`/`embed` when applicable).
Heavy imports go **inside methods**.

1. Create `local_llm_kit/backends/mybackend.py` following `echo.py` (the offline
   template — deterministic, stdlib-only, no network):

```python
"""My backend — template notes. No heavy deps at module top."""
from typing import Any, Dict, Iterator, Optional
from .base import BaseBackend

class MyBackend(BaseBackend):
    backend_name = "mybackend"

    def __init__(self, model_path: str = "default", **kwargs: Any):
        self.model_path = model_path
        # Real backends: lazy-import here, e.g.:
        #   try: import transformers
        #   except ImportError as e: raise ImportError(
        #       "This backend needs the 'transformers' extra: pip install 'local-llm-kit[transformers]'") from e

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"text": "echo: " + prompt}

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        yield {"text": self.generate(prompt, **kwargs)["text"]}

    def get_context_window(self) -> int:
        return 4096

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))
```

2. Register it in `local_llm_kit/backends/__init__.py` (`BACKENDS` + lazy imports)
   and accept it in `LLM(backend="mybackend", …)`.
3. Add `tests/test_mybackend.py`: `generate` returns `{"text": …}`, stream chunks
   concatenate to the non-stream text, `count_tokens`/`get_context_window` sane,
   and the test passes with **no extras installed**.
4. Document the extra (if any) in `README.md` install table + this file's §4, and show the
   `ImportError → pip install "local-llm-kit[<extra>]"` message.

## 8. Docs preview

Docs are Sphinx in `docs/source/` (built by Read the Docs per `.readthedocs.yml`).

```bash
pip install -e ".[dev]"
sphinx-build -b html docs/source docs/build/html
python -m http.server -d docs/build/html 8000
# open http://localhost:8000
```

If you change public API, update docstrings + `docs/source/` + `README.md` example.

## 9. Reporting bugs / requesting features

Use the issue templates (Bug / Feature / Backend Request). Include:
`pip show local-llm-kit`, Python version, OS, backend + extra versions,
minimal repro, expected vs actual, full traceback.

## 10. License

MIT. By contributing you agree your contributions are licensed under the repo's MIT license.
