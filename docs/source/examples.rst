Runnable examples
=================

The repository examples use the current ``LLM`` API and dictionary responses.
Run these from a checkout after ``pip install -e .``. The default echo path is
fully offline and deterministic; it does not measure model quality.

.. code-block:: bash

   python examples/simple_chat.py --backend echo --model echo
   python examples/tools_structured.py --backend echo --model echo
   python examples/rag_minimal.py --backend echo --model echo
   python examples/vision_chat.py --backend echo --model echo
   python examples/openai_compatible.py
   python examples/apple_foundation.py            # eligible Mac only; else setup steps
   python examples/apple_foundation.py --bridge   # needs `apfel --serve` running

``simple_chat.py`` demonstrates streaming and multi-turn calls.
``tools_structured.py`` demonstrates modern tools and JSON output.
``rag_minimal.py`` retrieves lexical matches from a small in-memory corpus.
``vision_chat.py`` shows text fallback; a meaningful image answer needs a vision
model and a supporting HTTP backend. ``openai_compatible.py`` attempts the local
HTTP server and falls back to echo if it is unavailable.
``apple_foundation.py`` uses Apple's on-device model natively
(``--bridge`` uses ``apfel --serve``); both print setup steps instead of
failing when the model or server is unavailable.

For a prepared Ollama model:

.. code-block:: bash

   python examples/simple_chat.py --backend ollama --model your-installed-model

Use :doc:`quickstart` for Python snippets, :doc:`tutorials` for complete small
flows and :doc:`backends` for runtime-specific configuration.
