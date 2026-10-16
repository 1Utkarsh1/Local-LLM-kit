Model files and caches
======================

In-process backends load local checkpoints or GGUF files. Server backends take
the model ID exposed by a running engine, not a client-side file to upload.
The kit does not choose a universal best model; use the upstream runtime's model
compatibility information and evaluate your own workload.

.. code-block:: python

   from local_llm_kit import LLM

   # Requires the corresponding extra and prepared local weights.
   llm = LLM("/path/to/checkpoint", backend="transformers")
   gguf = LLM("/path/to/model.gguf", backend="llamacpp")

Download a chosen file
----------------------

.. code-block:: bash

   local-llm-kit pull owner/repository --filename model.gguf --revision main
   local-llm-kit list
   local-llm-kit pull owner/repository --filename model.gguf --force

Or call the helper:

.. code-block:: python

   from local_llm_kit.models import download_model

   path = download_model("owner/repository", filename="model.gguf", revision="main")

Replace the placeholders with an accessible repository and filename. This step
uses the network. The model's license and hardware requirements still apply.

Cache identity and recovery
---------------------------

Repository downloads are scoped by repository and revision. Hugging Face cache
lookup follows the requested ref or commit. URL downloads include the full URL
hash in their cache path; older flat URL files are preserved but may be downloaded
once again. ``force=True`` bypasses a reusable download.

Downloads write to unique temporary siblings, check declared transfer size and
atomically replace the target on success. Failures clean up their temporary
file and preserve an existing complete target. Cache paths must remain inside
the configured cache root, including through symlinks.

See :doc:`api_reference` for model helper signatures and :doc:`backends` for
AirLLM shard preparation and offline loading.
