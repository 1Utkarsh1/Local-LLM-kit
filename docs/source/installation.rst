Installation
============

The core client requires Python 3.9 or newer and ``typing-extensions``. HTTP
backends use the standard library; selecting a server does not require its
inference engine to be installed in the client environment.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install local-llm-kit

For unreleased features, install from the branch or checkout containing them:

.. code-block:: bash

   git clone https://github.com/1Utkarsh1/Local-LLM-kit.git
   cd Local-LLM-kit
   pip install -e .

Select optional dependencies as needed:

.. code-block:: bash

   pip install "local-llm-kit[transformers]"
   pip install "local-llm-kit[llamacpp]"
   pip install "local-llm-kit[airllm]"
   pip install "local-llm-kit[server]"
   pip install "local-llm-kit[embeddings]"
   pip install "local-llm-kit[rag]"

AirLLM is experimental. Optional packages have their own Python, operating
system and accelerator requirements. Install a PyTorch build appropriate to
your hardware using its upstream instructions. See :doc:`backends` for setup
links and feature limits rather than installing every extra by default.

Verify without downloading a model:

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM("echo", backend="echo")
   response = llm.chat([{"role": "user", "content": "Hello"}])
   assert isinstance(response["choices"][0]["message"]["content"], str)

If an optional import fails, install its named extra in the same Python
environment. Prepare dependencies and weights before disconnecting. For Hugging
Face loaders, use local paths and the offline environment settings described
in :doc:`backends`.
