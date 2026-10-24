Quickstart
==========

Install and run offline
-----------------------

The core package needs no GPU or model download. The examples here target
Local LLM Kit 0.3.1 or newer.

.. code-block:: bash

   pip install --upgrade local-llm-kit

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM(model_path="echo", backend="echo")
   response = llm.chat([{"role": "user", "content": "Hello"}])
   print(response["choices"][0]["message"]["content"])

Echo is a deterministic test backend. For model-generated answers, select an
in-process backend or connect to a running local server.

Connect a local server
----------------------

Start your server and load a model using its own setup instructions. Supply the
model ID it serves; the kit does not install or launch third-party servers.

.. code-block:: python

   from local_llm_kit import LLM

   llm = LLM(model_path="your-served-model-id", backend="sglang")
   # Override any preset with backend_kwargs={"base_url": "http://host:port/v1"}.
   response = llm.chat([{"role": "user", "content": "Hello"}])
   print(response["choices"][0]["message"]["content"])

Presets include ``sglang``, ``vllm``, ``lmstudio``, ``localai``, ``llama-server``,
``jan``, ``gpt4all``, ``mlx-lm`` and ``koboldcpp``. Ollama has its own native adapter:

.. code-block:: python

   llm = LLM(model_path="your-ollama-model", backend="ollama")

See :doc:`backends` for URLs, upstream setup links, AirLLM and compatibility limits.

Stream text
-----------

.. code-block:: python

   for chunk in llm.chat([{"role": "user", "content": "Write a short greeting"}], stream=True):
       print(chunk["choices"][0]["delta"].get("content") or "", end="")

Register tools
--------------

.. code-block:: python

   from local_llm_kit import tool

   @tool
   def add(a: int, b: int) -> int:
       """Add two numbers."""
       return a + b

   response = llm.chat(
       [{"role": "user", "content": "What is 2 plus 3?"}],
       tools=[add],
       execute_tools=False,
   )
   print(response["choices"][0]["message"])

``execute_tools=False`` returns requested calls for the application to handle.
Set it to ``True`` to run trusted registered tools in the library's bounded loop.
Per-call callables stay scoped to that request; ``llm.add_tool(add)`` registers a
persistent tool. A timeout bounds response waiting and does not kill a Python
function that is already running.

Structured output
-----------------

.. code-block:: python

   response = llm.chat(
       [{"role": "user", "content": "Return JSON with a city name."}],
       response_format={
           "type": "json_schema",
           "json_schema": {
               "name": "city",
               "schema": {
                   "type": "object",
                   "properties": {"name": {"type": "string"}},
                   "required": ["name"],
                   "additionalProperties": False,
               },
           },
       },
   )
   print(response["choices"][0]["message"]["content"])

Native Ollama and OpenAI-compatible adapters forward schemas to the server.
Schema enforcement needs support from the selected server and model. Prompt-only
backends provide JSON prompting and syntax recovery, not guaranteed schema validation.

Command line
------------

.. code-block:: bash

   local-llm-kit chat --backend echo --model echo -q "Hello"
   local-llm-kit chat --backend sglang --model your-served-model-id -q "Hello"
   local-llm-kit pull owner/repo --filename model.gguf --revision main --force
   local-llm-kit serve --backend echo --port 8000

For offline model loading, download dependencies and weights first, use local
checkpoint paths, and enable the Hugging Face offline environment settings
``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``.
