Welcome to local_llm_kit's documentation!
=========================================

``local_llm_kit`` is a Python package that provides an OpenAI-like interface for local language models. It allows you to run language models locally while maintaining compatibility with OpenAI's API structure.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   backends
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   models
   api_reference
   examples
   mcp
   best_practices

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics:

   custom_models
   performance
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Development:

   contributing
   changelog

Features
--------

* OpenAI-compatible API interface
* Local model adapters: Transformers, llama.cpp and optional experimental AirLLM
* Native Ollama chat and HTTP presets covering 12 local runtimes (see :doc:`backends`)
* Chat completion API
* Function calling capability
* MCP (Model Context Protocol) host: discover and call tools from MCP servers
* Streaming responses
* JSON mode output
* Memory management
* Custom prompt formatting
* Extensive documentation and tutorials
* Performance optimization guides
* Community-driven development

Quick Installation
------------------

.. code-block:: bash

   pip install local-llm-kit

Quick Example
-------------

.. code-block:: python

   from local_llm_kit import LLM

   # Initialize the client
   client = LLM(model_path="echo", backend="echo")

   # Chat completion
   response = client.chat(
       messages=[
           {"role": "user", "content": "What is the capital of France?"}
       ]
   )

   print(response["choices"][0]["message"]["content"])

Getting Help
------------

If you need help using ``local_llm_kit``, you have several options:

1. Check the :doc:`tutorials` for step-by-step guides
2. Read the :doc:`api_reference` for detailed API documentation
3. Look through :doc:`examples` for common use cases
4. Visit our `GitHub Issues <https://github.com/1Utkarsh1/Local-LLM-kit/issues>`_ page
5. Join our community discussions

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details on how to:

* Report bugs
* Suggest features
* Submit pull requests
* Improve documentation

License
-------

``local_llm_kit`` is released under the MIT License. See the LICENSE file for more details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
