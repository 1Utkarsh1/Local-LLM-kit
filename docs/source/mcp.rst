MCP tools (Model Context Protocol)
==================================

`Model Context Protocol <https://modelcontextprotocol.io>`_ (MCP) is a standard
way for applications to expose *tools* to an LLM host. ``local_llm_kit`` can act
as an MCP **host**: it connects to one or more MCP servers, discovers the tools
they advertise, converts them to OpenAI ``tools`` specs, and feeds them to your
local model — including the auto-execution tool loop.

Why it matters: instead of hand-writing a tool for every capability, you point
Local LLM Kit at any MCP server (filesystem, git, sqlite, fetch, a private API,
...) and its tools become available to a local model.

Design notes
------------

* **Stdlib only.** The client lives in ``local_llm_kit.mcp`` and imports the
  Python standard library (``subprocess``, ``urllib``, ``json``, ...). The light
  core is unchanged and no new dependency is required. In particular, the
  official ``mcp`` SDK requires Python >= 3.10, while Local LLM Kit keeps
  working on Python 3.9.
* **Two transports.** ``stdio`` (a local server launched as a subprocess) and
  ``http`` (Streamable HTTP JSON-RPC, JSON body or request-scoped SSE).
* **Local by default.** A ``stdio`` server runs on your machine; nothing leaves
  the device unless you configure an ``http`` server yourself.

Quick start (Python)
--------------------

.. code-block:: python

   from local_llm_kit import LLM
   from local_llm_kit.mcp import MCPServer

   # 1. Connect to a server and see what it offers.
   with MCPServer(
       "fs",
       command="npx",
       args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
   ) as server:
       for tool in server.tools():
           print(tool.name, "-", tool.description)

   # 2. Register its tools on a model and chat normally.
   llm = LLM(model_path="qwen2.5-0.5b-instruct-q4_k_m.gguf", backend="llamacpp")
   llm.add_mcp_server("fs", command="npx", args=[...])   # or pass an MCPServer
   reply = llm.chat([{"role": "user", "content": "List the files in /tmp"}])
   llm.close()  # closes MCP connections too

Discovered tools are namespaced ``<server>__<tool>`` by default (so two servers
can both expose ``search``); pass ``prefix=False`` to keep the raw names, or a
string to choose the namespace. ``include=[...]`` / ``exclude=[...]`` filter by
the server's own tool names.

Quick start (CLI)
-----------------

Inspect and call tools without writing code:

.. code-block:: bash

   # List the tools a server advertises.
   local-llm-kit mcp tools --command "npx -y @modelcontextprotocol/server-filesystem /tmp"

   # Emit them as OpenAI tool specs (JSON).
   local-llm-kit mcp tools --command "uvx mcp-server-git" --json

   # Call one tool.
   local-llm-kit mcp call --command "uvx mcp-server-git" --tool git_status --args '{"repo_path": "."}'

   # Remote (Streamable HTTP) server, with an extra header.
   local-llm-kit mcp tools --url https://example.com/mcp --header "Authorization: Bearer $TOKEN"

Chat with a model and MCP tools attached:

.. code-block:: bash

   local-llm-kit chat -m ./models/qwen.gguf --backend llamacpp \
     --mcp-command "npx -y @modelcontextprotocol/server-filesystem /tmp"

``--mcp-command`` and ``--mcp-url`` may be repeated. When any server is
connected, tool execution is enabled for that session so the model can actually
use the tools.

Protocol coverage
-----------------

The client performs the ``initialize`` handshake on a best effort basis (the
2026 revision of the spec is stateless, so servers without ``initialize`` still
work), then uses:

* ``tools/list`` — with ``nextCursor`` pagination — to discover tools.
* ``tools/call`` — to invoke a tool; MCP content blocks are flattened to text
  (images/audio become ``[image: <mime>]`` placeholders, resources become
  ``[resource: <uri>]``).

Content-block flattening returns ``isError`` results as
:class:`~local_llm_kit.mcp.MCPToolError`, which the tool loop surfaces as an
ordinary tool-output string rather than a crash.

Security
--------

MCP servers are powerful — a filesystem server can read and write files, a shell
server can run commands. Treat a server command exactly like running that
program yourself:

* Prefer local ``stdio`` servers you trust; vet the package before ``npx``/``uvx``.
* Only point ``--url`` at endpoints you control.
* Tool execution can be disabled per call with ``execute_tools=False``.

Limitations
-----------

* The client implements the tools surface (``tools/list``, ``tools/call``).
  MCP *resources* and *prompts* are not exposed as tools.
* HTTP transport requires the server to answer a single POST with either a JSON
  body or an SSE stream; long-lived server-to-client streaming and OAuth
  authorization flows are not implemented.
* Not every local model is good at tool calling; a capable instruction-tuned
  model is still required. See :doc:`models`.
