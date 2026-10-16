Custom backend injection
========================

Prefer an existing :doc:`backends` adapter or the generic ``openai-compat``
client when your engine exposes HTTP. For a different local generation API,
implement the small ``BaseBackend`` contract and inject an instance. There is
no public ``LLM.register_backend`` or ``BaseTokenizer`` API.

.. code-block:: python

   from local_llm_kit import LLM
   from local_llm_kit.backends import BaseBackend

   class MyBackend(BaseBackend):
       def generate(self, prompt, **kwargs):
           return {"text": "received: " + prompt, "finish_reason": "stop"}

       def generate_stream(self, prompt, **kwargs):
           yield self.generate(prompt, **kwargs)

       def get_context_window(self):
           return 4096

       def count_tokens(self, text):
           return len(text.split())

   llm = LLM("custom-model", backend_instance=MyBackend(), use_cache=False)
   result = llm.complete("hello")
   print(result["choices"][0]["text"])

Generation results must provide ``text``; ``finish_reason`` is optional. Streams
must yield backend chunks, not OpenAI response envelopes. Heavy imports belong
inside backend construction so importing the core remains lightweight.

Optional methods include ``embed``, ``model_info`` and ``list_models``. Set
``supports_native_chat=True`` only when implementing compatible ``chat`` and
``chat_stream`` methods; native calls must return tool-call objects separately
from text. Otherwise the kit formats messages and tools as a text prompt.
See the existing adapters and their contract tests before extending a backend.
