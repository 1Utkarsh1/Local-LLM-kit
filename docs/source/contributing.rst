Contributing
============

The repository's `CONTRIBUTING.md
<https://github.com/1Utkarsh1/Local-LLM-kit/blob/main/CONTRIBUTING.md>`_
contains the contribution process, compatibility requirements and backend contract.

.. code-block:: bash

   pip install -e ".[dev]"
   pytest
   ruff check .
   black --check .
   python verify_install.py

HTTP server tests additionally need the server extra and ``httpx``. Core tests
must work without optional inference packages or model downloads. Public API
changes need a regression test, current examples and an Unreleased changelog entry.

Build documentation with:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html -W --keep-going docs/source docs/_build/html

Open a `GitHub issue <https://github.com/1Utkarsh1/Local-LLM-kit/issues>`_
with package/Python/backend versions and a minimal reproduction. Contributions
are licensed under MIT and follow the repository Code of Conduct.
