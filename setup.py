"""Backwards-compatibility shim.

All packaging metadata lives in ``pyproject.toml`` (PEP 621).
This file exists only so legacy workflows keep working::

    python setup.py --version
    pip install -e .            # old pip / old setuptools

Version is single-sourced from ``local_llm_kit/__init__.py``::

    __version__ = "0.2.0"

via ``[tool.setuptools.dynamic]`` in ``pyproject.toml``. Do NOT add
``install_requires`` / ``extras_require`` / ``version=`` here.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
