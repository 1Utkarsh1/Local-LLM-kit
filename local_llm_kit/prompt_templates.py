"""
Prompt template system for local_llm_kit.

Provides:
    * :class:`PromptTemplate` -- lightweight templates with ``{{variable}}``
      and ``{{variable|default}}`` plain-text substitution (only double-brace expressions are replaced).
      Values are not escaped; use json.dumps for JSON data.
    * :class:`TemplateRegistry` -- in-memory named-template registry with
      YAML/JSON file loading.
    * Module-level helpers (:func:`register_template`, :func:`get_template`,
      :func:`render_template`) and a default registry.

YAML/JSON loading uses only the standard library by default (``json``).
When the optional ``pyyaml`` package is installed, ``load_file`` also accepts
``.yaml``/``.yml`` files. Loading ``pyyaml`` is deferred to call time so the
light core install is never disturbed.

Python 3.9 compatible (``typing.Dict``/``List``/``Optional`` house style).
"""

import json
import re
from typing import Any, Dict, List, Optional

__all__ = [
    "PromptTemplate",
    "TemplateRegistry",
    "register_template",
    "get_template",
    "render_template",
    "DEFAULT_REGISTRY",
]

# Matches {{name}} or {{name|default}}; name is a bare identifier, default is
# everything after the first | up to the closing }}.
_VAR_RE = re.compile(r"\{\{([a-zA-Z_]\w*(?:\|[^}]*)?)\}\}")


class PromptTemplate:
    """A lightweight prompt template with ``{{variable}}`` substitution.

    Supports two placeholder forms:

    * ``{{name}}``  -- required variable (raises ``KeyError`` if absent).
    * ``{{name|fallback}}`` -- optional variable with a default value.

    Unlike ``str.format``, only double-brace expressions are replaced; bare
    single braces (common in JSON payloads) are left untouched.

    :param template_str: the raw template text.
    :param name: optional human-readable name.
    :param description: optional description.
    """

    def __init__(
        self,
        template_str: str,
        name: Optional[str] = None,
        description: str = "",
    ) -> None:
        if not isinstance(template_str, str):
            raise TypeError("template_str must be a string.")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("Template name must be a non-empty string or None.")
        if not isinstance(description, str):
            raise TypeError("description must be a string.")
        self.template_str = template_str
        self.name = name
        self.description = description

    def render(self, **kwargs: Any) -> str:
        """Render the template, substituting every ``{{...}}`` placeholder."""

        def _replace(match: "re.Match[str]") -> str:
            expr = match.group(1).strip()
            if "|" in expr:
                name_part, default_part = expr.split("|", 1)
                var_name = name_part.strip()
                default_val = default_part.strip()
                if not var_name:
                    raise ValueError(
                        "Empty variable name in template expression: {0!r}".format(match.group(0))
                    )
                if var_name in kwargs:
                    return str(kwargs[var_name])
                return default_val
            else:
                var_name = expr
                if not var_name:
                    raise ValueError(
                        "Empty variable name in template expression: {0!r}".format(match.group(0))
                    )
                if var_name not in kwargs:
                    raise KeyError(
                        "Required template variable '{0}' was not provided".format(var_name)
                    )
                return str(kwargs[var_name])

        return _VAR_RE.sub(_replace, self.template_str)

    @classmethod
    def from_string(
        cls,
        template_str: str,
        name: Optional[str] = None,
    ) -> "PromptTemplate":
        """Create a template from a string (alias for the constructor)."""
        return cls(template_str, name=name)

    @classmethod
    def from_file(cls, path: str, name: Optional[str] = None) -> "PromptTemplate":
        """Load a template from a JSON or YAML file.

        The file may contain either a bare template string, or an object with
        ``template`` / ``name`` / ``description`` keys.  ``.yaml``/``.yml``
        files require the optional ``pyyaml`` extra.
        """
        data = _load_template_file(path)
        if isinstance(data, str):
            return cls(data, name=name or _stem_name(path))
        if isinstance(data, dict):
            return cls(
                data.get("template", data.get("template_str")),
                name=name or data.get("name") or _stem_name(path),
                description=data.get("description", ""),
            )
        raise ValueError("Template file {0!r} must contain a string or object".format(path))

    def __repr__(self) -> str:
        label = self.name or "<unnamed>"
        return "PromptTemplate({0!r})".format(label)


class TemplateRegistry:
    """In-memory registry mapping names to :class:`PromptTemplate` instances.

    Supports bulk-loading templates from a directory of JSON or YAML files via
    :meth:`load_directory`.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}

    def register(
        self,
        template_or_name: Any,
        template_str: Optional[str] = None,
        description: str = "",
    ) -> PromptTemplate:
        """Register a template.

        Accepts either a :class:`PromptTemplate` instance or a
        ``(name, template_str)`` pair.
        """
        if isinstance(template_or_name, PromptTemplate):
            tmpl = template_or_name
            if tmpl.name is None:
                raise ValueError("Template must have a name when registered as an object")
            self._templates[tmpl.name] = tmpl
        else:
            name = template_or_name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("template name must be a non-empty string")
            if template_str is None:
                raise ValueError("template_str is required when registering by name")
            tmpl = PromptTemplate(template_str, name=name, description=description)
            self._templates[name] = tmpl
        return tmpl

    def get(self, name: str) -> PromptTemplate:
        """Return the template registered under *name* (``KeyError`` if absent)."""
        if name not in self._templates:
            raise KeyError("Template not found: %s" % name)
        return self._templates[name]

    def list_templates(self) -> Dict[str, PromptTemplate]:
        """Return a shallow copy of the registered templates."""
        return dict(self._templates)

    def render(self, name: str, **kwargs: Any) -> str:
        """Render the named template with *kwargs*."""
        return self.get(name).render(**kwargs)

    def register_from_file(self, path: str, name: Optional[str] = None) -> PromptTemplate:
        """Load and register a template from a JSON/YAML file."""
        tmpl = PromptTemplate.from_file(path, name=name)
        if tmpl.name is None:
            raise ValueError("Template loaded from {0} has no name".format(path))
        self._templates[tmpl.name] = tmpl
        return tmpl

    def load_directory(self, directory: str, recursive: bool = False) -> int:
        """Load all ``.json``/``.yaml``/``.yml`` template files from a directory.

        Returns the number of templates registered.
        """
        import os

        count = 0
        if not os.path.isdir(directory):
            raise FileNotFoundError("Not a directory: {0}".format(directory))
        entries: List[str] = []
        if recursive:
            for root, _dirs, files in os.walk(directory):
                for fname in sorted(files):
                    entries.append(os.path.join(root, fname))
        else:
            for fname in sorted(os.listdir(directory)):
                full = os.path.join(directory, fname)
                if os.path.isfile(full):
                    entries.append(full)

        for entry in entries:
            if not entry.endswith((".json", ".yaml", ".yml")):
                continue
            self.register_from_file(entry)
            count += 1
        return count

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY = TemplateRegistry()


def register_template(
    name: str,
    template_str: str,
    description: str = "",
) -> PromptTemplate:
    """Register a template in the default module-level registry."""
    return DEFAULT_REGISTRY.register(name, template_str, description=description)


def get_template(name: str) -> PromptTemplate:
    """Retrieve a template from the default module-level registry."""
    return DEFAULT_REGISTRY.get(name)


def render_template(name: str, **kwargs: Any) -> str:
    """Render a template from the default module-level registry."""
    return DEFAULT_REGISTRY.render(name, **kwargs)


# ---------------------------------------------------------------------------
# File loading helpers (optional-dep safe)
# ---------------------------------------------------------------------------


def _stem_name(path: str) -> str:
    """Derive a bare template name from a file path's stem."""
    import os

    return os.path.splitext(os.path.basename(path))[0]


def _load_template_file(path: str) -> Any:
    """Load template content from a JSON or YAML file.

    ``.json`` files always work (stdlib ``json``).  ``.yaml`` / ``.yml``
    files require the optional ``pyyaml`` package.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    if path.endswith(".json"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON in {0}: {1}".format(path, exc)) from exc

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "YAML support requires the optional 'pyyaml' package "
                "(pip install 'local-llm-kit[yaml]' or pip install pyyaml)."
            ) from exc
        data = yaml.safe_load(text)
        if data is None:
            raise ValueError("Empty YAML file: {0}".format(path))
        return data

    return text
