"""Offline tests for the prompt-template module.

Stdlib unittest only.  Python 3.9 compatible.

Run: python -m pytest tests/test_prompt_templates.py -q
"""

import json
import os
import tempfile
import unittest

from local_llm_kit.prompt_templates import (
    DEFAULT_REGISTRY,
    PromptTemplate,
    TemplateRegistry,
    get_template,
    register_template,
    render_template,
)


class TestPromptTemplate(unittest.TestCase):
    def test_required_variable(self):
        t = PromptTemplate("Hello, {{name}}!")
        self.assertEqual(t.render(name="World"), "Hello, World!")

    def test_required_variable_missing_raises(self):
        t = PromptTemplate("Hello, {{name}}!")
        with self.assertRaises(KeyError):
            t.render()

    def test_default_value(self):
        t = PromptTemplate("Hello, {{name|stranger}}!")
        self.assertEqual(t.render(name="Alice"), "Hello, Alice!")
        self.assertEqual(t.render(), "Hello, stranger!")

    def test_single_braces_preserved(self):
        t = PromptTemplate('{"key": "{{value}}"}')
        self.assertEqual(t.render(value="x"), '{"key": "x"}')

    def test_no_placeholders(self):
        t = PromptTemplate("static text")
        self.assertEqual(t.render(), "static text")

    def test_multiple_placeholders(self):
        t = PromptTemplate("{{greeting}}, {{name}}!")
        self.assertEqual(t.render(greeting="Hi", name="Bob"), "Hi, Bob!")

    def test_render_non_string_values(self):
        t = PromptTemplate("{{count}} items at {{price}}")
        self.assertEqual(t.render(count=42, price=3.14), "42 items at 3.14")

    def test_from_string(self):
        t = PromptTemplate.from_string("Hi {{name}}")
        self.assertEqual(t.render(name="X"), "Hi X")

    def test_from_file_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tmpl.json")
            with open(path, "w") as fh:
                json.dump(
                    {"template": "Hello {{name}}", "name": "greeting", "description": "d"}, fh
                )
            t = PromptTemplate.from_file(path)
            self.assertEqual(t.name, "greeting")
            self.assertEqual(t.description, "d")
            self.assertEqual(t.render(name="World"), "Hello World")

    def test_from_file_bare_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tmpl.txt")
            with open(path, "w") as fh:
                fh.write("Bare {{x}}")
            t = PromptTemplate.from_file(path, name="bare")
            self.assertEqual(t.name, "bare")
            self.assertEqual(t.render(x="val"), "Bare val")

    def test_repr(self):
        t = PromptTemplate("hi {{name}}", name="greet")
        self.assertIn("greet", repr(t))


class TestTemplateRegistry(unittest.TestCase):
    def test_register_and_get(self):
        reg = TemplateRegistry()
        t = reg.register("greet", "Hello {{name}}")
        self.assertIs(reg.get("greet"), t)
        with self.assertRaises(KeyError):
            reg.get("nope")

    def test_register_by_object_requires_name(self):
        reg = TemplateRegistry()
        with self.assertRaises(ValueError):
            reg.register(PromptTemplate("hi"))

    def test_register_by_object_with_name(self):
        reg = TemplateRegistry()
        t = reg.register(PromptTemplate("hi {{n}}", name="g"))
        self.assertIsNotNone(t.name)

    def test_register_by_name_requires_template_str(self):
        reg = TemplateRegistry()
        with self.assertRaises(ValueError):
            reg.register("x", None)

    def test_render(self):
        reg = TemplateRegistry()
        reg.register("greet", "Hello {{who}}")
        self.assertEqual(reg.render("greet", who="World"), "Hello World")

    def test_contains_and_len(self):
        reg = TemplateRegistry()
        reg.register("a", "A {{x}}")
        reg.register("b", "B {{x}}")
        self.assertIn("a", reg)
        self.assertNotIn("c", reg)
        self.assertEqual(len(reg), 2)

    def test_list_templates_is_copy(self):
        reg = TemplateRegistry()
        reg.register("a", "A {{x}}")
        templates = reg.list_templates()
        templates.clear()
        self.assertEqual(len(reg), 1)

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.json"), "w") as fh:
                json.dump({"template": "A{{x}}", "name": "a"}, fh)
            with open(os.path.join(tmp, "b.json"), "w") as fh:
                json.dump({"template": "B{{y}}", "name": "b"}, fh)
            with open(os.path.join(tmp, "c.txt"), "w") as fh:
                fh.write("should be ignored")
            reg = TemplateRegistry()
            count = reg.load_directory(tmp)
            self.assertEqual(count, 2)
            self.assertIn("a", reg)
            self.assertIn("b", reg)

    def test_register_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tmpl.json")
            with open(path, "w") as fh:
                json.dump({"template": "Hello {{who}}", "name": "greet"}, fh)
            reg = TemplateRegistry()
            reg.register_from_file(path)
            self.assertEqual(reg.render("greet", who="World"), "Hello World")


class TestModuleLevelHelpers(unittest.TestCase):
    def test_default_registry_exists(self):
        self.assertIsInstance(DEFAULT_REGISTRY, TemplateRegistry)

    def test_register_get_render(self):
        name = "test_helper_" + str(id(self))
        register_template(name, "Hi {{who}}")
        self.assertEqual(get_template(name).render(who="U"), "Hi U")
        self.assertEqual(render_template(name, who="U"), "Hi U")

    def test_get_template_missing_raises(self):
        with self.assertRaises(KeyError):
            get_template("definitely_not_registered")


class TestBackwardCompatReexports(unittest.TestCase):
    def test_conversation_log_module_reexports_templates(self):
        from local_llm_kit.conversation_log import (
            DEFAULT_REGISTRY as cl_registry,
        )
        from local_llm_kit.conversation_log import (
            PromptTemplate as CLPromptTemplate,
        )
        from local_llm_kit.conversation_log import (
            TemplateRegistry as CLTemplateRegistry,
        )
        from local_llm_kit.conversation_log import (
            get_template as cl_get,
        )
        from local_llm_kit.conversation_log import (
            register_template as cl_register,
        )
        from local_llm_kit.conversation_log import (
            render_template as cl_render,
        )

        self.assertIs(cl_registry, DEFAULT_REGISTRY)
        self.assertIs(CLPromptTemplate, PromptTemplate)
        self.assertIs(CLTemplateRegistry, TemplateRegistry)
        self.assertIs(cl_get, get_template)
        self.assertIs(cl_register, register_template)
        self.assertIs(cl_render, render_template)


if __name__ == "__main__":
    unittest.main()
