import json
import unittest
from utils.misc import *


class TestPreprocessVarRefs(unittest.TestCase):
    def test_no_change(self):
        json_str = """
        {
            "one": 1,
            "two": 2,
            "three": 3,
            "array": [ 1, 2, 3 ]
        }
        """
        self.json_equals(json_str, json_str)

    def test_basic_change(self):
        json_str = """
        {
            "vars": {
                "two": 2
            },
            "one": 1,
            "two": { "var_ref": "two" },
            "three": 3
        }
        """
        expected_json_str = """
        {
            "one": 1,
            "two": 2,
            "three": 3
        }
        """
        self.json_equals(json_str, expected_json_str)

    def test_recursive_change(self):
        json_str = """
        {
            "vars": {
                "foo": "bar",
                "two": 2
            },
            "obj": {
                "moop": { "var_ref": "foo" },
                "boop": { "var_ref": "two" },
                "obj2": {
                    "key": { "var_ref": "two" }
                }
            },
            "boop": { "var_ref": "foo" }
        }
        """
        expected_json_str = """
        {
            "obj": {
                "moop": "bar",
                "boop": 2,
                "obj2": {
                    "key": 2
                }
            },
            "boop": "bar"
        }
        """
        self.json_equals(json_str, expected_json_str)

    def test_recursive_override(self):
        json_str = """
        {
            "vars": {
                "foo": "bar",
                "two": 2
            },
            "obj": {
                "vars": {
                    "foo": "not bar"
                },
                "moop": { "var_ref": "foo" },
                "boop": { "var_ref": "two" }
            },
            "boop": { "var_ref": "foo" }
        }
        """
        expected_json_str = """
        {
            "obj": {
                "moop": "not bar",
                "boop": 2
            },
            "boop": "bar"
        }
        """
        self.json_equals(json_str, expected_json_str)

    def test_list_changes(self):
        json_str = """
        {
            "vars": {
                "foo": "bar",
                "two": 2
            },
            "obj": {
                "vars": {
                    "foo": "not bar",
                    "other_var": 3.14
                },
                "moop": [
                    {
                        "recursive": { "var_ref": "foo" }
                    },
                    { "var_ref": "other_var" }
                ]
            },
            "boop": [
                { "var_ref": "foo" },
                2
            ]
        }
        """
        expected_json_str = """
        {
            "obj": {
                "moop": [
                    {
                        "recursive": "not bar"
                    },
                    3.14
                ]
            },
            "boop": [
                "bar",
                2
            ]
        }
        """
        self.json_equals(json_str, expected_json_str)

    def json_equals(self, json_str, expected_json_str):
        content = json.loads(json_str)
        expected_content = json.loads(expected_json_str)
        preprocess_var_refs(content)
        self.assertDictEqual(expected_content, content)


class TestCompileArgs(unittest.TestCase):
    def test_number_arg(self):
        args = compile_args({'foo': 3e-4, 'bar': 200000})
        self.assertListEqual(['--bar=200000', '--foo=0.0003'], args)

    def test_bool_arg(self):
        args = compile_args({'foo': True})
        self.assertListEqual(['--foo=True'], args)

    def test_str_arg(self):
        args = compile_args({'foo': 'I am string.'})
        self.assertListEqual(['--foo="I am string."'], args)

    def test_multi_type_arg(self):
        args = compile_args({'foo': 'I am string.', 'bar': 3.14, 'bool': False})
        self.assertListEqual(['--bar=3.14', '--bool=False', '--foo="I am string."'], args)