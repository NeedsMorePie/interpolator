import unittest
from common.utils.config import *


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
        self.json_after_preprocessing_vars_equals(json_str, json_str)

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
        self.json_after_preprocessing_vars_equals(json_str, expected_json_str)

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
        self.json_after_preprocessing_vars_equals(json_str, expected_json_str)

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
        self.json_after_preprocessing_vars_equals(json_str, expected_json_str)

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
                        "recursive1": { "var_ref": "foo" },
                        "recursive2": { "var_ref": "other_var" }
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
                        "recursive1": "not bar",
                        "recursive2": 3.14
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
        self.json_after_preprocessing_vars_equals(json_str, expected_json_str)

    def json_after_preprocessing_vars_equals(self, json_str, expected_json_str):
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


class TestMergeDicts(unittest.TestCase):
    def test_value(self):
        source = {
            'foo': 'bar',
            'a': 'b'
        }
        destination = {
            'a': 'c'
        }
        merge_dicts(source, destination)
        self.assertDictEqual({
            'foo': 'bar',
            'a': 'c'
        }, destination)

    def test_array(self):
        source = {
            'foo': 'bar',
            'a': [1, 2, 3]
        }
        destination = {
            'a': [4, 5, 6]
        }
        merge_dicts(source, destination)
        self.assertDictEqual({
            'foo': 'bar',
            'a': [4, 5, 6]
        }, destination)

    def test_dict(self):
        source = {
            'foo': 'bar',
            'a': {
                'b': 'b'
            }
        }
        destination = {
            'foo': 'not_bar',
        }
        merge_dicts(source, destination)
        self.assertDictEqual({
            'foo': 'not_bar',
            'a': {
                'b': 'b'
            }
        }, destination)

    def test_deep(self):
        source = {
            'foo': 'bar',
            'a': {
                'foo': 'not_bar',
                'bar': 'foo',
                'deeper': {
                    'two': 2,
                    'me': 0
                }
            },
            'b': {
                'one': 1
            }
        }
        destination = {
            'a': {
                'bar': 'foo_bar',
                'deeper': {
                    'three': 3,
                    'me': 1
                }
            }
        }
        merge_dicts(source, destination)
        self.assertDictEqual({
            'foo': 'bar',
            'a': {
                'foo': 'not_bar',
                'bar': 'foo_bar',
                'deeper': {
                    'two': 2,
                    'three': 3,
                    'me': 1
                }
            },
            'b': {
                'one': 1
            }
        }, destination)

    def test_empty(self):
        source = {}
        destination = {}
        merge_dicts(source, destination)
        self.assertDictEqual({}, destination)

    def test_src_empty(self):
        source = {}
        destination = {
            'foo': 'bar',
            'a': {
                'foo': 'not_bar',
                'bar': 'foo_bar'
            }
        }
        merge_dicts(source, destination)
        self.assertDictEqual({
            'foo': 'bar',
            'a': {
                'foo': 'not_bar',
                'bar': 'foo_bar'
            }
        }, destination)

    def test_dst_empty(self):
        source = {
            'foo': 'bar',
            'a': {
                'foo': 'not_bar',
                'bar': 'foo_bar'
            }
        }
        destination = {}
        merge_dicts(source, destination)
        self.assertDictEqual({
            'foo': 'bar',
            'a': {
                'foo': 'not_bar',
                'bar': 'foo_bar'
            }
        }, destination)


class TestImportJson(unittest.TestCase):
    def setUp(self):
        self.base = 'common/utils/test_data/config/base.json'
        self.base_sub = 'common/utils/test_data/config/base_sub.json'
        self.base_sub_sub = 'common/utils/test_data/config/base_sub_sub.json'

    def test_identity(self):
        base_dict = import_json(self.base)
        self.assertDictEqual({
            'a': 1,
            'b': 2,
            'c': 3
        }, base_dict)

    def test_sub(self):
        base_dict = import_json(self.base_sub)
        self.assertDictEqual({
            'parent_json': 'common/utils/test_data/config/base.json',
            'a': 1,
            'b': 'b',
            'c': 'c',
            'deep': {
                'foo': 'bar',
                'bar': 'foobar'
            }
        }, base_dict)

    def test_deeper_sub(self):
        base_sub_dict = import_json(self.base_sub_sub)
        self.assertDictEqual({
            'parent_json': 'common/utils/test_data/config/base_sub.json',
            'a': 1,
            'b': 'b',
            'c': 'not_c',
            'deep': {
                'foo': 'bar',
                'bar': 'not_foobar',
                'deeper': {
                    'hamid': False
                }
            },
            'more': {
                'some_array': [
                    1, 2, 3, 4
                ]
            },
            'd': 'd'
        }, base_sub_dict)


if __name__ == '__main__':
    unittest.main()
