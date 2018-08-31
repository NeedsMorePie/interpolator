import copy
import json


def preprocess_var_refs(root, base_vars=None, remove_vars_dict=True):
    """
    Processes the dict by replacing var refs with variables declared in the 'vars' dict.
    For example, if given a dict (in json format):
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
    This function will process it into:
        {
            "obj": {
                "moop": "not bar",
                "boop": 2
            },
            "boop": "bar"
        }
    Notice that this works for nested objects. Vars in child objects will override vars in parent objects.
    :param root: Dict.
    :param base_vars: Dict. Vars to add to the dict.
    :param remove_vars_dict: Bool. Whether to remove the vars dict after processing.
    :return: Nothing
    """
    var_ref_str = 'var_ref'
    vars_str = 'vars'

    variables = {} if base_vars is None else base_vars

    # If the value is a dict, then it can either be replaced by a var or be recursively processed.
    def handle_dict(parent, indexer):
        assert isinstance(parent[indexer], dict)
        if var_ref_str in parent[indexer]:
            assert parent[indexer][var_ref_str] in variables
            parent[indexer] = variables[parent[indexer][var_ref_str]]
        else:
            preprocess_var_refs(parent[indexer], copy.deepcopy(variables), remove_vars_dict=remove_vars_dict)

    # If there is a 'vars' object, use it to override the base_vars.
    if vars_str in root:
        # If there is a 'vars' object, use it to override the base_vars.
        assert isinstance(root[vars_str], dict)
        for key, value in root[vars_str].items():
            assert not isinstance(value, dict) and not isinstance(value, list)
            variables[key] = value
        if remove_vars_dict:
            del root[vars_str]

    for root_key, root_value in root.items():
        if root_key == vars_str:
            continue
        elif isinstance(root_value, dict):
            handle_dict(root, root_key)
        elif isinstance(root_value, list):
            for i, item in enumerate(root_value):
                if isinstance(item, dict):
                    handle_dict(root_value, i)


def compile_args(args_dict):
    """
    Takes a dict of args and compiles it into a list of strings representing commandline arguments.
    { 'arg1': 1.0, 'arg2': 2.0 } would translate to ['--arg1=1.0', '--arg2=2.0'].
    :param args_dict: Dict.
    :return: List of str. Keys are in sorted order.
    """
    args = []
    for key, value in sorted(args_dict.items()):
        assert isinstance(key, str)
        assert not isinstance(value, dict) and not isinstance(value, list)
        str_value = str(value)
        if isinstance(value, str):
            str_value = '"' + str_value + '"'
        arg = '--' + key + '=' + str_value
        args.append(arg)
    return args


def read_raw_json(file_name):
    """
    :param file_name: Str.
    :return: Dict.
    """
    with open(file_name) as json_data:
        return json.load(json_data)


def import_json(file_name):
    """
    Reads and imports the specified JSON and it's parent dependencies.
    :param file_name: Str.
    :return: Dict.
    """
    parent_key = 'parent_json'

    json_dict = read_raw_json(file_name)
    if parent_key not in json_dict:
        return json_dict

    # Load the parent.
    parent_file = json_dict[parent_key]
    assert isinstance(parent_file, str)
    parent_json_dict = import_json(parent_file)

    merge_dicts(parent_json_dict, json_dict)
    return json_dict


def merge_dicts(source, destination):
    """
    Deep merges 2 dicts. Copies values from the source to the destination if they aren't already in the destination.
    i.e. Values in the destination take precedence.
    :param source: Dict.
    :param destination: Dict.
    :return: Nothing.
    """
    for key, value in source.items():
        if key not in destination:
            # Case where the source contains something that the destination does not.
            destination[key] = value
        elif isinstance(destination[key], dict) and isinstance(value, dict):
            # Case where source value and destination value are both dicts. Deep merge by recursing.
            merge_dicts(value, destination[key])
        # In all other cases, do nothing. The destination should override.
