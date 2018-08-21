import copy


# https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w
def sort_in_unison(key_list, lists):
    """
    :param key_list: The list whose elements we use to sort.
    :param lists: A list of lists, each of which will be sorted in unison with key_list.
    :return: (sorted_key_list, sorted_lists)
    """
    indices = [*range(len(key_list))]
    indices.sort(key=key_list.__getitem__)
    sorted_lists = []
    sorted_key_list = [key_list[indices[i]] for i in range(len(key_list))]
    for list in lists:
        sorted_list = [list[indices[i]] for i in range(len(list))]
        sorted_lists.append(sorted_list)
    return sorted_key_list, sorted_lists


# Initially copied from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress_bar(iteration, total, prefix= '', suffix='', decimals=1, length=100, fill='█', use_percentage=True):
    """
    Call in a loop to create terminal progress bar.
    :param iteration: Current iteration (Int)
    :param total: Total iterations (Int)
    :param prefix: Prefix string (Str)
    :param suffix: Suffix string (Str)
    :param decimals: Positive number of decimals in percent complete (Int)
    :param length: Character length of bar (Int)
    :param fill: Bar fill character (Str)
    :param use_percentage: Whether to print percentage, or 'iter of total'.
    """
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    if use_percentage:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    else:
        ratio = '%d of %d' % (iteration, total)
        print('\r%s |%s| %s %s' % (prefix, bar, ratio, suffix), end='\r')

    # Print New Line on Complete
    if iteration == total:
        print()


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
