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
