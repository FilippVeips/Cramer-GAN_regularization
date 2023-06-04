def flatten_dict_tree(tree):
    """
    Takes a tree in form of nested dicts and converts each leaf to
    a path-value pair as follows: `(key1, key2, ...), leaf_value`
    """
    result = []
    for k, v in tree.items():
        if not isinstance(v, dict):
            result.append(((k,), v))
        else:
            for path, value in flatten_dict_tree(v):
                result.append(((k,) + path, value))
    return result


def restore_dict_tree(flat_tree):
    """
    Inverse to `flatten_dict_tree`.
    """
    result = dict()
    for path, value in flat_tree:
        node = result
        for key in path[:-1]:
            if key not in node:
                node[key] = dict()
            node = node[key]
        node[path[-1]] = value
    return result


####################################################################################
# Taken from: https://stackoverflow.com/a/7205107/3801744


def merge_dicts(dest, src, path=None, overwrite=False):
    "merges src into dest"
    if path is None:
        path = []
    for key in src:
        if key in dest:
            if isinstance(dest[key], dict) and isinstance(dest[key], dict):
                merge_dicts(
                    dest[key], src[key], path=path + [str(key)], overwrite=overwrite
                )
            elif dest[key] == src[key]:
                pass  # same leaf value
            elif overwrite:
                dest[key] = src[key]
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            dest[key] = src[key]
    return dest


####################################################################################
