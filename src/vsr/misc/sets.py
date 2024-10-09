def set_diff_msg(iter_1, iter_2) -> str:
    err_msgs = []
    extra_1 = set(iter_1).difference(iter_2)
    if extra_1:
        err_msgs.append(f"Index 1 contains extra keys {extra_1}")
    extra_2 = set(iter_2).difference(iter_1)
    if extra_2:
        err_msgs.append(f"Index 2 contains extra keys {extra_2}")

    err_msg = "\nFurthermore:".join(err_msgs)

    return err_msg


def are_set_equal(iter_1, iter_2) -> bool:
    err_msg = set_diff_msg(iter_1, iter_2)
    if err_msg:
        return False
    return True


def check_set_equal(iter_1, iter_2):
    """Check if two set like object have identical keys. If not, raise a ValueError"""
    err_msg = set_diff_msg(iter_1, iter_2)

    if err_msg:
        raise ValueError(err_msg)
