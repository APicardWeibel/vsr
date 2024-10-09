def blab(silent: bool, *args, **kwargs) -> None:
    """
    Wrap up for print. If silents, does not print, else, prints.
    """
    if not silent:
        print(*args, **kwargs)


def tprint(msg, n: int = 20) -> None:
    """Title print"""
    print()
    print("-" * n, msg, "-" * n)
    print()
