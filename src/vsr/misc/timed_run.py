import signal
from contextlib import contextmanager


@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def timedrun(max_time: int):
    """Decorator to stop function calls if they last too long
    Args:
        max_time: the maximum time for a function call before raising an exception
    Returns:
        A decorator implementing the time out mechanism on the function
    """

    def deco(function):
        def wrap(*args, **kwargs):
            with timeout(max_time):
                return function(*args, **kwargs)

        return wrap

    return deco
