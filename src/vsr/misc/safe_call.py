import warnings
from typing import Any, Callable, Union

Input = Any
Output = Any


class SafeCallWarning(Warning):
    """Warning for safe_call context. Enables sending a warning of failure inside a safe_call
    decorated function even if UserWarning is filtered as an exception."""


def safe_call(fun: Callable[[Input], Output]) -> Callable[[Input], Union[None, Output]]:
    """
    Decorator to evaluate a function safely.
    If function call fails, throws a warning and returns None.

    Note:
    Decorated function can still fail IF SafeCallWarning is filtered as an error (which completely
    defeats SafeCallWarning purpose) inside fun.
    """

    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as exc:  # pylint: disable=W0703
            warnings.warn(
                f"Evaluation failed with inputs {args}, {kwargs}: {exc}",
                category=SafeCallWarning,
            )
            return None

    return wrapper
