from functools import partial
from typing import Any, Callable, Iterable

from multiprocess import Pool  # pylint:disable=E0611

Input = Any
Output = Any


def par_eval(
    fun: Callable[..., Output], xs: Iterable[Input], parallel: bool, *args, **kwargs
) -> list[Output]:
    """
    Evaluation of a function on a list of values. If parallel is True,
    computations are parallelized using multiprocess.Pool . Else list
    comprehension is used.

    Further arguments and keyword arguments are passed to fun.
    """
    if parallel:
        loc_fun = partial(fun, *args, **kwargs)
        with Pool() as pool:  # pylint: disable=E1102
            out = pool.map(loc_fun, xs)
    else:
        out = [fun(x, *args, **kwargs) for x in xs]
    return out
