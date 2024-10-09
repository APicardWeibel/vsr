from typing import Any, Callable

X1 = Any
X2 = Any
Y1 = Any
Y2 = Any


def interpretation(fun: Callable) -> Callable[[Callable[..., Y1]], Callable[..., Y1]]:
    """
    Decorator for composition on the right
    Take as argument fun, outputs a decorator transformin function g(x, ...) into g(fun(x), ...)
    Further arguments passed when constructing the decorator are passed to fun.
    """

    def deco(gfun: Callable[..., Y1], *fargs, **fkwargs) -> Callable[..., Y1]:
        def wrapper(x, *args, **kwargs):
            return gfun(fun(x, *fargs, **fkwargs), *args, **kwargs)

        return wrapper

    return deco


def post_modif(
    fun: Callable[[Y1], Y2]
) -> Callable[[Callable[..., Y1]], Callable[..., Y2]]:
    """
    Decorator for composition on the left.
    Take as argument fun, outputs a decorator transformin function g into fun(g( ...))
    Further arguments passed when constructing the decorator are passed to fun.
    """

    def deco(gfun: Callable[..., Y1], *fargs, **fkwargs) -> Callable[..., Y2]:
        def wrapper(*args, **kwargs):
            return fun(gfun(*args, **kwargs), *fargs, **fkwargs)

        return wrapper

    return deco
