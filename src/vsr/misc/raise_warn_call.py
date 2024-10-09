import inspect
import warnings


def raise_warn_call(*warn_cats):
    """Decorator to transform warnings into exception during a function call

    Usage:
        # This will transform all warnings into exception
        @raise_warn_call
        def fun(*args, **kwargs):
            pass

        # This will transform the specified warnings into exception
        @raise_warn_call(SyntaxWarning, RuntimeWarning)
        def fun(*args, **kwargs):
            pass
    """
    # If only a single element which is not a warning is passed, filter all warnings
    is_fun = (len(warn_cats) == 1) & (
        (not inspect.isclass(warn_cats[0])) or (not issubclass(warn_cats[0], Warning))
    )

    # A function is passed
    if is_fun:
        fun = warn_cats[0]

        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                print(*args, **kwargs)
                out = fun(*args, **kwargs)
            return out

        return wrapper

    # Warnings are passed
    def deco(fun):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for cat in warn_cats:
                    warnings.filterwarnings("error", category=cat)
                return fun(*args, **kwargs)

        return wrapper

    return deco
