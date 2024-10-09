import numpy as np


class ShapeError(ValueError):
    """Exception class when array shape is not as expected"""


def check_shape(xs: np.ndarray, shape_exp: tuple[int, ...]) -> None:
    """Check if xs has a given shape, raise ShapeError if not"""
    shape_obt = xs.shape
    if shape_obt != shape_exp:
        raise ShapeError(f"Shape mismatch: expected {shape_exp}, got {shape_obt}")


def prod(x: tuple[int, ...]) -> int:
    """Minor correction to np.prod function in the case where the shape is ().
    prod is used to ensure that the product of a tuple of int outputs an int."""
    return int(np.prod(x))


def _get_pre_shape(xs: np.ndarray, exp_shape: tuple[int, ...]) -> tuple[int, ...]:
    if exp_shape == ():
        return xs.shape
    n_dim = len(exp_shape)
    tot_shape = xs.shape

    if len(tot_shape) < n_dim:
        raise ShapeError("Shape of input array is not compliant with expected shape")

    if tot_shape[-n_dim:] != exp_shape:
        raise ShapeError("Shape of input array is not compliant with expected shape")

    return tot_shape[:-n_dim]
