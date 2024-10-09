from typing import Callable

import numpy as np

from vsr.misc.parallel import par_eval
from vsr.misc.shape import prod


def vectorize(
    fun: Callable,
    input_shape: tuple[int, ...],
    convert_input: bool = True,
    parallel: bool = True,
) -> Callable:
    """For a function fun which takes as input np.ndarray of shape input_shape and outputs
    arrays of shape output_shape, outputs the vectorized function which takes as input np.ndarray
    of shape (pre_shape, input_shape) and outputs np.ndarrat of shape (pre_shape, output_shape)
    """
    d = len(input_shape)

    def new_fun(xs) -> np.ndarray:
        if convert_input:
            xs = np.array(xs)
        pre_shape = xs.shape[:-d]
        xs.reshape(
            (
                (
                    prod(
                        pre_shape,
                    ),
                )
                + input_shape
            )
        )
        out = np.array(par_eval(fun, xs, parallel=parallel))
        out.reshape(pre_shape + out.shape[1:])
        return out

    return new_fun
