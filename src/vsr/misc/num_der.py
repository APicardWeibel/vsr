from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from vsr.misc.composition import interpretation
from vsr.misc.parallel import par_eval
from vsr.misc.shape import prod


def num_der(
    fun: Callable[[ArrayLike], ArrayLike],
    x0: ArrayLike,
    f0: Optional[ArrayLike] = None,  # pylint: disable=W0613
    rel_step: Optional[float] = None,
    parallel: bool = True,
) -> np.ndarray:
    """Return the Jacobian of a function
    If f : shape_x -> shape_y,
    the output is of shape (shape_x, shape_y)

    Arguments:
        fun: the function to derivate
        x0: the point at which to derivate the function
        f0: the value of fun at x0 (is not used since 2 point approximation of the derivative is used)
        parallel: should the evaluations of fun be parallelized
    Output:
        The approximated jacobian of fun at x0 as a np.ndarray of shape (shape_x, shape_y)
    """

    # Work with flat array to facilitate
    x0 = np.array(x0)
    shape_in = x0.shape
    x0 = x0.flatten()

    dim = prod(shape_in)
    loc_fun = interpretation(lambda x: x.reshape(shape_in))(fun)

    if rel_step is None:
        rel_step = float((np.finfo(x0.dtype).eps) ** (1 / 3))

    to_evaluate = np.full((2 * dim, dim), x0)

    delta_x = np.maximum(1.0, x0) * rel_step
    add_matrix = np.diag(delta_x)
    to_evaluate[::2] = to_evaluate[::2] + add_matrix

    to_evaluate[1::2] = to_evaluate[1::2] - add_matrix

    evals = np.array(par_eval(loc_fun, to_evaluate, parallel=parallel))

    der = evals[::2] - evals[1::2]

    for i, d_x in enumerate(delta_x):
        der[i] = der[i] / (2 * d_x)
    shape_out = der[0].shape
    return der.reshape(shape_in + shape_out)
