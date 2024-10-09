"""
Miscellanous functions used throughout the package

Functions:
    blab: wrap up for silent
    timedrun: TimeOut option for function call decorator
    interpretation: right composition decorator
    post_modif: left composition decorator
    safe_call: Evaluation of any function without failure (returns None if Exception occured)
    par_eval: Parallelisation switch for function evaluation
    num_der: numerical differentiation
    vectorize: function vectorization

Function implementations may change but input/output structure sholud remain stable.
"""
from vsr.misc.composition import interpretation, post_modif
from vsr.misc.dataframes import get_last_valid_index
from vsr.misc.errors import ImplementationError
from vsr.misc.lazy_accu import LazyAccu
from vsr.misc.matrix_inversion import safe_inverse_ps_matrix
from vsr.misc.num_der import num_der
from vsr.misc.parallel import par_eval
from vsr.misc.prints import blab, tprint
from vsr.misc.raise_warn_call import raise_warn_call
from vsr.misc.random_name import random_name
from vsr.misc.safe_call import SafeCallWarning, safe_call
from vsr.misc.sets import are_set_equal, check_set_equal, set_diff_msg
from vsr.misc.shape import ShapeError, _get_pre_shape, check_shape, prod
from vsr.misc.timed_run import timedrun, timeout
from vsr.misc.vectorize import vectorize
