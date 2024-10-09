import numpy as np


class LazyAccu:
    """Simplify accu management when iteratively storing arrays of identical shape

    Uses the syntax of python lists (append/extend methods), but using numpy ndarray
    to store the data (much more efficient).

    The shape of the arrays to be stored does not have to be known when constructing,
    but is inferred from first elements passed.

    No limit is given to number of elements to be accumulated. When allocated memory
    is full, all data is copied to an array twice as large. Since this takes time
    and should be avoided when possible, appropriate size should be selected at construction
    """

    def __init__(self, n: int = 100):
        assert n > 0
        self.n = n

        self._pos = 0
        self._data = None
        self._shape = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise Exception("No data yet")
        else:
            return self._data[: self._pos]

    def append(self, x):
        """Append element x to list"""
        x_arr = np.asarray(x)
        if self._data is None:
            self._shape = x_arr.shape
            self._data = np.zeros((self.n,) + self._shape)
        else:
            assert x_arr.shape == self._shape

        if self._pos == self.n:
            # Doubling time
            new_n = 2 * self.n
            new_data = np.zeros((new_n,) + self._shape)
            new_data[: self.n] = self._data
            self._data = new_data
            self.n = new_n

        self._data[self._pos] = x_arr
        self._pos += 1

    def extend(self, xs):
        """Add multiple elements to the accu"""
        xs_arr = np.asarray(xs)
        if self._data is None:
            self._shape = xs_arr.shape[1:]
            self._data = np.zeros((self.n,) + self._shape)
        else:
            assert self._shape == xs_arr.shape[1:]

        n_add = len(xs_arr)

        if self._pos + n_add > self.n:
            new_n = self.n
            while self._pos + n_add > new_n:
                new_n *= 2
            new_data = np.zeros((new_n,) + self._shape)
            new_data[: self.n] = self._data
            self._data, self.n = new_data, new_n

        self._data[self._pos : (self._pos + n_add)] = xs_arr
        self._pos += n_add
