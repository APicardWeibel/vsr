"""Custom numba typing nicknames shared throughout the package"""

import numba as nb

i = nb.int64
f = nb.float64
b = nb.boolean

i1D = nb.int64[:]
i1D_C = nb.types.Array(dtype=nb.int64, ndim=1, layout="C")
i1D_ru = nb.types.Array(dtype=nb.int64, ndim=1, layout="A", readonly=True)
i1D_C_ru = nb.types.Array(dtype=nb.int64, ndim=1, layout="C", readonly=True)

i2D = nb.int64[:, :]
i2D_C = nb.types.Array(dtype=nb.int64, ndim=2, layout="C")
i2D_C_ru = nb.types.Array(dtype=nb.int64, ndim=2, layout="C", readonly=True)

f1D = nb.float64[:]
f1D_C = nb.types.Array(dtype=nb.float64, ndim=1, layout="C")
f1Dru = nb.types.Array(dtype=nb.float64, ndim=1, layout="A", readonly=True)
f1D_C_ru = nb.types.Array(dtype=nb.float64, ndim=1, layout="C", readonly=True)

f2D = nb.float64[:, :]
f2D_C = nb.types.Array(dtype=nb.float64, ndim=2, layout="C")
f2Dru = nb.types.Array(dtype=nb.float64, ndim=2, layout="A", readonly=True)
f2D_C_ru = nb.types.Array(dtype=nb.float64, ndim=2, layout="C", readonly=True)

f3D = nb.float64[:, :, :]

b1D = nb.types.Array(dtype=nb.boolean, ndim=1, layout="A")
b1D_C = nb.types.Array(dtype=nb.boolean, ndim=1, layout="C")
b1D_C_ru = nb.types.Array(dtype=nb.boolean, ndim=1, layout="C", readonly=True)
Tuple = nb.types.Tuple
UTuple = nb.types.UniTuple
string = nb.types.unicode_type

void = nb.void
