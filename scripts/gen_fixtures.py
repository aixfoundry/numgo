#!/usr/bin/env python3
"""Generate golden fixtures from local numpy for the Go side (Phase 1).

Every generated case follows ONE convention, mirrored by fixtures_test.go:
  - `inputs` are arrays; METHODS are called on inputs[0]; package-level
    functions (Zeros, Arange, Concatenate, Stack, Tile, ...) receive all
    inputs (create functions get none).
  - `args` is the positional argument list, decoded by context in Go:
      * number                         -> float64 (int where an int is expected)
      * [n, n, ...]                    -> []int   (axes / shape / Take indices)
      * [true, false, ...]             -> []bool  (Compress mask)
      * {"start":s, "stop":e, "step":k} -> Range  (Slice; keys default to 0).
        CAUTION: all-zero start/stop means WHOLE AXIS (numgo Range
        convention), never the empty slice a[0:0]; open-ended slices are
        emitted with explicit stop == dim.
  - `want` is always an array; NaN/Inf encode as the strings
    "NaN"/"Inf"/"-Inf" (JSON has no NaN literal; allow_nan=False keeps the
    dump portable).
  - Go compares exact shape plus |got-want| <= 1e-12 + 1e-12*|want|,
    NaN == NaN.

Value semantics only: *Into mutators, panic paths, multi-value returns
(Split, UniqueCounts, MeshGrid) and Partition/ArgPartition (numpy leaves the
non-kth slots unspecified, so whole-array comparison is invalid) are
deliberately NOT covered here — those belong to unit tests.

Usage: python3 scripts/gen_fixtures.py   (numpy >= 2.x required)
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "testdata", "fixtures")

WHOLE = {"start": 0, "stop": 0, "step": 1}  # numgo zero-Range = whole axis


def data(shape, lo=-3.0, hi=3.0):
    """Deterministic uniform test data; a fresh fixed-seed generator per call."""
    global _seed
    _seed += 1
    return np.random.default_rng(_seed).uniform(lo, hi, size=shape)


_seed = 1000


def clean(x):
    """Encode floats, mapping non-finite values to their string form."""
    if isinstance(x, list):
        return [clean(v) for v in x]
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return x


def enc(a):
    return {"shape": [int(d) for d in a.shape], "data": clean(a.ravel(order="C").tolist())}


def scalar(x):
    return np.array([float(x)])


def go_index_layout(idx2d, axis):
    """Reshape numpy's argsort/argmax result into numgo's flat index layout:
    one entry per element, kept slots in C order, channel-major."""
    moved = np.moveaxis(idx2d, axis, -1)
    return {"shape": [int(moved.size)], "data": [float(v) for v in moved.ravel(order="C")]}


domains = {}


def add(domain, name, func, args, inputs, want):
    domains.setdefault(domain, []).append({
        "name": name,
        "func": func,
        "args": args,
        "inputs": [enc(a) for a in inputs],
        "want": want if isinstance(want, dict) else enc(want),
    })


# ---------------------------------------------------------------- create ----
add("create", "zeros_2d", "Zeros", [[2, 3]], [], np.zeros((2, 3)))
add("create", "zeros_3d", "Zeros", [[2, 3, 4]], [], np.zeros((2, 3, 4)))
add("create", "ones_2d", "Ones", [[3, 2]], [], np.ones((3, 2)))
add("create", "ones_3d", "Ones", [[2, 3, 2]], [], np.ones((2, 3, 2)))
add("create", "full_value", "Full", [1.5, [2, 3]], [], np.full((2, 3), 1.5))
add("create", "full_3d", "Full", [-2.25, [2, 1, 3]], [], np.full((2, 1, 3), -2.25))
add("create", "arange_quarter", "Arange", [0.0, 2.0, 0.25], [], np.arange(0.0, 2.0, 0.25))
add("create", "arange_negative", "Arange", [3.0, 0.0, -0.5], [], np.arange(3.0, 0.0, -0.5))
add("create", "arange_fractional", "Arange", [0.0, 1.0, 0.3], [], np.arange(0.0, 1.0, 0.3))
add("create", "linspace_basic", "Linspace", [-1.0, 3.0, 9], [], np.linspace(-1.0, 3.0, 9))
add("create", "linspace_reverse", "Linspace", [2.5, -2.5, 7], [], np.linspace(2.5, -2.5, 7))
add("create", "linspace_single", "Linspace", [1.25, 9.75, 1], [], np.linspace(1.25, 9.75, 1))
add("create", "linspace_empty", "Linspace", [1.25, 9.75, 0], [], np.linspace(1.25, 9.75, 0))
add("create", "eye_4", "Eye", [4], [], np.eye(4))
add("create", "identity_3", "Identity", [3], [], np.identity(3))

# ----------------------------------------------------------------- views ----
a34 = data((3, 4))
add("views", "slice_rows", "Slice", [{"start": 0, "stop": 2, "step": 1}], [a34], a34[0:2])
add("views", "slice_rows_strided_cols", "Slice",
    [{"start": 1, "stop": 3, "step": 1}, {"start": 0, "stop": 4, "step": 2}],
    [a34], a34[1:3, ::2])
add("views", "slice_negative_start", "Slice", [{"start": -2, "stop": 3, "step": 1}], [a34], a34[-2:])
add("views", "slice_strided", "Slice", [{"start": 0, "stop": 3, "step": 2}], [a34], a34[::2])
add("views", "slice_reverse_step", "Slice", [{"start": 0, "stop": 0, "step": -1}], [a34], a34[::-1])
add("views", "slice_tail_cols", "Slice",
    [WHOLE, {"start": 0, "stop": 0, "step": -1}], [a34], a34[:, ::-1])
a234 = data((2, 3, 4))
add("views", "slice_3d_middle", "Slice",
    [WHOLE, {"start": 1, "stop": 3, "step": 1}, WHOLE], [a234], a234[:, 1:3, :])
add("views", "slice_3d_leading", "Slice", [{"start": 1, "stop": 2, "step": 1}], [a234], a234[1:])
add("views", "transpose_default_2d", "Transpose", [[]], [a34], a34.T)
add("views", "transpose_axes_3d", "Transpose", [[1, 0, 2]], [a234], np.transpose(a234, (1, 0, 2)))
add("views", "transpose_default_3d", "Transpose", [[]], [a234], np.transpose(a234))
add("views", "reshape_2d_to_2d", "Reshape", [[2, 6]], [a34], a34.reshape(2, 6))
add("views", "reshape_minus1_flat", "Reshape", [[-1]], [a34], a34.reshape(-1))
add("views", "reshape_3d_infer", "Reshape", [[2, -1]], [a234], a234.reshape(2, -1))
a314 = data((3, 1, 4))
add("views", "squeeze_all", "Squeeze", [[]], [a314], np.squeeze(a314))
add("views", "squeeze_axis", "Squeeze", [[1]], [a314], np.squeeze(a314, axis=1))
a231 = data((2, 3, 1))
add("views", "squeeze_axis_negative", "Squeeze", [[-1]], [a231], np.squeeze(a231, axis=-1))
a34b = data((3, 4))
add("views", "expanddims_0", "ExpandDims", [0], [a34b], np.expand_dims(a34b, 0))
add("views", "expanddims_1", "ExpandDims", [1], [a34b], np.expand_dims(a34b, 1))
add("views", "expanddims_negative", "ExpandDims", [-1], [a34b], np.expand_dims(a34b, -1))
add("views", "swapaxes_0_2", "SwapAxes", [0, 2], [a234], np.swapaxes(a234, 0, 2))
add("views", "swapaxes_negative", "SwapAxes", [-1, -2], [a234], np.swapaxes(a234, 2, 1))
add("views", "ravel_3d", "Ravel", [], [a234], a234.ravel())
add("views", "ravel_2d", "Ravel", [], [a34], a34.ravel())
v14 = data((1, 4))
add("views", "broadcastto_rows", "BroadcastTo", [[3, 4]], [v14], np.broadcast_to(v14, (3, 4)))
v31 = data((3, 1))
add("views", "broadcastto_cols", "BroadcastTo", [[3, 4]], [v31], np.broadcast_to(v31, (3, 4)))
v4 = data((4,))
add("views", "broadcastto_1d_to_3d", "BroadcastTo", [[2, 3, 4]], [v4], np.broadcast_to(v4, (2, 3, 4)))
v131 = data((1, 3, 1))
add("views", "broadcastto_mixed", "BroadcastTo", [[2, 3, 4]], [v131], np.broadcast_to(v131, (2, 3, 4)))

# ----------------------------------------------------------------- ufunc ----
b34 = data((3, 4))
add("ufunc", "add_same", "Add", [], [a34, b34], a34 + b34)
row4 = data((4,))
add("ufunc", "add_broadcast_row", "Add", [], [a34, row4], a34 + row4)
col41 = data((4, 1))
row13 = data((1, 3))
add("ufunc", "add_broadcast_outer", "Add", [], [col41, row13], col41 + row13)
a234b = data((2, 3, 4))
add("ufunc", "add_broadcast_3d", "Add", [], [a234, a234b], a234 + a234b)
add("ufunc", "sub_broadcast", "Sub", [], [a34, row4], a34 - row4)
add("ufunc", "mul_broadcast", "Mul", [], [col41, row13], col41 * row13)
add("ufunc", "div_same", "Div", [], [a34, b34], a34 / b34)
dz_a = np.array([[1.0, -1.0, 0.0], [2.5, -0.5, 0.0]])
dz_b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, -1.0]])
with np.errstate(all="ignore"):
    add("ufunc", "div_zero", "Div", [], [dz_a, dz_b], dz_a / dz_b)
add("ufunc", "addvalue", "AddValue", [2.5], [a34], a34 + 2.5)
add("ufunc", "subvalue", "SubValue", [-1.25], [a34], a34 - (-1.25))
add("ufunc", "mulvalue", "MulValue", [0.5], [a34], a34 * 0.5)
add("ufunc", "divvalue", "DivValue", [4.0], [a34], a34 / 4.0)
with np.errstate(all="ignore"):
    add("ufunc", "divvalue_zero", "DivValue", [0.0], [dz_a], dz_a / 0.0)

# ---------------------------------------------------------------- reduce ----
add("reduce", "sum_all", "Sum", [[]], [a34], scalar(a34.sum()))
add("reduce", "sum_axis0", "Sum", [[0]], [a34], a34.sum(axis=0))
add("reduce", "sum_axis_last", "Sum", [[-1]], [a34], a34.sum(axis=-1))
add("reduce", "sum_axes_0_2", "Sum", [[0, 2]], [a234], a234.sum(axis=(0, 2)))
add("reduce", "sum_3d_axis1", "Sum", [[1]], [a234], a234.sum(axis=1))
add("reduce", "sumkeepdims_axis0", "SumKeepdims", [[0]], [a34], a34.sum(axis=0, keepdims=True))
add("reduce", "sumkeepdims_0_2", "SumKeepdims", [[0, 2]], [a234], a234.sum(axis=(0, 2), keepdims=True))
add("reduce", "prod_all", "Prod", [[]], [a34], scalar(a34.prod()))
add("reduce", "prod_axis1", "Prod", [[1]], [a34], a34.prod(axis=1))
add("reduce", "prodkeepdims_all", "ProdKeepdims", [[]], [a34], a34.prod(keepdims=True))
add("reduce", "max_all", "Max", [[]], [a34], scalar(a34.max()))
add("reduce", "max_axis0", "Max", [[0]], [a34], a34.max(axis=0))
add("reduce", "maxkeepdims_last", "MaxKeepdims", [[-1]], [a34], a34.max(axis=-1, keepdims=True))
add("reduce", "min_all", "Min", [[]], [a34], scalar(a34.min()))
add("reduce", "min_axis_last", "Min", [[-1]], [a34], a34.min(axis=-1))
add("reduce", "mean_all", "Mean", [[]], [a34], scalar(a34.mean()))
add("reduce", "mean_axis0", "Mean", [[0]], [a34], a34.mean(axis=0))
add("reduce", "meankeepdims_axis0", "MeanKeepdims", [[0]], [a34], a34.mean(axis=0, keepdims=True))
add("reduce", "var_all", "Var", [[]], [a34], scalar(a34.var()))
add("reduce", "var_axis0", "Var", [[0]], [a34], a34.var(axis=0))
add("reduce", "varkeepdims_last", "VarKeepdims", [[-1]], [a34], a34.var(axis=-1, keepdims=True))
add("reduce", "varddof_axis0", "VarDDof", [1, [0]], [a34], a34.var(axis=0, ddof=1))
add("reduce", "std_all", "Std", [[]], [a34], scalar(a34.std()))
add("reduce", "stdddof_all", "StdDDof", [1, []], [a34], scalar(a34.std(ddof=1)).reshape(1))
add("reduce", "argmax_axis0", "ArgMax", [0], [a34], go_index_layout(a34.argmax(axis=0), 0))
add("reduce", "argmax_last", "ArgMax", [-1], [a34], go_index_layout(a34.argmax(axis=-1), -1))
add("reduce", "argmin_axis0", "ArgMin", [0], [a34], go_index_layout(a34.argmin(axis=0), 0))
def bval(b):
    return {"shape": [], "data": [bool(b)]}

add("reduce", "any_mixed", "Any", [], [np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -2.0]])], bval(True))
add("reduce", "any_zero", "Any", [], [np.zeros((2, 3))], bval(False))
add("reduce", "all_nonzero", "All", [], [data((2, 3))], bval(True))
add("reduce", "all_with_zero", "All", [], [np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 1.0]])], bval(False))
add("reduce", "cumsum_flat", "CumSum", [[]], [a34], np.cumsum(a34))
add("reduce", "cumsum_axis0", "CumSum", [[0]], [a34], np.cumsum(a34, axis=0))
add("reduce", "cumsum_axis_last", "CumSum", [[-1]], [a34], np.cumsum(a34, axis=-1))
cp_in = data((2, 3))
add("reduce", "cumprod_flat", "CumProd", [[]], [cp_in], np.cumprod(cp_in))
add("reduce", "cumprod_axis1", "CumProd", [[1]], [cp_in], np.cumprod(cp_in, axis=1))

# ----------------------------------------------------------------- manip ----
c23 = data((2, 3))
c13 = data((1, 3))
add("manip", "concatenate_axis0", "Concatenate", [0], [c23, c13], np.concatenate([c23, c13], axis=0))
c22 = data((2, 2))
add("manip", "concatenate_axis1", "Concatenate", [1], [c23, c22], np.concatenate([c23, c22], axis=1))
d1, d2, d3 = data((2, 3, 2)), data((2, 3, 2)), data((2, 3, 2))
add("manip", "concatenate_3d_last", "Concatenate", [-1], [d1, d2, d3],
    np.concatenate([d1, d2, d3], axis=-1))
s1, s2 = data((2, 3)), data((2, 3))
add("manip", "stack_axis0", "Stack", [0], [s1, s2], np.stack([s1, s2], axis=0))
add("manip", "stack_last", "Stack", [-1], [s1, s2], np.stack([s1, s2], axis=-1))
add("manip", "tile_2d", "Tile", [[2, 1]], [c23], np.tile(c23, (2, 1)))
v4t = data((4,))
add("manip", "tile_1d", "Tile", [[3]], [v4t], np.tile(v4t, 3))
add("manip", "tile_extra_axis", "Tile", [[2, 1, 2]], [c23], np.tile(c23, (2, 1, 2)))
add("manip", "repeat_axis0", "Repeat", [2, 0], [c23], np.repeat(c23, 2, axis=0))
add("manip", "repeat_last", "Repeat", [3, -1], [c23], np.repeat(c23, 3, axis=-1))
add("manip", "flip_default", "Flip", [[]], [c23], np.flip(c23))
add("manip", "flip_axis0", "Flip", [[0]], [c23], np.flip(c23, axis=0))
add("manip", "flip_both", "Flip", [[0, 1]], [c23], np.flip(c23, axis=(0, 1)))
v6 = data((6,))
add("manip", "roll_1d", "Roll", [2, []], [v6], np.roll(v6, 2))
add("manip", "roll_axis0_negative", "Roll", [-1, [0]], [c23], np.roll(c23, -1, axis=0))
add("manip", "roll_flat", "Roll", [3, []], [c23], np.roll(c23, 3))
add("manip", "roll_two_axes", "Roll", [1, [0, 1]], [c23], np.roll(c23, 1, axis=(0, 1)))

# ------------------------------------------------------------------ sort ----
add("sort", "sort_default", "Sort", [[]], [a34], np.sort(a34))
add("sort", "sort_axis0", "Sort", [[0]], [a34], np.sort(a34, axis=0))
add("sort", "sort_last_negative", "Sort", [[-1]], [a34], np.sort(a34, axis=-1))
add("sort", "sort_3d_axis1", "Sort", [[1]], [a234], np.sort(a234, axis=1))
add("sort", "argsort_axis0", "Argsort", [0], [a34],
    go_index_layout(a34.argsort(axis=0), 0))
add("sort", "argsort_last", "Argsort", [-1], [a34], a34.argsort(axis=-1).ravel())
dup = np.round(data((3, 4)), 1)
add("sort", "unique_dupes", "Unique", [], [dup], np.unique(dup))
dup3 = np.round(data((2, 3, 2)), 1)
add("sort", "unique_3d_flat", "Unique", [], [dup3], np.unique(dup3))
uniq_neg = np.array([[-1.5, 2.0, -1.5], [0.0, 2.0, -3.25]])
add("sort", "unique_negatives", "Unique", [], [uniq_neg], np.unique(uniq_neg))

# ------------------------------------------------------------------ math ----
_math_inputs = {
    "Abs": lambda: data((3, 4)),
    "Sqrt": lambda: data((3, 4), 0.1, 9.0),
    "Cbrt": lambda: data((3, 4), -8.0, 8.0),
    "Exp": lambda: data((3, 4), -2.0, 2.0),
    "Exp2": lambda: data((3, 4), -3.0, 3.0),
    "Log": lambda: data((3, 4), 0.1, 50.0),
    "Log2": lambda: data((3, 4), 0.1, 64.0),
    "Log10": lambda: data((3, 4), 0.1, 100.0),
    "Log1p": lambda: data((3, 4), -0.9, 3.0),
    "Expm1": lambda: data((3, 4), -1.0, 1.0),
    "Sin": lambda: data((3, 4), -1.3, 1.3),
    "Cos": lambda: data((3, 4), -1.3, 1.3),
    "Tan": lambda: data((3, 4), -1.3, 1.3),
    "Asin": lambda: data((3, 4), -0.95, 0.95),
    "Acos": lambda: data((3, 4), -0.95, 0.95),
    "Atan": lambda: data((3, 4), -5.0, 5.0),
    "Sinh": lambda: data((3, 4), -2.0, 2.0),
    "Cosh": lambda: data((3, 4), -2.0, 2.0),
    "Tanh": lambda: data((3, 4), -3.0, 3.0),
    "Asinh": lambda: data((3, 4), -5.0, 5.0),
    "Acosh": lambda: data((3, 4), 1.01, 6.0),
    "Atanh": lambda: data((3, 4), -0.95, 0.95),
    "Floor": lambda: data((3, 4), -3.9, 3.9),
    "Ceil": lambda: data((3, 4), -3.9, 3.9),
    "Trunc": lambda: data((3, 4), -3.9, 3.9),
}
_np_unary = {
    "Abs": np.abs, "Sqrt": np.sqrt, "Cbrt": np.cbrt, "Exp": np.exp, "Exp2": np.exp2,
    "Log": np.log, "Log2": np.log2, "Log10": np.log10, "Log1p": np.log1p, "Expm1": np.expm1,
    "Sin": np.sin, "Cos": np.cos, "Tan": np.tan, "Asin": np.arcsin, "Acos": np.arccos,
    "Atan": np.arctan, "Sinh": np.sinh, "Cosh": np.cosh, "Tanh": np.tanh,
    "Asinh": np.arcsinh, "Acosh": np.arccosh, "Atanh": np.arctanh,
    "Floor": np.floor, "Ceil": np.ceil, "Trunc": np.trunc,
}
for fn in _np_unary:
    x = _math_inputs[fn]()
    add("math", fn.lower(), fn, [], [x], _np_unary[fn](x))

sq_neg = np.array([[-1.0, -2.0, -0.25, -4.0], [-9.0, -0.01, -16.0, -1.5]])
with np.errstate(all="ignore"):
    add("math", "sqrt_negative", "Sqrt", [], [sq_neg], np.sqrt(sq_neg))
halves = np.array([[0.5, 1.5, 2.5, -0.5], [-1.5, 2.25, -2.5, 0.5]])
add("math", "round_halves", "Round", [], [halves], np.round(halves))
lg = data((3, 4), 1.0, 3.0)
add("math", "round_random", "Round", [], [lg], np.round(lg))

# ------------------------------------------------------------------- dot ----
v5a, v5b = data((5,)), data((5,))
add("dot", "dot_1d_1d", "Dot", [], [v5a, v5b], scalar(np.dot(v5a, v5b)))
m34, m45 = data((3, 4)), data((4, 5))
add("dot", "dot_2d_2d", "Dot", [], [m34, m45], np.dot(m34, m45))
v4d = data((4,))
add("dot", "dot_2d_1d", "Dot", [], [m34, v4d], np.dot(m34, v4d))
add("dot", "dot_1d_2d", "Dot", [], [v4d, m45], np.dot(v4d, m45))
t234, m45b = data((2, 3, 4)), data((4, 5))
add("dot", "dot_3d_2d", "Dot", [], [t234, m45b], np.dot(t234, m45b))
m25, t453 = data((2, 5)), data((4, 5, 3))
add("dot", "dot_2d_3d_b_leads", "Dot", [], [m25, t453], np.dot(m25, t453))
add("dot", "matmul_2d_2d", "Matmul", [], [m34, m45], m34 @ m45)
add("dot", "matmul_2d_1d", "Matmul", [], [m34, v4d], m34 @ v4d)
add("dot", "matmul_1d_2d", "Matmul", [], [v4d, m45], v4d @ m45)
b1, b2 = data((2, 3, 4)), data((2, 4, 5))
add("dot", "matmul_batched", "Matmul", [], [b1, b2], b1 @ b2)
add("dot", "matmul_broadcast", "Matmul", [], [b1, m45], b1 @ m45)
va, vb = data((3, 4)), data((2, 6))
add("dot", "vdot_mixed_shapes", "Vdot", [], [va, vb], scalar(np.vdot(va, vb)))

# ----------------------------------------------------------------- fancy ----
add("fancy", "take_axis0", "Take", [[1, 0, 1], 0], [c23], np.take(c23, [1, 0, 1], axis=0))
add("fancy", "take_negative", "Take", [[-1, 0], 0], [c23], np.take(c23, [-1, 0], axis=0))
add("fancy", "take_last", "Take", [[0, 2, 2], -1], [c23], np.take(c23, [0, 2, 2], axis=-1))
add("fancy", "take_repeat", "Take", [[1, 1, 0], 0], [c23], np.take(c23, [1, 1, 0], axis=0))
t3 = data((2, 3, 4))
add("fancy", "take_3d_axis1", "Take", [[2, 0], 1], [t3], np.take(t3, [2, 0], axis=1))
add("fancy", "compress_axis0", "Compress", [[True, False], 0], [c23], np.compress([True, False], c23, axis=0))
add("fancy", "compress_last", "Compress", [[True, False, True], -1], [c23],
    np.compress([True, False, True], c23, axis=-1))
add("fancy", "compress_none", "Compress", [[False, False], 0], [c23], np.compress([False, False], c23, axis=0))
add("fancy", "compress_3d_axis1", "Compress", [[True, False, True], 1], [t3],
    np.compress([True, False, True], t3, axis=1))

# ------------------------------------------------------------------ emit ----
os.makedirs(OUT, exist_ok=True)
total = 0
for domain, cases in sorted(domains.items()):
    # cumprod needs its input referenced; rebuild it here to keep the case tidy.
    path = os.path.join(OUT, domain + ".json")
    with open(path, "w") as f:
        json.dump(cases, f, indent=1, allow_nan=False, sort_keys=False)
        f.write("\n")
    print(f"{domain}: {len(cases)} cases -> {path}")
    total += len(cases)
print(f"total: {total} cases")
