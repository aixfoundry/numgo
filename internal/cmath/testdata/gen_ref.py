#!/usr/bin/env python3
"""Generate testdata/cmath_ref.json: glibc reference sweeps for exp/log/log1p/pow.

Dense grid (as in the task spec: values across [1e-3, 1e3]) plus random bit
patterns in the domains exercised by the random package. Reference values come
from CPython's math module, which calls the platform glibc.
"""
import json
import math
import random
import struct

def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]

random.seed(12345)
out = {}

# exp: dense grid over [-745, 709.9] plus randoms
xs = []
n = 20000
for i in range(n):
    xs.append(-745.0 + (709.78 + 745.0) * i / n)
for _ in range(5000):
    v = (random.random() * 2 - 1) * 710
    xs.append(709.7 if v > 709.7 else v)
out["exp"] = {
    "xs": xs,
    "ys": [0.0] * len(xs),
    "rs": [bits(math.exp(x)) for x in xs],
}

# log: dense grid over [1e-300, 1e300] log-spaced, near 1.0, near 0, plus randoms
xs = []
for i in range(n):
    e = -300 + 600 * i / n
    xs.append(math.exp(e))
for i in range(10000):
    # near 1.0, both sides (avoid crossing 0/negative due to precision)
    d = (random.random() - 0.5) * 1e-6 * float(1 << (i % 40))
    v = 1.0 + d
    if v > 0:
        xs.append(v)
for _ in range(5000):
    xs.append(math.exp((random.random() - 1) * 700))
out["log"] = {
    "xs": xs,
    "ys": [0.0] * len(xs),
    "rs": [bits(math.log(x)) for x in xs],
}

# log1p: x in (-1, 0] (numpy domain) plus positives and near ±1
xs = []
for i in range(n):
    xs.append(-random.random())          # (-1, 0]
for i in range(5000):
    xs.append(-random.random() * 1e-8)   # tiny negatives
for i in range(5000):
    xs.append(random.random() * 1e6)     # large positives
out["log1p"] = {
    "xs": xs,
    "ys": [0.0] * len(xs),
    "rs": [bits(math.log1p(x)) for x in xs],
}

# pow: x in (0, 2) plus small/large, y in ranges used by gamma/beta/poisson
xs, ys = [], []
for i in range(20000):
    xs.append(random.random() * 2)
    ys.append(random.random() * 4)
for i in range(5000):
    xs.append(math.exp((random.random() - 0.5) * 40))
    ys.append((random.random() - 0.5) * 30)
for i in range(5000):
    xs.append(random.random())           # [0,1): gamma<1, beta
    ys.append(1.0 / random.random() * 0.1)
xs += [0.5, 2.0, 1e-300, 1e300, 0.9999999999, 1.0000000001]
ys += [0.5, 0.5, 1e-10, 1e-10, 1e10, 1e10]
out["pow"] = {
    "xs": xs,
    "ys": ys,
    "rs": [bits(math.pow(x, y)) for x, y in zip(xs, ys)],
}

with open(__file__.replace("gen_ref.py", "cmath_ref.json"), "w") as f:
    json.dump(out, f)
total = sum(len(v["xs"]) for v in out.values())
print(f"wrote cmath_ref.json: {total} reference values")
