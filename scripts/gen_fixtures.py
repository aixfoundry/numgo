#!/usr/bin/env python3
"""Generate golden fixtures from local numpy for the Go tests (Phase 1+).

Usage: python3 scripts/gen_fixtures.py
"""
import json
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

print(f"numpy {np.__version__}")

out_dir = os.path.join(os.path.dirname(__file__), "..", "testdata", "fixtures")
os.makedirs(out_dir, exist_ok=True)

hello = {
    "arange": np.arange(10).tolist(),
    "sum": int(np.arange(10).sum()),
}
with open(os.path.join(out_dir, "hello.json"), "w") as f:
    json.dump(hello, f, indent=2)
print(f"wrote {os.path.join(out_dir, 'hello.json')}")
