"""
examples/day01_demo.py
-----------------------
Day 1 demo — showing exactly what the Value class can do today.

Run:
    python examples/day01_demo.py
"""

import math
from micrograd import Value


def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ── 1. Basic wrapping ──────────────────────────────────────────
separator("1. Wrapping scalars in Value")

a = Value(2.0)
b = Value(3,   label="b")       # int → promoted to float
c = Value(-1.5, label="c")
pi = Value(3.14159).named("π")  # fluent label assignment

print(f"  a  = {a!r}")
print(f"  b  = {b!r}")
print(f"  c  = {c!r}")
print(f"  pi = {pi!r}")


# ── 2. Human-friendly printing ─────────────────────────────────
separator("2. str() output")

for val in [a, b, c, pi]:
    print(f"  str → {val}")


# ── 3. Accessing .data and .label ─────────────────────────────
separator("3. .data and .label attributes")

x = Value(42.0, label="answer")
print(f"  x.data  = {x.data}")
print(f"  x.label = {x.label!r}")


# ── 4. Type promotion: int → float ────────────────────────────
separator("4. Int → float promotion")

int_val = Value(7)
print(f"  Value(7).data = {int_val.data}  (type: {type(int_val.data).__name__})")


# ── 5. Unary operations ────────────────────────────────────────
separator("5. Unary operations")

v = Value(5.0, label="v")
print(f"  -v           = {-v!r}")
print(f"  abs(c)       = {abs(c)!r}")
print(f"  +v           = {+v!r}")


# ── 6. Numeric identity unwrapping ────────────────────────────
separator("6. float() / int() / .item()")

q = Value(9.81, label="g")
print(f"  float(q)   = {float(q)}")
print(f"  int(q)     = {int(q)}")
print(f"  q.item()   = {q.item()}")


# ── 7. Comparisons ────────────────────────────────────────────
separator("7. Comparison operators")

p = Value(2.0, label="p")
r = Value(5.0, label="r")
print(f"  p < r  → {p < r}")
print(f"  p > r  → {p > r}")
print(f"  p == 2.0 → {p == 2.0}")

vals = [Value(3.0), Value(1.0), Value(4.0), Value(1.5)]
sorted_vals = sorted(vals)
print(f"  sorted → {[v.data for v in sorted_vals]}")


# ── 8. Utility checks ─────────────────────────────────────────
separator("8. is_finite() / is_nan() helpers")

normal = Value(1.0,       label="normal")
inf_v  = Value(math.inf,  label="inf")
nan_v  = Value(math.nan,  label="nan")

for val in [normal, inf_v, nan_v]:
    print(f"  {val.label:8s} → finite={val.is_finite()}, nan={val.is_nan()}")


# ── 9. Type safety ────────────────────────────────────────────
separator("9. Type safety (TypeError on bad input)")

try:
    bad = Value("hello")
except TypeError as e:
    print(f"  Caught → {e}")


# ── 10. What comes next ───────────────────────────────────────
separator("10. The road ahead")

print("""
  Day 1  ✅  Value wraps a scalar — identity, labels, comparisons
  Day 2  ⏳  Arithmetic: +, -, *, /  — Values start computing
  Day 3  ⏳  .grad attribute         — Every Value tracks its gradient
  Day 4  ⏳  .backward()             — Automatic differentiation!
  Day 5  ⏳  Activations             — tanh, relu, sigmoid, exp
  Day 6  ⏳  Neuron / Layer / MLP    — A real neural net from scratch
""")
