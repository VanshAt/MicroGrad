"""
examples/day02_demo.py
-----------------------
Day 2 demo — showing what the Value class can do after arithmetic.

Run:
    python examples/day02_demo.py
"""

from micrograd import Value


def separator(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print('─' * 55)


# ── 1. The task-brief example ──────────────────────────────────
separator("1. Karpathy sanity check: a * b")

a = Value(2.0)
b = Value(3.0)
c = a * b
print(f"  a = {a!r}")
print(f"  b = {b!r}")
print(f"  c = a * b = {c!r}")
print(f"  c.data → {c.data}")          # should print 6.0
assert c.data == 6.0, "multiplication broken!"


# ── 2. All four operators ──────────────────────────────────────
separator("2. All four operators")

x = Value(10.0, label="x")
y = Value(4.0,  label="y")

add = x + y
sub = x - y
mul = x * y
div = x / y

print(f"  x + y = {add.data}")    # 14.0
print(f"  x - y = {sub.data}")    #  6.0
print(f"  x * y = {mul.data}")    # 40.0
print(f"  x / y = {div.data}")    #  2.5


# ── 3. Scalar interoperability ────────────────────────────────
separator("3. Scalars on either side (int / float)")

v = Value(5.0, label="v")
print(f"  v + 3    = {(v + 3).data}")       # 8.0   __add__
print(f"  3 + v    = {(3 + v).data}")       # 8.0   __radd__
print(f"  v * 2.0  = {(v * 2.0).data}")     # 10.0  __mul__
print(f"  2.0 * v  = {(2.0 * v).data}")     # 10.0  __rmul__
print(f"  10 - v   = {(10 - v).data}")      # 5.0   __rsub__
print(f"  10 / v   = {(10 / v).data}")      # 2.0   __rtruediv__


# ── 4. Power operator ─────────────────────────────────────────
separator("4. Power operator")

p = Value(3.0, label="p")
print(f"  p ** 2   = {(p ** 2).data}")      # 9.0
print(f"  p ** 0.5 = {(p ** 0.5).data:.4f}")# 1.7320...
print(f"  p ** -1  = {(p ** -1).data:.4f}") # 0.3333...


# ── 5. Computation-graph metadata ─────────────────────────────
separator("5. _op and _prev — the graph skeleton")

a = Value(2.0, label="a")
b = Value(3.0, label="b")
c = a * b                    # leaf nodes have empty _op / _prev

print(f"  a._op  = {a._op!r}      (leaf — no op)")
print(f"  a._prev = {a._prev}    (leaf — no parents)")
print(f"  c._op  = {c._op!r}")
print(f"  c._prev contains a? {a in c._prev}")
print(f"  c._prev contains b? {b in c._prev}")


# ── 6. Chained / nested expression ───────────────────────────
separator("6. Realistic neuron: out = w*x + b")

w   = Value(0.5,  label="w")
x   = Value(2.0,  label="x")
b   = Value(0.3,  label="b")

wx  = w * x          # 1.0
out = wx + b         # 1.3

print(f"  w   = {w.data}")
print(f"  x   = {x.data}")
print(f"  b   = {b.data}")
print(f"  w*x = {wx.data}")
print(f"  out = w*x + b = {out.data}")   # 1.3


# ── 7. Road ahead ─────────────────────────────────────────────
separator("7. The road ahead")

print("""
  Day 1  ✅  Value wraps a scalar — identity, labels, comparisons
  Day 2  ✅  Arithmetic: +, -, *, /, **  — Values compute, graph forms
  Day 3  ⏳  .grad attribute             — Every Value tracks its gradient
  Day 4  ⏳  .backward()                 — Automatic differentiation!
  Day 5  ⏳  Activations                 — tanh, relu, sigmoid, exp
  Day 6  ⏳  Neuron / Layer / MLP        — A real neural net from scratch
""")
