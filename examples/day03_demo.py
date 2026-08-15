"""
examples/day03_demo.py
-----------------------
Day 3 demo — gradients arrive: every Value now knows its .grad and how
to propagate it one step backwards.

Run:
    python examples/day03_demo.py
"""

import sys

from micrograd import Value

# Force UTF-8 output — needed on Windows where the default codec is cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def separator(title: str) -> None:
    print(f"\n{'-' * 58}")
    print(f"  {title}")
    print('-' * 58)


# ── 1. Every Value has a .grad ─────────────────────────────────
separator("1. Every Value starts with .grad = 0.0")

x = Value(3.0, label="x")
y = Value(4.0, label="y")
z = x * y

print(f"  x.grad = {x.grad}   (leaf — not yet touched by backprop)")
print(f"  y.grad = {y.grad}   (leaf — same)")
print(f"  z.grad = {z.grad}   (result of x*y — also 0)")
print()
print("  Gradients start at zero and are ACCUMULATED by ._backward()")


# ── 2. One-step backward on x * y ─────────────────────────────
separator("2. Single backward hop: c = a * b")

a = Value(2.0, label="a")
b = Value(3.0, label="b")
c = a * b   # c = 6.0

print(f"  a = {a.data},  b = {b.data},  c = a * b = {c.data}")
print()
print("  We want ∂L/∂a and ∂L/∂b where L = c.")
print("  Seed: c.grad = 1.0  (gradient of L w.r.t. itself is always 1)")
c.grad = 1.0
c._backward()

print(f"  After c._backward():")
print(f"    a.grad = {a.grad}  (= c.grad * b.data = 1 × 3 = 3)  ✓")
print(f"    b.grad = {b.grad}  (= c.grad * a.data = 1 × 2 = 2)  ✓")


# ── 3. Walk a chain step by step ──────────────────────────────
separator("3. Chain: out = w*x + b   (a linear neuron)")

w   = Value(0.5,  label="w")
x   = Value(2.0,  label="x")
b   = Value(0.3,  label="b")
wx  = w * x           # 1.0   — intermediate node
out = wx + b          # 1.3   — output

print(f"  w={w.data}, x={x.data}, b={b.data}")
print(f"  wx  = w * x  = {wx.data}")
print(f"  out = wx + b = {out.data}")
print()

# Seed the output
out.grad = 1.0
print(f"  Step 1  →  seed out.grad = 1.0")

# Propagate through addition (out = wx + b)
out._backward()
print(f"  Step 2  →  out._backward()  [addition]")
print(f"    wx.grad = {wx.grad}   (d(out)/d(wx) = 1)")
print(f"    b.grad  = {b.grad}    (d(out)/d(b)  = 1)")

# Propagate through multiplication (wx = w * x)
wx._backward()
print(f"  Step 3  →  wx._backward()   [multiplication]")
print(f"    w.grad = {w.grad}   (= wx.grad * x.data = 1 × 2 = 2.0)")
print(f"    x.grad = {x.grad}   (= wx.grad * w.data = 1 × 0.5 = 0.5)")


# ── 4. Accumulation — shared node ─────────────────────────────
separator("4. Gradient accumulation: y = x * x  (x used twice)")

x2 = Value(3.0, label="x")
y2 = x2 * x2   # 9.0  — both parents are the SAME node

y2.grad = 1.0
y2._backward()

print(f"  x = {x2.data},  y = x*x = {y2.data}")
print(f"  dy/dx = 2*x = {2 * x2.data}")
print(f"  After y._backward(): x.grad = {x2.grad}")
print()
print("  Both branches contributed x.grad += 3  →  total = 6.0  ✓")
print("  This is why we use += not = in every ._backward closure.")


# ── 5. Zero grad ──────────────────────────────────────────────
separator("5. zero_grad() — reset the whole graph at once")

p = Value(4.0, label="p")
q = Value(2.0, label="q")
r = p * q   # 8.0

r.grad = 1.0
r._backward()
print(f"  After backward: p.grad={p.grad}, q.grad={q.grad}, r.grad={r.grad}")

r.zero_grad()
print(f"  After zero_grad():  p.grad={p.grad}, q.grad={q.grad}, r.grad={r.grad}")
print()
print("  Now safe to start a new forward + backward pass from scratch.")


# ── 6. Division & subtraction (chain through pow and neg) ─────
separator("6. Division and subtraction inherit correct gradients")

numerator   = Value(6.0, label="n")
denominator = Value(3.0, label="d")
quot = numerator / denominator   # 6/3 = 2.0

quot.grad = 1.0
# Division is n * d**-1, so we need to propagate through the intermediate **-1 node too.
# Walk the chain manually: quot → (n * d_inv) → d_inv = d**-1
d_inv = denominator ** -1
d_inv_node = list(quot._prev - {numerator})[0]   # pick the d**-1 node from prev

quot._backward()
d_inv_node._backward()

print(f"  n={numerator.data}, d={denominator.data}, n/d={quot.data}")
print(f"  d(n/d)/dn = 1/d = {1/denominator.data:.4f}   → n.grad = {numerator.grad:.4f}  ✓")
print(f"  d(n/d)/dd = -n/d² = {-numerator.data / denominator.data**2:.4f} → d.grad = {denominator.grad:.4f}  ✓")


# ── 7. Road ahead ─────────────────────────────────────────────
separator("7. The road ahead")

print("""
  Day 1  ✅  Value wraps a scalar — identity, labels, comparisons
  Day 2  ✅  Arithmetic: +, -, *, /, **  — Values compute, graph forms
  Day 3  ✅  .grad + ._backward()        — Every node tracks its gradient
  Day 4  ⏳  .backward()                 — Full auto-diff in one call!
  Day 5  ⏳  Activations                 — tanh, relu, sigmoid, exp
  Day 6  ⏳  Neuron / Layer / MLP        — A real neural net from scratch
""")
