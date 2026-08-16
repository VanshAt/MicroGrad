"""
examples/day04_demo.py
-----------------------
Day 4 demo — `.backward()` arrives: one call to propagate gradients
through the entire computation graph automatically.

Run:
    python examples/day04_demo.py
"""

import sys

from micrograd import Value

# Force UTF-8 output — needed on Windows where the default codec is cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def separator(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print('-' * 60)


# ── 1. The problem Day 4 solves ───────────────────────────────────────
separator("1. The problem: manual backprop was tedious")

print("""
  Day 3 gave us ._backward() on every node, but to get all gradients
  you had to manually walk the graph, calling _backward() node by node
  in the correct order.  For out = w*x + b that looked like:

      out.grad = 1.0     # seed manually
      out._backward()    # hop 1: addition
      wx._backward()     # hop 2: multiplication

  For deeper graphs this becomes unmanageable.  Day 4 automates it:

      out.backward()     # that's it — the whole graph in one call
""")


# ── 2. Single call: a linear neuron ──────────────────────────────────
separator("2. out = w*x + b   →   out.backward()")

w   = Value(0.5,  label="w")
x   = Value(2.0,  label="x")
b   = Value(0.3,  label="b")
out = w * x + b

print(f"  Forward pass:  w={w.data}, x={x.data}, b={b.data}")
print(f"  out = w*x + b = {out.data}")
print()

out.backward()

print(f"  After out.backward():")
print(f"    w.grad = {w.grad:.4f}  (∂out/∂w = x = 2.0)  ✓")
print(f"    x.grad = {x.grad:.4f}  (∂out/∂x = w = 0.5)  ✓")
print(f"    b.grad = {b.grad:.4f}  (∂out/∂b = 1.0)       ✓")
print(f"    out.grad = {out.grad:.4f}  (∂out/∂out = 1.0 — always)  ✓")


# ── 3. Shared node: gradient accumulation ────────────────────────────
separator("3. y = x * x   →   dy/dx should be 2x")

x2 = Value(3.0, label="x")
y  = x2 * x2     # x appears in BOTH parent slots

print(f"  x = {x2.data},  y = x*x = {y.data}")
print(f"  Analytical dy/dx = 2*x = {2 * x2.data}")

y.backward()

print(f"  After y.backward():  x.grad = {x2.grad:.4f}  ✓")
print()
print("  Note: each parent slot contributed x.grad += 3 (= out.grad * other.data)")
print("  Because we use += in ._backward, both paths add correctly → 6.0")


# ── 4. Deeper polynomial ──────────────────────────────────────────────
separator("4. Polynomial: f(x) = 3x³ - 2x² + x - 5   at x = 1.5")

xp = Value(1.5, label="x")
f  = Value(3.0) * xp**3 - Value(2.0) * xp**2 + xp - Value(5.0)

print(f"  f(1.5)  = {f.data:.6f}")
print(f"  f'(x)   = 9x² - 4x + 1")

analytical_expected = 9 * 1.5**2 - 4 * 1.5 + 1
print(f"  f'(1.5) = {analytical_expected:.6f}  (expected)")

f.backward()
print(f"  x.grad  = {xp.grad:.6f}  (from backward())  ✓")


# ── 5. Numerical gradient check ───────────────────────────────────────
separator("5. Numerical gradient check (finite differences vs .backward)")

print("""
  We verify that our analytical gradients match numerical approximations:

      grad_numerical = (f(x+h) - f(x-h)) / (2h)   with h = 1e-5

  Agreement within 4 decimal places is the gold standard.
""")

h = 1e-5

checks = [
    ("x²   at x=3", lambda v: v**2,      3.0),
    ("x³   at x=2", lambda v: v**3,      2.0),
    ("1/x  at x=4", lambda v: Value(1.0) / v, 4.0),
    ("x⁴   at x=2", lambda v: (v**2)**2, 2.0),
]

all_pass = True
for name, fn, xv in checks:
    # Analytical
    xnode = Value(xv)
    out_node = fn(xnode)
    out_node.backward()
    analytical = xnode.grad

    # Numerical
    def f_plain(v, _fn=fn):
        return _fn(Value(v)).data

    numerical = (f_plain(xv + h) - f_plain(xv - h)) / (2 * h)

    ok = abs(analytical - numerical) < 1e-3
    all_pass = all_pass and ok
    status = "✓" if ok else "✗"
    print(f"  f(x) = {name:12s}  analytical={analytical:10.5f}  numerical={numerical:10.5f}  {status}")

print()
print(f"  {'All checks passed! ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")


# ── 6. zero_grad() + backward() loop ────────────────────────────────
separator("6. Training loop pattern: zero_grad → forward → backward")

print("""
  In real training you iterate:
    1. zero_grad()   — reset stale gradients
    2. forward pass  — compute the loss
    3. backward()    — accumulate fresh gradients
    4. update params — nudge weights by -lr * grad
""")

w_train = Value(0.0,  label="w")   # weight to learn
target  = 3.0                       # we want w*2 ≈ 3 → w → 1.5
lr      = 0.1

print(f"  Goal: learn w such that w * 2.0 = {target}")
print(f"  Starting w = {w_train.data}")
print()

for step in range(8):
    x_train = Value(2.0)
    out_train = w_train * x_train
    loss = (out_train - Value(target)) ** 2   # MSE

    # Backward pass
    loss.zero_grad()
    loss.backward()

    # Gradient descent step (manual update, not through Value)
    w_train.data -= lr * w_train.grad
    w_train.grad  = 0.0   # clear manually after update

    print(f"  Step {step+1:2d}:  w={w_train.data:.6f}   loss={loss.data:.6f}")

print()
print(f"  Converged to w ≈ {w_train.data:.4f}  (target: 1.5000)  ✓")


# ── 7. Road ahead ────────────────────────────────────────────────────
separator("7. The road ahead")

print("""
  Day 1  ✅  Value wraps a scalar — identity, labels, comparisons
  Day 2  ✅  Arithmetic: +, -, *, /, **  — Values compute, graph forms
  Day 3  ✅  .grad + ._backward()        — Every node tracks its gradient
  Day 4  ✅  .backward()                 — Full auto-diff in one call!
  Day 5  ⏳  Activations                 — tanh, relu, sigmoid, exp
  Day 6  ⏳  Neuron / Layer / MLP        — A real neural net from scratch
""")
