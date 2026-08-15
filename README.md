# MicroGrad 🔬

> *A minimal scalar-valued autograd engine — built from scratch, one day at a time.*

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), this project rebuilds automatic differentiation from first principles. Every concept is introduced incrementally so the mechanics are never magic.

---

## 🗓️ Day-by-Day Roadmap

| Day | Topic | Status |
|-----|-------|--------|
| **1** | `Value` — wrap a scalar in an object | ✅ Done |
| **2** | Arithmetic operators (`+`, `−`, `×`, `÷`, `**`) | ✅ Done |
| **3** | `.grad` & `._backward` — every node tracks its gradient | ✅ Done |
| 4 | `.backward()` — full reverse-mode autodiff | ⏳ |
| 5 | Activation functions (`tanh`, `relu`, `sigmoid`, `exp`) | ⏳ |
| 6 | `Neuron` / `Layer` / `MLP` — a real tiny neural net | ⏳ |

---

## 📦 Installation (dev mode)

```bash
# Clone
git clone https://github.com/VanshAt/MicroGrad.git
cd MicroGrad

# Install in editable mode with dev tools
pip install -e ".[dev]"
```

---

## 🚀 Day 1 — `Value`: Wrapping a Scalar

The single insight of Day 1: **a raw number knows nothing about itself**.
A `Value` gives every number:

- a **`.data`** attribute — the float it wraps
- a **`.label`** — a human-readable name (critical when we visualise graphs)
- **type safety** — rejects anything that isn't a number
- **comparison operators** — so Values can be sorted and compared naturally
- **utility helpers** — `.item()`, `.is_finite()`, `.is_nan()`, `.named()`

```python
from micrograd import Value

x = Value(2.0, label="x")
print(x)          # Value 'x' = 2.0
print(x.data)     # 2.0
print(repr(x))    # Value(data=2.0, label='x')

# Fluent label assignment
pi = Value(3.14159).named("π")

# Unary ops return new Values
neg_x = -x        # Value(data=-2.0)

# Type safety
Value("oops")     # TypeError: must be int or float

# Comparison
Value(1.0) < Value(2.0)   # True
sorted([Value(3), Value(1), Value(2)])  # [1.0, 2.0, 3.0]
```

---

## 🧮 Day 3 — `.grad` & `._backward`: Local Gradients

Day 3 wires the **gradient side** of every node — the other half of automatic differentiation.

Every `Value` now has:

- **`.grad`** — a float, starting at `0.0`, that accumulates ∂L/∂self during backprop.
- **`._backward()`** — a zero-argument closure that propagates *this node's* gradient one step
  backwards to its parents using the **local chain-rule formula** for that specific operation.
- **`.zero_grad()`** — walks the entire ancestor graph and resets all `.grad`s to `0.0`.
  Mirrors PyTorch's `optimizer.zero_grad()`.

> Day 3 does **not** include the full traversal — that's Day 4.
> You can already walk backwards manually, one hop at a time.

```python
from micrograd import Value

# Forward pass — same as Day 2
a = Value(2.0, label="a")
b = Value(3.0, label="b")
c = a * b   # c.data = 6.0

# Seed the output gradient (∂L/∂c = 1 always)
c.grad = 1.0

# Propagate one step back — local chain rule for multiplication
c._backward()

print(a.grad)   # 3.0  (∂c/∂a = b)
print(b.grad)   # 2.0  (∂c/∂b = a)

# Reset before the next pass
c.zero_grad()
print(a.grad)   # 0.0
```

**Key insight: gradients accumulate, not overwrite.**
When the same node appears in multiple places (e.g. `x * x`), each branch
contributes its own partial derivative — they must be *added*:

```python
x   = Value(3.0)
y   = x * x     # x appears twice
y.grad = 1.0
y._backward()
print(x.grad)   # 6.0  (= 3 + 3, both branches fire +3)
```

---

## 🚀 Day 2 — Arithmetic Operators

Day 2 makes Values actually compute. All four operators plus power are supported.
Every result node records `_op` and `_prev` — the skeleton of the computation
graph that backprop will traverse.

```python
from micrograd import Value

x = Value(10.0, label="x")
y = Value(4.0,  label="y")

print((x + y).data)  # 14.0
print((x - y).data)  #  6.0
print((x * y).data)  # 40.0
print((x / y).data)  #  2.5
print((x ** 2).data) # 100.0

# Plain scalars work on either side
print((3 + x).data)  # 13.0   (__radd__)
print((10 / y).data) #  2.5   (__rtruediv__)

# Graph metadata
w, b = Value(0.5, label="w"), Value(0.3, label="b")
out = w * x + b
print(out._op)           # '+'
print(w in out._prev)    # False  (w is two hops back)
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_day01.py ...  PASSED
tests/test_day02.py ...  PASSED
tests/test_day03.py ...  PASSED
≥ 70 passed in 0.XXs
```

---

## 🎬 Running the Demos

```bash
python examples/day01_demo.py
python examples/day02_demo.py
python examples/day03_demo.py
```

---

## 🏗️ Project Structure

```
MicroGrad/
├── micrograd/
│   ├── __init__.py       # Public API
│   └── engine.py         # Value class (the heart of everything)
├── tests/
│   ├── test_day01.py     # Day 1 test suite
│   ├── test_day02.py     # Day 2 test suite
│   └── test_day03.py     # Day 3 test suite
├── examples/
│   ├── day01_demo.py     # Day 1 interactive walkthrough
│   ├── day02_demo.py     # Day 2 interactive walkthrough
│   └── day03_demo.py     # Day 3 interactive walkthrough
├── pyproject.toml        # Build config & metadata
├── .gitignore
└── README.md
```

---

## 💡 Key Design Decisions

**Why `float()` promotion?**
Storing data as `float` consistently means arithmetic in later days won't break on mixed int/float inputs.

**Why a `label`?**
Every serious autograd engine uses variable names for graph visualisation. Introducing it on Day 1 costs nothing and pays dividends when we draw the computation graph.

**Why comparison operators now?**
Sorting and comparing Values will be needed once we track multiple parameters. Implementing them on Day 1 keeps the later days clean.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
