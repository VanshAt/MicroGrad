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
| **4** | `.backward()` — full reverse-mode autodiff | ✅ Done |
| **5** | Activation functions (`tanh`, `relu`, `sigmoid`, `exp`) | ✅ Done |
| **6** | `Neuron` / `Layer` / `MLP` — a real tiny neural net | ✅ Done |

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

## 🧠 Day 6 — `Neuron` / `Layer` / `MLP`

Day 6 assembles the autograd engine from Days 1–5 into a complete, trainable
neural network — written entirely in terms of `Value` arithmetic so every
weight and bias is automatically differentiable.

| Class | What it does |
|---|---|
| `Neuron(n_in, activation)` | `out = activation(w·x + b)`, Kaiming-init |
| `Layer(n_in, n_out, activation)` | `n_out` independent Neurons, same input |
| `MLP(n_in, layer_sizes, ...)` | Stack of Layers — a full MLP |

Every class exposes:
- **`__call__(x)`** — forward pass, accepts `list[Value | float]`
- **`.parameters()`** — flat list of all trainable `Value` nodes
- **`.zero_grad()`** (MLP) — O(params) gradient reset

```python
import random
from micrograd import MLP

random.seed(42)
model = MLP(2, [4, 4, 1])   # 2 inputs → hidden(4) → hidden(4) → output(1)
print(model)                 # MLP([Layer(..., tanh), Layer(..., tanh), Layer(..., linear)])
print(len(model.parameters()))  # 37

# Training loop
xs = [[2.0,  3.0], [-1.0, -1.0], [1.0, -2.0], [-3.0, 1.0]]
ys = [1.0,          -1.0,          -1.0,          1.0]

for step in range(20):
    preds = [model(x) for x in xs]
    loss  = sum((p - t)**2 for p, t in zip(preds, ys)) / len(ys)

    model.zero_grad()    # 1. clear gradients
    loss.backward()      # 2. backprop

    for p in model.parameters():
        p.data -= 0.05 * p.grad   # 3. gradient descent

    if step % 5 == 0:
        print(f"step {step:2d}  loss={loss.data:.4f}")
# step  0  loss=0.7062
# step  5  loss=0.1589
# step 10  loss=0.0690
# step 15  loss=0.0539
```

**Key insights:**
- Weights initialised with **Kaiming uniform** (`±1/√n_in`) — activations
  stay well-scaled at random init.
- The output layer defaults to **`linear`** (no activation) so the network
  can produce unbounded values for regression or feed directly into a loss.
- `MLP.zero_grad()` is O(params) — it only resets parameter leaves, unlike
  `loss.zero_grad()` which walks the entire computation graph.

---

## 🔥 Day 5 — Activation Functions

Day 5 wires five non-linearities into the autograd engine — the building
blocks every real neural network needs to learn complex functions.

| Method | Forward | Backward (chain rule) |
|---|---|---|
| `.exp()` | `eˣ` | `out.grad × out.data` |
| `.log()` | `ln x` (x > 0) | `out.grad / x` |
| `.tanh()` | `(e²ˣ−1)/(e²ˣ+1)` | `out.grad × (1 − t²)` |
| `.relu()` | `max(0, x)` | `out.grad if x > 0 else 0` |
| `.sigmoid()` | `1/(1+e⁻ˣ)` | `out.grad × s × (1−s)` |

`tanh` and `sigmoid` are **fused** — single graph nodes rather than
composed sub-graphs — for a cleaner computation graph and better
numerical stability.

```python
from micrograd import Value

# Every activation is fully differentiable
x = Value(1.0, label="x")

print(x.tanh().data)     # 0.7616  — classic neuron activation
print(x.relu().data)     # 1.0     — positive pass-through
print(x.sigmoid().data)  # 0.7311  — logistic output
print(x.exp().data)      # 2.7183
print(x.log().data)      # 0.0  (ln(1) = 0)

# Full backward pass — single neuron with tanh
x1 = Value(2.0,  label="x1")
x2 = Value(0.0,  label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0,  label="w2")
b  = Value(6.8813735870195432, label="b")

n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()

# d(o)/d(n) = 1 - tanh²(n) ≈ 0.5
print(w1.grad)  # -1.5  (≈ x1 × d(tanh)/dn)
print(x1.grad)  # -1.5  (≈ w1 × d(tanh)/dn)
```

**Key insight:** `relu` blocks gradients entirely for negative inputs (the
"dead neuron" problem); `tanh` and `sigmoid` squash gradients near
saturation (the "vanishing gradient" problem). Both are important to
understand before building deep networks in Day 6.

---

## ⚡ Day 4 — `.backward()`: Full Reverse-Mode Autodiff

Day 4 wires together all the pieces built in Days 1–3 into a **single call** that
propagates gradients through the entire computation graph automatically.

Internally, `.backward()` does three things:
1. **Topological sort** — DFS post-order traversal of the `_prev` graph.
2. **Seed** — sets `self.grad = 1.0` (∂L/∂L is always 1).
3. **Reverse walk** — calls each node's `._backward()` in reverse topo order.

Because a node's `._backward()` only fires *after* all its consumers have
already accumulated their contributions into its `.grad`, the chain rule is
always applied correctly — even for shared nodes and diamond-shaped graphs.

```python
from micrograd import Value

# A linear neuron: out = w*x + b
w   = Value(0.5, label="w")
x   = Value(2.0, label="x")
b   = Value(0.3, label="b")
out = w * x + b

out.backward()   # one call — that's it

print(w.grad)   # 2.0  (∂out/∂w = x)
print(x.grad)   # 0.5  (∂out/∂x = w)
print(b.grad)   # 1.0  (∂out/∂b = 1)

# Gradient accumulation works for shared nodes
x2 = Value(3.0)
y  = x2 * x2      # x2 appears in both parent slots
y.backward()
print(x2.grad)  # 6.0  (dy/dx = 2x = 2*3)

# Training loop pattern
for step in range(10):
    loss.zero_grad()   # 1. clear stale gradients
    # forward pass ...  # 2. compute loss
    loss.backward()    # 3. compute all gradients
    # param.data -= lr * param.grad  # 4. update
```

**Key insight:** `backward()` also extracts the topological-sort DFS into a
shared `_topo_sort()` helper, so `zero_grad()` reuses the same traversal logic.

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
tests/test_day04.py ...  PASSED
tests/test_day05.py ...  PASSED
tests/test_day06.py ...  PASSED
≥ 321 passed in 0.XXs
```

---

## 🎬 Running the Demos

```bash
python examples/day01_demo.py
python examples/day02_demo.py
python examples/day03_demo.py
python examples/day04_demo.py
python examples/day06_demo.py
```

---

## 🏗️ Project Structure

```
MicroGrad/
├── micrograd/
│   ├── __init__.py       # Public API (Value, Neuron, Layer, MLP)
│   └── engine.py         # Value class + Neuron / Layer / MLP
├── tests/
│   ├── test_day01.py     # Day 1 test suite
│   ├── test_day02.py     # Day 2 test suite
│   ├── test_day03.py     # Day 3 test suite
│   ├── test_day04.py     # Day 4 test suite
│   ├── test_day05.py     # Day 5 test suite
│   └── test_day06.py     # Day 6 test suite
├── examples/
│   ├── day01_demo.py     # Day 1 interactive walkthrough
│   ├── day02_demo.py     # Day 2 interactive walkthrough
│   ├── day03_demo.py     # Day 3 interactive walkthrough
│   ├── day04_demo.py     # Day 4 interactive walkthrough
│   └── day06_demo.py     # Day 6 interactive walkthrough
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
