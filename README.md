# MicroGrad 🔬

> *A minimal scalar-valued autograd engine — built from scratch, one day at a time.*

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), this project rebuilds automatic differentiation from first principles. Every concept is introduced incrementally so the mechanics are never magic.

---

## 🗓️ Day-by-Day Roadmap

| Day | Topic | Status |
|-----|-------|--------|
| **1** | `Value` — wrap a scalar in an object | ✅ Done |
| 2 | Arithmetic operators (`+`, `−`, `×`, `÷`) | ⏳ |
| 3 | `.grad` attribute — every node tracks its gradient | ⏳ |
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

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_day01.py::TestConstruction::test_float_stored_as_float  PASSED
tests/test_day01.py::TestConstruction::test_int_promoted_to_float   PASSED
...
34 passed in 0.XXs
```

---

## 🎬 Running the Demo

```bash
python examples/day01_demo.py
```

---

## 🏗️ Project Structure

```
MicroGrad/
├── micrograd/
│   ├── __init__.py       # Public API
│   └── engine.py         # Value class (the heart of everything)
├── tests/
│   └── test_day01.py     # Comprehensive test suite
├── examples/
│   └── day01_demo.py     # Interactive walkthrough
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
