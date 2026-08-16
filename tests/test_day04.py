"""
tests/test_day04.py
-------------------
Tests for Day 4 — `.backward()`: full reverse-mode autodiff.

Philosophy:
  - Call `out.backward()` and assert the resulting `.grad` values on
    all ancestor nodes in one shot.
  - A numerical gradient check (central finite differences) proves that
    the analytical gradients match numerical approximations to high
    precision — the gold-standard correctness test for any autograd engine.

Run with:
    python -m pytest tests/ -v
"""

import math
import pytest
from micrograd import Value


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def numerical_grad(f, x: Value, h: float = 1e-5) -> float:
    """
    Central finite-difference approximation of df/dx at the current x.

        grad ≈ (f(x + h) - f(x - h)) / (2h)

    `f` is a callable that takes a plain float and returns a plain float.
    `x` is a Value whose `.data` is the evaluation point.
    """
    xv = x.data
    return (f(xv + h) - f(xv - h)) / (2 * h)


# ──────────────────────────────────────────────────────────────────────
# 1. Leaf node — backward() seeds itself
# ──────────────────────────────────────────────────────────────────────

class TestLeafBackward:
    def test_leaf_backward_sets_grad_to_one(self):
        """A standalone leaf's backward() seeds self.grad = 1.0."""
        v = Value(5.0)
        v.backward()
        assert v.grad == pytest.approx(1.0)

    def test_leaf_backward_overwrites_existing_grad(self):
        """backward() always sets self.grad = 1.0, ignoring prior value."""
        v = Value(3.0)
        v.grad = 99.0
        v.backward()
        assert v.grad == pytest.approx(1.0)

    def test_leaf_backward_does_not_alter_other_leaves(self):
        """Calling backward() on one Value should not touch unrelated nodes."""
        a = Value(1.0)
        b = Value(2.0)
        a.backward()
        assert b.grad == 0.0


# ──────────────────────────────────────────────────────────────────────
# 2. Single operations
# ──────────────────────────────────────────────────────────────────────

class TestSingleOpBackward:
    def test_add_backward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b        # c = 5.0
        c.backward()
        # ∂c/∂a = 1,  ∂c/∂b = 1
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(1.0)
        assert c.grad == pytest.approx(1.0)

    def test_mul_backward(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b        # c = 6.0
        c.backward()
        # ∂c/∂a = b = 3,  ∂c/∂b = a = 2
        assert a.grad == pytest.approx(3.0)
        assert b.grad == pytest.approx(2.0)

    def test_pow_backward(self):
        a = Value(3.0)
        out = a ** 2     # out = 9.0
        out.backward()
        # ∂out/∂a = 2a = 6
        assert a.grad == pytest.approx(6.0)

    def test_sub_backward(self):
        a = Value(5.0)
        b = Value(2.0)
        c = a - b        # c = 3.0
        c.backward()
        # ∂c/∂a = 1,  ∂c/∂b = -1
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(-1.0)

    def test_div_backward(self):
        a = Value(6.0)
        b = Value(2.0)
        c = a / b        # c = 3.0
        c.backward()
        # ∂c/∂a = 1/b = 0.5,  ∂c/∂b = -a/b² = -1.5
        assert a.grad == pytest.approx(0.5)
        assert b.grad == pytest.approx(-1.5)

    def test_scalar_add_backward(self):
        a = Value(3.0)
        out = a + 5.0
        out.backward()
        assert a.grad == pytest.approx(1.0)

    def test_scalar_mul_backward(self):
        a = Value(3.0)
        out = a * 4.0
        out.backward()
        assert a.grad == pytest.approx(4.0)

    def test_radd_backward(self):
        a = Value(3.0)
        out = 5.0 + a    # __radd__
        out.backward()
        assert a.grad == pytest.approx(1.0)

    def test_rmul_backward(self):
        a = Value(3.0)
        out = 4.0 * a    # __rmul__
        out.backward()
        assert a.grad == pytest.approx(4.0)


# ──────────────────────────────────────────────────────────────────────
# 3. Chained operations (linear neuron style)
# ──────────────────────────────────────────────────────────────────────

class TestChainedBackward:
    def test_linear_neuron(self):
        """out = w * x + b;  verify all leaf grads."""
        w   = Value(0.5,  label="w")
        x   = Value(2.0,  label="x")
        b   = Value(0.3,  label="b")
        out = w * x + b  # 1.3

        out.backward()

        # ∂out/∂w = x = 2.0
        # ∂out/∂x = w = 0.5
        # ∂out/∂b = 1
        assert w.grad == pytest.approx(2.0)
        assert x.grad == pytest.approx(0.5)
        assert b.grad == pytest.approx(1.0)

    def test_quadratic_chain(self):
        """out = x**2 + 2*x + 1;  at x=3 → dout/dx = 2x+2 = 8."""
        x   = Value(3.0)
        out = x ** 2 + 2.0 * x + 1.0
        out.backward()
        assert x.grad == pytest.approx(8.0)

    def test_three_hop_chain(self):
        """a → b=a*2 → c=b*3 → d=c*4;  dd/da = 2*3*4 = 24."""
        a = Value(1.0)
        d = a * 2.0 * 3.0 * 4.0
        d.backward()
        assert a.grad == pytest.approx(24.0)

    def test_nested_pow(self):
        """out = (x**2)**2 = x**4;  at x=2 → d/dx = 4x**3 = 32."""
        x   = Value(2.0)
        out = (x ** 2) ** 2
        out.backward()
        assert x.grad == pytest.approx(32.0)


# ──────────────────────────────────────────────────────────────────────
# 4. Shared / reused nodes (gradient accumulation)
# ──────────────────────────────────────────────────────────────────────

class TestSharedNodes:
    def test_square_via_mul(self):
        """y = x * x  →  dy/dx = 2x."""
        x = Value(3.0)
        y = x * x
        y.backward()
        assert x.grad == pytest.approx(6.0)

    def test_square_via_pow(self):
        """y = x**2  →  dy/dx = 2x."""
        x = Value(4.0)
        y = x ** 2
        y.backward()
        assert x.grad == pytest.approx(8.0)

    def test_add_self(self):
        """y = x + x  →  dy/dx = 2."""
        x = Value(5.0)
        y = x + x
        y.backward()
        assert x.grad == pytest.approx(2.0)

    def test_cubic_shared(self):
        """y = x * x * x  →  dy/dx = 3x²."""
        x = Value(2.0)
        y = x * x * x
        y.backward()
        assert x.grad == pytest.approx(12.0)

    def test_mixed_shared(self):
        """out = x**2 + x;  at x=3 → d/dx = 2x+1 = 7."""
        x   = Value(3.0)
        out = x ** 2 + x
        out.backward()
        assert x.grad == pytest.approx(7.0)


# ──────────────────────────────────────────────────────────────────────
# 5. Diamond graph
# ──────────────────────────────────────────────────────────────────────

class TestDiamondGraph:
    def test_diamond_add(self):
        """
        Graph:
            a ──┬── m = a + b ──┬── out = m + n
            b ──┘               │
            c ──── n = b + c ──┘

        Here b appears in both m and n (diamond).
        ∂out/∂a = 1
        ∂out/∂b = 2   (contributes through both m and n)
        ∂out/∂c = 1
        """
        a = Value(1.0, label="a")
        b = Value(2.0, label="b")
        c = Value(3.0, label="c")
        m   = a + b
        n   = b + c
        out = m + n
        out.backward()

        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(2.0)
        assert c.grad == pytest.approx(1.0)

    def test_diamond_mul(self):
        """
        z = x * y + x * y  (x and y each appear twice)
        dz/dx = y + y = 2y,  dz/dy = x + x = 2x
        """
        x = Value(3.0)
        y = Value(4.0)
        z = x * y + x * y
        z.backward()
        assert x.grad == pytest.approx(8.0)   # 2 * y = 2*4
        assert y.grad == pytest.approx(6.0)   # 2 * x = 2*3


# ──────────────────────────────────────────────────────────────────────
# 6. backward() with zero_grad() — idempotence across passes
# ──────────────────────────────────────────────────────────────────────

class TestBackwardIdempotence:
    def test_second_call_doubles_gradients(self):
        """Calling backward() twice accumulates — user must zero_grad()."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        # First call: a.grad = 3.0
        first_a = a.grad
        c.backward()
        # Second call (no zero_grad): adds another 3.0
        assert a.grad == pytest.approx(2 * first_a)

    def test_zero_grad_then_backward_correct(self):
        """zero_grad() + backward() gives clean, correct result."""
        w = Value(0.5)
        x = Value(2.0)
        out = w * x

        # First pass
        out.backward()
        assert w.grad == pytest.approx(2.0)

        # Reset and repeat
        out.zero_grad()
        out.backward()
        assert w.grad == pytest.approx(2.0)
        assert x.grad == pytest.approx(0.5)

    def test_three_passes_stay_correct_with_zero_grad(self):
        a = Value(3.0)
        out = a ** 2   # da/dx = 2*3 = 6

        for _ in range(3):
            out.zero_grad()
            out.backward()
            assert a.grad == pytest.approx(6.0)


# ──────────────────────────────────────────────────────────────────────
# 7. Karpathy reference examples
# ──────────────────────────────────────────────────────────────────────

class TestKarpathyExamples:
    def test_karpathy_neuron(self):
        """
        Exactly the example from Karpathy's 'micrograd' lecture.

        Forward:
            x1, x2 = 2.0, 0.0
            w1, w2 = -3.0, 1.0
            b = 6.8813735870195432

            x1w1 = x1 * w1   = -6.0
            x2w2 = x2 * w2   = 0.0
            x1w1x2w2 = x1w1 + x2w2 = -6.0
            n = x1w1x2w2 + b  ≈ 0.8814
            o = tanh(n)  — we skip tanh here, using a linear approx

        We test a simpler version without tanh since tanh is Day 5.
        """
        x1 = Value(2.0, label="x1")
        x2 = Value(0.0, label="x2")
        w1 = Value(-3.0, label="w1")
        w2 = Value(1.0, label="w2")
        b  = Value(6.7, label="b")

        x1w1   = x1 * w1
        x2w2   = x2 * w2
        x1w1x2w2 = x1w1 + x2w2
        n      = x1w1x2w2 + b   # n ≈ 0.7

        n.backward()

        # ∂n/∂x1 = w1 = -3.0
        # ∂n/∂x2 = w2 = 1.0
        # ∂n/∂w1 = x1 = 2.0
        # ∂n/∂w2 = x2 = 0.0
        assert x1.grad == pytest.approx(-3.0)
        assert x2.grad == pytest.approx(1.0)
        assert w1.grad == pytest.approx(2.0)
        assert w2.grad == pytest.approx(0.0)
        assert b.grad  == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# 8. Numerical gradient check
#    (finite differences vs analytical .grad)
# ──────────────────────────────────────────────────────────────────────

class TestNumericalGradientCheck:
    """
    Gold-standard test: compare .backward() gradient to a numerical
    finite-difference approximation.

        grad_numerical = (f(x+h) - f(x-h)) / 2h   (central differences)

    Agreement to 4 significant figures is expected.
    """

    def _check(self, f_lambda, x_val: float, tol: float = 1e-4):
        """
        Helper that:
          1. Builds a Value graph, runs backward(), reads x.grad.
          2. Computes the numerical gradient via central differences.
          3. Asserts they agree to `tol` relative tolerance.
        """
        x = Value(x_val)
        out = f_lambda(x)
        out.backward()
        analytical = x.grad

        f_plain = lambda v: f_lambda(Value(v)).data
        numerical = numerical_grad(f_plain, x)

        assert analytical == pytest.approx(numerical, rel=tol), (
            f"analytical={analytical}, numerical={numerical}"
        )

    # Individual cases ─────────────────────────────────────────────────

    def test_numgrad_square(self):
        self._check(lambda x: x ** 2, x_val=3.0)

    def test_numgrad_cube(self):
        self._check(lambda x: x ** 3, x_val=2.0)

    def test_numgrad_reciprocal(self):
        self._check(lambda x: Value(1.0) / x, x_val=4.0)

    def test_numgrad_polynomial(self):
        """f(x) = 3x³ - 2x² + x - 5;  f'(x) = 9x² - 4x + 1."""
        self._check(
            lambda x: Value(3.0) * x ** 3 - Value(2.0) * x ** 2 + x - Value(5.0),
            x_val=1.5,
        )

    def test_numgrad_nested_pow(self):
        """f(x) = (x²)²= x⁴;  f'(x) = 4x³."""
        self._check(lambda x: (x ** 2) ** 2, x_val=2.0)

    def test_numgrad_division(self):
        """f(x) = x / 3;  f'(x) = 1/3."""
        self._check(lambda x: x / Value(3.0), x_val=5.0)

    def test_numgrad_linear(self):
        """f(x) = 2x + 7;  f'(x) = 2."""
        self._check(lambda x: Value(2.0) * x + Value(7.0), x_val=0.0)

    def test_numgrad_fraction(self):
        """f(x) = 1/x²;  f'(x) = -2/x³."""
        self._check(lambda x: Value(1.0) / (x ** 2), x_val=3.0)

    def test_numgrad_at_multiple_points(self):
        """f(x) = x³ checked at several x values."""
        for xv in [-2.0, -0.5, 1.0, 2.5, 4.0]:
            self._check(lambda x: x ** 3, x_val=xv)

    # Two-variable numerical check ─────────────────────────────────────

    def test_numgrad_two_vars(self):
        """
        f(a, b) = a * b²;  ∂f/∂a = b², ∂f/∂b = 2ab.
        Perturb each variable independently.
        """
        h = 1e-5
        a_val, b_val = 3.0, 4.0

        def f(a_v, b_v):
            return (Value(a_v) * Value(b_v) ** 2).data

        num_da = (f(a_val + h, b_val) - f(a_val - h, b_val)) / (2 * h)
        num_db = (f(a_val, b_val + h) - f(a_val, b_val - h)) / (2 * h)

        a = Value(a_val)
        b = Value(b_val)
        out = a * b ** 2
        out.backward()

        assert a.grad == pytest.approx(num_da, rel=1e-4)
        assert b.grad == pytest.approx(num_db, rel=1e-4)


# ──────────────────────────────────────────────────────────────────────
# 9. backward() produces correct self.grad on root
# ──────────────────────────────────────────────────────────────────────

class TestRootGrad:
    def test_root_grad_is_one(self):
        """After backward(), the node it's called on always has .grad == 1."""
        a = Value(2.0)
        b = Value(3.0)
        out = a * b + a ** 2
        out.backward()
        assert out.grad == pytest.approx(1.0)

    def test_intermediate_node_grad_correct(self):
        """Intermediate nodes should also accumulate correct gradients."""
        x  = Value(2.0)
        wx = x * Value(3.0)  # wx = 6; d(out)/d(wx) will be 1
        out = wx + Value(1.0)
        out.backward()
        assert wx.grad == pytest.approx(1.0)
