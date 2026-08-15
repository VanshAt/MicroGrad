"""
tests/test_day03.py
-------------------
Tests for Day 3 — .grad attribute and local ._backward closures.

Philosophy:
  - We test each op's ._backward in isolation by manually seeding
    out.grad = 1.0 and calling out._backward(), then asserting the
    expected gradients on the parent nodes.
  - We never call the full backward() here — that's Day 4's job.
    All traversal is done by hand, one hop at a time.

Run with:
    python -m pytest tests/ -v
"""

import math
import pytest
from micrograd import Value


# ──────────────────────────────────────────────────────────────
# 1. .grad attribute basics
# ──────────────────────────────────────────────────────────────

class TestGradAttribute:
    def test_grad_starts_at_zero(self):
        v = Value(3.0)
        assert v.grad == 0.0

    def test_grad_is_float(self):
        v = Value(5.0)
        assert isinstance(v.grad, float)

    def test_grad_is_mutable(self):
        v = Value(2.0)
        v.grad = 1.5
        assert v.grad == 1.5

    def test_grad_independent_across_nodes(self):
        a = Value(1.0)
        b = Value(1.0)
        a.grad = 42.0
        assert b.grad == 0.0

    def test_op_node_grad_starts_at_zero(self):
        out = Value(2.0) + Value(3.0)
        assert out.grad == 0.0

    def test_multiple_ops_all_start_at_zero(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        d = c * Value(3.0)
        assert a.grad == 0.0
        assert b.grad == 0.0
        assert c.grad == 0.0
        assert d.grad == 0.0


# ──────────────────────────────────────────────────────────────
# 2. Leaf ._backward is a no-op
# ──────────────────────────────────────────────────────────────

class TestLeafBackward:
    def test_leaf_backward_is_callable(self):
        v = Value(5.0)
        assert callable(v._backward)

    def test_leaf_backward_does_not_raise(self):
        v = Value(5.0)
        v.grad = 99.0
        v._backward()  # should silently do nothing

    def test_leaf_backward_does_not_change_grad(self):
        v = Value(5.0)
        v.grad = 7.0
        v._backward()
        assert v.grad == 7.0


# ──────────────────────────────────────────────────────────────
# 3. Addition backward
# ──────────────────────────────────────────────────────────────

class TestAdditionBackward:
    def test_add_backward_unit_seed(self):
        a = Value(2.0)
        b = Value(3.0)
        out = a + b
        out.grad = 1.0
        out._backward()
        # d(a+b)/da = 1, d(a+b)/db = 1
        assert a.grad == pytest.approx(1.0)
        assert b.grad == pytest.approx(1.0)

    def test_add_backward_scaled_seed(self):
        a = Value(2.0)
        b = Value(3.0)
        out = a + b
        out.grad = 5.0
        out._backward()
        assert a.grad == pytest.approx(5.0)
        assert b.grad == pytest.approx(5.0)

    def test_add_backward_negative_seed(self):
        a = Value(1.0)
        b = Value(1.0)
        out = a + b
        out.grad = -2.0
        out._backward()
        assert a.grad == pytest.approx(-2.0)
        assert b.grad == pytest.approx(-2.0)

    def test_scalar_add_backward(self):
        a = Value(3.0)
        out = a + 5.0   # 5.0 gets wrapped in a Value internally
        out.grad = 1.0
        out._backward()
        assert a.grad == pytest.approx(1.0)

    def test_radd_backward(self):
        a = Value(3.0)
        out = 5.0 + a   # __radd__
        out.grad = 1.0
        out._backward()
        assert a.grad == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────
# 4. Multiplication backward
# ──────────────────────────────────────────────────────────────

class TestMultiplicationBackward:
    def test_mul_backward_unit_seed(self):
        a = Value(2.0)
        b = Value(3.0)
        out = a * b
        out.grad = 1.0
        out._backward()
        # d(a*b)/da = b = 3.0,  d(a*b)/db = a = 2.0
        assert a.grad == pytest.approx(3.0)
        assert b.grad == pytest.approx(2.0)

    def test_mul_backward_scaled_seed(self):
        a = Value(4.0)
        b = Value(5.0)
        out = a * b
        out.grad = 2.0
        out._backward()
        assert a.grad == pytest.approx(10.0)   # 2.0 * 5.0
        assert b.grad == pytest.approx(8.0)    # 2.0 * 4.0

    def test_mul_backward_with_scalar(self):
        a = Value(3.0)
        out = a * 4.0
        out.grad = 1.0
        out._backward()
        assert a.grad == pytest.approx(4.0)

    def test_rmul_backward(self):
        a = Value(3.0)
        out = 4.0 * a
        out.grad = 1.0
        out._backward()
        assert a.grad == pytest.approx(4.0)

    def test_karpathy_example_mul_grad(self):
        # From Karpathy's video: a=2, b=3, c=a*b
        # dc/da should be b=3, dc/db should be a=2
        a = Value(2.0, label="a")
        b = Value(3.0, label="b")
        c = a * b
        c.grad = 1.0
        c._backward()
        assert a.grad == pytest.approx(3.0)
        assert b.grad == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────────
# 5. Power backward
# ──────────────────────────────────────────────────────────────

class TestPowerBackward:
    def test_square_backward(self):
        a = Value(3.0)
        out = a ** 2
        out.grad = 1.0
        out._backward()
        # d(a**2)/da = 2*a = 6.0
        assert a.grad == pytest.approx(6.0)

    def test_cube_backward(self):
        a = Value(2.0)
        out = a ** 3
        out.grad = 1.0
        out._backward()
        # d(a**3)/da = 3*a**2 = 12.0
        assert a.grad == pytest.approx(12.0)

    def test_sqrt_backward(self):
        a = Value(9.0)
        out = a ** 0.5
        out.grad = 1.0
        out._backward()
        # d(a**0.5)/da = 0.5 * a**(-0.5) = 0.5/3 ≈ 0.1667
        assert a.grad == pytest.approx(0.5 * 9.0 ** -0.5)

    def test_reciprocal_backward(self):
        a = Value(2.0)
        out = a ** -1
        out.grad = 1.0
        out._backward()
        # d(a**-1)/da = -1 * a**(-2) = -0.25
        assert a.grad == pytest.approx(-0.25)

    def test_pow_backward_scaled_seed(self):
        a = Value(3.0)
        out = a ** 2
        out.grad = 3.0
        out._backward()
        # 3.0 * 2*3 = 18.0
        assert a.grad == pytest.approx(18.0)


# ──────────────────────────────────────────────────────────────
# 6. Gradient accumulation (shared nodes)
# ──────────────────────────────────────────────────────────────

class TestGradientAccumulation:
    def test_add_self_accumulates(self):
        # a + a → grad should accumulate from both branches
        a = Value(3.0)
        out = a + a
        out.grad = 1.0
        out._backward()
        # d(a+a)/da = 1 + 1 = 2
        assert a.grad == pytest.approx(2.0)

    def test_mul_self_accumulates(self):
        # a * a → grad should accumulate from both branches
        a = Value(3.0)
        out = a * a
        out.grad = 1.0
        out._backward()
        # d(a*a)/da = a + a = 2*a = 6.0
        assert a.grad == pytest.approx(6.0)

    def test_multiple_calls_accumulate(self):
        # Calling _backward twice should ADD to existing .grad
        a = Value(2.0)
        b = Value(3.0)
        out = a + b
        out.grad = 1.0
        out._backward()
        out._backward()   # second call — gradients should double
        assert a.grad == pytest.approx(2.0)
        assert b.grad == pytest.approx(2.0)

    def test_subtraction_accumulation(self):
        # a - a → a used as both self and -other; net grad = 0
        # Sub is a + (-other), and neg wraps in a new Value so a is used once.
        # This is a structural property test — just ensure no crash.
        a = Value(5.0)
        b = Value(5.0)
        out = a - b
        out.grad = 1.0
        out._backward()
        # d(a-b)/da = 1, d(a-b)/db = -1 (via chain through negation + add)
        assert a.grad == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────
# 7. Two-hop chain: manual step-by-step
# ──────────────────────────────────────────────────────────────

class TestChainedBackward:
    def test_linear_two_hop(self):
        # z = w*x + b;  seed z.grad=1, walk backwards manually.
        w = Value(0.5,  label="w")
        x = Value(2.0,  label="x")
        b = Value(0.3,  label="b")
        wx  = w * x          # wx = 1.0
        out = wx + b         # out = 1.3

        # Step 1 — seed the output
        out.grad = 1.0
        # Step 2 — propagate through addition
        out._backward()      # wx.grad += 1.0, b.grad += 1.0
        # Step 3 — propagate through multiplication
        wx._backward()       # w.grad += 1.0 * x.data = 2.0
                             # x.grad += 1.0 * w.data = 0.5

        assert b.grad  == pytest.approx(1.0)
        assert w.grad  == pytest.approx(2.0)
        assert x.grad  == pytest.approx(0.5)

    def test_quadratic_two_hop(self):
        # y = x**2 + x;  at x=3 → dy/dx = 2*3 + 1 = 7
        x  = Value(3.0)
        x2 = x ** 2         # x**2
        y  = x2 + x         # x**2 + x (x is shared — accumulation test too)

        y.grad = 1.0
        y._backward()       # x2.grad += 1, x.grad += 1 (from + branch)
        x2._backward()      # x.grad += 1 * 2*3 = 6  (from ** branch)

        # x.grad = 1 (from +) + 6 (from **) = 7 ✓
        assert x.grad == pytest.approx(7.0)


# ──────────────────────────────────────────────────────────────
# 8. zero_grad()
# ──────────────────────────────────────────────────────────────

class TestZeroGrad:
    def test_zero_grad_on_leaf(self):
        v = Value(3.0)
        v.grad = 5.0
        v.zero_grad()
        assert v.grad == 0.0

    def test_zero_grad_clears_parent(self):
        a = Value(2.0)
        b = Value(3.0)
        out = a * b
        a.grad = 9.0
        b.grad = 4.0
        out.grad = 1.0
        out.zero_grad()
        assert out.grad == 0.0
        assert a.grad   == 0.0
        assert b.grad   == 0.0

    def test_zero_grad_deep_graph(self):
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0)
        d = (a + b) * c
        a.grad = 1.0; b.grad = 2.0; c.grad = 3.0; d.grad = 4.0
        d.zero_grad()
        assert a.grad == 0.0
        assert b.grad == 0.0
        assert c.grad == 0.0
        assert d.grad == 0.0

    def test_zero_grad_then_backward_still_works(self):
        w = Value(0.5)
        x = Value(2.0)
        out = w * x
        out.grad = 1.0
        out._backward()
        assert w.grad == pytest.approx(2.0)
        # now reset and redo
        out.zero_grad()
        assert w.grad == 0.0
        out.grad = 1.0
        out._backward()
        assert w.grad == pytest.approx(2.0)

    def test_zero_grad_on_fresh_graph_is_noop(self):
        a = Value(1.0)
        b = Value(2.0)
        out = a + b
        out.zero_grad()   # everything already 0 — should not raise
        assert a.grad == 0.0
        assert b.grad == 0.0
        assert out.grad == 0.0
