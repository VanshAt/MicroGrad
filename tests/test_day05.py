"""
tests/test_day05.py
--------------------
Tests for Day 5 — activation functions: exp, log, tanh, relu, sigmoid.

Test philosophy (same as earlier days):
  - Forward values are checked analytically.
  - Backward correctness is verified by calling .backward() and reading
    .grad, compared against closed-form derivatives.
  - A numerical gradient check (central finite differences) provides an
    independent, gold-standard verification for every activation.
  - Edge cases (boundary inputs, zero, saturation regions) are included
    to catch numerical issues early.

Run with:
    python -m pytest tests/ -v
"""

import math
import pytest
from micrograd import Value


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def numerical_grad(f, x: Value, h: float = 1e-5) -> float:
    """
    Central finite-difference approximation of df/dx at x.data.

        grad ≈ (f(x + h) - f(x - h)) / (2h)

    `f` maps a plain float → plain float.
    """
    xv = x.data
    return (f(xv + h) - f(xv - h)) / (2 * h)


def numgrad_check(f_value, x_val: float, tol: float = 1e-4) -> None:
    """
    Full numerical gradient check helper:
      1. Build the Value graph, run backward(), read x.grad.
      2. Compute the numerical gradient via central differences.
      3. Assert they agree to `tol` relative tolerance.
    """
    x = Value(x_val)
    out = f_value(x)
    out.backward()
    analytical = x.grad

    f_plain = lambda v: f_value(Value(v)).data
    numerical = numerical_grad(f_plain, x)

    assert analytical == pytest.approx(numerical, rel=tol), (
        f"analytical={analytical:.8f}, numerical={numerical:.8f}"
    )


# ──────────────────────────────────────────────────────────────────────
# 1. exp()
# ──────────────────────────────────────────────────────────────────────

class TestExp:
    # ── Forward ───────────────────────────────────────────────────────

    def test_exp_zero(self):
        """e^0 = 1"""
        assert Value(0.0).exp().data == pytest.approx(1.0)

    def test_exp_one(self):
        """e^1 = e"""
        assert Value(1.0).exp().data == pytest.approx(math.e)

    def test_exp_negative(self):
        """e^{-1} = 1/e"""
        assert Value(-1.0).exp().data == pytest.approx(1 / math.e)

    def test_exp_ln2(self):
        """e^{ln 2} = 2"""
        assert Value(math.log(2)).exp().data == pytest.approx(2.0)

    def test_exp_large_positive(self):
        """e^{10} should be finite and match math.exp."""
        assert Value(10.0).exp().data == pytest.approx(math.exp(10.0))

    def test_exp_preserves_op(self):
        assert Value(1.0).exp()._op == "exp"

    def test_exp_has_one_parent(self):
        x = Value(1.0)
        assert x in Value(1.0).exp()._prev or len(x.exp()._prev) == 1

    # ── Backward (closed-form) ─────────────────────────────────────────

    def test_exp_backward_at_zero(self):
        """d/dx e^x at x=0 = e^0 = 1."""
        x = Value(0.0)
        x.exp().backward()
        assert x.grad == pytest.approx(1.0)

    def test_exp_backward_at_one(self):
        """d/dx e^x at x=1 = e."""
        x = Value(1.0)
        x.exp().backward()
        assert x.grad == pytest.approx(math.e)

    def test_exp_backward_at_negative(self):
        """d/dx e^x at x=-2 = e^{-2}."""
        x = Value(-2.0)
        x.exp().backward()
        assert x.grad == pytest.approx(math.exp(-2.0))

    def test_exp_chained(self):
        """d/dx (e^x * 3) at x=0 = 3."""
        x = Value(0.0)
        out = x.exp() * Value(3.0)
        out.backward()
        assert x.grad == pytest.approx(3.0)

    def test_exp_squared_chain(self):
        """d/dx (e^x)^2 = 2*e^{2x}.  At x=1: 2e²."""
        x = Value(1.0)
        out = x.exp() ** 2
        out.backward()
        assert x.grad == pytest.approx(2 * math.exp(2.0))

    # ── Numerical gradient check ───────────────────────────────────────

    @pytest.mark.parametrize("xv", [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    def test_exp_numgrad(self, xv):
        numgrad_check(lambda x: x.exp(), xv)

    # ── Error conditions ───────────────────────────────────────────────

    def test_exp_overflow_raises(self):
        """Input large enough to overflow math.exp raises OverflowError.

        Note: 1e309 becomes float('inf') before reaching math.exp, so
        we use 710.0 — the smallest integer that reliably overflows on
        all platforms (math.exp(709) is finite; math.exp(710) is not).
        """
        with pytest.raises(OverflowError):
            Value(710.0).exp()


# ──────────────────────────────────────────────────────────────────────
# 2. log()
# ──────────────────────────────────────────────────────────────────────

class TestLog:
    # ── Forward ───────────────────────────────────────────────────────

    def test_log_of_one(self):
        """ln(1) = 0"""
        assert Value(1.0).log().data == pytest.approx(0.0)

    def test_log_of_e(self):
        """ln(e) = 1"""
        assert Value(math.e).log().data == pytest.approx(1.0)

    def test_log_of_exp(self):
        """ln(e^x) = x — round-trips."""
        for xv in [0.5, 1.0, 2.5, 10.0]:
            assert Value(math.exp(xv)).log().data == pytest.approx(xv, rel=1e-9)

    def test_log_small_positive(self):
        """ln(0.1) matches math.log."""
        assert Value(0.1).log().data == pytest.approx(math.log(0.1))

    def test_log_preserves_op(self):
        assert Value(1.0).log()._op == "log"

    # ── Backward ──────────────────────────────────────────────────────

    def test_log_backward_at_one(self):
        """d/dx ln(x) at x=1 = 1/1 = 1."""
        x = Value(1.0)
        x.log().backward()
        assert x.grad == pytest.approx(1.0)

    def test_log_backward_at_two(self):
        """d/dx ln(x) at x=2 = 0.5."""
        x = Value(2.0)
        x.log().backward()
        assert x.grad == pytest.approx(0.5)

    def test_log_backward_at_e(self):
        """d/dx ln(x) at x=e = 1/e."""
        x = Value(math.e)
        x.log().backward()
        assert x.grad == pytest.approx(1 / math.e)

    def test_log_chained_with_exp(self):
        """d/dx ln(e^x) = 1 — log and exp cancel."""
        x = Value(2.0)
        out = x.exp().log()
        out.backward()
        assert x.grad == pytest.approx(1.0, rel=1e-5)

    # ── Numerical gradient check ───────────────────────────────────────

    @pytest.mark.parametrize("xv", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_log_numgrad(self, xv):
        numgrad_check(lambda x: x.log(), xv)

    # ── Error conditions ───────────────────────────────────────────────

    def test_log_of_zero_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Value(0.0).log()

    def test_log_of_negative_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Value(-3.0).log()


# ──────────────────────────────────────────────────────────────────────
# 3. tanh()
# ──────────────────────────────────────────────────────────────────────

class TestTanh:
    # ── Forward ───────────────────────────────────────────────────────

    def test_tanh_zero(self):
        """tanh(0) = 0"""
        assert Value(0.0).tanh().data == pytest.approx(0.0)

    def test_tanh_one(self):
        """tanh(1) matches math.tanh(1)."""
        assert Value(1.0).tanh().data == pytest.approx(math.tanh(1.0))

    def test_tanh_negative(self):
        """tanh is an odd function: tanh(-x) = -tanh(x)."""
        for xv in [0.5, 1.0, 2.0]:
            assert Value(-xv).tanh().data == pytest.approx(-math.tanh(xv))

    def test_tanh_saturates_positive(self):
        """tanh → +1 for large positive inputs."""
        assert Value(100.0).tanh().data == pytest.approx(1.0, abs=1e-10)

    def test_tanh_saturates_negative(self):
        """tanh → -1 for large negative inputs."""
        assert Value(-100.0).tanh().data == pytest.approx(-1.0, abs=1e-10)

    def test_tanh_output_range(self):
        """tanh output is always in (-1, 1)."""
        for xv in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            t = Value(xv).tanh().data
            assert -1.0 < t < 1.0 or abs(t) == pytest.approx(1.0, abs=1e-9)

    def test_tanh_preserves_op(self):
        assert Value(1.0).tanh()._op == "tanh"

    # ── Backward ──────────────────────────────────────────────────────

    def test_tanh_backward_at_zero(self):
        """d/dx tanh(x) at x=0 = 1 - tanh²(0) = 1."""
        x = Value(0.0)
        x.tanh().backward()
        assert x.grad == pytest.approx(1.0)

    def test_tanh_backward_at_one(self):
        """d/dx tanh(x) at x=1 = 1 - tanh²(1)."""
        t = math.tanh(1.0)
        x = Value(1.0)
        x.tanh().backward()
        assert x.grad == pytest.approx(1.0 - t ** 2)

    def test_tanh_backward_saturation(self):
        """Gradient ≈ 0 when tanh saturates (vanishing gradient)."""
        x = Value(10.0)
        x.tanh().backward()
        assert x.grad == pytest.approx(0.0, abs=1e-8)

    def test_tanh_backward_chained(self):
        """d/dx 2*tanh(x) at x=0 = 2."""
        x = Value(0.0)
        out = Value(2.0) * x.tanh()
        out.backward()
        assert x.grad == pytest.approx(2.0)

    def test_tanh_karpathy_neuron(self):
        """
        Replicates the exact forward+backward from Karpathy's lecture.

        x1w1 + x2w2 + b ≈ 0.8814,  tanh ≈ 0.7071
        d(tanh)/d(n) = 1 - 0.7071² ≈ 0.5
        """
        x1 = Value(2.0,  label="x1")
        x2 = Value(0.0,  label="x2")
        w1 = Value(-3.0, label="w1")
        w2 = Value(1.0,  label="w2")
        b  = Value(6.8813735870195432, label="b")

        n  = x1 * w1 + x2 * w2 + b
        o  = n.tanh()
        o.backward()

        # d(o)/d(x1) = w1 * d(tanh)/d(n),  d(tanh)/d(n) ≈ 0.5
        dt_dn = 1.0 - o.data ** 2
        assert x1.grad == pytest.approx(w1.data * dt_dn, rel=1e-5)
        assert x2.grad == pytest.approx(w2.data * dt_dn, rel=1e-5)
        assert w1.grad == pytest.approx(x1.data * dt_dn, rel=1e-5)
        assert w2.grad == pytest.approx(x2.data * dt_dn, rel=1e-5)

    # ── Numerical gradient check ───────────────────────────────────────

    @pytest.mark.parametrize("xv", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_tanh_numgrad(self, xv):
        numgrad_check(lambda x: x.tanh(), xv)


# ──────────────────────────────────────────────────────────────────────
# 4. relu()
# ──────────────────────────────────────────────────────────────────────

class TestRelu:
    # ── Forward ───────────────────────────────────────────────────────

    def test_relu_positive(self):
        """relu(x) = x for x > 0."""
        assert Value(3.0).relu().data == pytest.approx(3.0)

    def test_relu_negative(self):
        """relu(x) = 0 for x < 0."""
        assert Value(-5.0).relu().data == pytest.approx(0.0)

    def test_relu_zero(self):
        """relu(0) = 0 (boundary)."""
        assert Value(0.0).relu().data == pytest.approx(0.0)

    def test_relu_large_positive(self):
        assert Value(1000.0).relu().data == pytest.approx(1000.0)

    def test_relu_large_negative(self):
        assert Value(-1000.0).relu().data == pytest.approx(0.0)

    def test_relu_float_small(self):
        assert Value(1e-10).relu().data == pytest.approx(1e-10)

    def test_relu_output_non_negative(self):
        for xv in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert Value(xv).relu().data >= 0.0

    def test_relu_preserves_op(self):
        assert Value(1.0).relu()._op == "relu"

    # ── Backward ──────────────────────────────────────────────────────

    def test_relu_backward_positive(self):
        """d/dx relu(x) = 1 for x > 0."""
        x = Value(3.0)
        x.relu().backward()
        assert x.grad == pytest.approx(1.0)

    def test_relu_backward_negative(self):
        """d/dx relu(x) = 0 for x < 0 (dead neuron)."""
        x = Value(-3.0)
        x.relu().backward()
        assert x.grad == pytest.approx(0.0)

    def test_relu_backward_zero(self):
        """Sub-gradient at 0 is defined as 0 (PyTorch convention)."""
        x = Value(0.0)
        x.relu().backward()
        assert x.grad == pytest.approx(0.0)

    def test_relu_backward_chained_with_mul(self):
        """d/dx 5*relu(x) at x=2 = 5."""
        x = Value(2.0)
        out = Value(5.0) * x.relu()
        out.backward()
        assert x.grad == pytest.approx(5.0)

    def test_relu_backward_dead_neuron_no_propagation(self):
        """For x<0 gradient should NOT propagate to upstream nodes."""
        w = Value(2.0, label="w")
        x = Value(-3.0, label="x")
        out = (w * x).relu()   # pre-activation = -6, dead neuron
        out.backward()
        assert w.grad == pytest.approx(0.0)
        assert x.grad == pytest.approx(0.0)

    def test_relu_backward_active_neuron_propagates(self):
        """For x>0 gradient should propagate normally."""
        w = Value(3.0, label="w")
        x = Value(2.0, label="x")
        out = (w * x).relu()   # pre-activation = 6, active
        out.backward()
        # d(relu(w*x))/dw = x (since relu is identity here)
        assert w.grad == pytest.approx(x.data)
        assert x.grad == pytest.approx(w.data)

    # ── Numerical gradient check ───────────────────────────────────────

    @pytest.mark.parametrize("xv", [-3.0, -1.0, 0.5, 1.0, 3.0])
    def test_relu_numgrad(self, xv):
        # Skip x=0 for finite-diff (not differentiable there)
        numgrad_check(lambda x: x.relu(), xv)

    # ── Composition ───────────────────────────────────────────────────

    def test_double_relu_identity(self):
        """relu(relu(x)) = relu(x) for all x."""
        for xv in [-2.0, 0.0, 2.0]:
            expected = max(0.0, xv)
            assert Value(xv).relu().relu().data == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────
# 5. sigmoid()
# ──────────────────────────────────────────────────────────────────────

class TestSigmoid:
    # ── Forward ───────────────────────────────────────────────────────

    def test_sigmoid_zero(self):
        """sigmoid(0) = 0.5"""
        assert Value(0.0).sigmoid().data == pytest.approx(0.5)

    def test_sigmoid_one(self):
        """sigmoid(1) = 1/(1+e^{-1})."""
        expected = 1.0 / (1.0 + math.exp(-1.0))
        assert Value(1.0).sigmoid().data == pytest.approx(expected)

    def test_sigmoid_negative_one(self):
        """sigmoid(-1) = 1 - sigmoid(1)."""
        s1 = Value(1.0).sigmoid().data
        s_neg1 = Value(-1.0).sigmoid().data
        assert s_neg1 == pytest.approx(1.0 - s1, rel=1e-9)

    def test_sigmoid_symmetry(self):
        """sigmoid(-x) = 1 - sigmoid(x) for any x."""
        for xv in [0.5, 1.0, 2.0, 5.0]:
            s_pos = Value(xv).sigmoid().data
            s_neg = Value(-xv).sigmoid().data
            assert s_pos + s_neg == pytest.approx(1.0, abs=1e-12)

    def test_sigmoid_saturates_positive(self):
        """sigmoid → 1 for large positive inputs."""
        assert Value(100.0).sigmoid().data == pytest.approx(1.0, abs=1e-10)

    def test_sigmoid_saturates_negative(self):
        """sigmoid → 0 for large negative inputs."""
        assert Value(-100.0).sigmoid().data == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_output_range(self):
        """sigmoid output is always in (0, 1)."""
        for xv in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            s = Value(xv).sigmoid().data
            assert 0.0 < s < 1.0

    def test_sigmoid_preserves_op(self):
        assert Value(1.0).sigmoid()._op == "sigmoid"

    # ── Backward ──────────────────────────────────────────────────────

    def test_sigmoid_backward_at_zero(self):
        """d/dx sigmoid at x=0 = 0.5*(1-0.5) = 0.25."""
        x = Value(0.0)
        x.sigmoid().backward()
        assert x.grad == pytest.approx(0.25)

    def test_sigmoid_backward_at_one(self):
        """d/dx sigmoid at x=1 = s*(1-s)."""
        s = 1.0 / (1.0 + math.exp(-1.0))
        x = Value(1.0)
        x.sigmoid().backward()
        assert x.grad == pytest.approx(s * (1.0 - s))

    def test_sigmoid_backward_saturation(self):
        """Gradient ≈ 0 at saturation (vanishing gradient)."""
        x = Value(10.0)
        x.sigmoid().backward()
        assert x.grad == pytest.approx(0.0, abs=1e-4)

    def test_sigmoid_backward_chained(self):
        """d/dx 4*sigmoid(x) at x=0 = 4*0.25 = 1."""
        x = Value(0.0)
        out = Value(4.0) * x.sigmoid()
        out.backward()
        assert x.grad == pytest.approx(1.0)

    # ── Numerical gradient check ───────────────────────────────────────

    @pytest.mark.parametrize("xv", [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    def test_sigmoid_numgrad(self, xv):
        numgrad_check(lambda x: x.sigmoid(), xv)


# ──────────────────────────────────────────────────────────────────────
# 6. Cross-activation compositions & graph correctness
# ──────────────────────────────────────────────────────────────────────

class TestCompositions:
    def test_exp_log_identity(self):
        """exp(log(x)) = x — round-trip."""
        for xv in [0.5, 1.0, 2.0, 5.0]:
            result = Value(xv).log().exp().data
            assert result == pytest.approx(xv, rel=1e-9)

    def test_log_exp_identity_grad(self):
        """d/dx log(exp(x)) = 1 for any x."""
        for xv in [-1.0, 0.0, 1.0, 2.0]:
            x = Value(xv)
            x.log().exp() if False else x.exp().log()
            out = x.exp().log()
            out.backward()
            assert x.grad == pytest.approx(1.0, rel=1e-5)

    def test_sigmoid_via_tanh_relation(self):
        """
        sigmoid(x) = (tanh(x/2) + 1) / 2 — mathematical identity.
        Both fused ops must agree on the forward value.
        """
        for xv in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            s = Value(xv).sigmoid().data
            t = (Value(xv / 2.0).tanh().data + 1.0) / 2.0
            assert s == pytest.approx(t, rel=1e-9)

    def test_relu_composed_with_tanh(self):
        """relu(tanh(x)) — forward and backward consistency."""
        x = Value(1.0)
        out = x.tanh().relu()
        out.backward()
        t = math.tanh(1.0)
        # tanh(1) > 0 so relu is identity; d/dx = d(tanh)/dx = 1-t²
        assert x.grad == pytest.approx(1.0 - t ** 2)

    def test_relu_dead_when_tanh_negative(self):
        """For x very negative, tanh < 0 so relu kills the gradient."""
        x = Value(-5.0)
        out = x.tanh().relu()   # tanh(-5) ≈ -1, relu → 0
        out.backward()
        assert x.grad == pytest.approx(0.0)

    def test_all_activations_graph_op(self):
        """Each activation produces a node with a distinct _op string."""
        x = Value(0.5)
        assert x.exp()._op == "exp"
        assert x.log()._op == "log"
        assert x.tanh()._op == "tanh"
        assert x.relu()._op == "relu"
        assert x.sigmoid()._op == "sigmoid"

    def test_activations_do_not_mutate_input(self):
        """Applying an activation must not modify self.data or self.grad."""
        x = Value(1.5)
        original_data = x.data
        original_grad = x.grad
        x.exp(); x.log(); x.tanh(); x.relu(); x.sigmoid()
        assert x.data == original_data
        assert x.grad == original_grad

    def test_neuron_with_tanh_full_backward(self):
        """
        Full single-neuron forward + backward through tanh.
        Mirrors Karpathy's lecture example — all 5 leaf grads verified.
        """
        x1 = Value(2.0,  label="x1")
        x2 = Value(0.0,  label="x2")
        w1 = Value(-3.0, label="w1")
        w2 = Value(1.0,  label="w2")
        b  = Value(6.8813735870195432, label="b")

        n = x1 * w1 + x2 * w2 + b
        o = n.tanh()
        o.backward()

        dt = 1.0 - o.data ** 2  # local gradient of tanh

        assert x1.grad == pytest.approx(w1.data * dt, rel=1e-5)
        assert x2.grad == pytest.approx(w2.data * dt, rel=1e-5)
        assert w1.grad == pytest.approx(x1.data * dt, rel=1e-5)
        assert w2.grad == pytest.approx(x2.data * dt, rel=1e-5)
        assert b.grad  == pytest.approx(dt, rel=1e-5)

    def test_neuron_with_relu_full_backward(self):
        """Single-neuron forward + backward through relu (active neuron)."""
        w = Value(0.5,  label="w")
        x = Value(2.0,  label="x")
        b = Value(0.3,  label="b")

        pre = w * x + b   # 1.3 > 0 → relu is identity
        o = pre.relu()
        o.backward()

        assert w.grad == pytest.approx(x.data)   # ∂o/∂w = x
        assert x.grad == pytest.approx(w.data)   # ∂o/∂x = w
        assert b.grad == pytest.approx(1.0)       # ∂o/∂b = 1

    def test_neuron_with_sigmoid_full_backward(self):
        """Single-neuron forward + backward through sigmoid."""
        w = Value(0.5,  label="w")
        x = Value(2.0,  label="x")
        b = Value(0.1,  label="b")

        pre = w * x + b
        o   = pre.sigmoid()
        o.backward()

        s    = o.data
        dsdn = s * (1.0 - s)   # local gradient of sigmoid

        assert w.grad == pytest.approx(x.data * dsdn, rel=1e-6)
        assert x.grad == pytest.approx(w.data * dsdn, rel=1e-6)
        assert b.grad == pytest.approx(dsdn, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────
# 7. zero_grad() compatibility with activations
# ──────────────────────────────────────────────────────────────────────

class TestZeroGradWithActivations:
    @pytest.mark.parametrize("activation", ["exp", "tanh", "relu", "sigmoid"])
    def test_zero_grad_resets_activation_grad(self, activation):
        """zero_grad() + backward() gives clean grads for activation nodes."""
        x = Value(1.0)
        out = getattr(x, activation)()

        out.backward()
        first_grad = x.grad

        out.zero_grad()
        out.backward()

        assert x.grad == pytest.approx(first_grad)

    def test_log_zero_grad_reset(self):
        x = Value(2.0)
        out = x.log()
        out.backward()
        first_grad = x.grad
        out.zero_grad()
        out.backward()
        assert x.grad == pytest.approx(first_grad)
