"""
tests/test_day02.py
-------------------
Tests for Day 2 — arithmetic operators and computation-graph metadata.

Run with:
    python -m pytest tests/ -v
"""

import math
import pytest
from micrograd import Value


# ──────────────────────────────────────────────────────────────
# 1. Addition
# ──────────────────────────────────────────────────────────────

class TestAddition:
    def test_value_plus_value(self):
        assert (Value(2.0) + Value(3.0)).data == 5.0

    def test_value_plus_scalar(self):
        assert (Value(2.0) + 3.0).data == 5.0

    def test_scalar_plus_value(self):
        # __radd__: plain float on the left
        assert (3.0 + Value(2.0)).data == 5.0

    def test_int_plus_value(self):
        assert (3 + Value(2.0)).data == 5.0

    def test_add_negative(self):
        assert (Value(5.0) + Value(-3.0)).data == 2.0

    def test_add_zero(self):
        assert (Value(4.0) + Value(0.0)).data == 4.0

    def test_add_returns_value(self):
        result = Value(1.0) + Value(2.0)
        assert isinstance(result, Value)

    def test_add_op_recorded(self):
        result = Value(1.0) + Value(2.0)
        assert result._op == "+"

    def test_add_prev_recorded(self):
        a, b = Value(1.0), Value(2.0)
        result = a + b
        assert a in result._prev
        assert b in result._prev

    def test_chained_add(self):
        a = Value(1.0) + Value(2.0) + Value(3.0)
        assert a.data == 6.0


# ──────────────────────────────────────────────────────────────
# 2. Multiplication
# ──────────────────────────────────────────────────────────────

class TestMultiplication:
    def test_value_times_value(self):
        assert (Value(2.0) * Value(3.0)).data == 6.0

    def test_value_times_scalar(self):
        assert (Value(2.0) * 3.0).data == 6.0

    def test_scalar_times_value(self):
        assert (3.0 * Value(2.0)).data == 6.0

    def test_mul_by_zero(self):
        assert (Value(99.0) * Value(0.0)).data == 0.0

    def test_mul_by_negative(self):
        assert (Value(3.0) * Value(-2.0)).data == -6.0

    def test_mul_returns_value(self):
        result = Value(2.0) * Value(3.0)
        assert isinstance(result, Value)

    def test_mul_op_recorded(self):
        result = Value(2.0) * Value(3.0)
        assert result._op == "*"

    def test_mul_prev_recorded(self):
        a, b = Value(2.0), Value(3.0)
        result = a * b
        assert a in result._prev
        assert b in result._prev

    def test_mul_karpathy_example(self):
        # The exact example from the task brief
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0


# ──────────────────────────────────────────────────────────────
# 3. Subtraction
# ──────────────────────────────────────────────────────────────

class TestSubtraction:
    def test_value_minus_value(self):
        assert (Value(5.0) - Value(3.0)).data == 2.0

    def test_value_minus_scalar(self):
        assert (Value(5.0) - 3.0).data == 2.0

    def test_scalar_minus_value(self):
        assert (10.0 - Value(3.0)).data == 7.0

    def test_sub_negative_result(self):
        assert (Value(2.0) - Value(5.0)).data == -3.0

    def test_sub_returns_value(self):
        assert isinstance(Value(5.0) - Value(3.0), Value)

    def test_sub_self(self):
        a = Value(7.0)
        assert (a - a).data == 0.0


# ──────────────────────────────────────────────────────────────
# 4. Division
# ──────────────────────────────────────────────────────────────

class TestDivision:
    def test_value_div_value(self):
        assert (Value(6.0) / Value(2.0)).data == 3.0

    def test_value_div_scalar(self):
        assert (Value(9.0) / 3.0).data == 3.0

    def test_scalar_div_value(self):
        result = 12.0 / Value(4.0)
        assert result.data == pytest.approx(3.0)

    def test_div_returns_value(self):
        assert isinstance(Value(6.0) / Value(2.0), Value)

    def test_div_fractional(self):
        assert (Value(1.0) / Value(3.0)).data == pytest.approx(1 / 3)

    def test_div_by_zero_raises(self):
        with pytest.raises((ZeroDivisionError, ValueError)):
            _ = Value(1.0) / Value(0.0)


# ──────────────────────────────────────────────────────────────
# 5. Power
# ──────────────────────────────────────────────────────────────

class TestPower:
    def test_square(self):
        assert (Value(3.0) ** 2).data == 9.0

    def test_cube(self):
        assert (Value(2.0) ** 3).data == 8.0

    def test_fractional_exponent(self):
        assert (Value(9.0) ** 0.5).data == pytest.approx(3.0)

    def test_negative_exponent(self):
        assert (Value(2.0) ** -1).data == pytest.approx(0.5)

    def test_pow_returns_value(self):
        assert isinstance(Value(2.0) ** 3, Value)

    def test_pow_op_recorded(self):
        result = Value(2.0) ** 3
        assert result._op == "**3"

    def test_pow_prev_recorded(self):
        a = Value(2.0)
        result = a ** 3
        assert a in result._prev

    def test_pow_invalid_exponent_raises(self):
        with pytest.raises(TypeError):
            _ = Value(2.0) ** "three"  # type: ignore


# ──────────────────────────────────────────────────────────────
# 6. Mixed arithmetic expressions
# ──────────────────────────────────────────────────────────────

class TestMixedExpressions:
    def test_linear_combo(self):
        # 2*x + 1 where x = 3  ->  7
        x = Value(3.0)
        result = Value(2.0) * x + Value(1.0)
        assert result.data == 7.0

    def test_quadratic(self):
        # x^2 + 2x + 1 where x = 4  ->  25
        x = Value(4.0)
        result = x ** 2 + Value(2.0) * x + Value(1.0)
        assert result.data == 25.0

    def test_neuron_like_expression(self):
        # w*x + b  ->  typical linear neuron
        w = Value(0.5, label="w")
        x = Value(2.0, label="x")
        b = Value(1.0, label="b")
        out = w * x + b
        assert out.data == pytest.approx(2.0)

    def test_scalar_on_both_sides(self):
        # 3 + Value(2) * 4 == 11
        result = 3 + Value(2.0) * 4
        assert result.data == 11.0

    def test_graph_depth(self):
        # Make sure _prev chains correctly through 3 ops
        a = Value(1.0)
        b = Value(2.0)
        c = Value(3.0)
        d = (a + b) * c   # d._prev should contain (a+b) and c
        assert c in d._prev


# ──────────────────────────────────────────────────────────────
# 7. Graph metadata on leaf nodes
# ──────────────────────────────────────────────────────────────

class TestLeafMetadata:
    def test_leaf_op_empty(self):
        v = Value(5.0)
        assert v._op == ""

    def test_leaf_prev_empty(self):
        v = Value(5.0)
        assert len(v._prev) == 0
