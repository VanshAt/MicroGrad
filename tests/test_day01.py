"""
tests/test_day01.py
-------------------
Tests for Day 1 — Value class: scalar wrapping and identity.

Run with:
    python -m pytest tests/ -v
"""

import math
import pytest
from micrograd import Value


# ──────────────────────────────────────────────────────────────
# 1. Construction & .data attribute
# ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_float_stored_as_float(self):
        v = Value(2.0)
        assert v.data == 2.0
        assert isinstance(v.data, float)

    def test_int_promoted_to_float(self):
        v = Value(3)
        assert v.data == 3.0
        assert isinstance(v.data, float)

    def test_negative_value(self):
        v = Value(-7.5)
        assert v.data == -7.5

    def test_zero(self):
        v = Value(0)
        assert v.data == 0.0

    def test_large_float(self):
        v = Value(1e38)
        assert v.data == 1e38

    def test_label_default_empty(self):
        v = Value(1.0)
        assert v.label == ""

    def test_label_stored(self):
        v = Value(1.0, label="x")
        assert v.label == "x"

    def test_invalid_type_string_raises(self):
        with pytest.raises(TypeError, match="int or float"):
            Value("hello")

    def test_invalid_type_list_raises(self):
        with pytest.raises(TypeError):
            Value([1, 2, 3])

    def test_invalid_type_none_raises(self):
        with pytest.raises(TypeError):
            Value(None)


# ──────────────────────────────────────────────────────────────
# 2. Representation
# ──────────────────────────────────────────────────────────────

class TestRepresentation:
    def test_repr_no_label(self):
        v = Value(2.0)
        assert repr(v) == "Value(data=2.0)"

    def test_repr_with_label(self):
        v = Value(2.0, label="x")
        assert repr(v) == "Value(data=2.0, label='x')"

    def test_str_no_label(self):
        v = Value(5.0)
        assert "5.0" in str(v)

    def test_str_with_label(self):
        v = Value(5.0, label="y")
        assert "y" in str(v)
        assert "5.0" in str(v)


# ──────────────────────────────────────────────────────────────
# 3. Numeric identity helpers
# ──────────────────────────────────────────────────────────────

class TestNumericHelpers:
    def test_float_unwrap(self):
        assert float(Value(3.7)) == 3.7

    def test_int_unwrap(self):
        assert int(Value(3.9)) == 3

    def test_abs_positive(self):
        v = abs(Value(-4.0))
        assert v.data == 4.0
        assert isinstance(v, Value)

    def test_abs_negative(self):
        v = abs(Value(4.0))
        assert v.data == 4.0

    def test_neg(self):
        v = -Value(5.0)
        assert v.data == -5.0
        assert isinstance(v, Value)

    def test_pos(self):
        v = +Value(5.0)
        assert v.data == 5.0

    def test_item(self):
        assert Value(9.1).item() == 9.1

    def test_is_finite_normal(self):
        assert Value(3.14).is_finite() is True

    def test_is_finite_inf(self):
        assert Value(math.inf).is_finite() is False

    def test_is_nan(self):
        assert Value(math.nan).is_nan() is True

    def test_named(self):
        v = Value(2.0).named("bias")
        assert v.label == "bias"
        assert v.data == 2.0


# ──────────────────────────────────────────────────────────────
# 4. Comparison operators
# ──────────────────────────────────────────────────────────────

class TestComparisons:
    def test_eq_value_value(self):
        assert Value(3.0) == Value(3.0)

    def test_eq_value_float(self):
        assert Value(3.0) == 3.0

    def test_lt(self):
        assert Value(1.0) < Value(2.0)

    def test_le(self):
        assert Value(2.0) <= Value(2.0)

    def test_gt(self):
        assert Value(3.0) > Value(2.0)

    def test_ge(self):
        assert Value(3.0) >= Value(3.0)

    def test_sort_values(self):
        values = [Value(3.0), Value(1.0), Value(2.0)]
        sorted_vals = sorted(values)
        assert [v.data for v in sorted_vals] == [1.0, 2.0, 3.0]
