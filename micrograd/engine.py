"""
micrograd/engine.py
--------------------
Day 2 — Arithmetic: Values can now add, subtract, multiply and divide.

Philosophy
----------
Every number in this autograd engine is a `Value`.
Raw floats and ints are "dead" — they carry no history, no identity, no
future.  A `Value` gives a number a name, a type guarantee, and a
place to grow as we add math, gradients, and backprop in later days.

Today we hand the actors their lines:
    - Arithmetic operators return new Value nodes.
    - Each result remembers *which operation* created it (_op).
    - Each result remembers *which parent nodes* fed into it (_prev).
    → These two fields are the skeleton of the computation graph that
      backpropagation will traverse in later days.

No gradients yet — that's Day 3/4.
"""

from __future__ import annotations

import math
from typing import Union


# Supported scalar types that can be wrapped inside a Value.
_SCALAR = Union[int, float]


class Value:
    """
    A scalar value node in a computational graph.

    Attributes
    ----------
    data : float
        The actual numeric scalar this node wraps.
    label : str
        An optional human-readable name — extremely helpful when
        visualising the computation graph later.
    _op : str
        The operation that *produced* this node (e.g. '+', '*').
        Empty string for leaf nodes created directly by the user.
    _prev : frozenset[Value]
        The parent Value nodes that were inputs to the operation.
        Empty for leaf nodes.

    Notes
    -----
    Day 1 scope  — wrapping + identity.
    Day 2 scope  — arithmetic operators + computation graph skeleton.
    Days ahead   — gradients, backward pass, activations.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: _SCALAR,
        label: str = "",
        _op: str = "",
        _prev: tuple["Value", ...] = (),
    ) -> None:
        """
        Wrap a scalar number inside a Value node.

        Parameters
        ----------
        data : int | float
            The numeric value to store.  Automatically promoted to float
            so every Value is consistently typed.
        label : str, optional
            A human-readable name for this node (e.g. "x", "weight",
            "loss").  Defaults to an empty string.
        _op : str, optional
            The operation that produced this node.  Set automatically by
            arithmetic operators — users should not pass this directly.
        _prev : tuple[Value, ...], optional
            Parent nodes that were inputs to this operation.  Set
            automatically — users should not pass this directly.

        Raises
        ------
        TypeError
            If `data` is not an int or float.

        Examples
        --------
        >>> v = Value(2.0)
        >>> v.data
        2.0

        >>> v = Value(3, label="x")
        >>> v.label
        'x'
        """
        if not isinstance(data, (int, float)):
            raise TypeError(
                f"Value.data must be a scalar number (int or float), "
                f"got {type(data).__name__!r} instead."
            )

        # Always store as float — keeps arithmetic consistent later.
        self.data: float = float(data)

        # A friendly name — invaluable when we draw computation graphs.
        self.label: str = label

        # Computation-graph metadata (populated by arithmetic operators).
        self._op: str = _op                          # e.g. '+', '*', '**'
        self._prev: frozenset["Value"] = frozenset(_prev)  # parent nodes

    # ------------------------------------------------------------------
    # Identity & display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise representation — shows label when available."""
        if self.label:
            return f"Value(data={self.data}, label={self.label!r})"
        return f"Value(data={self.data})"

    def __str__(self) -> str:
        """Human-friendly string — what you'd print in a notebook."""
        name = f"'{self.label}' " if self.label else ""
        return f"Value {name}= {self.data}"

    # ------------------------------------------------------------------
    # Numeric identity helpers
    # ------------------------------------------------------------------

    def __float__(self) -> float:
        """Allow float(Value(...)) to unwrap the scalar."""
        return self.data

    def __int__(self) -> int:
        """Allow int(Value(...)) to unwrap the scalar."""
        return int(self.data)

    def __abs__(self) -> "Value":
        """Return a new Value with the absolute value."""
        return Value(abs(self.data), label=f"|{self.label}|" if self.label else "")

    def __neg__(self) -> "Value":
        """Unary negation — returns -self as a new Value."""
        return Value(-self.data, label=f"-{self.label}" if self.label else "")

    def __pos__(self) -> "Value":
        """Unary positive — returns a copy."""
        return Value(self.data, label=self.label)

    # ------------------------------------------------------------------
    # Comparison operators (so we can sort / compare Values naturally)
    # ------------------------------------------------------------------

    # __hash__ must be defined explicitly because we defined __eq__.
    # Python nullifies __hash__ when __eq__ is overridden.
    # We use identity-based hashing (default object id) so that Value
    # nodes can be stored in sets / frozensets (used by _prev), while
    # __eq__ still compares .data values for user convenience.
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Value):
            return self.data == other.data
        if isinstance(other, (int, float)):
            return self.data == float(other)
        return NotImplemented

    def __lt__(self, other: "Value | float") -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data < other_val

    def __le__(self, other: "Value | float") -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data <= other_val

    def __gt__(self, other: "Value | float") -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data > other_val

    def __ge__(self, other: "Value | float") -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data >= other_val

    # ------------------------------------------------------------------
    # Utility / introspection helpers
    # ------------------------------------------------------------------

    def item(self) -> float:
        """
        Return the Python float inside this Value.

        Mirrors PyTorch's `.item()` — so later code feels familiar.
        """
        return self.data

    def is_finite(self) -> bool:
        """Return True if data is a finite number (not inf or nan)."""
        return math.isfinite(self.data)

    def is_nan(self) -> bool:
        """Return True if data is NaN."""
        return math.isnan(self.data)

    def named(self, label: str) -> "Value":
        """
        Return a *new* Value with the given label — fluent builder style.

        Lets you do: ``x = Value(3.14).named("pi")`` in one line.
        """
        return Value(self.data, label=label)

    # ------------------------------------------------------------------
    # Arithmetic operators  (Day 2)
    # ------------------------------------------------------------------
    # Design rules:
    #   1. Always return a *new* Value — Values are immutable nodes.
    #   2. Record _op and _prev on every result so the graph is complete.
    #   3. Accept plain int/float on either side via __radd__/__rmul__ etc.
    #      (Python calls the reflected dunder when the left operand can't
    #       handle the operation, e.g.  3.0 + Value(2.0).)

    def _coerce(self, other: object) -> "Value":
        """Wrap a raw scalar in a Value so arithmetic is uniform."""
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(float(other))
        return NotImplemented  # type: ignore[return-value]

    # ── Addition ──────────────────────────────────────────────────

    def __add__(self, other: object) -> "Value":
        """self + other"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Value(self.data + other.data, _op="+", _prev=(self, other))

    def __radd__(self, other: object) -> "Value":
        """other + self  (called when `other` is a plain number)"""
        return self.__add__(other)

    # ── Subtraction ───────────────────────────────────────────────

    def __sub__(self, other: object) -> "Value":
        """self - other  (implemented as self + (-other))"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: object) -> "Value":
        """other - self"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    # ── Multiplication ────────────────────────────────────────────

    def __mul__(self, other: object) -> "Value":
        """self * other"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return Value(self.data * other.data, _op="*", _prev=(self, other))

    def __rmul__(self, other: object) -> "Value":
        """other * self  (called when `other` is a plain number)"""
        return self.__mul__(other)

    # ── Division ──────────────────────────────────────────────────

    def __truediv__(self, other: object) -> "Value":
        """self / other  (implemented as self * other**-1)"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self * other ** -1

    def __rtruediv__(self, other: object) -> "Value":
        """other / self"""
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other * self ** -1

    # ── Power ─────────────────────────────────────────────────────

    def __pow__(self, exponent: _SCALAR) -> "Value":
        """
        self ** exponent

        Only supports scalar (int / float) exponents — Value exponents
        would need separate treatment during backprop and are not needed
        for a standard MLP.
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                f"Exponent must be int or float, got {type(exponent).__name__!r}."
            )
        return Value(self.data ** exponent, _op=f"**{exponent}", _prev=(self,))

    # ------------------------------------------------------------------
    # Future hook placeholders
    # ------------------------------------------------------------------
    # Day 3 additions → .grad attribute, ._backward callable
    # Day 4 additions → .backward() full backpropagation
    # Day 5 additions → activation functions: tanh, relu, sigmoid, exp
