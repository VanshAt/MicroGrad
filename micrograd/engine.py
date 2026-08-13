"""
micrograd/engine.py
--------------------
Day 1 — The foundation: wrapping a scalar number in a Python object.

Philosophy
----------
Every number in this autograd engine is a `Value`.
Raw floats and ints are "dead" — they carry no history, no identity, no
future.  A `Value` gives a number a *name*, a *type guarantee*, and a
*place to grow* as we add math, gradients, and backprop in later days.

Think of it like this:
    - A plain number  →  an unnamed actor with no script
    - A Value object  →  an actor with a name, a role, and a backstory

Today we only write the actor's name on the door.
Tomorrow we hand them lines.
"""

from __future__ import annotations

import math
from typing import Union


# Supported scalar types that can be wrapped inside a Value.
_SCALAR = Union[int, float]


class Value:
    """
    A scalar value node in a future computational graph.

    Attributes
    ----------
    data : float
        The actual numeric scalar this node wraps.
    label : str
        An optional human-readable name — extremely helpful when
        visualising the computation graph later.

    Notes
    -----
    Day 1 scope  — only wrapping + identity.
    Days ahead   — arithmetic, gradients, backward pass.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data: _SCALAR, label: str = "") -> None:
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
    # Future hook placeholders
    # ------------------------------------------------------------------
    # These stubs document the road ahead without implementing anything.
    # They'll be fleshed out in later days.

    # Day 2 additions → __add__, __mul__, __sub__, __truediv__, __pow__
    # Day 3 additions → .grad attribute, ._backward callable, ._prev set
    # Day 4 additions → .backward() full backpropagation
    # Day 5 additions → activation functions: tanh, relu, sigmoid, exp
