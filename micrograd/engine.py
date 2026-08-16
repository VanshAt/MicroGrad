"""
micrograd/engine.py
--------------------
Day 4 — `.backward()`: full reverse-mode autodiff in one call.

Philosophy
----------
Every number in this autograd engine is a `Value`.
Raw floats and ints are "dead" — they carry no history, no identity, no
future.  A `Value` gives a number a name, a type guarantee, and a
place to grow as we add math, gradients, and backprop in later days.

What's new in Day 4:
    - `.backward()` — seeds self.grad = 1.0, performs a topological
      sort of the entire computation graph, then calls every node's
      `._backward()` closure in *reverse* topological order so that by
      the time a node's `._backward()` fires, its own `.grad` has
      already been fully accumulated from all its consumers.

What was added previously:
    Day 3:
    - `.grad` attribute — accumulated gradient ∂L/∂self, starts at 0.0.
    - `._backward` callable — a zero-arg closure attached to every
      operation node.  When called it propagates *this node's* gradient
      one step backwards to its immediate parents using the local
      chain-rule formula for that specific operation.
    - `zero_grad()` — resets .grad to 0.0 on this node and every
      ancestor.  Like PyTorch's optimizer.zero_grad().

Note on accumulation (+=, not =)
---------------------------------
Gradients are *summed*, not replaced, because the same Value node can
appear multiple times in an expression (e.g. x*x).  Each path through
the graph contributes its own partial derivative and they must all be
added together (multivariate chain rule).

Topological order
-----------------
A computation graph is a DAG (directed acyclic graph): edges point
from parent (input) to child (output).  Backprop must visit nodes in
reverse — outputs before inputs — so that when we call a node's
._backward(), the downstream gradients that feed into it are already
fully computed.  A DFS post-order traversal produces exactly this
ordering (children before their dependants), and reversing it gives us
the correct backprop order.
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

    grad : float
        Accumulated gradient ∂L/∂self.  Starts at 0.0 and is filled in
        by calling `._backward()` on descendant nodes (or `backward()`
        on the root, which is Day 4's job).
    _backward : Callable[[], None]
        Zero-argument closure that propagates *this* node's `.grad` one
        step backwards to its immediate parents.  Leaf nodes use a no-op
        lambda.  Set automatically by arithmetic operators.

    Notes
    -----
    Day 1 scope  — wrapping + identity.
    Day 2 scope  — arithmetic operators + computation graph skeleton.
    Day 3 scope  — .grad attribute + local ._backward closures + zero_grad().
    Day 4 scope  — .backward() full reverse-mode autodiff.
    Days ahead   — activation functions (tanh, relu, sigmoid, exp).
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

        # ── Day 3 additions ──────────────────────────────────────────────
        # Accumulated gradient ∂L/∂self.  Starts at zero; backprop fills it.
        self.grad: float = 0.0

        # Local chain-rule step — propagates *this* node's .grad one hop
        # backwards to its parents.  Leaf nodes use the no-op lambda.
        # Arithmetic operators overwrite this with their own closure.
        self._backward = lambda: None

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
        """Unary negation — returns -self as a proper graph node.

        Chain rule: d(-a)/da = -1
        """
        out = Value(-self.data, _op="neg", _prev=(self,),
                    label=f"-{self.label}" if self.label else "")

        def _backward() -> None:
            self.grad += out.grad * -1.0

        out._backward = _backward
        return out

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
        out = Value(self.data + other.data, _op="+", _prev=(self, other))

        # Chain rule: d(a+b)/da = 1,  d(a+b)/db = 1
        # We capture `self` and `other` by closing over them.
        def _backward() -> None:
            self.grad  += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

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
        out = Value(self.data * other.data, _op="*", _prev=(self, other))

        # Chain rule: d(a*b)/da = b,  d(a*b)/db = a
        def _backward() -> None:
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

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
        out = Value(self.data ** exponent, _op=f"**{exponent}", _prev=(self,))

        # Chain rule: d(a**n)/da = n * a**(n-1)
        def _backward() -> None:
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Gradient utilities  (Day 3 & 4)
    # ------------------------------------------------------------------

    def _topo_sort(self) -> list["Value"]:
        """
        Return all nodes reachable from `self` in topological order.

        The list is ordered so that every node appears *after* all of
        its parents (i.e. DFS post-order over the ``_prev`` edges).  To
        get the correct backprop order, reverse this list.

        Shared by `zero_grad()` and `backward()` so the traversal logic
        lives in exactly one place.
        """
        visited: set["Value"] = set()
        topo: list["Value"] = []

        def _build(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    _build(parent)
                topo.append(node)

        _build(self)
        return topo

    def zero_grad(self) -> None:
        """
        Reset `.grad` to 0.0 on this node and every ancestor in the graph.

        Walks the full computation graph (topological order, ancestors
        first) so every parameter's gradient is cleared before a new
        forward + backward pass.

        Mirrors PyTorch's ``optimizer.zero_grad()``.

        Examples
        --------
        >>> w = Value(0.5, label="w")
        >>> x = Value(2.0, label="x")
        >>> out = w * x
        >>> out.grad = 1.0
        >>> out._backward()
        >>> w.grad
        2.0
        >>> out.zero_grad()  # reset the whole graph
        >>> w.grad
        0.0
        """
        for node in self._topo_sort():
            node.grad = 0.0

    def backward(self) -> None:
        """
        Run full reverse-mode automatic differentiation from this node.

        After this call every ancestor's ``.grad`` holds the partial
        derivative of *this* output w.r.t. that ancestor, i.e.:

            node.grad  ==  ∂(self) / ∂(node)

        Algorithm
        ---------
        1. Build the full computation graph in topological order using
           :meth:`_topo_sort`.
        2. Seed ``self.grad = 1.0``  — the gradient of the output
           w.r.t. itself is always 1.
        3. Walk the nodes in *reverse* topological order (outputs
           before inputs) and call each node's ``._backward()``
           closure.  Because a node's ``.grad`` is fully accumulated
           by all of its consumers before its own ``._backward()``
           fires, the chain rule is applied correctly.

        Notes
        -----
        - Gradients are **accumulated** (``+=``), not overwritten.  If
          you call ``backward()`` twice you will double the gradients.
          Call :meth:`zero_grad` first to reset.
        - This method seeds ``self.grad = 1.0`` unconditionally; any
          value you set before calling ``backward()`` is overwritten.

        Examples
        --------
        >>> a = Value(2.0, label="a")
        >>> b = Value(3.0, label="b")
        >>> c = a * b           # c.data == 6.0
        >>> c.backward()
        >>> a.grad              # ∂c/∂a = b = 3.0
        3.0
        >>> b.grad              # ∂c/∂b = a = 2.0
        2.0

        >>> x = Value(3.0)
        >>> y = x * x           # y = x²
        >>> y.backward()
        >>> x.grad              # dy/dx = 2x = 6.0
        6.0
        """
        topo = self._topo_sort()

        # Seed: ∂self/∂self = 1
        self.grad = 1.0

        # Walk in reverse topological order — outputs before inputs.
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Future hook placeholders
    # ------------------------------------------------------------------
    # Day 5 additions → activation functions: tanh, relu, sigmoid, exp
    # Day 6 additions → Neuron / Layer / MLP classes
