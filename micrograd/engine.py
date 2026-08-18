"""
micrograd/engine.py
--------------------
Day 6 — Neuron / Layer / MLP: a real tiny neural network.

Philosophy
----------
Every number in this autograd engine is a `Value`.
Raw floats and ints are "dead" — they carry no history, no identity, no
future.  A `Value` gives a number a name, a type guarantee, and a
place to grow as we add math, gradients, and backprop in later days.

What's new in Day 6:
    - `Neuron`  — single unit: activation(w·x + b), random init,
                  supports tanh / relu / sigmoid / linear activations.
    - `Layer`   — n_out Neurons reading the same input vector.
    - `MLP`     — a stack of Layers; a full Multi-Layer Perceptron.
    - Every class exposes `.parameters()` (flat list of Value weights)
      and `.__call__()` (forward pass accepting list[Value|float]).

What was added in Day 5:
    - `.exp()`     — eˣ,        backward: out.grad * out.data
    - `.log()`     — natural log, backward: out.grad / self.data
    - `.tanh()`    — hyperbolic tangent (fused), backward: out.grad*(1-t²)
    - `.relu()`    — max(0, x), backward: out.grad if x > 0 else 0
    - `.sigmoid()` — 1/(1+e⁻ˣ) (fused), backward: out.grad*s*(1-s)

What was added previously:
    Day 4:
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
import random
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
    Day 5 scope  — activation functions: exp, log, tanh, relu, sigmoid.
    Day 6 scope  — Neuron / Layer / MLP.
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
    # Activation functions  (Day 5)
    # ------------------------------------------------------------------
    # Design rules (same as arithmetic operators):
    #   1. Always return a new Value — Values are immutable nodes.
    #   2. Record _op and _prev so the computation graph stays complete.
    #   3. The _backward closure captures all values needed for the
    #      chain-rule formula via closure (no extra state on the node).
    #   4. Fused implementations (tanh, sigmoid) are preferred over
    #      composing existing ops: they produce a single, clean graph
    #      node and avoid numerical issues from intermediate exp().

    def exp(self) -> "Value":
        """
        Elementwise natural exponential: e^self.

        Forward:  out = e^x
        Backward: d(e^x)/dx = e^x = out  →  self.grad += out.grad * out.data

        Notes
        -----
        For very large positive inputs (x > ~709) Python's math.exp raises
        OverflowError.  This mirrors PyTorch's behaviour — callers must
        ensure inputs are in a reasonable range.

        Examples
        --------
        >>> v = Value(1.0)
        >>> v.exp().data
        2.718281828459045

        >>> v = Value(0.0)
        >>> v.exp().data
        1.0
        """
        out = Value(math.exp(self.data), _op="exp", _prev=(self,))

        # d(e^x)/dx = e^x — we reuse the already-computed out.data.
        def _backward() -> None:
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self) -> "Value":
        """
        Elementwise natural logarithm: ln(self).

        Forward:  out = ln(x)       (x must be > 0)
        Backward: d(ln x)/dx = 1/x  →  self.grad += out.grad / self.data

        Raises
        ------
        ValueError
            If self.data <= 0 (log is undefined for non-positive reals).

        Examples
        --------
        >>> import math
        >>> v = Value(math.e)
        >>> round(v.log().data, 10)
        1.0

        >>> Value(1.0).log().data
        0.0
        """
        if self.data <= 0:
            raise ValueError(
                f"log() requires a strictly positive input, got {self.data}."
            )
        out = Value(math.log(self.data), _op="log", _prev=(self,))

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        """
        Hyperbolic tangent activation (fused).

        Forward:  t = tanh(x) = (e^2x - 1) / (e^2x + 1)
        Backward: d(tanh x)/dx = 1 - tanh²(x) = 1 - t²
                  self.grad += out.grad * (1 - out.data ** 2)

        Why fused?
        ----------
        Computing tanh via existing ops (exp, div, sub, add) would create
        a multi-node sub-graph and obscure the structure of the network.
        A fused node keeps the graph clean and is numerically equivalent.

        Examples
        --------
        >>> Value(0.0).tanh().data
        0.0

        >>> round(Value(1.0).tanh().data, 6)
        0.761594
        """
        t = math.tanh(self.data)
        out = Value(t, _op="tanh", _prev=(self,))

        # 1 - t² is the sech²(x) derivative of tanh.
        def _backward() -> None:
            self.grad += out.grad * (1.0 - out.data ** 2)

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """
        Rectified Linear Unit activation.

        Forward:  out = max(0, x)
        Backward: d/dx max(0,x) = 1 if x > 0 else 0
                  self.grad += out.grad * (1 if self.data > 0 else 0)

        Notes
        -----
        - The subgradient at x=0 is defined to be 0 (consistent with
          PyTorch's default).
        - ReLU is the most common hidden-layer activation in modern MLPs
          because it is cheap to compute and avoids the vanishing-gradient
          problem that tanh/sigmoid suffer from in deep networks.

        Examples
        --------
        >>> Value(3.0).relu().data
        3.0

        >>> Value(-2.0).relu().data
        0.0

        >>> Value(0.0).relu().data
        0.0
        """
        out = Value(max(0.0, self.data), _op="relu", _prev=(self,))

        def _backward() -> None:
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        """
        Logistic sigmoid activation (fused).

        Forward:  s = 1 / (1 + e^{-x})
        Backward: d/dx sigmoid(x) = s * (1 - s)
                  self.grad += out.grad * out.data * (1 - out.data)

        Why fused?
        ----------
        Same reasoning as tanh — keeps the graph compact and avoids
        precision loss from chaining multiple exp/div nodes.

        Notes
        -----
        For numerical stability the implementation uses the standard
        formula which is well-conditioned for all finite x:
          - Large positive x → e^{-x} ≈ 0 → s ≈ 1
          - Large negative x → e^{-x} is huge → s ≈ 0
        Neither case causes overflow because we only ever compute e^{-x},
        which is bounded above (1) when x ≥ 0.

        Examples
        --------
        >>> Value(0.0).sigmoid().data
        0.5

        >>> round(Value(2.0).sigmoid().data, 6)
        0.880797
        """
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, _op="sigmoid", _prev=(self,))

        # s*(1-s) is the classic sigmoid derivative.
        def _backward() -> None:
            self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = _backward
        return out


# ==========================================================================
# Day 6 — Neural network building blocks
# ==========================================================================
#
# Three classes built entirely on `Value` arithmetic, so every weight and
# bias is automatically differentiable through the autograd engine above.
#
# Design rules (consistent with the Value class above):
#   1. All learnable parameters are `Value` nodes — no raw floats sneak in.
#   2. __call__ accepts list[Value | float] for ergonomic use.
#   3. parameters() returns a flat list so a training loop only needs:
#          for p in model.parameters():
#              p.data -= lr * p.grad
#   4. Weights are initialised with Kaiming-style uniform ±1/√n_in scaling
#      so that activations neither explode nor vanish at random init.


_ACTIVATIONS = frozenset({"tanh", "relu", "sigmoid", "linear"})


class Neuron:
    """
    A single artificial neuron: out = activation(w · x + b).

    Parameters
    ----------
    n_in : int
        Number of scalar inputs the neuron receives.
    activation : str, optional
        Non-linearity to apply after the dot product.
        One of ``'tanh'`` (default), ``'relu'``, ``'sigmoid'``,
        or ``'linear'`` (no activation — useful for output neurons).

    Attributes
    ----------
    weights : list[Value]
        One weight per input, initialised uniformly in
        ``[-1/√n_in, +1/√n_in]``.
    bias : Value
        Scalar bias, initialised to 0.0.

    Examples
    --------
    >>> import random; random.seed(0)
    >>> n = Neuron(3)
    >>> out = n([1.0, 2.0, 3.0])
    >>> isinstance(out, Value)
    True
    >>> -1.0 <= out.data <= 1.0  # tanh output is bounded
    True
    """

    def __init__(self, n_in: int, activation: str = "tanh") -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, "
                f"got {activation!r}."
            )
        if not isinstance(n_in, int) or n_in < 1:
            raise ValueError(f"n_in must be a positive integer, got {n_in!r}.")

        self.activation = activation

        # Kaiming-style uniform: ±1/√n_in keeps variance ~1 at init.
        limit = 1.0 / math.sqrt(n_in)
        self.weights: list[Value] = [
            Value(random.uniform(-limit, limit), label=f"w{i}")
            for i in range(n_in)
        ]
        self.bias: Value = Value(0.0, label="b")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: list) -> Value:
        """
        Compute the neuron's output for input vector *x*.

        Parameters
        ----------
        x : list[Value | int | float]
            Input activations.  Plain numbers are wrapped automatically.

        Returns
        -------
        Value
            The scalar output after applying the activation function.

        Raises
        ------
        ValueError
            If ``len(x) != len(self.weights)``.
        """
        if len(x) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(x)}."
            )

        # Dot product: w · x  (coerce plain numbers to Value on the fly)
        act: Value = sum(
            (wi * (xi if isinstance(xi, Value) else Value(float(xi)))
             for wi, xi in zip(self.weights, x)),
            self.bias,
        )

        # Apply chosen non-linearity
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        if self.activation == "sigmoid":
            return act.sigmoid()
        # 'linear' — no activation
        return act

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> list[Value]:
        """Return all trainable Value nodes: weights + bias."""
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return (
            f"Neuron(n_in={len(self.weights)}, activation={self.activation!r})"
        )


class Layer:
    """
    A fully-connected layer: a list of *n_out* independent Neurons that
    all receive the same input vector and each produce one scalar output.

    Parameters
    ----------
    n_in : int
        Number of inputs per neuron (= width of the previous layer).
    n_out : int
        Number of neurons in this layer (= output width).
    activation : str, optional
        Activation for every neuron. Default ``'tanh'``.

    Examples
    --------
    >>> import random; random.seed(0)
    >>> layer = Layer(2, 3)
    >>> outs = layer([1.0, -1.0])
    >>> len(outs)
    3
    >>> all(isinstance(o, Value) for o in outs)
    True
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "tanh") -> None:
        self.neurons: list[Neuron] = [
            Neuron(n_in, activation=activation) for _ in range(n_out)
        ]

    def __call__(self, x: list) -> list[Value]:
        """
        Run each neuron and return a list of scalar Value outputs.

        If there is only one neuron the *list* is still returned (MLP
        unpacks it correctly) — a single Value is never auto-unwrapped
        here so that Layer behaviour is consistent regardless of width.
        """
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        """Flat list of all Value parameters across every neuron."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self) -> str:
        return (
            f"Layer(n_in={len(self.neurons[0].weights)}, "
            f"n_out={len(self.neurons)}, "
            f"activation={self.neurons[0].activation!r})"
        )


class MLP:
    """
    Multi-Layer Perceptron: a stack of fully-connected Layers.

    Parameters
    ----------
    n_in : int
        Number of scalar inputs to the network.
    layer_sizes : list[int]
        Width of each successive layer.  The final entry is the output
        width.  Example: ``[4, 4, 1]`` creates two hidden layers of
        width 4 followed by a single-output layer.
    activation : str, optional
        Activation for all *hidden* layers. Default ``'tanh'``.
    out_activation : str, optional
        Activation for the *output* layer. Default ``'linear'`` so the
        network can produce unbounded values suitable for regression or
        direct loss computation (e.g. MSE).  Set to ``'sigmoid'`` for
        binary classification.

    Attributes
    ----------
    layers : list[Layer]
        The ordered list of layers.

    Examples
    --------
    >>> import random; random.seed(42)
    >>> model = MLP(2, [4, 4, 1])
    >>> out = model([1.0, 2.0])
    >>> isinstance(out, Value)      # single-output network unwraps the list
    True
    >>> len(model.parameters())    # (2*4+4) + (4*4+4) + (4*1+1) = 41
    41
    """

    def __init__(
        self,
        n_in: int,
        layer_sizes: list[int],
        activation: str = "tanh",
        out_activation: str = "linear",
    ) -> None:
        if not layer_sizes:
            raise ValueError("layer_sizes must contain at least one entry.")

        # Build layers: all hidden layers use `activation`, output uses
        # `out_activation`.
        sizes = [n_in] + list(layer_sizes)
        self.layers: list[Layer] = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation=(
                    activation if i < len(layer_sizes) - 1 else out_activation
                ),
            )
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x: list) -> "Value | list[Value]":
        """
        Run a forward pass through every layer in sequence.

        Parameters
        ----------
        x : list[Value | int | float]
            Input feature vector.

        Returns
        -------
        Value
            If the output layer has exactly one neuron the single Value
            is returned directly (most common case for scalar regression
            / binary classification).
        list[Value]
            Otherwise a list of output Values is returned.
        """
        out: list[Value] = list(x)  # copy so we don't mutate the input
        for layer in self.layers:
            out = layer(out)
        # Unwrap single-output networks for ergonomics
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        """Flat list of every trainable Value in the network."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        """
        Zero the gradient on every parameter in the network.

        Convenience shortcut — equivalent to calling
        ``p.grad = 0.0`` for every ``p`` in ``self.parameters()``.
        Unlike ``Value.zero_grad()`` this does **not** walk the full
        computation graph; it only resets the parameter leaves.
        Use this instead of ``loss.zero_grad()`` in a training loop
        — it is O(parameters) rather than O(graph nodes).
        """
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self) -> str:
        layer_strs = ", ".join(repr(l) for l in self.layers)
        return f"MLP([{layer_strs}])"
