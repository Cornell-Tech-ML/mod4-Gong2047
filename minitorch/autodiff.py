from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    h_vals = list(vals)
    h_vals[arg] += epsilon
    y_h = f(*h_vals)
    y = f(*vals)
    return (y_h - y) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier for the variable."""

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of the current variable."""

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for backpropagation."""


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    stack = []

    def dfs(v: Variable) -> None:
        if v.is_constant() or v.unique_id in visited:
            return
        visited.add(v.unique_id)
        for p in v.parents:
            dfs(p)
        stack.append(v)

    dfs(variable)
    return stack


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable
    deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.

    derivatives = {variable.unique_id: deriv}
    nodes = topological_sort(variable)
    for node in reversed(nodes):
        d_output = derivatives[node.unique_id]
        if node.history is not None and node.history.last_fn is not None:
            for parent, d_parent in node.chain_rule(d_output):
                if not parent.is_constant():
                    if parent.unique_id in derivatives:
                        derivatives[parent.unique_id] += d_parent
                    else:
                        derivatives[parent.unique_id] = d_parent
    for node in nodes:
        if node.is_leaf() and node.unique_id in derivatives:
            node.accumulate_derivative(derivatives[node.unique_id])


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for backpropagation."""
        return self.saved_values
