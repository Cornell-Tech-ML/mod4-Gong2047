"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        tensor_vals = []
        for v in vals:
            if isinstance(v, minitorch.Tensor):
                raw_vals.append(v.detach())
                tensor_vals.append(v)
                if v.requires_grad():
                    need_grad = True
            else:
                raw_vals.append(v)
        ctx = Context(not need_grad)
        c = cls._forward(ctx, *raw_vals)
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, tensor_vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for negation."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for negation."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for Inv."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for Inv."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise addition of two tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for addition."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for element-wise multiplication."""
        t1, t2 = ctx.saved_values
        grad_t1 = grad_output.f.mul_zip(grad_output, t2)
        grad_t2 = grad_output.f.mul_zip(grad_output, t1)
        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Element-wise sigmoid function."""
        result = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid."""
        (t,) = ctx.saved_values
        return grad_output.f.mul_zip(
            grad_output, grad_output.f.mul_zip(t, 1 + grad_output.f.neg_map(t))
        )


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Element-wise ReLU function."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU."""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Element-wise natural logarithm."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for logarithm."""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Element-wise exponential function."""
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for exponential."""
        (result,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, result)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: int) -> Tensor:
        """Sum over a specified dimension."""
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sum."""
        shape, dim = ctx.saved_values
        if dim is None:
            out = grad_output.zeros(shape)
            out._tensor._storage[:] = grad_output[0]
            return out
        else:
            return grad_output


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise 'less than' comparison."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for 'less than' comparison (zero gradients)."""
        t1_shape, t2_shape = ctx.saved_values
        zeros_t1 = zeros(t1_shape, backend=grad_output.backend)
        zeros_t2 = zeros(t2_shape, backend=grad_output.backend)
        return zeros_t1, zeros_t2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise equality comparison."""
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison (zero gradients)."""
        t1_shape, t2_shape = ctx.saved_values
        zeros_t1 = zeros(t1_shape, backend=grad_output.backend)
        zeros_t2 = zeros(t2_shape, backend=grad_output.backend)
        return zeros_t1, zeros_t2


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise comparison for closeness."""
        return t1.f.is_close_zip(t1, t2)

    # No backward method for IsClose as specified


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tuple[int, ...]) -> Tensor:
        """Permute the dimensions of a tensor."""
        ctx.save_for_backward(order)
        return t1._new(t1._tensor.permute(*order))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for permute."""
        (order,) = ctx.saved_values
        # Compute inverse permutation
        inv_order = tuple(order.index(i) for i in range(len(order)))
        return grad_output._new(grad_output._tensor.permute(*inv_order))


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward pass for the view function.
        This function reshapes the input tensor `a` to the specified `shape`.
        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(dim) for dim in shape]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient using central difference method.

    Args:
    ----
        f (Any): The function to compute the gradient for.
        vals (Tensor): The input tensors.
        arg (int): The argument index to compute the gradient for.
        epsilon (float): The epsilon value for central difference.
        ind (UserIndex): The index to compute the gradient at.

    Returns:
    -------
        float: The computed gradient.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
