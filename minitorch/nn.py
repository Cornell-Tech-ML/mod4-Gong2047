from typing import Tuple

import numpy as np

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    new_height = height // kh
    new_width = width // kw

    input = input.contiguous()

    out = input.view(batch, channel, new_height, kh, new_width, kw)

    out = out.permute(0, 1, 2, 4, 3, 5)

    out = out.contiguous()

    out = out.view(batch, channel, new_height, new_width, kh * kw)
    return out, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """2D Average Pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: (kh, kw)

    Returns:
    -------
        A tensor of shape (batch, channel, new_height, new_width)
        where new_height = height // kh and new_width = width // kw

    """
    out, nh, nw = tile(input, kernel)  # out shape: (batch, channel, nh, nw, kh*kw)
    batch, channel, _, _, size = out.shape
    out = out.sum(dim=4) / size
    out = out.view(batch, channel, nh, nw)
    return out


# TODO: Implement for Task 4.4.


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: int = None) -> Tensor:
        """Perform the forward pass of the max function.

        Args:
        ----
            ctx: Context to save information for backward computation.
            a: Input tensor.
            dim: Dimension along which to compute the max. If None, compute the global max.

        Returns:
        -------
            Tensor containing the maximum values

        """
        np_a = a.to_numpy()
        if dim is None:
            max_val = np.max(np_a)
            out = a.zeros((1,))
            out[0] = max_val
            argmax_idx = np.argmax(np_a.ravel())
            ctx.save_for_backward(a, dim, argmax_idx)
            return out
        else:
            max_val = np.max(np_a, axis=dim, keepdims=True)
            argmax_arr = np.argmax(np_a, axis=dim)
            argmax_arr = np.expand_dims(argmax_arr, axis=dim)
            ctx.save_for_backward(a, dim, argmax_arr)
            out = Tensor.make(max_val.ravel(), max_val.shape, backend=a.backend)
            return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """Perform the backward pass of the max function.

        Args:
        ----
            ctx: Context with saved information from the forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple containing the gradient of the loss with respect to the input.

        """
        a, dim, argmax_data = ctx.saved_values
        grad = a.zeros()
        np_grad = grad.to_numpy()
        np_go = grad_output.to_numpy()
        shape = a.shape

        if dim is None:
            argmax_idx = argmax_data
            pos = np.unravel_index(argmax_idx, shape)
            np_grad[pos] = np_go.item()
        else:
            expand_shape = list(shape)
            expand_shape[dim] = 1
            np_go_expanded = np.broadcast_to(np_go, tuple(expand_shape))

            it = np.nditer(argmax_data, flags=["multi_index"])
            while not it.finished:
                idx = list(it.multi_index)
                argm = it[0]
                idx[dim] = argm
                idx = tuple(idx)
                np_grad[idx] = np_go_expanded[it.multi_index]
                it.iternext()

        return (Tensor.make(np_grad.ravel(), np_grad.shape, backend=a.backend),)


def max(a: Tensor, dim: int = None) -> Tensor:
    """Compute the maximum value along a specified dimension.

    Args:
    ----
        a: Input tensor.
        dim: Dimension along which to compute the maximum. If None, compute the global maximum.

    Returns:
    -------
        A tensor containing the maximum values.

    """
    return Max.apply(a, dim)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """2D Max Pooling
    Similar to avgpool2d but taking max instead of average.
    """
    out, nh, nw = tile(input, kernel)
    out = max(out, dim=4)
    batch, channel, _, _, _ = out.shape
    out = out.view(batch, channel, nh, nw)
    return out


def softmax(a: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor along a specified dimension.

    Args:
    ----
        a: Input tensor.
        dim: Dimension along which to compute the softmax.

    Returns:
    -------
        A tensor with the same shape as `a` with softmax applied along the specified dimension.

    """
    max_val = max(a, dim)
    a_shifted = a - max_val
    exp_val = a_shifted.exp()
    sum_exp = exp_val.sum(dim)
    return exp_val / sum_exp


def logsoftmax(a: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of a tensor along a specified dimension.

    Args:
    ----
        a: Input tensor.
        dim: Dimension along which to compute the logsoftmax.

    Returns:
    -------
        A tensor with the same shape as `a` with logsoftmax applied along the specified dimension.

    """
    max_val = max(a, dim)
    a_shifted = a - max_val
    sum_exp = a_shifted.exp().sum(dim)
    log_sum_exp = sum_exp.log()
    return a_shifted - log_sum_exp


def dropout(a: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout:
    With probability p set elements to zero.
    If ignore=True, then do nothing.
    """
    if ignore:
        return a
    if p >= 1.0:
        return a.zeros()
    if p <= 0.0:
        return a
    mask = (np.random.rand(*a.shape) > p).astype(np.float64)
    mask_t = Tensor(a.backend.from_numpy(mask), backend=a.backend)
    return a * mask_t
