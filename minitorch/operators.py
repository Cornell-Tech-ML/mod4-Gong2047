"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
#
# mul - Multiplies two numbers
# id - Returns the input unchanged
# add - Adds two numbers
# neg - Negates a number
# lt - Checks if one number is less than another
# eq - Checks if two numbers are equal
# max - Returns the larger of two numbers
# is_close - Checks if two numbers are close in value
# sigmoid - Calculates the sigmoid function
# relu - Applies the ReLU activation function
# log - Calculates the natural logarithm
# exp - Calculates the exponential function
# inv - Calculates the reciprocal
# log_back - Computes the derivative of log times a second arg
# inv_back - Computes the derivative of reciprocal times a second arg
# relu_back - Computes the derivative of ReLU times a second arg
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers.
    $f(x, y) = x * y$
    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.
    $f(x) = x$
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.
    $f(x, y) = x + y$
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.
    $f(x) = -x$
    """
    return float(-x)


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x >= y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value.
    $f(x) = |x - y| < 1e-2$
    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""Calculates the sigmoid function.
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg."""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Complete the following functions in minitorch/operators.py and pass tests marked as tasks0_3.

# map - Higher-order function that applies a given function to each element of an iterable
# zipWith - Higher-order function that combines elements from two iterables using a given function
# reduce - Higher-order function that reduces an iterable to a single value using a given function

# Using the above functions, implement:

# negList - Negate all elements in a list using map
# addLists - Add corresponding elements from two lists using zipWith
# sum - Sum all elements in a list using reduce
# prod - Calculate the product of all elements in a list using reduce


# TODO: Implement for Task 0.3.


def map(l: Iterable, fn: Callable) -> Iterable:
    """Higher-order function that applies a given function to each element of an iterable."""
    return [fn(x) for x in l]


def zipWith(l1: Iterable, l2: Iterable, fn: Callable) -> Iterable:
    """Higher-order function that combines elements from two iterables using a given function."""
    l1 = list(l1)
    l2 = list(l2)
    return [fn(l1[i], l2[i]) for i in range(len(list(l1)))]


def reduce(l: Iterable, fn: Callable) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function."""
    l = list(l)
    result = l[0]
    for i in range(1, len(l)):
        result = fn(result, l[i])
    return result


def negList(l: Iterable) -> Iterable:
    """Negate all elements in a list using map."""
    return map(l, neg)


def addLists(l1: Iterable, l2: Iterable) -> Iterable:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(l1, l2, add)


def sum(ls: Iterable) -> float:
    """Sum all elements in a list using reduce."""
    return reduce(ls, add)


def prod(ls: Iterable) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(ls, mul)
