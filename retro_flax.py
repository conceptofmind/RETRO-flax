from flax.linen import nn

import jax.numpy as jnp

import numpy

def exists(x):
    return x is not None

def default(x, default_value):
    return x if exists(x) else default_value

def divisible_by(x, divisor):
    return (x / divisor).is_integer()

def cast_tuple(x, num = 1):
    return x if isinstance(x, tuple) else ((x,) * num)