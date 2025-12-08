from flax import nnx
import numpy as np
import torch

from jax import numpy as jnp
import jax
from jax import Array
from einops import rearrange

from typing import Callable, Optional


def get_batch_sampler(
    sampler_fn: Callable,
    ncond: int,
):
    @jax.jit
    def sampler(key) -> Array:
        return sampler_fn(key, ncond)

    # Vectorize sampler_fn over batch dimension
    batched_sampler = jax.vmap(sampler)
    return batched_sampler


class PosteriorWrapper:
    def __init__(self, pipeline, *args, rngs: nnx.Rngs, **kwargs):
        """ """

        self.pipeline = pipeline
        self.args = args
        self.kwargs = kwargs
        self.default_x = None
        self.rngs = rngs

    def set_default_x(self, x):
        self.default_x = x

    def sample(
        self,
        sample_shape,
        x: Optional[torch.Tensor] = None,
        show_progress_bars: bool = False,
    ):
        key = self.rngs.sample()
        if x is None:
            cond = self.default_x.numpy()
        else:
            cond = x.numpy()

        res = self.pipeline.sample(
            key, cond, sample_shape[0], *self.args, **self.kwargs
        )
        res = res.reshape((sample_shape[0], -1))
        return torch.from_numpy(np.array(res))

    def sample_batched(
        self,
        sample_shape,
        x: Optional[torch.Tensor] = None,
        show_progress_bars: bool = False,
    ):
        if x is None:
            cond = np.array(self.default_x)
        else:
            cond = np.array(x)

        sampler = self.pipeline.get_sampler(self.rngs.sample(), cond)
        batched_sampler = get_batch_sampler(
            sampler,
            ncond=cond.shape[0],
        )

        keys = jax.random.split(self.rngs.sample(), sample_shape[0])
        res = batched_sampler(
            keys,
        )
        res = rearrange(res, "... f c -> ... (f c)")
        return torch.from_numpy(np.array(res))
