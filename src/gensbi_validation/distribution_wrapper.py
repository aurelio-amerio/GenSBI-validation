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
        
    def ravel(self,x):
        return x.reshape(x.shape[0], -1)
    
    def unravel_theta(self,x):
        return x.reshape(x.shape[0], self.pipeline.dim_obs, -1)
    
    def unravel_xs(self,x):
        return x.reshape(x.shape[0], self.pipeline.dim_cond, -1)

    def set_default_x(self, x):
        assert x.ndim in (2, 3), "x must be of shape (batch, dim) or (batch, dim, ch)"
        self.default_x = self.ravel(x)

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
            
        if cond.ndim == 2:
            cond = self.unravel_xs(cond)

        res = self.pipeline.sample(
            key, cond, sample_shape[0], *self.args, **self.kwargs
        )
        res = self.ravel(res)
        return torch.from_numpy(np.array(res))

    def sample_batched(
        self,
        sample_shape,
        x: Optional[torch.Tensor] = None,
        show_progress_bars: bool = False,
    ):
        if x is None:
            cond = self.default_x.numpy()
        else:
            cond = x.numpy()
            
        if cond.ndim == 2:
            cond = self.unravel_xs(cond)

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
