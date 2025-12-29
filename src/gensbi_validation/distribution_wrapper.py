from flax import nnx
import numpy as np
import torch

from jax import numpy as jnp
import jax
from jax import Array
from einops import rearrange

from typing import Callable, Optional
from gensbi.recipes.pipeline import AbstractPipeline

from jax import lax

from tqdm import tqdm

# def get_batch_sampler(
#     sampler_fn: Callable,
#     ncond: int,
# ):
#     @jax.jit
#     def sampler(key) -> Array:
#         return sampler_fn(key, ncond)

#     # Vectorize sampler_fn over batch dimension
#     batched_sampler = jax.vmap(sampler)
#     return batched_sampler

# def get_batch_sampler(
#     sampler_fn: Callable,
#     ncond: int,
# ):
#     @jax.jit
#     def sampler(key) -> Array:
#         return sampler_fn(key, ncond)

#     def batched_sampler(keys):
#         def body_fn(carry, key):
#             res = sampler(key)
#             return carry, res
#         _, results = jax.lax.scan(body_fn, None, keys)
#         return results

#     return batched_sampler

# v3
# def get_batch_sampler(
#     sampler_fn: Callable,
#     ncond: int,
#     chunk_size: Optional[int] = None,
# ):
#     def single_sampler(key):
#         return sampler_fn(key, ncond)

#     @jax.jit
#     def batched_sampler(keys):
#         n_samples = keys.shape[0]

#         # --- Path 1: Pure Vmap ---
#         if chunk_size is None:
#             return jax.vmap(single_sampler)(keys)

#         # --- Path 2: Chunked Map ---
#         if n_samples % chunk_size != 0:
#             raise ValueError(f"Batch size {n_samples} must be divisible by chunk_size {chunk_size}")

#         n_chunks = n_samples // chunk_size
#         keys_reshaped = keys.reshape(n_chunks, chunk_size, *keys.shape[1:])

#         # Define the function for a single chunk
#         # No 'carry' argument needed anymore!
#         def chunk_fn(key_chunk):
#             return jax.vmap(single_sampler)(key_chunk)

#         # map loops over the first axis (chunks) sequentially
#         results_stacked = lax.map(chunk_fn, keys_reshaped)

#         return results_stacked.reshape(n_samples, *results_stacked.shape[2:])

#     return batched_sampler


# def get_batch_sampler(
#     sampler_fn: Callable,
#     ncond: int,
#     chunk_size: Optional[int] = None,
# ):
#     # Wrapper for a single sample (will be vectorized internally)
#     def single_sampler(key):
#         return sampler_fn(key, ncond)

#     @jax.jit
#     def batched_sampler(keys):
#         n_samples = keys.shape[0]

#         # --- Path 1: Pure Vmap (Unlimited Memory) ---
#         if chunk_size is None:
#             return jax.vmap(single_sampler)(keys)

#         # --- Path 2: Chunked Scan (Memory Efficient) ---
        
#         # 1. Enforce Divisibility
#         # This acts as a compile-time assertion for static shapes
#         if n_samples % chunk_size != 0:
#             raise ValueError(
#                 f"Input batch size ({n_samples}) must be divisible by chunk_size ({chunk_size})."
#             )

#         # 2. Reshape keys: (N, ...) -> (Num_Chunks, Chunk_Size, ...)
#         # This works for both old keys (N, 2) and new Array keys (N,)
#         n_chunks = n_samples // chunk_size
#         keys_reshaped = keys.reshape(n_chunks, chunk_size, *keys.shape[1:])

#         # 3. Define the scan loop (Sequential Chunks, Parallel Inside)
#         def scan_body(carry, key_chunk):
#             # vmap processes the specific chunk in parallel
#             chunk_results = jax.vmap(single_sampler)(key_chunk)
#             return carry, chunk_results

#         # 4. Run Scan
#         # scan stacks results along axis 0 automatically
#         _, results_stacked = jax.lax.scan(scan_body, None, keys_reshaped)

#         # 5. Flatten Output: (Num_Chunks, Chunk_Size, Output_Dim) -> (N, Output_Dim)
#         return results_stacked.reshape(n_samples, *results_stacked.shape[2:])

#     return batched_sampler

# v4 moved to the pipeline
# def get_batch_sampler(
#     sampler_fn: Callable,
#     ncond: int,
#     chunk_size: int,
#     show_progress_bars: bool = True,
# ):
#     # JIT the chunk processor
#     @jax.jit
#     def process_chunk(key_batch):
#         return jax.vmap(lambda k: sampler_fn(k, ncond))(key_batch)

#     def sampler(keys):
#         n_samples = keys.shape[0]
#         results = []
        
#         # Calculate total chunks for tqdm
#         # We use ceil division to handle remainders
#         n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
#         # Using a tqdm loop
        
#         if show_progress_bars:
#             loop = tqdm(
#                 range(0, n_samples, chunk_size),
#                 total=n_chunks,
#                 desc="Sampling",
#             )
#         else:
#             loop = range(0, n_samples, chunk_size)
        
#         for i in loop:
#             batch_keys = keys[i : i + chunk_size]
            
#             chunk_out = process_chunk(batch_keys)
            
#             # CRITICAL: Wait for GPU to finish this chunk before updating bar
#             # This makes the progress bar accurate.
#             chunk_out.block_until_ready()
            
#             results.append(chunk_out)
            
#         return jnp.concatenate(results, axis=0)

#     return sampler

class PosteriorWrapper:
    def __init__(self, pipeline: AbstractPipeline, *args, rngs: nnx.Rngs, theta_shape = None, x_shape = None, **kwargs):
        """ 
        Wrap a GenSBI pipeline into a distribution compatible with sbi.
        """

        self.pipeline = pipeline
        self.args = args
        self.kwargs = kwargs
        self.default_x = None
        self.rngs = rngs
        
        if theta_shape is not None:
            self.dim_theta = theta_shape[0]
            self.ch_theta = theta_shape[1]
        else:
            self.ch_theta = self.pipeline.ch_obs
            self.dim_theta = self.pipeline.dim_obs
            
        if x_shape is not None:
            self.dim_x = x_shape[0]
            self.ch_x = x_shape[1]
        else:
            if self.pipeline.ch_cond is None:
                self.ch_x = self.ch_theta
            else:
                self.ch_x = self.pipeline.ch_cond
            self.dim_x = self.pipeline.dim_cond

    def _ravel(self, x):
        return x.reshape(x.shape[0], -1)

    def _unravel_theta(self, x):
        return x.reshape(x.shape[0], self.dim_theta, self.ch_theta)

    def _unravel_xs(self, x):
        return x.reshape(x.shape[0], self.dim_x, self.ch_x)

    def _process_x(self, x):
        assert x.ndim in (2, 3), "x must be of shape (batch, dim) or (batch, dim, ch)"


        if x.ndim == 3:
            assert (
                x.shape[2] == self.ch_x
            ), f"Wrong number of channels, expected {self.ch_x}, got {x.shape[2]}"

        if x.ndim == 2:
            x = self._unravel_xs(x)

        return self._ravel(x)

    def set_default_x(self, x):

        self.default_x = self._process_x(x)

    def sample(
        self,
        sample_shape,
        x: Optional[torch.Tensor] = None,
        **kwargs, # does nothing, for compatibility
    ):
        key = self.rngs.sample()
        if x is None:
            cond = self.default_x.numpy()
        else:
            cond = x.numpy()

        if cond.ndim == 2:
            cond = self._unravel_xs(cond)

        res = self.pipeline.sample(
            key, cond, sample_shape[0], *self.args, **self.kwargs
        )
        res = self._ravel(res)
        return torch.from_numpy(np.array(res))

    def sample_batched(
        self,
        sample_shape,
        x: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = 50,
        show_progress_bars = True,
        **kwargs, # does nothing, for compatibility
    ):
        if x is None:
            cond = self.default_x.numpy()
        else:
            cond = x.numpy()
            
        if cond.ndim == 2:
            cond = self._unravel_xs(cond)

        # TODO: we will have to implement a seed in the get sampler method once we enable latent diffusion, as it is needed for the encoder
        # Possibly fixed by passing the kwargs, which should include the encoder_key
        # sampler = self.pipeline.get_sampler(cond, **self.kwargs)
        # batched_sampler = get_batch_sampler(
        #     sampler,
        #     ncond=cond.shape[0],
        #     chunk_size=chunk_size,
        #     show_progress_bars=show_progress_bars,
        # )

        # keys = jax.random.split(self.rngs.sample(), sample_shape[0])
        
        key = self.rngs.sample()
        res = self.pipeline.sample_batched(
            key,
            cond,
            sample_shape[0],
            chunk_size=chunk_size,
            show_progress_bars=show_progress_bars,
            **self.kwargs,  
        )

        res = rearrange(res, "... f c -> ... (f c)")
        return torch.from_numpy(np.array(res))
