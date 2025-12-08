# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import torch
from typing import Callable, Optional
from jax import Array

from gensbi.recipes import (
    ConditionalFlowPipeline,
)

from gensbi.models import Simformer, SimformerParams, Flux1, Flux1Params

import grain
import numpy as np

from gensbi_validation.distribution_wrapper import PosteriorWrapper


nsamples = 1000
rng = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 7
dim_joint = dim_obs + dim_cond

ch = 1


theta = jax.random.normal(rng, (nsamples, dim_obs, ch))
x = jax.random.normal(rng, (nsamples, dim_cond, ch))

data = jnp.concatenate([theta, x], axis=1)


def split_obs_cond(data):
    return (
        data[:, :dim_obs],
        data[:, dim_obs:],
    )  # assuming first dim_obs are obs, last dim_cond are cond


train_dataset_cond = (
    grain.MapDataset.source(np.array(data)[:800])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)

val_dataset_cond = (
    grain.MapDataset.source(np.array(data)[800:])
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(32)
    .map(split_obs_cond)
    # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
)
# we define a conditional and a joint model for testing


params = Flux1Params(
    in_channels=ch,
    vec_in_dim=None,
    context_in_dim=ch,
    mlp_ratio=4,
    num_heads=4,
    depth=1,
    depth_single_blocks=2,
    axes_dim=[
        2,
    ],
    obs_dim=dim_obs,
    cond_dim=dim_cond,
    qkv_bias=True,
    guidance_embed=False,
    rngs=nnx.Rngs(0),
    param_dtype=jnp.float32,
)

model_conditional = Flux1(params)
# %%
training_config = ConditionalFlowPipeline._get_default_training_config()

pipeline = ConditionalFlowPipeline(
    model_conditional,
    train_dataset_cond,
    val_dataset_cond,
    dim_obs,
    dim_cond,
    ch_obs=ch,
    ch_cond=ch,
    training_config=training_config,
)
# %%

pipeline.train(nnx.Rngs(0), nsteps=100, save_model=False)
# %%
posterior = PosteriorWrapper(pipeline, rngs=nnx.Rngs(1))
# %%
cond = jnp.zeros((10, dim_cond, ch))
#%%
res = posterior.sample((10,), x=torch.from_numpy(np.array(cond)))
#%%
res.shape
#%%
cond = jnp.zeros((10, dim_cond, ch))
res2 = posterior.sample_batched((20,), x=torch.from_numpy(np.array(cond)))
#%%
res2.shape
#%%
# samples = sampler(key, 10)
# cond = np.zeros((1, dim_cond, ch))
# # %%
# samples = posterior.sample((10,), x=torch.from_numpy(cond))
# # %%
# samples.shape


# %%



def get_batch_sampler(
    sampler_fn: Callable,
    # nsamples: int,
):  
    @jax.jit(static_argnums=[2])
    def sampler(key: Array, cond: Array, nsamples: int) -> Array:
        return sampler_fn(key, cond[None, ...], nsamples)

    # Vectorize sampler_fn over batch dimension
    batched_sampler = jax.vmap(sampler, in_axes=(0, 0, None))
    return batched_sampler

def get_batch_sampler2(
    sampler_fn: Callable,
    nsamples: int,
):  
    @jax.jit
    def sampler(key: Array, cond: Array) -> Array:
        return sampler_fn(key, cond[None,...], nsamples)

    # Vectorize sampler_fn over batch dimension
    batched_sampler = jax.vmap(sampler)
    return batched_sampler


def get_batch_sampler3(
    sampler_fn: Callable,
    ncond: int,
):  
    @jax.jit
    def sampler(key) -> Array:
        return sampler_fn(key, ncond)

    # Vectorize sampler_fn over batch dimension
    batched_sampler = jax.vmap(sampler)
    return batched_sampler


# %%
ncond = 3
batch_size = 20
cond = jnp.zeros((ncond, dim_cond, ch))
sampler_ = pipeline.get_sampler(jax.random.PRNGKey(0), cond)
#%%
batched_sampler = get_batch_sampler(
    pipeline.sample,
    # nsamples=10,
)

batched_sampler2 = get_batch_sampler2(
    pipeline.sample,
    nsamples=batch_size,
)

batched_sampler3 = get_batch_sampler3(
    sampler_,
    ncond=ncond,
)
#%%

keys = jax.random.split(jax.random.PRNGKey(0), cond.shape[0])
res1 = batched_sampler(
    keys,
    cond,
    10,
)

keys = jax.random.split(jax.random.PRNGKey(0), cond.shape[0])
res2 = batched_sampler2(
    keys,
    cond,
)

keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
res3 = batched_sampler3(
    keys,
)
#%%
res1.shape, res2.shape, res3.shape, (batch_size, ncond, dim_obs, ch)

# %%
%%timeit
keys = jax.random.split(jax.random.PRNGKey(0), cond.shape[0])
res1 = batched_sampler(
    keys,
    cond,
    batch_size,
)
# %%
%%timeit
keys = jax.random.split(jax.random.PRNGKey(0), cond.shape[0])
res2 = batched_sampler2(
    keys,
    cond,
)
#%%
%%timeit 
keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
res3 = batched_sampler3(
    keys,
)
# %%
