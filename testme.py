# %%
import os

# os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import torch
from typing import Callable, Optional
from jax import Array

from gensbi.recipes import (
    ConditionalFlowPipeline,
)

from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp

from gensbi.models import Simformer, SimformerParams, Flux1, Flux1Params

import grain
import numpy as np

from gensbi_validation.distribution_wrapper import PosteriorWrapper


nsamples = 1000
key = jax.random.PRNGKey(0)

dim_obs = 2
dim_cond = 7
dim_joint = dim_obs + dim_cond

ch = 1


theta = jax.random.normal(key, (nsamples, dim_obs, ch))
x = jax.random.normal(key, (nsamples, dim_cond, ch))

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
    dim_obs=dim_obs,
    dim_cond=dim_cond,
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
cond = jnp.zeros((1, dim_cond, ch))
#%%
res = posterior.sample((100,), x=torch.from_numpy(np.array(cond)))
#%%
res.shape
#%%
cond = jnp.zeros((10, dim_cond, ch))
res2 = posterior.sample_batched((20,), x=torch.from_numpy(np.array(cond)))
#%%
# %%
obs, cond = next(iter(val_dataset_cond))
# %%
thetas, xs = posterior.ravel(obs), posterior.ravel(cond)
thetas = torch.Tensor(np.array(thetas))
xs = torch.Tensor(np.array(xs))
#%%
ranks, dap_samples = run_sbc(thetas, xs, posterior)
check_stats = check_sbc(ranks, thetas, dap_samples, 1_000)
print(check_stats)
f, ax = sbc_rank_plot(ranks, 1_000, plot_type="hist", num_bins=30)
#%%
ecp, alpha = run_tarp(
    torch.Tensor(thetas),
    torch.Tensor(xs),
    posterior,
    references=None,  # will be calculated automatically.
    num_posterior_samples=100,
)

# %%
atc, ks_pval = check_tarp(ecp, alpha)
print(atc, "Should be close to 0")
print(ks_pval, "Should be larger than 0.05")
# %%
from sbi.analysis.plot import plot_tarp

plot_tarp(ecp, alpha)
# %%
