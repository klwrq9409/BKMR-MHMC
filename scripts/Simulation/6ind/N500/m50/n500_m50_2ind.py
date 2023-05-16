#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:15:56 2022

@author: ruiqi
"""
import os
# import tempfile
import numpy as np
import jax.numpy as jnp
import jax
import joblib
import sys
from momentum.hmc.mixed_hmc_jax3 import mixed_hmc_on_joint
from momentum.utils4 import generate_bkmr_potential
from momentum.utils import jax_prng_key
# from momentum.diagnostics.ess import get_min_ess
import json
from typing import Callable
import time
# import collections
# from tqdm import tqdm
# from itertools import chain
# import matplotlib.pyplot as plt
# import pymc3
# import arviz

jax.config.update("jax_enable_x64", False)
id = os.getenv('SGE_TASK_ID')

filenames='/Simulation/6ind/N500/m50/data/data_n500_m50_6ind_regular_{}.json'.format(id)
with open(filenames) as f:
    l = json.load(f)

y=np.array(l['y'])
X=np.array(l['X'])
M=l['M'][0]
N=l['n'][0]
P=1
Z=np.array(l['Z'])
beta0=np.array(l['beta.true'])
sigmasq0=np.array(l['sigsq.true'])

sigma_r=5.0
lambda_a=1.0
sigma_a=0.001
sigma_b=0.001
lambda_b=0.1
r_a=2
r_b=1
lambda10=np.array([3.0])
# r0=np.random.normal(0,5,M)
r0=np.random.gamma(shape=r_a,scale=1/r_b,size=(M))
# r0=np.ones(M)
eta=np.concatenate((beta0,sigmasq0,lambda10,r0))
eta
# delta=np.random.binomial(1, 0.5,M)
delta=np.zeros(M)


pot=generate_bkmr_potential(X,Z, y,sigma_a,sigma_b,lambda_a,lambda_b,r_a,r_b)
pot(delta,eta)

# n_samples=int(10)
epsilon = 0.005
# total_travel_time = 0.1
L = 10
key=jax_prng_key()
# key=jax.random.PRNGKey(100)
mode='gibbs'
adaptive_step_size=None
progbar=True
labels_for_discrete = jax.device_put(np.tile(np.arange(2), (M, 1)))
n_discrete_to_update=1
n_warm_up_samples=0
n_samples=5000
start = time.time()
z_samples, eta_samples, accept_array= mixed_hmc_on_joint(
        q0_discrete=delta,
        q0_continuous=eta,
        n_samples=n_warm_up_samples+ n_samples,
        epsilon=epsilon,
        L=L,
        key=key,
        labels_for_discrete=labels_for_discrete,
        potential=pot,
        grad_potential=jax.jit(jax.grad(pot,argnums=1)),
        # grad_potential=jax.grad(pot,argnums=1),
        # grad_potential=jacfwd(pot,argnums=1),
        n_discrete_to_update=1,
        mode=mode,
        progbar=progbar,
        adaptive_step_size=adaptive_step_size,
    )
time_used=time.time() - start

delta_true=delta
delta_true[:2,]=1.0
if n_warm_up_samples + n_samples == z_samples.shape[0]:
        samples = z_samples[n_warm_up_samples:,]
else:
        samples = z_samples

samples = samples.reshape((-1, samples.shape[-1]))
bin_count = np.bincount(
    np.sum(samples == (delta_true > 0).astype(np.int32).reshape((1, -1)), axis=1),
    minlength=delta_true.shape[0],
)
# print('Bin count: {}, total {}'.format(bin_count, np.sum(bin_count)))
# print('Correct percentage: {}'.format(bin_count[-1] / np.sum(bin_count)))
# print(
#     'Hamming distance: {}'.format(
#         np.mean(
#             np.sum(samples != (delta_true > 0).astype(np.int32).reshape((1, -1)), axis=1)
#         )
#         / delta_true.shape[0]
#     )
# )
results = {
            'beta': eta_samples,
            'gamma': z_samples,
            'n_warm_up_samples': n_warm_up_samples,
            'bin_count': bin_count,
            'Correct percentage':bin_count[-1] / np.sum(bin_count),
            'Hamming distance': np.mean(np.sum(samples != (delta_true > 0).astype(np.int32).reshape((1, -1)), axis=1))/ delta_true.shape[0] ,    
            'time_used':time_used
        }


results_fname = '/Simulation/6ind/N500/m50/results/bkmr-mhmc/results.joblib_{}'.format(id)
joblib.dump(results, results_fname)
# ass=joblib.load("/Users/ruiqi/Documents/github/mixed_hmc/results.joblib_4")
