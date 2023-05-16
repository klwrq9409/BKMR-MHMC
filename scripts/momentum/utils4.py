#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:15:02 2022

@author: ruiqi
"""
# for the component-wise variable selection
import jax
# import joblib
import jax.numpy as jnp
# from jax import jit

import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.spatial.distance import pdist, squareform
from typing import Callable
# from momentum.utils import jax_prng_key

def generate_bkmr_potential(X,Z, y, sigma_a,sigma_b,lambda_a,lambda_b,r_a,r_b, use_jax=True, use_dhmc=False):

# def generate_bkmr_potential(X,Z, y, sigma_r,sigma_a,sigma_b,lambda_a,lambda_b,r_a,r_b, use_jax=True, use_dhmc=False):
  # X is N*p
  # Z is N*M
  # y is N*1
  # sigma is a scalar
    # if use_jax:
    #     import jax.numpy as jnp

    X = jax.device_put(X)
    Z = jax.device_put(Z)
    y = jax.device_put(y)
    # sigma_r = jax.device_put(sigma_r)
    sigma_a = jax.device_put(sigma_a)
    sigma_b = jax.device_put(sigma_b)
    lambda_a = jax.device_put(lambda_a)
    lambda_b = jax.device_put(lambda_b)
    r_a=jax.device_put(r_a)
    r_b=jax.device_put(r_b)
    P=X.shape[1]
    N=X.shape[0]
    M=Z.shape[1]
    
    def sqeuclidean_distance(x: np.array, y: np.array) -> float:
        return jnp.sum((x - y) ** 2)
    def euclidean_distance(x: np.array, y: np.array) -> float:
        return sqeuclidean_distance(x, y)

    def distmat(func: Callable, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """distance matrix"""
        return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)
    def pdist_squareform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """squared euclidean distance matrix

        Notes
        -----
        This is equivalent to the scipy commands

        >>> from scipy.spatial.distance import pdist, squareform
        >>> dists = squareform(pdist(X, metric='sqeuclidean')
        """
        return distmat(euclidean_distance, x, y)


    def makeKpart(r, delta, Z1, Z2 = None):
        # Z1=jax.device_put(Z1)
        # r=jax.device_put(r)
        # delta=jax.device_put(delta)
        # Z1r= Z1 * jnp.array(jnp.sqrt(delta*r))[jnp.newaxis, :] # sweep along the columns
        # ss=jnp.multiply(delta,r)
        # Z1r=jnp.multiply(Z1,jnp.sqrt(ss)))  # sweep along the columns
        # ss=jnp.atleast_2d(jnp.sqrt(abs(delta*r))).repeat(repeats=N, axis=0)
        # Z1r=jnp.multiply(Z1,ss)
        Z1r=jnp.multiply(Z1,jnp.sqrt(abs(r)))
        Z1r=jnp.multiply(Z1r,delta)
        # Z1r= Z1*ss# sweep along the columns

        if (Z2 is None):
            Z2r = Z1r
        # else:
        #     Z2=jax.device_put(Z2)
        #     # Z2r = Z2 * jnp.array(jnp.sqrt(delta*r))[jnp.newaxis, :]
        #     Z2r = Z2 * ss
        Kpart=pdist_squareform(Z1r,Z2r)
        return Kpart
    
    # def rbf_kernel(x: jnp.array, y: jnp.array, ell,delta) -> jnp.array:
    #     # ell, sigma = params["lengthscale"], params["variance"]
    #     tau = jnp.sum(delta*jnp.square(x *ell - y *ell))
    #     return  jnp.exp(-1.0 * tau)

    # def evaluate_kernel(
    #     kernel_fn: Callable, x: jnp.array, y: jnp.array,ell,delta) -> jnp.array:
    #     K = jax.vmap(lambda x1: jax.vmap(lambda y1: kernel_fn(x1, y1, ell,delta))(y))(x)
    #     return K 
    
    def bkmr_potential(delta,eta):
        # eta=jax.device_put(eta)
        # delta=jax.device_put(delta)
        # [beta,sigmasq,lambda1,r]=eta
        beta=eta[0:P]
        sigmasq=eta[P]
        # lambda1=eta[P+1]
        tau1=eta[P+1]
        r=eta[P+2:]
        
        # if use_dhmc:
        #     z = map_embedded_to_discrete(z)
        #     boundary_potential = (jnp.sum(z < 0) + jnp.sum(z >= K)) * jnp.exp(80)
        # else:
        boundary_potential = 0
        # [Sigma_inv,logdetV]=makeVcomps(r,delta,lambda1,Z
        Kpart = makeKpart(r, delta,Z1=Z)
    
        # V2 = jnp.diag(jnp.ones(Z.shape[0])) + lambda1*jnp.exp(-Kpart)
        # Kpart=evaluate_kernel(rbf_kernel,Z,Z,jnp.sqrt(abs(r)),delta)
        # V = (jnp.diag(jnp.ones(Z.shape[0])) + lambda1*Kpart)*sigmasq
        V= sigmasq*jnp.diag(jnp.ones(Z.shape[0])) + tau1*jnp.exp(-Kpart)
        # V= jnp.diag(jnp.ones(Z.shape[0]))*sigmasq + tau1*jnp.exp(-Kpart)
        # Sigma_inv=jnp.linalg.inv(V)
        # logdetV=jnp.log(jnp.linalg.det(V))
        
        mean_deviation = y - jnp.dot(X, beta)
        # r_prior_potential = jnp.sum(0.5 * jnp.log(2 * jnp.pi * sigma_r ** 2) + 0.5 * r ** 2 / sigma_r ** 2)
        r_prior_potential = r_b*jnp.sum(r)-(r_a-1)*jnp.sum(jnp.log(r))
        # sigmasq_prior_potential=sigma_b/sigmasq-(sigma_a-1)*jnp.log(1/sigmasq)
        sigmasq_prior_potential=sigma_b*sigmasq-(sigma_a-1)*jnp.log(sigmasq)

        # lambda1_prior_potential=lambda_b*lambda1-(lambda_a-1)*jnp.log(lambda1)
        lambda1_prior_potential=lambda_b*tau1-(lambda_a-1)*jnp.log(tau1)
        # Vinv=jnp.linalg.inv(V)
        # gaussian_potential = 0.5 * jnp.log(jnp.prod(2 * jnp.pi * jnp.ones_like(y))*sigmasq* jnp.linalg.det(V))+ 0.5 * jnp.dot(jnp.linalg.solve(V, mean_deviation), mean_deviation)
        # gaussian_potential = 0.5 * jnp.log(sigmasq)+0.5* jnp.linalg.slogdet(V)[1]+ 0.5 * jnp.dot(jnp.linalg.solve(V, mean_deviation), mean_deviation)/sigmasq
        gaussian_potential = 0.5* jnp.linalg.slogdet(V)[1]+ 0.5 * jnp.dot( mean_deviation,jnp.linalg.solve(V, mean_deviation))
        # gaussian_potential = 0.5* jnp.linalg.slogdet(V)[1]+ 0.5 * jnp.dot(jnp.dot(mean_deviation,Vinv), mean_deviation)

        # gaussian_potential = -jax.scipy.stats.multivariate_normal.logpdf(x=y.T, mean=jnp.dot(X, beta), cov=V)

        # +0.5*jnp.linalg.slogdet(V)[1]
        # -jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(V))))
        
        potential = gaussian_potential + r_prior_potential + sigmasq_prior_potential + lambda1_prior_potential+ boundary_potential
        # potential = jnp.asarray(  r_prior_potential + sigmasq_prior_potential + lambda1_prior_potential+ boundary_potential)

        return np.reshape(potential, ())
    return bkmr_potential