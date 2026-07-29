import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)

# Test Rayleigh Quotient Warm-Starting vs Full Eigvalsh
dim = 8 # 3-qubit marginal
num_states = 2000

# Generate 2,000 random density matrices
key = jax.random.PRNGKey(42)
z = jax.random.normal(key, (num_states, dim, dim)) + 1j * jax.random.normal(key, (num_states, dim, dim))
rhos_0 = jnp.matmul(z, jnp.conj(jnp.swapaxes(z, -1, -2)))
rhos_0 = rhos_0 / jnp.trace(rhos_0, axis1=-2, axis2=-1)[:, None, None]

# Compute exact initial eigendecomposition at Step 0
t0 = time.time()
evals_0, evecs_0 = jnp.linalg.eigh(rhos_0)
evals_0 = evals_0.block_until_ready()
t_exact_0 = time.time() - t0
print(f"Exact eigvalsh on Step 0 for 2000 states: {t_exact_0:.4f}s")

# Perturb density matrices slightly (simulating step t -> t+1)
delta = 0.01 * (jax.random.normal(key, (num_states, dim, dim)) + 1j * jax.random.normal(key, (num_states, dim, dim)))
delta = delta + jnp.conj(jnp.swapaxes(delta, -1, -2))
rhos_1 = rhos_0 + delta
rhos_1 = rhos_1 / jnp.trace(rhos_1, axis1=-2, axis2=-1)[:, None, None]

# Method A: Full Eigvalsh on Step 1
t0 = time.time()
evals_1_exact, _ = jnp.linalg.eigh(rhos_1)
evals_1_exact = evals_1_exact.block_until_ready()
t_full = time.time() - t0
print(f"Full eigvalsh on Step 1: {t_full:.4f}s")

# Method B: Rayleigh Quotient Warm-Start (V_0^dagger @ rho_1 @ V_0)
@jax.jit
def warmstart_rayleigh_evals(rhos, evecs_prev):
    # evecs_prev shape: (num_states, dim, dim)
    # rhos shape: (num_states, dim, dim)
    # V^\dagger @ rho @ V
    rho_v = jnp.matmul(rhos, evecs_prev)
    v_rho_v = jnp.matmul(jnp.conj(jnp.swapaxes(evecs_prev, -1, -2)), rho_v)
    # Extract real diagonal
    evals_approx = jnp.real(jnp.diagonal(v_rho_v, axis1=-2, axis2=-1))
    return evals_approx

t0 = time.time()
evals_1_warm = warmstart_rayleigh_evals(rhos_1, evecs_0).block_until_ready()
t_warm = time.time() - t0
print(f"Warm-started Rayleigh Quotient on Step 1: {t_warm:.4f}s ({t_full/t_warm:.1f}x FASTER!)")

# Error evaluation
max_err = float(jnp.max(jnp.abs(evals_1_exact - evals_1_warm)))
mean_err = float(jnp.mean(jnp.abs(evals_1_exact - evals_1_warm)))
print(f"Max Eigenvalue Absolute Error: {max_err:.2e}")
print(f"Mean Eigenvalue Absolute Error: {mean_err:.2e}")
