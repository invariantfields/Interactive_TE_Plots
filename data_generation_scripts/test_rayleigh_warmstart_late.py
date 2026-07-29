import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

dim = 8
num_states = 2000

key = jax.random.PRNGKey(42)
z = jax.random.normal(key, (num_states, dim, dim)) + 1j * jax.random.normal(key, (num_states, dim, dim))
rhos_0 = jnp.matmul(z, jnp.conj(jnp.swapaxes(z, -1, -2)))
rhos_0 = rhos_0 / jnp.trace(rhos_0, axis1=-2, axis2=-1)[:, None, None]

evals_0, evecs_0 = jnp.linalg.eigh(rhos_0)

# Simulate LATE-STAGE optimization step where delta_rho is very small (1e-4)
delta_late = 1e-4 * (jax.random.normal(key, (num_states, dim, dim)) + 1j * jax.random.normal(key, (num_states, dim, dim)))
delta_late = delta_late + jnp.conj(jnp.swapaxes(delta_late, -1, -2))
rhos_late = rhos_0 + delta_late
rhos_late = rhos_late / jnp.trace(rhos_late, axis1=-2, axis2=-1)[:, None, None]

# Exact eigvalsh
evals_late_exact, _ = jnp.linalg.eigh(rhos_late)

# Rayleigh Quotient Warm-Start
@jax.jit
def warmstart_rayleigh_evals(rhos, evecs_prev):
    rho_v = jnp.matmul(rhos, evecs_prev)
    v_rho_v = jnp.matmul(jnp.conj(jnp.swapaxes(evecs_prev, -1, -2)), rho_v)
    return jnp.real(jnp.diagonal(v_rho_v, axis1=-2, axis2=-1))

evals_late_warm = warmstart_rayleigh_evals(rhos_late, evecs_0)

max_err_late = float(jnp.max(jnp.abs(evals_late_exact - evals_late_warm)))
mean_err_late = float(jnp.mean(jnp.abs(evals_late_exact - evals_late_warm)))

print(f"Late-stage (delta=1e-4) Max Eigenvalue Absolute Error: {max_err_late:.2e}")
print(f"Late-stage (delta=1e-4) Mean Eigenvalue Absolute Error: {mean_err_late:.2e}")
