import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import jaxopt
from itertools import combinations

jax.config.update("jax_enable_x64", True)

# Benchmark 2000 seeds, 7 qubits
n_qubits = 7
num_starts = 2000
n_dim = 2**n_qubits
k = n_qubits // 2
k_dim = 2**k

combos = list(combinations(range(n_qubits), k))
perms_list = []
for combo in combos:
    keep = list(combo)
    trace = [i for i in range(n_qubits) if i not in keep]
    perms_list.append(tuple(trace + keep))

# Create dummy rhos & evecs for 2000 seeds
key = jax.random.PRNGKey(42)
z = jax.random.normal(key, (num_starts, len(perms_list), k_dim, k_dim)) + 1j * jax.random.normal(key, (num_starts, len(perms_list), k_dim, k_dim))
rhos = jnp.matmul(z, jnp.conj(jnp.swapaxes(z, -1, -2)))
rhos = rhos / jnp.trace(rhos, axis1=-2, axis2=-1)[:, :, None, None]

# 1. Benchmark Exact Eigvalsh
@jax.jit
def exact_eigvalsh_batch(rhos_batch):
    ex = jnp.linalg.eigvalsh(rhos_batch)
    rhs = ex[:, :, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, :, 0] * ex[:, :, 2], 1e-15))
    viols = jnp.maximum(0.0, ex[:, :, -1] - rhs)
    return jnp.sum(viols**2, axis=-1)

# Warmup JIT
_ = exact_eigvalsh_batch(rhos).block_until_ready()

t0 = time.time()
for _ in range(50):
    _ = exact_eigvalsh_batch(rhos)
_ = _.block_until_ready()
t_exact_50 = time.time() - t0

# 2. Benchmark Warm-Started Rayleigh Quotient
# Pre-compute evecs on Step 0
_, evecs_prev = jnp.linalg.eigh(rhos)

@jax.jit
def warmstart_rayleigh_batch(rhos_batch, evecs_batch):
    rho_v = jnp.matmul(rhos_batch, evecs_batch)
    v_rho_v = jnp.matmul(jnp.conj(jnp.swapaxes(evecs_batch, -1, -2)), rho_v)
    ex = jnp.real(jnp.diagonal(v_rho_v, axis1=-2, axis2=-1))
    rhs = ex[:, :, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, :, 0] * ex[:, :, 2], 1e-15))
    viols = jnp.maximum(0.0, ex[:, :, -1] - rhs)
    return jnp.sum(viols**2, axis=-1)

# Warmup JIT
_ = warmstart_rayleigh_batch(rhos, evecs_prev).block_until_ready()

t0 = time.time()
for _ in range(50):
    _ = warmstart_rayleigh_batch(rhos, evecs_prev)
_ = _.block_until_ready()
t_warm_50 = time.time() - t0

print(f"50 iterations of Exact Eigvalsh for 2000 seeds: {t_exact_50:.4f}s ({t_exact_50/50*1000:.2f} ms / step)")
print(f"50 iterations of Warm-Started Rayleigh for 2000 seeds: {t_warm_50:.4f}s ({t_warm_50/50*1000:.2f} ms / step)")
print(f"Speedup Factor: {t_exact_50 / t_warm_50:.2f}x")
