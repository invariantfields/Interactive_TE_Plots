import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
from scipy.linalg import hadamard

jax.config.update("jax_enable_x64", True)

# Test 7 qubits, 100 seeds, 50 steps
n_qubits = 7
num_starts = 100
n_dim = 2**n_qubits
k = n_qubits - n_qubits // 2
k_dim = 2**k
rem_dim = 2**(n_qubits - k)

from itertools import combinations
combos = list(combinations(range(n_qubits), k))
perms_list = []
inv_perms_list = []
for combo in combos:
    rem = [i for i in range(n_qubits) if i not in combo]
    per = tuple(rem + list(combo))
    inv_per = tuple(np.argsort(per))
    perms_list.append(per)
    inv_perms_list.append(inv_per)

# Pure JAX single-step transform for 1 state
def apply_entanglement_step_single(psi, eps=1e-10):
    psi_tensor = psi.reshape((2,) * n_qubits)
    for per, inv_per in zip(perms_list, inv_perms_list):
        psi_perm = psi_tensor.transpose(per).reshape(rem_dim, k_dim)
        rho_A = psi_perm @ psi_perm.conj().T # (rem_dim, rem_dim)
        rho_A = (rho_A + rho_A.conj().T) / 2.0
        
        evals, evecs = jnp.linalg.eigh(rho_A)
        evals_inv_sqrt = (jnp.maximum(evals, 0.0) + eps) ** (-0.5)
        x_inv_sqrt = evecs @ jnp.diag(evals_inv_sqrt) @ evecs.conj().T
        
        # Apply (x_inv_sqrt tensor I) to psi_perm
        psi_perm_transformed = x_inv_sqrt @ psi_perm
        psi_tensor = psi_perm_transformed.reshape((2,) * n_qubits).transpose(inv_per)
        
    psi_flat = psi_tensor.flatten()
    return psi_flat / jnp.linalg.norm(psi_flat)

vmapped_step = jax.jit(vmap(apply_entanglement_step_single))

# Benchmark
key = random.PRNGKey(42)
psi_batch = random.normal(key, (num_starts, n_dim)) + 1j * random.normal(key, (num_starts, n_dim))
psi_batch = psi_batch / jnp.linalg.norm(psi_batch, axis=1, keepdims=True)

print("Warming up JIT for iterative entanglement step...")
t0 = time.time()
psi_batch = vmapped_step(psi_batch).block_until_ready()
print(f"JIT Warmup complete in {time.time() - t0:.2f}s")

t0 = time.time()
for step in range(50):
    psi_batch = vmapped_step(psi_batch)
psi_batch = psi_batch.block_until_ready()
t_50 = time.time() - t0
print(f"50 steps execution for 100 seeds: {t_50:.3f}s ({t_50/50*1000:.2f}ms per step)")
