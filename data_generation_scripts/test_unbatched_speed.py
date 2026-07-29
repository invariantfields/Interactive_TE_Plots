import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
import jaxopt

jax.config.update("jax_enable_x64", True)

n_qubits = 7
num_starts = 2000
n_dim = 128
k = 7
chunk_size = 50

# Generate 2000 states & generators
key = random.PRNGKey(42)
pauli_matrices = [
    jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128),
    jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
    jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128),
    jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
]

def get_pauli_string(indices):
    mat = jnp.array([[1.0]], dtype=jnp.complex128)
    for idx in indices:
        mat = jnp.kron(mat, pauli_matrices[idx])
    return mat

states = []
generators_list = []
for start_idx in range(num_starts):
    key, subkey1, subkey2 = random.split(key, 3)
    pauli_indices = random.randint(subkey1, (k, n_qubits), 0, 4)
    generators = [get_pauli_string(indices) for indices in pauli_indices]
    psi = random.normal(subkey2, (n_dim,)) + 1j * random.normal(subkey2, (n_dim,))
    psi = psi / jnp.linalg.norm(psi)
    states.append(psi)
    generators_list.append(jnp.stack(generators))

states_jax = jnp.stack(states)
generators_jax = jnp.stack(generators_list)

params_batch = jnp.hstack([jnp.real(states_jax), jnp.imag(states_jax)])

def objective_fn(params, n_dim, generators, k):
    real_part = params[:n_dim]
    imag_part = params[n_dim:]
    psi = real_part + 1j * imag_part
    norm_sq = jnp.sum(real_part**2 + imag_part**2)
    psi_normed = psi / jnp.sqrt(jnp.maximum(norm_sq, 1e-12))
    purity_loss = (norm_sq - 1.0)**2
    expvals = jnp.real(jnp.einsum('i,kij,j->k', jnp.conj(psi_normed), generators, psi_normed))
    violation_loss = jnp.sum((expvals - 1.0)**2)
    return purity_loss + violation_loss

def single_opt(p, g):
    solver = jaxopt.LBFGS(
        fun=lambda x: objective_fn(x, n_dim, g, k),
        maxiter=chunk_size,
        tol=1e-11,
        history_size=5
    )
    return solver.run(p).params

vmapped_run = vmap(single_opt, in_axes=(0, 0))

print("Warming up JIT...")
t0 = time.time()
params_opt = vmapped_run(params_batch, generators_jax).block_until_ready()
print(f"JIT Warmup + 50 steps: {time.time() - t0:.2f}s")

# Test 1 chunk (50 steps)
t0 = time.time()
params_opt = vmapped_run(params_batch, generators_jax).block_until_ready()
t_chunk = time.time() - t0
print(f"50 steps execution for ALL 2000 seeds in 1 vmap: {t_chunk:.3f}s")
print(f"Est. 1500 steps (30 chunks) per gap: {t_chunk * 30:.1f}s ({t_chunk * 30 / 60:.2f} min)")
