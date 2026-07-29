import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import jaxopt
from itertools import combinations

jax.config.update("jax_enable_x64", True)

# Per-Seed Independent Warm-Start Transition Test
n_qubits = 7
num_starts = 100
n_dim = 2**n_qubits
k = n_qubits // 2
k_dim = 2**k

combos = list(combinations(range(n_qubits), k))
perms_list = []
for combo in combos:
    keep = list(combo)
    trace = [i for i in range(n_qubits) if i not in keep]
    perms_list.append(tuple(trace + keep))

# Exact Eigvalsh Objective for single state
def objective_exact_single(params):
    psi = params[:n_dim] + 1j * params[n_dim:]
    psi /= jnp.linalg.norm(psi)
    psi_tensor = psi.reshape((2,) * n_qubits)

    rhos = [
        psi_tensor.transpose(perm).reshape(-1, k_dim).conj().T
        @ psi_tensor.transpose(perm).reshape(-1, k_dim)
        for perm in perms_list
    ]
    batch_rho = jnp.stack(rhos, axis=0)

    ex = jnp.linalg.eigvalsh(batch_rho)
    rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
    viols = jnp.maximum(0.0, ex[:, -1] - rhs)
    return jnp.sum(viols**2)

# Warm-Started Rayleigh Objective for single state
def objective_warmstart_single(params, evecs_prev):
    psi = params[:n_dim] + 1j * params[n_dim:]
    psi /= jnp.linalg.norm(psi)
    psi_tensor = psi.reshape((2,) * n_qubits)

    rhos = [
        psi_tensor.transpose(perm).reshape(-1, k_dim).conj().T
        @ psi_tensor.transpose(perm).reshape(-1, k_dim)
        for perm in perms_list
    ]
    batch_rho = jnp.stack(rhos, axis=0)

    rho_v = jnp.matmul(batch_rho, evecs_prev)
    v_rho_v = jnp.matmul(jnp.conj(jnp.swapaxes(evecs_prev, -1, -2)), rho_v)
    ex = jnp.real(jnp.diagonal(v_rho_v, axis1=-2, axis2=-1))

    rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
    viols = jnp.maximum(0.0, ex[:, -1] - rhs)
    return jnp.sum(viols**2)

# Hybrid Per-Seed Adaptive Objective
def objective_adaptive_single(params, evecs_prev, is_warmstart):
    return jnp.where(
        is_warmstart,
        objective_warmstart_single(params, evecs_prev),
        objective_exact_single(params)
    )

def single_opt_adaptive(p, evecs_prev, is_warmstart):
    def obj(params):
        return objective_adaptive_single(params, evecs_prev, is_warmstart)
    
    solver = jaxopt.LBFGS(
        fun=obj,
        maxiter=50,
        tol=1e-11,
        history_size=5
    )
    return solver.run(p).params

vmapped_run_adaptive = jax.jit(vmap(single_opt_adaptive))

print("=======================================================================")
print("TESTING PER-SEED INDEPENDENT WARM-START TRANSITION ENGINE (100 SEEDS)")
print("=======================================================================")
print("Per-seed transition tracks each individual seed's violation rate of change!")
