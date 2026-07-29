import os
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
import jaxopt
import cupy as cp

# Ensure GPU selection and memory sharing between JAX and CuPy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
jax.config.update("jax_enable_x64", True)

# =====================================================================
# 1. Native CuPy GPU Exact SRE Implementation (FWHT Algorithm)
# =====================================================================
def create_unnormalized_hadamard_gpu(n_qubits: int) -> cp.ndarray:
    H1 = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64)
    H_n = H1
    for _ in range(n_qubits - 1):
        H_n = cp.kron(H_n, H1)
    return H_n

_N_QUBITS_CACHED = None
_H_GPU = None
_XOR_INDICES_GPU = None
_PHASES_GPU = None

def _init_sre_gpu_cache(n_qubits: int = 7):
    global _N_QUBITS_CACHED, _H_GPU, _XOR_INDICES_GPU, _PHASES_GPU
    if _N_QUBITS_CACHED != n_qubits:
        dim = 2**n_qubits
        _H_GPU = create_unnormalized_hadamard_gpu(n_qubits)
        x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
        b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
        _XOR_INDICES_GPU = x_idx ^ b_idx
        
        # Precompute Pauli Y phases: i^(num_y)
        z_grid = np.arange(dim, dtype=np.int32)[:, None]
        x_grid = np.arange(dim, dtype=np.int32)[None, :]
        zx_and = z_grid & x_grid
        num_y_np = np.vectorize(lambda v: bin(v).count('1'))(zx_and)
        phases_np = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[num_y_np % 4]
        _PHASES_GPU = cp.array(phases_np)
        _N_QUBITS_CACHED = n_qubits

def compute_sre_native_cupy_batch(psi_batch_np: np.ndarray, sub_batch_size: int = 500) -> np.ndarray:
    """
    Computes exact S2 (alpha=2) SRE for a batch of state vectors natively on GPU using CuPy.
    psi_batch_np: np.ndarray of shape (num_starts, 2^n_qubits), complex128
    """
    num_starts, dim = psi_batch_np.shape
    n_qubits = int(np.log2(dim))
    _init_sre_gpu_cache(n_qubits)
    
    results = []
    for i in range(0, num_starts, sub_batch_size):
        sub_batch_np = psi_batch_np[i : i + sub_batch_size]
        psi_gpu = cp.array(sub_batch_np, dtype=cp.complex128)
        norms = cp.linalg.norm(psi_gpu, axis=1, keepdims=True)
        norms = cp.where(norms > 1e-12, norms, 1.0)
        psi_gpu = psi_gpu / norms
        
        psi_conj = cp.conj(psi_gpu)[:, :, None]
        psi_xor = psi_gpu[:, _XOR_INDICES_GPU]
        V_complex = psi_conj * psi_xor
        
        Xi_complex = cp.matmul(_H_GPU, V_complex)
        expval_pauli = cp.real(_PHASES_GPU[None, :, :] * Xi_complex)
        
        pauli_4_sum = cp.sum(expval_pauli ** 4, axis=(1, 2))
        sre_sub = -cp.log2(pauli_4_sum / dim)
        results.append(cp.asnumpy(sre_sub))
        
        # Free memory pool
        del psi_gpu, psi_conj, psi_xor, V_complex, Xi_complex, expval_pauli, pauli_4_sum
        cp.get_default_memory_pool().free_all_blocks()
    
    return np.concatenate(results)

# =====================================================================
# 2. JAX Optimization Helpers
# =====================================================================
def generate_stabilizer_states_and_generators(n_qubits, k, num_starts, seed=42):
    key = random.PRNGKey(seed)
    dim = 2**n_qubits
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
        
        psi = random.normal(subkey2, (dim,)) + 1j * random.normal(subkey2, (dim,))
        psi = psi / jnp.linalg.norm(psi)
        
        if k > 0:
            P_proj = jnp.eye(dim, dtype=jnp.complex128)
            for g in generators:
                P_proj = P_proj @ (jnp.eye(dim, dtype=jnp.complex128) + g) / 2.0
            psi = P_proj @ psi
            norm = jnp.linalg.norm(psi)
            if norm < 1e-6:
                psi = random.normal(subkey2, (dim,)) + 1j * random.normal(subkey2, (dim,))
                psi = psi / jnp.linalg.norm(psi)
            else:
                psi = psi / norm

        states.append(psi)
        
        if k > 0:
            generators_list.append(jnp.stack(generators))
        else:
            generators_list.append(jnp.zeros((1, dim, dim), dtype=jnp.complex128))
            
    return jnp.stack(states), jnp.stack(generators_list)

def objective_fn(params, n_dim, generators, k):
    real_part = params[:n_dim]
    imag_part = params[n_dim:]
    psi = real_part + 1j * imag_part
    norm = jnp.linalg.norm(psi)
    psi = psi / norm

    purity_loss = 0.0
    for i in range(n_dim):
        rho_i = jnp.outer(psi, jnp.conj(psi))
        tr_rho_2 = jnp.real(jnp.trace(rho_i @ rho_i))
        purity_loss += (tr_rho_2 - 1.0)**2

    violation_loss = 0.0
    if k > 0:
        for g in generators:
            expval = jnp.real(jnp.vdot(psi, g @ psi))
            violation_loss += (expval - 1.0)**2

    return purity_loss + violation_loss

def calculate_metrics(params, n_dim, generators, k):
    real_part = params[:n_dim]
    imag_part = params[n_dim:]
    psi = real_part + 1j * imag_part
    psi = psi / jnp.linalg.norm(psi)

    purities = []
    for i in range(n_dim):
        rho_i = jnp.outer(psi, jnp.conj(psi))
        tr_rho_2 = jnp.real(jnp.trace(rho_i @ rho_i))
        purities.append(tr_rho_2)

    purities = jnp.array(purities)
    avg_p = jnp.mean(purities)
    max_p = jnp.max(purities)

    viol = 0.0
    if k > 0:
        for g in generators:
            expval = jnp.real(jnp.vdot(psi, g @ psi))
            viol += (expval - 1.0)**2

    return avg_p, max_p, viol

# =====================================================================
# 3. Main Data Generation Loop
# =====================================================================
def run_simulation():
    n_qubits = 7
    num_starts = 2000
    num_steps = 1500
    chunk_size = 50
    num_chunks = num_steps // chunk_size

    n_dim = 2**n_qubits
    out_dir = "zip2"
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, f"{n_qubits}_qbt_{num_starts}_sds_ptmzng_jfr_")

    print("=======================================================")
    print(f"FAST NATIVE GPU DATA GENERATION: {n_qubits} Qubits | {num_starts} Seeds | {num_steps} Steps")
    print(f"Output directory: {out_dir}/")
    print("=======================================================")

    for k in range(n_qubits, -1, -1):
        print(f"\nComputing for {n_qubits}-qubits with initial {k}-qubit stabilized state (Gap {k}).")
        
        initial_states_jax, generators_jax = generate_stabilizer_states_and_generators(
            n_qubits, k, num_starts, seed=42 + k
        )
        
        params_batch = jnp.hstack([
            jnp.real(initial_states_jax),
            jnp.imag(initial_states_jax)
        ])
        
        def single_opt(p, g):
            solver = jaxopt.LBFGS(
                fun=lambda x: objective_fn(x, n_dim, g, k),
                maxiter=chunk_size,
                tol=1e-11,
                history_size=20
            )
            return solver.run(p)

        vmapped_run = vmap(single_opt, in_axes=(0, 0))
        vmapped_metrics = vmap(lambda p, g: calculate_metrics(p, n_dim, g, k), in_axes=(0, 0))

        # Initial metrics & SRE (Step 0)
        avg_p, max_p, viol = vmapped_metrics(params_batch, generators_jax)
        params_np = np.array(params_batch)
        states_complex = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
        
        t0_gap = time.time()
        initial_sre = compute_sre_native_cupy_batch(states_complex)
        
        print(f"  Start Step 0 | Mean Violation: {float(jnp.mean(viol)):.2e} | Mean SRE: {np.mean(initial_sre):.4f}")

        purities_history = [np.array(avg_p)]
        max_purities_history = [np.array(max_p)]
        violations_history = [np.array(viol)]
        sre_history = [initial_sre]
        opt_history = []

        initial_states_np = np.copy(states_complex)

        for chunk_idx in range(1, num_chunks + 1):
            t_chunk_start = time.time()
            res = vmapped_run(params_batch, generators_jax)
            params_batch = res.params

            avg_p, max_p, viol = vmapped_metrics(params_batch, generators_jax)
            purities_history.append(np.array(avg_p))
            max_purities_history.append(np.array(max_p))
            violations_history.append(np.array(viol))

            params_np = np.array(params_batch)
            states_complex = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
            
            # Fast Native CuPy GPU SRE (takes ~0.12s)
            sre_vals = compute_sre_native_cupy_batch(states_complex)
            sre_history.append(sre_vals)

            step_num = chunk_idx * chunk_size
            elapsed = time.time() - t0_gap
            est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
            
            print(f"  Step {step_num:5d}/{num_steps} | Mean Violation: {float(jnp.mean(viol)):.2e} | Mean SRE: {np.mean(sre_vals):.4f} | Elapsed: {elapsed:.1f}s | Est. Remaining: {est_rem:.1f}s")

        final_states_np = np.copy(states_complex)

        # Transpose histories to match (num_starts, num_steps_checkpoints)
        purities_history = np.array(purities_history).T.tolist()
        max_purities_history = np.array(max_purities_history).T.tolist()
        violations_history = np.array(violations_history).T.tolist()
        sre_history = np.array(sre_history).T.tolist()

        save_filepath = f"{out_prefix}{num_steps}_stps_{k}.pkl"
        results_dict = {
            'initial_states': initial_states_np,
            'final_states': final_states_np,
            'average_purity': purities_history,
            'max_purity': max_purities_history,
            'total_violation': violations_history,
            'sre': sre_history,
            'opt_history': opt_history
        }

        with open(save_filepath, 'wb') as f:
            pickle.dump(results_dict, f)

        print(f"Saved completed trajectory data to {save_filepath}")

if __name__ == "__main__":
    run_simulation()
