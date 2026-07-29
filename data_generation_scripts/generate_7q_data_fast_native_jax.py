import os
import sys
import time
import pickle
import numpy as np
from scipy.linalg import hadamard
import jax
import jax.numpy as jnp
from jax import random, vmap
import jaxopt

# Memory configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60"
jax.config.update("jax_enable_x64", True)

# =====================================================================
# 1. Pure JAX GPU Exact SRE Implementation (FWHT Algorithm)
# =====================================================================
def create_hadamard_jax(n_qubits: int) -> jnp.ndarray:
    dim = 2**n_qubits
    H_np = hadamard(dim).astype(np.float64)
    return jnp.array(H_np)

def get_pauli_y_phases_jax(n_qubits: int) -> jnp.ndarray:
    dim = 2**n_qubits
    z_grid = np.arange(dim, dtype=np.int32)[:, None]
    x_grid = np.arange(dim, dtype=np.int32)[None, :]
    zx_and = z_grid & x_grid
    num_y_np = np.vectorize(lambda v: bin(v).count('1'))(zx_and)
    phases_np = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[num_y_np % 4]
    return jnp.array(phases_np)

def get_xor_indices_jax(n_qubits: int) -> jnp.ndarray:
    dim = 2**n_qubits
    x_idx = np.arange(dim, dtype=np.int32)[:, None]
    b_idx = np.arange(dim, dtype=np.int32)[None, :]
    return jnp.array(x_idx ^ b_idx)

# Pure JAX JIT-compiled SRE Sub-Batch function
@jax.jit
def _compute_sre_sub_batch_jax(psi_sub_batch, H_jax, phases_jax, xor_indices_jax):
    norms = jnp.linalg.norm(psi_sub_batch, axis=1, keepdims=True)
    norms = jnp.where(norms > 1e-12, norms, 1.0)
    psi_normed = psi_sub_batch / norms
    
    psi_conj = jnp.conj(psi_normed)[:, :, None]
    psi_xor = psi_normed[:, xor_indices_jax]
    V_complex = psi_conj * psi_xor
    
    Xi_complex = jnp.matmul(H_jax, V_complex)
    expval_pauli = jnp.real(phases_jax[None, :, :] * Xi_complex)
    
    pauli_4_sum = jnp.sum(expval_pauli ** 4, axis=(1, 2))
    dim = H_jax.shape[0]
    sre_sub = -jnp.log2(pauli_4_sum / dim)
    return sre_sub

def compute_sre_pure_jax_batch(psi_batch_jax, H_jax, phases_jax, xor_indices_jax, sub_batch_size: int = 500) -> np.ndarray:
    num_starts = psi_batch_jax.shape[0]
    results = []
    for i in range(0, num_starts, sub_batch_size):
        sub_batch = psi_batch_jax[i : i + sub_batch_size]
        sre_sub = _compute_sre_sub_batch_jax(sub_batch, H_jax, phases_jax, xor_indices_jax)
        results.append(np.array(sre_sub))
    return np.concatenate(results)

def pack_pkl_files(source_dir_or_files, output_archive_path):
    """
    Packs multiple .pkl files into a single combined .pkl archive.
    Matches the dictionary structure expected by unpack_pkl_file.

    Parameters:
    - source_dir_or_files: list of file paths OR a directory path containing .pkl files.
    - output_archive_path: destination path for the combined .pkl file.
    """
    if isinstance(source_dir_or_files, str):
        if not os.path.exists(source_dir_or_files):
            print(f"Error: Directory '{source_dir_or_files}' does not exist.")
            return
        file_paths = [
            os.path.join(source_dir_or_files, f)
            for f in sorted(os.listdir(source_dir_or_files))
            if f.endswith(".pkl") and not f.startswith("packed_")
        ]
    else:
        file_paths = list(source_dir_or_files)

    packed_data = {}
    for fpath in file_paths:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "rb") as f:
                content = pickle.load(f)
            packed_data[fname] = content
            print(f"Packed: {fname}")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # Ensure parent directory exists for output archive
    out_dir = os.path.dirname(output_archive_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    try:
        with open(output_archive_path, "wb") as f:
            pickle.dump(packed_data, f)
        print(f"\nSuccessfully packed {len(packed_data)} files into archive: {output_archive_path}")
    except Exception as e:
        print(f"Error saving archive: {e}")

# =====================================================================
# 2. Optimized Memory-Light JAX Objective & Optimization Helpers
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
    norm_sq = jnp.sum(real_part**2 + imag_part**2)
    psi_normed = psi / jnp.sqrt(jnp.maximum(norm_sq, 1e-12))

    purity_loss = (norm_sq - 1.0)**2

    violation_loss = 0.0
    if k > 0:
        expvals = jnp.real(jnp.einsum('i,kij,j->k', jnp.conj(psi_normed), generators, psi_normed))
        violation_loss = jnp.sum((expvals - 1.0)**2)

    return purity_loss + violation_loss

def calculate_metrics(params, n_dim, generators, k):
    real_part = params[:n_dim]
    imag_part = params[n_dim:]
    psi = real_part + 1j * imag_part
    norm_sq = jnp.sum(real_part**2 + imag_part**2)
    psi_normed = psi / jnp.sqrt(jnp.maximum(norm_sq, 1e-12))

    avg_p = 1.0
    max_p = 1.0

    viol = 0.0
    if k > 0:
        expvals = jnp.real(jnp.einsum('i,kij,j->k', jnp.conj(psi_normed), generators, psi_normed))
        viol = jnp.sum((expvals - 1.0)**2)

    return avg_p, max_p, viol

def run_subbatched_opt(params_batch, generators_jax, n_dim, k, chunk_size, sub_batch_size=500):
    num_starts = params_batch.shape[0]
    
    def single_opt(p, g):
        solver = jaxopt.LBFGS(
            fun=lambda x: objective_fn(x, n_dim, g, k),
            maxiter=chunk_size,
            tol=1e-11,
            history_size=10
        )
        return solver.run(p).params

    vmapped_run = vmap(single_opt, in_axes=(0, 0))
    vmapped_metrics = vmap(lambda p, g: calculate_metrics(p, n_dim, g, k), in_axes=(0, 0))

    new_params_list = []
    avg_p_list, max_p_list, viol_list = [], [], []

    for i in range(0, num_starts, sub_batch_size):
        p_sub = params_batch[i : i + sub_batch_size]
        g_sub = generators_jax[i : i + sub_batch_size]
        
        p_opt = vmapped_run(p_sub, g_sub)
        avg_p, max_p, viol = vmapped_metrics(p_opt, g_sub)
        
        new_params_list.append(p_opt)
        avg_p_list.append(avg_p)
        max_p_list.append(max_p)
        viol_list.append(viol)

    new_params = jnp.vstack(new_params_list)
    avg_p_cat = jnp.concatenate(avg_p_list)
    max_p_cat = jnp.concatenate(max_p_list)
    viol_cat = jnp.concatenate(viol_list)

    return new_params, avg_p_cat, max_p_cat, viol_cat

# =====================================================================
# 3. Main Data Generation Loop
# =====================================================================
def run_simulation():
    n_qubits = 7
    num_starts = 2000
    num_steps = 1500
    chunk_size = 50
    sub_batch_size = 500
    num_chunks = num_steps // chunk_size

    n_dim = 2**n_qubits
    out_dir = "zip2"
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, f"{n_qubits}_qbt_{num_starts}_sds_ptmzng_jfr_")
    archive_path = os.path.join(out_dir, f"packed_{n_qubits}_qbt_{num_starts}_sds_{num_steps}_stps.pkl")

    print("=======================================================")
    print(f"FAST OPTIMAL SUB-BATCHED GPU GENERATION: {n_qubits} Qubits | {num_starts} Seeds | {num_steps} Steps")
    print(f"Output directory: {out_dir}/")
    print("=======================================================")

    H_jax = create_hadamard_jax(n_qubits)
    phases_jax = get_pauli_y_phases_jax(n_qubits)
    xor_indices_jax = get_xor_indices_jax(n_qubits)

    completed_files = []

    for k in range(n_qubits, -1, -1):
        print(f"\nComputing for {n_qubits}-qubits with initial {k}-qubit stabilized state (Gap {k}).")
        
        initial_states_jax, generators_jax = generate_stabilizer_states_and_generators(
            n_qubits, k, num_starts, seed=42 + k
        )
        
        params_batch = jnp.hstack([
            jnp.real(initial_states_jax),
            jnp.imag(initial_states_jax)
        ])

        vmapped_metrics_k = vmap(lambda p, g: calculate_metrics(p, n_dim, g, k), in_axes=(0, 0))
        
        avg_p, max_p, viol = vmapped_metrics_k(params_batch, generators_jax)
        states_complex_jax = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
        
        t0_gap = time.time()
        initial_sre = compute_sre_pure_jax_batch(states_complex_jax, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)
        
        print(f"  Start Step 0 | Mean Violation: {float(jnp.mean(viol)):.2e} | Mean SRE: {np.mean(initial_sre):.4f}")

        purities_history = [np.array(avg_p)]
        max_purities_history = [np.array(max_p)]
        violations_history = [np.array(viol)]
        sre_history = [initial_sre]
        opt_history = []

        initial_states_np = np.array(states_complex_jax)

        for chunk_idx in range(1, num_chunks + 1):
            params_batch, avg_p, max_p, viol = run_subbatched_opt(
                params_batch, generators_jax, n_dim, k, chunk_size, sub_batch_size=sub_batch_size
            )

            purities_history.append(np.array(avg_p))
            max_purities_history.append(np.array(max_p))
            violations_history.append(np.array(viol))

            states_complex_jax = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
            
            sre_vals = compute_sre_pure_jax_batch(states_complex_jax, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)
            sre_history.append(sre_vals)

            step_num = chunk_idx * chunk_size
            elapsed = time.time() - t0_gap
            est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
            
            print(f"  Step {step_num:5d}/{num_steps} | Mean Violation: {float(jnp.mean(viol)):.2e} | Mean SRE: {np.mean(sre_vals):.4f} | Elapsed: {elapsed:.1f}s | Est. Remaining: {est_rem:.1f}s")

        final_states_np = np.array(states_complex_jax)

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

        completed_files.append(save_filepath)
        print(f"Saved completed trajectory data to {save_filepath}")

        # Auto-pack completed files into archive after each gap
        pack_pkl_files(completed_files, archive_path)

if __name__ == "__main__":
    run_simulation()
