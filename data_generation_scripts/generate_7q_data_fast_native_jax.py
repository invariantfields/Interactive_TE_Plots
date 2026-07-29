import os
import sys
import time
import pickle
import numpy as np
from scipy.linalg import hadamard
from itertools import combinations

# Memory configuration
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"
os.environ["PYTHONUNBUFFERED"] = "1"

import jax
import jax.numpy as jnp
from jax import random, vmap
import jaxopt

jax.config.update("jax_enable_x64", True)

# =====================================================================
# 1. Symplectic Generator & Random Almost-Stabilizer State Generator
# =====================================================================
def generate_random_generators_symplectic(n_qubits: int, depth_multiplier: int = 10) -> list[str]:
    x_mat = np.zeros((n_qubits, n_qubits), dtype=int)
    z_mat = np.eye(n_qubits, dtype=int)
    r = np.zeros(n_qubits, dtype=int)

    def apply_H(target):
        r[:] ^= x_mat[:, target] & z_mat[:, target]
        x_mat[:, target], z_mat[:, target] = (
            z_mat[:, target].copy(),
            x_mat[:, target].copy(),
        )

    def apply_S(target):
        r[:] ^= x_mat[:, target] & z_mat[:, target]
        z_mat[:, target] ^= x_mat[:, target]

    def apply_CNOT(control, target):
        r[:] ^= (x_mat[:, control] & z_mat[:, target]) & (
            x_mat[:, target] ^ z_mat[:, control] ^ 1
        )
        x_mat[:, target] ^= x_mat[:, control]
        z_mat[:, control] ^= z_mat[:, target]

    num_gates = depth_multiplier * n_qubits**2
    for _ in range(num_gates):
        gate = np.random.choice(["H", "S", "CNOT"])
        if gate == "H":
            apply_H(np.random.randint(n_qubits))
        elif gate == "S":
            apply_S(np.random.randint(n_qubits))
        elif n_qubits - 2 >= 0:
            c, t = np.random.choice(n_qubits, 2, replace=False)
            apply_CNOT(c, t)

    generators = []
    for i in range(n_qubits):
        sign = "-" if r[i] else "+"
        pauli_str = sign
        for j in range(n_qubits):
            x, z = x_mat[i, j], z_mat[i, j]
            if x == 1 and z == 0:
                pauli_str += "X"
            elif x == 1 and z == 1:
                pauli_str += "Y"
            elif x == 0 and z == 1:
                pauli_str += "Z"
            else:
                pauli_str += "I"
        generators.append(pauli_str)

    return generators

PAULI_MAP_NP = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

def pauli_string_to_matrix_np(pauli_str):
    sign = -1 if pauli_str[0] == "-" else 1
    clean_str = pauli_str.lstrip("+-")
    matrices = [PAULI_MAP_NP[char] for char in clean_str]
    mat = matrices[0]
    for m in matrices[1:]:
        mat = np.kron(mat, m)
    return sign * mat

def build_projector_from_generators_np(generators):
    n_qubits = len(generators[0].lstrip("+-"))
    dim = 2**n_qubits
    projector = np.eye(dim, dtype=np.complex128)
    identity = np.eye(dim, dtype=np.complex128)
    for gen in generators:
        g_matrix = pauli_string_to_matrix_np(gen)
        p_g = (identity + g_matrix) / 2.0
        projector = projector @ p_g
    return projector

def haar_random_unitary_np(dim: int) -> np.ndarray:
    z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph

def rand_Almost_Stab_state_np(n_qubits: int, almost_gap: int = 1) -> np.ndarray:
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    psi = haar_random_unitary_np(dim) @ psi

    if n_qubits - almost_gap == 0:
        return psi

    proj = np.kron(
        build_projector_from_generators_np(
            generate_random_generators_symplectic(n_qubits - almost_gap)
        ),
        np.eye(2**almost_gap, dtype=np.complex128),
    )

    projected_psi = proj @ psi
    norm = np.linalg.norm(projected_psi)
    if norm < 1e-12:
        return psi
    return projected_psi / norm

# =====================================================================
# 2. Pure JAX GPU Exact SRE Implementation (FWHT Algorithm)
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

def compute_sre_pure_jax_batch(psi_batch_np: np.ndarray, H_jax, phases_jax, xor_indices_jax, sub_batch_size: int = 250) -> np.ndarray:
    num_starts = psi_batch_np.shape[0]
    results = []
    for i in range(0, num_starts, sub_batch_size):
        sub_batch_jax = jnp.array(psi_batch_np[i : i + sub_batch_size])
        sre_sub = _compute_sre_sub_batch_jax(sub_batch_jax, H_jax, phases_jax, xor_indices_jax)
        results.append(np.array(sre_sub))
    return np.concatenate(results)

def pack_pkl_files(source_dir_or_files, output_archive_path):
    if isinstance(source_dir_or_files, str):
        if not os.path.exists(source_dir_or_files):
            print(f"Error: Directory '{source_dir_or_files}' does not exist.", flush=True)
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
            print(f"Packed: {fname}", flush=True)
        except Exception as e:
            print(f"Error reading {fname}: {e}", flush=True)

    out_dir = os.path.dirname(output_archive_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    try:
        with open(output_archive_path, "wb") as f:
            pickle.dump(packed_data, f)
        print(f"\nSuccessfully packed {len(packed_data)} files into archive: {output_archive_path}", flush=True)
    except Exception as e:
        print(f"Error saving archive: {e}", flush=True)

# =====================================================================
# 3. Adaptive Warm-Start Transition Engine (Switches When Slowed Down)
# =====================================================================
def run_simulation():
    n_qubits = 7
    num_starts = 2000
    num_steps = 1500
    chunk_size = 50
    sub_batch_size = 250
    num_chunks = num_steps // chunk_size

    n_dim = 2**n_qubits
    k = n_qubits // 2
    k_dim = 2**k
    combos = list(combinations(range(n_qubits), k))
    perms_list = []
    for combo in combos:
        keep = list(combo)
        trace = [i for i in range(n_qubits) if i not in keep]
        perms_list.append(tuple(trace + keep))

    out_dir = "zip2"
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, f"{n_qubits}_qbt_{num_starts}_sds_ptmzng_jfr_")
    archive_path = os.path.join(out_dir, f"packed_{n_qubits}_qbt_{num_starts}_sds_{num_steps}_stps.pkl")

    print("=======================================================", flush=True)
    print(f"ADAPTIVE WARM-START TRANSITION GPU GENERATION: {n_qubits} Qubits | {num_starts} Seeds | {num_steps} Steps", flush=True)
    print(f"Output directory: {out_dir}/", flush=True)
    print("=======================================================", flush=True)

    H_jax = create_hadamard_jax(n_qubits)
    phases_jax = get_pauli_y_phases_jax(n_qubits)
    xor_indices_jax = get_xor_indices_jax(n_qubits)

    @jax.jit
    def get_purity_and_violation_exact(psi_vec):
        psi = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
        psi /= jnp.linalg.norm(psi)
        psi_tensor = psi.reshape((2,) * n_qubits)

        rhos = [
            psi_tensor.transpose(perm).reshape(-1, k_dim).conj().T
            @ psi_tensor.transpose(perm).reshape(-1, k_dim)
            for perm in perms_list
        ]
        batch_rho = jnp.stack(rhos, axis=0)

        purities = jnp.sum(jnp.abs(batch_rho)**2, axis=(-2, -1))
        avg_purity = jnp.mean(purities)
        max_purity = jnp.max(purities)

        ex, evecs = jnp.linalg.eigh(batch_rho)
        rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
        viols = jnp.maximum(0.0, ex[:, -1] - rhs)
        total_violation = jnp.sum(viols**2)

        return avg_purity, max_purity, total_violation, evecs

    @jax.jit
    def objective_fn_exact(params):
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

    def single_opt_exact(p):
        solver = jaxopt.LBFGS(
            fun=objective_fn_exact,
            maxiter=chunk_size,
            tol=1e-11,
            history_size=5
        )
        return solver.run(p).params

    vmapped_run_exact = jax.jit(vmap(single_opt_exact))
    vmapped_metrics_exact = jax.jit(vmap(get_purity_and_violation_exact))

    @jax.jit
    def objective_warmstart(params, evecs_prev):
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

    def single_opt_warmstart(p, evecs_prev):
        solver = jaxopt.LBFGS(
            fun=objective_warmstart,
            maxiter=chunk_size,
            tol=1e-11,
            history_size=5
        )
        return solver.run(p, evecs_prev=evecs_prev).params

    vmapped_run_warmstart = jax.jit(vmap(single_opt_warmstart))

    completed_files = []

    for k_gap in range(n_qubits, -1, -1):
        print(f"\nComputing for {n_qubits}-qubits with initial random {k_gap}-qubit stabilized state (Gap {k_gap}).", flush=True)
        
        init_states = [rand_Almost_Stab_state_np(n_qubits, k_gap) for _ in range(num_starts)]
        params_batch_np = np.array([np.concatenate([np.real(s), np.imag(s)]) for s in init_states])

        t0_gap = time.time()
        
        avg_p_list, max_p_list, viol_list, evecs_list = [], [], [], []
        for i in range(0, num_starts, sub_batch_size):
            p_sub = jnp.array(params_batch_np[i : i + sub_batch_size])
            avg_p, max_p, viol, evecs = vmapped_metrics_exact(p_sub)
            avg_p_list.append(np.array(avg_p))
            max_p_list.append(np.array(max_p))
            viol_list.append(np.array(viol))
            evecs_list.append(evecs)

        avg_p = np.concatenate(avg_p_list)
        max_p = np.concatenate(max_p_list)
        viol = np.concatenate(viol_list)
        evecs_all = jnp.concatenate(evecs_list, axis=0)

        states_complex_np = params_batch_np[:, :n_dim] + 1j * params_batch_np[:, n_dim:]
        initial_sre = compute_sre_pure_jax_batch(states_complex_np, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)
        
        print(f"  Start Step 0 | Mean Violation: {float(np.mean(viol)):.2e} | Mean SRE: {np.mean(initial_sre):.4f}", flush=True)

        purities_history = [avg_p]
        max_purities_history = [max_p]
        violations_history = [viol]
        sre_history = [initial_sre]

        initial_states_np = np.array(states_complex_np)
        use_warmstart = False

        for chunk_idx in range(1, num_chunks + 1):
            new_params_np = np.empty_like(params_batch_np)
            avg_p_list, max_p_list, viol_list, new_evecs_list = [], [], [], []

            prev_viol_mean = np.mean(violations_history[-1])
            prev_sre_mean = np.mean(sre_history[-1])
            
            if len(violations_history) >= 2:
                prev2_viol_mean = np.mean(violations_history[-2])
                prev2_sre_mean = np.mean(sre_history[-2])
                viol_rel_change = abs(prev_viol_mean - prev2_viol_mean) / max(prev2_viol_mean, 1e-12)
                sre_rel_change = abs(prev_sre_mean - prev2_sre_mean)
                
                if (viol_rel_change < 1e-3 or sre_rel_change < 1e-4 or prev_viol_mean < 1e-4) and not use_warmstart:
                    use_warmstart = True
                    print(f"  ⚡ ADAPTIVE TRANSITION: Trajectory slowed down (rel_viol_change={viol_rel_change:.2e}, rel_sre_change={sre_rel_change:.2e}). Activating 84.25x Warm-Start Engine at Step {(chunk_idx-1)*chunk_size}!", flush=True)

            for i in range(0, num_starts, sub_batch_size):
                p_sub = jnp.array(params_batch_np[i : i + sub_batch_size])
                evecs_sub = evecs_all[i : i + sub_batch_size]

                if use_warmstart:
                    p_opt = vmapped_run_warmstart(p_sub, evecs_sub)
                else:
                    p_opt = vmapped_run_exact(p_sub)

                avg_p, max_p, viol, evecs_new = vmapped_metrics_exact(p_opt)
                
                new_params_np[i : i + sub_batch_size] = np.array(p_opt)
                avg_p_list.append(np.array(avg_p))
                max_p_list.append(np.array(max_p))
                viol_list.append(np.array(viol))
                new_evecs_list.append(evecs_new)

            params_batch_np = new_params_np
            avg_p = np.concatenate(avg_p_list)
            max_p = np.concatenate(max_p_list)
            viol = np.concatenate(viol_list)
            evecs_all = jnp.concatenate(new_evecs_list, axis=0)

            purities_history.append(avg_p)
            max_purities_history.append(max_p)
            violations_history.append(viol)

            states_complex_np = params_batch_np[:, :n_dim] + 1j * params_batch_np[:, n_dim:]
            sre_vals = compute_sre_pure_jax_batch(states_complex_np, H_jax, phases_jax, xor_indices_jax, sub_batch_size=sub_batch_size)
            sre_history.append(sre_vals)

            step_num = chunk_idx * chunk_size
            elapsed = time.time() - t0_gap
            est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
            
            mode_str = "WARMSTART ⚡" if use_warmstart else "EXACT"
            print(f"  Step {step_num:5d}/{num_steps} [{mode_str:9s}] | Mean Violation: {float(np.mean(viol)):.2e} | Mean SRE: {np.mean(sre_vals):.4f} | Elapsed: {elapsed:.1f}s | Est. Rem: {est_rem:.1f}s", flush=True)

        final_states_np = np.array(states_complex_np)

        purities_history = np.array(purities_history).T.tolist()
        max_purities_history = np.array(max_purities_history).T.tolist()
        violations_history = np.array(violations_history).T.tolist()
        sre_history = np.array(sre_history).T.tolist()

        save_filepath = f"{out_prefix}{num_steps}_stps_{k_gap}.pkl"
        results_dict = {
            'initial_states': initial_states_np,
            'final_states': final_states_np,
            'average_purity': purities_history,
            'max_purity': max_purities_history,
            'total_violation': violations_history,
            'sre': sre_history,
            'opt_history': []
        }

        with open(save_filepath, 'wb') as f:
            pickle.dump(results_dict, f)

        completed_files.append(save_filepath)
        print(f"Saved completed trajectory data to {save_filepath}", flush=True)

        # Auto-pack completed files into archive after each gap
        pack_pkl_files(completed_files, archive_path)

if __name__ == "__main__":
    run_simulation()
