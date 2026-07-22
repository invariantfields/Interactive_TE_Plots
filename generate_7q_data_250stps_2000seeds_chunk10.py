import os
import sys
import time
import pickle
import re
from itertools import combinations
from functools import reduce
import numpy as np
import cupy as cp
import jax
import jax.numpy as jnp
from jaxopt import LBFGS

# VRAM/XLA Configurations
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
jax.config.update("jax_enable_x64", True)

# =====================================================================
# 1. Symplectic Generator (NumPy / CPU)
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

# =====================================================================
# 2. Dense Projector Construction (GPU / CuPy)
# =====================================================================
PAULI_MAP = {
    "I": cp.array([[1, 0], [0, 1]], dtype=complex),
    "X": cp.array([[0, 1], [1, 0]], dtype=complex),
    "Y": cp.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": cp.array([[1, 0], [0, -1]], dtype=complex),
}

def pauli_string_to_matrix(pauli_str):
    sign = -1 if pauli_str[0] == "-" else 1
    clean_str = pauli_str.lstrip("+-")
    matrices = [PAULI_MAP[char] for char in clean_str]
    return sign * reduce(cp.kron, matrices)

def build_projector_from_generators(generators):
    n_qubits = len(generators[0].lstrip("+-"))
    dim = 2**n_qubits

    projector = cp.eye(dim, dtype=complex)
    identity = cp.eye(dim, dtype=complex)

    for gen in generators:
        g_matrix = pauli_string_to_matrix(gen)
        p_g = (identity + g_matrix) / 2.0
        projector = projector @ p_g

    return projector

def haar_random_unitary_gpu(dim: int) -> cp.ndarray:
    z = cp.random.randn(dim, dim) + 1j * cp.random.randn(dim, dim)
    q, r = cp.linalg.qr(z)
    d = cp.diagonal(r)
    ph = d / cp.abs(d)
    return q * ph

def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1) -> cp.ndarray:
    psi = cp.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0
    psi = haar_random_unitary_gpu(2**n_qubits) @ psi

    if n_qubits - almost_gap == 0:
        return psi

    proj = cp.kron(
        build_projector_from_generators(
            generate_random_generators_symplectic(n_qubits - almost_gap)
        ),
        cp.eye(2**almost_gap, dtype=complex),
    )

    projected_psi = proj @ psi
    return projected_psi / cp.linalg.norm(projected_psi)

# =====================================================================
# 3. JAX L-BFGS Optimization with Julia CUDA Batch SRE Trajectories
# =====================================================================
def run_jax_gpu_optimization(
    n_qubits: int,
    num_starts: int,
    num_loops: int,
    gap: int,
    step_mes: int = 10,
    filename: str = "jax_trajectory_data.pkl",
):
    n = n_qubits
    k = n // 2
    combos = list(combinations(range(n), k))
    n_dim = 2**n

    perms_list = []
    for combo in combos:
        keep = list(combo)
        trace = [i for i in range(n) if i not in keep]
        perms_list.append(tuple(trace + keep))

    @jax.jit
    def get_purity_and_violation(psi_vec):
        psi = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
        psi /= jnp.linalg.norm(psi)
        psi_tensor = psi.reshape((2,) * n)

        rhos = [
            psi_tensor.transpose(perm).reshape(-1, 2**k).conj().T
            @ psi_tensor.transpose(perm).reshape(-1, 2**k)
            for perm in perms_list
        ]
        batch_rho = jnp.stack(rhos, axis=0)
        ex = jnp.linalg.eigvalsh(batch_rho)

        purities = jnp.sum(ex**2, axis=-1)
        avg_purity = jnp.mean(purities)
        max_purity = jnp.max(purities)

        rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
        viols = jnp.maximum(0.0, ex[:, -1] - rhs)
        total_violation = jnp.sum(viols**2)

        return avg_purity, max_purity, total_violation

    @jax.jit
    def objective(params):
        _, _, total_viol = get_purity_and_violation(params)
        return total_viol

    solver = LBFGS(fun=objective, maxiter=step_mes, tol=1e-11)

    print("Initializing Julia/HadaMAG.jl...")
    from juliacall import Main as jl
    jl.seval("using Logging; disable_logging(Logging.Error)")
    jl.seval("using HadaMAG, CUDA")
    jl.seval("using LinearAlgebra: norm")
    jl.seval("""
    function jl_compute_sre_batch(psi_batch_np, alpha, n_qubits, dim, num_starts)
        results = zeros(Float64, num_starts)
        for i in 1:num_starts
            psi_row = psi_batch_np[i, :]
            psi_jl = Vector{ComplexF64}(psi_row)
            nrm = norm(psi_jl)
            if nrm > 1e-12
                psi_jl ./= nrm
            end
            psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            sre_result, lost_norm = SRE(psi_sv, alpha, backend= :CUDA, progress=false)
            results[i] = sre_result
        end
        return results
    end
    """)
    print("Julia HadaMAG batch SRE initialized.")

    print(f"Generating {num_starts} initial states...")
    init_params_list = []
    for st in range(num_starts):
        psi_init = rand_Almost_Stab_state(n, gap)
        if hasattr(psi_init, "get"):
            psi_np = psi_init.get()
        else:
            psi_np = np.array(psi_init)
        p = np.concatenate([np.real(psi_np), np.imag(psi_np)])
        init_params_list.append(p)
    params_batch = jnp.array(init_params_list, dtype=jnp.float64)

    vmapped_metrics = jax.vmap(get_purity_and_violation)
    vmapped_run = jax.vmap(solver.run)

    traj_avg_p = [[] for _ in range(num_starts)]
    traj_max_p = [[] for _ in range(num_starts)]
    traj_viol = [[] for _ in range(num_starts)]
    traj_sre = [[] for _ in range(num_starts)]

    start_time = time.time()

    avg_p, max_p, viol = vmapped_metrics(params_batch)
    avg_p_cpu, max_p_cpu, viol_cpu = np.array(avg_p), np.array(max_p), np.array(viol)

    params_np = np.array(params_batch)
    states_complex_batch = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
    sre_vals = jl.jl_compute_sre_batch(states_complex_batch, 2, n, n_dim, num_starts)
    sre_vals_cpu = np.array(sre_vals)

    for i in range(num_starts):
        traj_avg_p[i].append(float(avg_p_cpu[i]))
        traj_max_p[i].append(float(max_p_cpu[i]))
        traj_viol[i].append(float(viol_cpu[i]))
        traj_sre[i].append(float(sre_vals_cpu[i]))
    print(f"  Start Step | Mean Violation: {float(np.mean(viol_cpu)):.2e} | Mean SRE: {float(np.mean(sre_vals_cpu)):.4f}")

    num_chunks = max(1, num_loops // step_mes)

    for chunk_idx in range(1, num_chunks + 1):
        res = vmapped_run(params_batch)
        params_batch = res.params

        avg_p, max_p, viol = vmapped_metrics(params_batch)
        avg_p_cpu, max_p_cpu, viol_cpu = np.array(avg_p), np.array(max_p), np.array(viol)

        params_np = np.array(params_batch)
        states_complex_batch = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
        sre_vals = jl.jl_compute_sre_batch(states_complex_batch, 2, n, n_dim, num_starts)
        sre_vals_cpu = np.array(sre_vals)

        for i in range(num_starts):
            traj_avg_p[i].append(float(avg_p_cpu[i]))
            traj_max_p[i].append(float(max_p_cpu[i]))
            traj_viol[i].append(float(viol_cpu[i]))
            traj_sre[i].append(float(sre_vals_cpu[i]))

        if chunk_idx % 5 == 0 or chunk_idx == num_chunks:
            elapsed = time.time() - start_time
            rate = elapsed / chunk_idx
            remaining = rate * (num_chunks - chunk_idx)
            print(
                f"  Step {chunk_idx * step_mes:5d}/{num_loops} | "
                f"Mean Violation: {float(np.mean(viol_cpu)):.2e} | "
                f"Mean SRE: {float(np.mean(sre_vals_cpu)):.4f} | "
                f"Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s"
            )

    final_params_np = np.array(params_batch)
    final_states_batch = final_params_np[:, :n_dim] + 1j * final_params_np[:, n_dim:]
    final_states_normalized = final_states_batch / np.linalg.norm(final_states_batch, axis=1, keepdims=True)

    final_states_list = [final_states_normalized[i] for i in range(num_starts)]

    trajectories = {
        "average_purity": traj_avg_p,
        "max_purity": traj_max_p,
        "sre": traj_sre,
        "total_violation": traj_viol,
        "final_states": final_states_list,
    }

    with open(filename, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Saved completed SRE-updated data to {filename}")
    return trajectories

# =====================================================================
# 4. Data Generator Loop
# =====================================================================
def gen_data(n_qubits: int, num_seeds: int, num_steps: int, step_mes: int, f_name: str):
    gaps = range(0, n_qubits + 1)
    for _gap in gaps:
        _ffname = f_name + str(num_steps) + "_stps_" + str(n_qubits - _gap) + ".pkl"

        if os.path.exists(_ffname):
            try:
                with open(_ffname, "rb") as _f:
                    _d = pickle.load(_f)
                if len(_d["sre"][0]) > 10:
                    print(f"Skipping gap {n_qubits - _gap}: full trajectory file {_ffname} already exists.")
                    continue
            except Exception:
                pass

        print(
            f"\ncomputing for {n_qubits}-qubits with initial random {_gap}-qubit stabilized state."
        )
        run_jax_gpu_optimization(
            n_qubits,
            num_seeds,
            num_steps,
            n_qubits - _gap,
            step_mes=step_mes,
            filename=_ffname,
        )

if __name__ == "__main__":
    os.makedirs("correct_data", exist_ok=True)

    stpsi = [250]
    qbt_nmbs = [7]
    num_seds = [2000]
    chunk_size = 10

    for idx in range(len(qbt_nmbs)):
        flnm = f"correct_data/{qbt_nmbs[idx]}_qbt_{num_seds[idx]}_sds_ptmzng_jfr_"
        print(f"\n=======================================================")
        print(f"Running simulation for {qbt_nmbs[idx]} Qubits | {num_seds[idx]} Seeds | {stpsi[idx]} Steps | Chunk size = {chunk_size}")
        print(f"Output prefix: {flnm}")
        print(f"=======================================================")
        
        gen_data(
            n_qubits=qbt_nmbs[idx],
            num_seeds=num_seds[idx],
            num_steps=stpsi[idx],
            step_mes=chunk_size,
            f_name=flnm
        )
    print("All simulation runs and SRE computations completed successfully!")
