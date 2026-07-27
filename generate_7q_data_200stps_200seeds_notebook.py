# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "cupy-cuda12x",
#     "jax",
#     "jaxopt",
#     "juliacall==0.9.35",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
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
    import marimo as mo

    # VRAM/XLA Configurations
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    jax.config.update("jax_enable_x64", True)
    return LBFGS, combinations, cp, jax, jnp, mo, np, os, pickle, reduce, time


@app.cell
def _(mo):
    mo.md("""
    # 🧪 7-Qubit State Optimization & SRE Trajectory Generator

    Generate $N$-qubit trajectories initialized with random stabilizer-gap states, optimizing total Hildebrand violation via **JAX L-BFGS** and computing exact SRE at each step via **CUDA HadaMAG.jl**.
    """)
    return


@app.cell
def _(mo):
    n_qubits_slider = mo.ui.slider(start=2, stop=9, step=1, value=7, label="Qubits (N)")
    num_seeds_input = mo.ui.number(start=1, stop=5000, value=200, step=1, label="Num Seeds (Samples)")
    num_steps_input = mo.ui.number(start=1, stop=2000, value=200, step=1, label="Optimization Steps")
    chunk_size_input = mo.ui.number(start=1, stop=100, value=1, step=1, label="Chunk Size (step_mes)")
    run_button = mo.ui.run_button(label="🚀 Run Data Generation")

    mo.hstack(
        [
            mo.vstack([n_qubits_slider, num_seeds_input]),
            mo.vstack([num_steps_input, chunk_size_input]),
            run_button,
        ],
        align="center",
        justify="start",
        gap=3,
    )
    return (
        chunk_size_input,
        n_qubits_slider,
        num_seeds_input,
        num_steps_input,
        run_button,
    )


@app.cell
def _(np):
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

    return (generate_random_generators_symplectic,)


@app.cell
def _(cp, generate_random_generators_symplectic, reduce):
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

    return (rand_Almost_Stab_state,)


@app.cell
def _(LBFGS, combinations, jax, jnp, np, pickle, rand_Almost_Stab_state, time):
    def run_jax_gpu_optimization(
        n_qubits: int,
        num_starts: int,
        num_loops: int,
        gap: int,
        step_mes: int = 1,
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

        try:
            from juliacall import Main as jl
            jl.seval("using Logging; disable_logging(Logging.Error)")
            jl.seval("using HadaMAG")
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
            has_julia = True
        except Exception as e:
            print(f"Julia/HadaMAG.jl not available: {e}. SRE will be saved as 0.0.")
            has_julia = False

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
        if has_julia:
            sre_vals = jl.jl_compute_sre_batch(states_complex_batch, 2, n, n_dim, num_starts)
            sre_vals_cpu = np.array(sre_vals)
        else:
            sre_vals_cpu = np.zeros(num_starts)

        for i in range(num_starts):
            traj_avg_p[i].append(float(avg_p_cpu[i]))
            traj_max_p[i].append(float(max_p_cpu[i]))
            traj_viol[i].append(float(viol_cpu[i]))
            traj_sre[i].append(float(sre_vals_cpu[i]))

        num_chunks = max(1, num_loops // step_mes)

        for chunk_idx in range(1, num_chunks + 1):
            res = vmapped_run(params_batch)
            params_batch = res.params

            avg_p, max_p, viol = vmapped_metrics(params_batch)
            avg_p_cpu, max_p_cpu, viol_cpu = np.array(avg_p), np.array(max_p), np.array(viol)

            params_np = np.array(params_batch)
            states_complex_batch = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
            if has_julia:
                sre_vals = jl.jl_compute_sre_batch(states_complex_batch, 2, n, n_dim, num_starts)
                sre_vals_cpu = np.array(sre_vals)
            else:
                sre_vals_cpu = np.zeros(num_starts)

            for i in range(num_starts):
                traj_avg_p[i].append(float(avg_p_cpu[i]))
                traj_max_p[i].append(float(max_p_cpu[i]))
                traj_viol[i].append(float(viol_cpu[i]))
                traj_sre[i].append(float(sre_vals_cpu[i]))

            if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
                elapsed = time.time() - start_time
                est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
                print(f"  Step {chunk_idx * step_mes:5d}/{num_loops} | Mean Violation: {float(np.mean(viol_cpu)):.2e} | Mean SRE: {float(np.mean(sre_vals_cpu)):.4f} | Elapsed: {elapsed:.1f}s | Est. Remaining: {est_rem:.1f}s")

        final_states_list = [np.array(params_batch[i][:n_dim] + 1j * params_batch[i][n_dim:]) for i in range(num_starts)]

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

    return (run_jax_gpu_optimization,)


@app.cell
def _(os, run_jax_gpu_optimization):
    def gen_data(n_qubits: int, num_seeds: int, num_steps: int, step_mes: int, f_name: str):
        gaps = range(0, n_qubits + 1)
        for _gap in gaps:
            _ffname = f_name + str(num_steps) + "_stps_" + str(n_qubits - _gap) + ".pkl"
            if os.path.exists(_ffname):
                print(f"Skipping gap {n_qubits - _gap}: file {_ffname} already exists.")
                continue

            print(
                f"\nComputing for {n_qubits}-qubits with initial random {_gap}-qubit stabilized state."
            )
            run_jax_gpu_optimization(
                n_qubits,
                num_seeds,
                num_steps,
                n_qubits - _gap,
                step_mes=step_mes,
                filename=_ffname,
            )

    return (gen_data,)


@app.cell
def _(
    chunk_size_input,
    gen_data,
    mo,
    n_qubits_slider,
    num_seeds_input,
    num_steps_input,
    os,
    run_button,
):
    is_script_mode = mo.app_meta().mode == "script"
    should_run = is_script_mode or run_button.value

    if should_run:
        os.makedirs("correct_data", exist_ok=True)
        N = n_qubits_slider.value
        seeds = num_seeds_input.value
        steps = num_steps_input.value
        step_mes = chunk_size_input.value

        prefix = f"zip1/{N}_qbt_{seeds}_sds_ptmzng_jfr_"
        print(f"\n=======================================================")
        print(f"Running simulation for {N} Qubits | {seeds} Seeds | {steps} Steps | Chunk size = {step_mes}")
        print(f"Output prefix: {prefix}")
        print(f"=======================================================")

        gen_data(
            n_qubits=N,
            num_seeds=seeds,
            num_steps=steps,
            step_mes=step_mes,
            f_name=prefix,
        )
        status_md = mo.md(f"✅ **Completed simulation run!** Data saved under prefix `{prefix}`")
    else:
        status_md = mo.md("👆 Adjust parameters above and click **Run Data Generation** to start simulation.")

    status_md
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
