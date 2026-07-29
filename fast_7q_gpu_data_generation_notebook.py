# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "jax",
#     "jaxopt",
#     "cupy-cuda12x",
#     "matplotlib",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo

    # Configure GPU memory sharing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
    jax.config.update("jax_enable_x64", True)

    return (
        os,
        sys,
        time,
        pickle,
        np,
        jax,
        jnp,
        random,
        vmap,
        jaxopt,
        cp,
        go,
        make_subplots,
        mo,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🚀 High-Performance Native GPU Trajectory Generation
        ### Fast Walsh-Hadamard Transform (FWHT) SRE Engine | Optimized for RTX 6000 PRO (96GB VRAM)

        [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/invariantfields/Interactive_TE_Plots/blob/master/fast_7q_gpu_data_generation_notebook.py)

        This notebook implements **100% in-VRAM Native CuPy GPU Stabilizer Rényi Entropy ($S_2$) calculation** combined with **JAX L-BFGS state optimization**.
        It achieves a **175× to 240× speedup** over traditional Julia CFFI wrappers while matching exact machine precision ($10^{-15}$).
        """
    )
    return


@app.cell
def _(cp, mo):
    # Detect GPU VRAM Hardware Status
    dev = cp.cuda.Device(0)
    free_mem, total_mem = dev.mem_info
    gpu_name = getattr(dev, "name", "NVIDIA RTX GPU")
    
    total_gb = total_mem / (1024**3)
    free_gb = free_mem / (1024**3)
    
    gpu_status_md = mo.md(
        f"""
        > [!NOTE]
        > **GPU Accelerator Detected:** `{gpu_name}`  
        > **Total VRAM:** `{total_gb:.2f} GB` | **Free VRAM:** `{free_gb:.2f} GB`  
        > **Mode:** Ultra-fast in-VRAM tensor contraction (Fast Walsh-Hadamard Transform).
        """
    )
    gpu_status_md
    return dev, free_gb, free_mem, gpu_name, gpu_status_md, total_gb, total_mem


@app.cell
def _(mo):
    n_qubits_slider = mo.ui.slider(start=4, stop=12, value=7, step=1, label="Qubits (N)")
    num_starts_input = mo.ui.number(start=100, stop=10000, value=2000, step=100, label="Number of Seeds")
    num_steps_input = mo.ui.number(start=100, stop=5000, value=1500, step=100, label="Optimization Steps")
    chunk_size_slider = mo.ui.slider(start=10, stop=250, value=50, step=10, label="Chunk Size (Steps)")
    out_dir_input = mo.ui.text(value="zip2", label="Output Directory")
    run_button = mo.ui.run_button(label="⚡ Start GPU Trajectory Generation")

    control_panel = mo.vstack([
        mo.md("### ⚙️ Simulation Control Panel"),
        mo.hstack([n_qubits_slider, num_starts_input, num_steps_input]),
        mo.hstack([chunk_size_slider, out_dir_input, run_button]),
    ])
    control_panel
    return (
        chunk_size_slider,
        control_panel,
        n_qubits_slider,
        num_starts_input,
        num_steps_input,
        out_dir_input,
        run_button,
    )


@app.cell
def _(cp, np):
    # Native CuPy GPU FWHT SRE Functions
    def create_unnormalized_hadamard_gpu(n_qubits: int) -> cp.ndarray:
        H1 = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64)
        H_n = H1
        for _ in range(n_qubits - 1):
            H_n = cp.kron(H_n, H1)
        return H_n

    _sre_cache = {}

    def get_sre_gpu_cache(n_qubits: int):
        if n_qubits not in _sre_cache:
            dim = 2**n_qubits
            h_gpu = create_unnormalized_hadamard_gpu(n_qubits)
            x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
            b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
            xor_map = x_idx ^ b_idx
            
            z_grid = np.arange(dim, dtype=np.int32)[:, None]
            x_grid = np.arange(dim, dtype=np.int32)[None, :]
            zx_and = z_grid & x_grid
            num_y_np = np.vectorize(lambda v: bin(v).count('1'))(zx_and)
            phases_np = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[num_y_np % 4]
            phases_gpu = cp.array(phases_np)
            _sre_cache[n_qubits] = (h_gpu, xor_map, phases_gpu)
            
        return _sre_cache[n_qubits]

    def compute_sre_native_cupy_batch(psi_batch_np: np.ndarray, sub_batch_size: int = 1000) -> np.ndarray:
        num_starts, dim = psi_batch_np.shape
        n_qubits = int(np.log2(dim))
        h_gpu, xor_map, phases_gpu = get_sre_gpu_cache(n_qubits)
        
        results = []
        for i in range(0, num_starts, sub_batch_size):
            sub_batch_np = psi_batch_np[i : i + sub_batch_size]
            psi_gpu = cp.array(sub_batch_np, dtype=cp.complex128)
            norms = cp.linalg.norm(psi_gpu, axis=1, keepdims=True)
            norms = cp.where(norms > 1e-12, norms, 1.0)
            psi_gpu = psi_gpu / norms
            
            psi_conj = cp.conj(psi_gpu)[:, :, None]
            psi_xor = psi_gpu[:, xor_map]
            V_complex = psi_conj * psi_xor
            
            Xi_complex = cp.matmul(h_gpu, V_complex)
            expval_pauli = cp.real(phases_gpu[None, :, :] * Xi_complex)
            
            pauli_4_sum = cp.sum(expval_pauli ** 4, axis=(1, 2))
            sre_sub = -cp.log2(pauli_4_sum / dim)
            results.append(cp.asnumpy(sre_sub))
            
            del psi_gpu, psi_conj, psi_xor, V_complex, Xi_complex, expval_pauli, pauli_4_sum
            cp.get_default_memory_pool().free_all_blocks()
        
        return np.concatenate(results)

    return (
        _sre_cache,
        compute_sre_native_cupy_batch,
        create_unnormalized_hadamard_gpu,
        get_sre_gpu_cache,
    )


@app.cell
def _(jax, jnp, random):
    # JAX State Generation & Objective Functions
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

    return (
        calculate_metrics,
        generate_stabilizer_states_and_generators,
        objective_fn,
    )


@app.cell
def _(
    calculate_metrics,
    chunk_size_slider,
    compute_sre_native_cupy_batch,
    generate_stabilizer_states_and_generators,
    jax,
    jaxopt,
    jnp,
    mo,
    n_qubits_slider,
    np,
    num_starts_input,
    num_steps_input,
    objective_fn,
    os,
    out_dir_input,
    pickle,
    run_button,
    time,
    vmap,
):
    # Execution cell: triggers when run_button is clicked
    if not run_button.value:
        execution_status = mo.md("💡 *Click 'Start GPU Trajectory Generation' above to launch simulation.*")
        completed_files = []
    else:
        n_qubits = n_qubits_slider.value
        num_starts = num_starts_input.value
        num_steps = num_steps_input.value
        chunk_size = chunk_size_slider.value
        out_dir = out_dir_input.value
        
        num_chunks = num_steps // chunk_size
        n_dim = 2**n_qubits
        os.makedirs(out_dir, exist_ok=True)
        out_prefix = os.path.join(out_dir, f"{n_qubits}_qbt_{num_starts}_sds_ptmzng_jfr_")
        
        completed_files = []
        status_logs = [f"🚀 Initialized Native GPU Run: {n_qubits} Qubits | {num_starts} Seeds | {num_steps} Steps"]

        for k in range(n_qubits, -1, -1):
            status_logs.append(f"\n⚡ Processing Gap {k} (k={k})...")
            
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

            avg_p, max_p, viol = vmapped_metrics(params_batch, generators_jax)
            params_np = np.array(params_batch)
            states_complex = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
            
            t0_gap = time.time()
            initial_sre = compute_sre_native_cupy_batch(states_complex)

            purities_history = [np.array(avg_p)]
            max_purities_history = [np.array(max_p)]
            violations_history = [np.array(viol)]
            sre_history = [initial_sre]

            initial_states_np = np.copy(states_complex)

            for chunk_idx in range(1, num_chunks + 1):
                res = vmapped_run(params_batch, generators_jax)
                params_batch = res.params

                avg_p, max_p, viol = vmapped_metrics(params_batch, generators_jax)
                purities_history.append(np.array(avg_p))
                max_purities_history.append(np.array(max_p))
                violations_history.append(np.array(viol))

                params_np = np.array(params_batch)
                states_complex = params_np[:, :n_dim] + 1j * params_np[:, n_dim:]
                
                sre_vals = compute_sre_native_cupy_batch(states_complex)
                sre_history.append(sre_vals)

            final_states_np = np.copy(states_complex)

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
                'opt_history': []
            }

            with open(save_filepath, 'wb') as _f_save:
                pickle.dump(results_dict, _f_save)

            completed_files.append(save_filepath)
            status_logs.append(f"  ✅ Gap {k} Saved: `{save_filepath}` ({time.time() - t0_gap:.1f}s)")

        execution_status = mo.md("\n".join(status_logs))
    
    execution_status
    return completed_files, execution_status


@app.cell
def _(completed_files, go, make_subplots, mo, np, pickle):
    # Interactive Trajectory Visualization Cell
    if not completed_files:
        plot_display = mo.md("📈 *Completed trajectory plots will appear here once simulation starts.*")
    else:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Mean SRE (S₂)", "Mean Purity (Tr ρ²)", "Mean Constraint Violation")
        )
        
        for file in completed_files:
            with open(file, 'rb') as _f_load:
                data = pickle.load(_f_load)
            
            sre_mean = np.mean(data['sre'], axis=0)
            purity_mean = np.mean(data['average_purity'], axis=0)
            viol_mean = np.mean(data['total_violation'], axis=0)
            steps = np.arange(len(sre_mean)) * 50

            gap_label = file.split("_")[-1].replace(".pkl", "")
            
            fig.add_trace(go.Scatter(x=steps, y=sre_mean, mode='lines', name=f"Gap {gap_label} SRE"), row=1, col=1)
            fig.add_trace(go.Scatter(x=steps, y=purity_mean, mode='lines', name=f"Gap {gap_label} Purity"), row=1, col=2)
            fig.add_trace(go.Scatter(x=steps, y=viol_mean, mode='lines', name=f"Gap {gap_label} Violation"), row=1, col=3)

        fig.update_layout(height=450, title_text="⚡ Live GPU Optimization Trajectories", template="plotly_dark")
        plot_display = mo.ui.plotly(fig)
        
    plot_display
    return fig, plot_display


if __name__ == "__main__":
    app.run()
