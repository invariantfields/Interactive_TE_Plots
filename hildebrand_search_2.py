# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "jax[cuda12]",
#     "jaxopt",
#     "plotly",
#     "scipy",
#     "cupy-cuda12x",
#     "juliacall",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import site

    # Fix for JAX: Point to pip-installed CUDA libraries if they exist
    try:
        _site_packages = site.getsitepackages()[0]
        _nvidia_base = os.path.join(_site_packages, "nvidia")
        if os.path.exists(_nvidia_base):
            _library_paths = []
            for _subdir in os.listdir(_nvidia_base):
                _lib_dir = os.path.join(_nvidia_base, _subdir, "lib")
                if os.path.exists(_lib_dir):
                    _library_paths.append(_lib_dir)

            _nvvm_dir = os.path.join(_nvidia_base, "cuda_nvcc", "nvvm", "lib64")
            if os.path.exists(_nvvm_dir):
                _library_paths.append(_nvvm_dir)

            _current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            _new_ld = ":".join(_library_paths + ([_current_ld] if _current_ld else []))
            os.environ["LD_LIBRARY_PATH"] = _new_ld
    except Exception:
        pass

    sys.setrecursionlimit(10000)

    from juliacall import Main as jl

    # Suppress implementation warnings
    jl.seval("using Logging; disable_logging(Logging.Warn)")
    jl.seval("using HadaMAG")
    jl.seval("using CUDA")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import cupy as cp
    from functools import reduce
    from jaxopt import LBFGS
    import plotly.graph_objects as go
    import pickle
    from itertools import combinations
    import time
    import traceback

    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    # Print JAX device info
    print(f"JAX Devices: {jax.devices()}")
    return (
        LBFGS,
        combinations,
        cp,
        go,
        jax,
        jl,
        jnp,
        mo,
        np,
        os,
        pickle,
        reduce,
    )


@app.cell
def _(mo):
    mo.md("""
    # Optimized Hildebrand Search v2 (Vectorized JAX)
    This notebook uses **Vectorized JAX** to optimize hundreds of quantum states simultaneously.

    ### Key Optimizations:
    1.  **Parallel Marginal Checks:** Checks all qubit combinations in parallel using `jax.vmap`.
    2.  **Batched Solver:** Optimizes all random samples for a given gap in a single GPU pass.
    3.  **Batched SRE:** Computes Stabilizer Rényi Entropy in bulk using Julia.
    """)
    return


@app.cell
def _(cp, jl, np):
    def compute_sre_batch(psi_matrix, alpha=2):
        """psi_matrix shape: (2^N, n_samples)"""
        try:
            dim, n_samples = psi_matrix.shape
            n_qubits = int(np.log2(dim))

            jl.psi_matrix = np.array(psi_matrix)
            jl.alpha = alpha
            jl.n_qubits = n_qubits
            jl.dim = dim
            jl.n_samples = n_samples

            # Bulk compute SRE in Julia to minimize overhead
            jl.seval(""" 
                sre_results = zeros(Float64, n_samples)
                for i in 1:n_samples
                    psi_jl = Vector{ComplexF64}(psi_matrix[:, i])
                    psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, n_qubits, dim)
                    sre_val, _ = SRE(psi_sv, alpha, backend= :CUDA)
                    sre_results[i] = isnothing(sre_val) ? NaN : sre_val
                end
            """)

            return np.array(jl.sre_results)
        except Exception as e:
            print(f"SRE Batch Error: {e}")
            return np.full(psi_matrix.shape[1], np.nan)

    def par_trace(psi, dim, n, n_parties):
        """Optimized partial trace for pure states."""
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def get_hildebrand_violation_batch(rho_batch, k_dim):
        """rho_batch shape: (n_perms, dim_k, dim_k)"""
        ex = cp.linalg.eigvalsh(rho_batch)
        rhs = ex[:, 1] + 2 * cp.sqrt(cp.maximum(ex[:, 0] * ex[:, 2], 0))
        violation = cp.maximum(0, ex[:, -1] - rhs)
        return violation

    return (compute_sre_batch,)


@app.cell
def _(cp, np, reduce):
    PAULI_MAP = {
        "I": cp.array([[1, 0], [0, 1]], dtype=complex),
        "X": cp.array([[0, 1], [1, 0]], dtype=complex),
        "Y": cp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": cp.array([[1, 0], [0, -1]], dtype=complex),
    }

    def pauli_string_to_matrix(pauli_str):
        sign = -1 if pauli_str[0] == "-" else 1
        matrices = [PAULI_MAP[char] for char in pauli_str.lstrip("+-")]
        return sign * reduce(cp.kron, matrices)

    def generate_random_generators_symplectic(n_qubits: int) -> list[str]:
        x_mat = np.zeros((n_qubits, n_qubits), dtype=int)
        z_mat = np.eye(n_qubits, dtype=int)
        r = np.zeros(n_qubits, dtype=int)
        def apply_H(target):
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            x_mat[:, target], z_mat[:, target] = z_mat[:, target].copy(), x_mat[:, target].copy()
        def apply_S(target):
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            z_mat[:, target] ^= x_mat[:, target]
        def apply_CNOT(control, target):
            r[:] ^= (x_mat[:, control] & z_mat[:, target]) & (x_mat[:, target] ^ x_mat[:, control] ^ 1)
            x_mat[:, target] ^= x_mat[:, control]
            z_mat[:, control] ^= z_mat[:, target]
        for _ in range(10 * n_qubits**2):
            gate = np.random.choice(["H", "S", "CNOT"])
            if gate == "H": apply_H(np.random.randint(n_qubits))
            elif gate == "S": apply_S(np.random.randint(n_qubits))
            elif n_qubits > 1:
                c, t = np.random.choice(n_qubits, 2, replace=False)
                apply_CNOT(c, t)
        generators = []
        for i in range(n_qubits):
            sign = "-" if r[i] else "+"
            pauli_str = sign
            for j in range(n_qubits):
                x, z = x_mat[i, j], z_mat[i, j]
                if x == 1 and z == 0: pauli_str += "X"
                elif x == 1 and z == 1: pauli_str += "Y"
                elif x == 0 and z == 1: pauli_str += "Z"
                else: pauli_str += "I"
            generators.append(pauli_str)
        return generators

    def build_projector_from_generators(generators):
        n_qubits = len(generators[0].lstrip("+-"))
        dim = 2**n_qubits
        projector = cp.eye(dim, dtype=complex)
        identity = cp.eye(dim, dtype=complex)
        for gen in generators:
            projector = projector @ ((identity + pauli_string_to_matrix(gen)) / 2.0)
        return projector

    def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1) -> cp.ndarray:
        dim = 2**n_qubits
        z = cp.random.randn(dim, dim) + 1j * cp.random.randn(dim, dim)
        q, r = cp.linalg.qr(z)
        psi = q[:, 0]
        if n_qubits - almost_gap <= 0: return psi
        proj = cp.kron(build_projector_from_generators(generate_random_generators_symplectic(n_qubits - almost_gap)), cp.eye(2**almost_gap, dtype=complex))
        projected_psi = proj @ psi
        return projected_psi / cp.linalg.norm(projected_psi)

    return (rand_Almost_Stab_state,)


@app.cell
def _(combinations, jax, jnp):
    def get_search_logic_vectorized(n, k):
        combos = list(combinations(range(n), k))
        # Keep perms as a list of concrete tuples for JAX static indexing
        perms = [tuple([i for i in range(n) if i not in combo] + list(combo)) for combo in combos]

        @jax.jit
        def objective_single(params):
            n_dim = 2**n
            psi = params[:n_dim] + 1j * params[n_dim:]
            psi /= jnp.linalg.norm(psi)
            psi_tensor = psi.reshape((2,) * n)

            total_violation = 0.0

            # Use a Python loop to keep the indices concrete during JIT
            for perm in perms:
                psi_perm = psi_tensor.transpose(perm).reshape(-1, 2**k)
                rho = psi_perm.conj().T @ psi_perm
                ex = jnp.linalg.eigvalsh(rho)
                # Sqrt regularization fix
                rhs = ex[1] + 2 * jnp.sqrt(ex[0] * ex[2] + 1e-15)
                viol = jnp.maximum(0, ex[-1] - rhs)
                total_violation += viol**2

            return total_violation

        return objective_single, combos

    return (get_search_logic_vectorized,)


@app.cell
def _(mo):
    n_qubits_slider = mo.ui.number(3, 10, step=1, value=4, label="Number of Qubits (N)")
    n_samples_input = mo.ui.number(1, 10000, step=1, value=20, label="Samples per Gap")
    max_iter_input = mo.ui.number(10, 1000, step=10, value=200, label="Max Iterations")
    run_batch_button = mo.ui.run_button(label="🚀 Run Optimized Batch Analysis")

    mo.vstack([
        mo.md("### Search Parameters"),
        mo.hstack([n_qubits_slider, n_samples_input, max_iter_input], gap=2),
        run_batch_button
    ])
    return max_iter_input, n_qubits_slider, n_samples_input, run_batch_button


@app.cell
def _(
    LBFGS,
    compute_sre_batch,
    cp,
    get_search_logic_vectorized,
    go,
    jax,
    jnp,
    max_iter_input,
    mo,
    n_qubits_slider,
    n_samples_input,
    np,
    os,
    pickle,
    rand_Almost_Stab_state,
    run_batch_button,
):
    mo.stop(not run_batch_button.value)

    n = n_qubits_slider.value
    n_samples = n_samples_input.value
    k_sub = n // 2
    n_dim = 2**n

    obj_fn, perms = get_search_logic_vectorized(n, k_sub)
    solver = LBFGS(fun=obj_fn, maxiter=max_iter_input.value, tol=1e-12)
    vmapped_solver = jax.vmap(solver.run)

    gaps = list(range(n + 1))
    means_initial, stds_initial = [], []
    means_final, stds_final = [], []
    means_filtered, stds_filtered = [], []
    filtered_counts = []

    os.makedirs("data", exist_ok=True)

    for gap in gaps:
        with mo.status.spinner(title=f"Optimizing Gap {gap}/{n} (Batch Size: {n_samples})..."):
            # 1. Generate Batch of States
            batch_psi_np = []
            for _ in range(n_samples):
                psi_cp = rand_Almost_Stab_state(n, gap)
                p_np = cp.asnumpy(psi_cp)
                noise = np.random.normal(size=p_np.shape) + 1j * np.random.normal(size=p_np.shape)
                p_np = p_np + 1e-5 * noise
                batch_psi_np.append(p_np / np.linalg.norm(p_np))

            batch_psi_np = np.stack(batch_psi_np) # (n_samples, n_dim)

            # 2. Batch Initial SRE
            init_sres = compute_sre_batch(batch_psi_np.T)

            # 3. Batch Optimize
            init_params = jnp.concatenate([jnp.real(batch_psi_np), jnp.imag(batch_psi_np)], axis=1)
            results = vmapped_solver(init_params)

            # 4. Process Results
            final_params = results.params
            final_psi = np.array(final_params[:, :n_dim] + 1j * final_params[:, n_dim:])
            final_psi /= np.linalg.norm(final_psi, axis=1, keepdims=True)

            # 5. Batch Final SRE & Violations
            final_sres = compute_sre_batch(final_psi.T)

            # For violations, we need a vmapped obj_fn that returns scores
            batch_violations = jax.vmap(obj_fn)(final_params)

            filtered_sres = final_sres[batch_violations < 1e-8]

            # 6. Save Data
            batch_data = {
                "gap": gap,
                "initial_sre": init_sres,
                "final_sre": final_sres,
                "filtered_sre": filtered_sres,
                "violations": batch_violations
            }
            with open(f"data/opt_v2_{n}q_{n_samples}s_gap{gap}.pkl", "wb") as f:
                pickle.dump(batch_data, f)

            means_initial.append(np.nanmean(init_sres))
            stds_initial.append(np.nanstd(init_sres))
            means_final.append(np.nanmean(final_sres))
            stds_final.append(np.nanstd(final_sres))
            means_filtered.append(np.nanmean(filtered_sres) if len(filtered_sres) > 0 else 0)
            stds_filtered.append(np.nanstd(filtered_sres) if len(filtered_sres) > 0 else 0)
            filtered_counts.append(len(filtered_sres))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Initial', x=gaps, y=means_initial, error_y=dict(type='data', array=stds_initial), marker_color='indianred'))
    fig.add_trace(go.Bar(name='Final', x=gaps, y=means_final, error_y=dict(type='data', array=stds_final), marker_color='lightsalmon'))
    fig.add_trace(go.Bar(
        name='Filtered', 
        x=gaps, 
        y=means_filtered, 
        error_y=dict(type='data', array=stds_filtered), 
        marker_color='seagreen',
        text=filtered_counts,
        textposition='inside'
    ))

    fig.update_layout(
        title=f"Optimized Batch SRE for {n}-Qubits",
        barmode='group', template="plotly_white",
        yaxis_title="Average SRE (α=2)", xaxis_title="Gap (k)"
    )

    batch_view = mo.vstack([mo.md("### Vectorized Results"), mo.as_html(fig)])
    return (batch_view,)


@app.cell
def _(batch_view):
    batch_view
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
