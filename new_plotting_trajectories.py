# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "cupy-cuda12x",
#     "plotly",
#     "scipy",
#     "fsspec",
#     "requests",
#     "juliacall==0.9.35",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    try:
        from juliacall import Main as jl
    except ImportError:
        jl = None
    return (jl,)


@app.cell
def _(jl, mo):
    if jl is not None:
        jl.seval("using HadaMAG")
        jl.seval("using CUDA")
        jl.seval("""
        function jl_compute_sre_exact(psi_np, alpha, n_qubits, dim)
            psi_jl = Vector{ComplexF64}(psi_np)
            psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            sre_result, lost_norm = SRE(psi_sv, alpha, backend= :CUDA)
            return (sre_result, lost_norm)
        end
        """)
        status = mo.md("⚡ **Julia SRE Direct Bridge:** Initialized successfully (`HadaMAG` + `CUDA` backend)")
    else:
        status = mo.md("⚠️ **Julia SRE Direct Bridge:** Julia/JuliaCall not available (falling back to placeholders)")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cupy as cp
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc
    import pickle
    from itertools import combinations
    import os
    from fsspec.implementations.github import GithubFileSystem
    import warnings
    import matplotlib.pyplot as plt

    # Suppress all warnings globally to avoid logging overhead in loops
    warnings.filterwarnings("ignore")

    _CACHE_FILE = "_plot_cache.pkl"

    class _DiskCache(dict):
        def __init__(self, path: str):
            super().__init__()
            self._path = path
            self._load()

        def _load(self):
            if os.path.exists(self._path):
                try:
                    with open(self._path, "rb") as _f:
                        self.update(pickle.load(_f))
                    print(f"[cache] Loaded {len(self)} entries from {self._path}")
                except Exception as _e:
                    print(f"[cache] Could not load disk cache: {_e}. Starting fresh.")

        def _save(self):
            try:
                with open(self._path, "wb") as _f:
                    pickle.dump(dict(self), _f)
            except Exception as _e:
                print(f"[cache] Could not save to disk: {_e}")

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            self._save()

        def clear(self):
            super().clear()
            try:
                if os.path.exists(self._path):
                    os.remove(self._path)
                    print(f"[cache] Removed disk cache file {self._path}")
            except Exception as _e:
                print(f"[cache] Could not remove disk cache: {_e}")

    # One global instance — loaded from disk once per kernel session
    plot_cache = globals().get("plot_cache", _DiskCache(_CACHE_FILE))
    return (
        GithubFileSystem,
        combinations,
        cp,
        go,
        make_subplots,
        mo,
        np,
        os,
        pc,
        pickle,
        plot_cache,
        plt,
    )


@app.cell
def _(cp):
    def par_trace(psi, dim, n, n_parties):
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def is_appt(x) -> bool:
        xp = cp.get_array_module(x)
        _purity = xp.sum(x.real**2 + x.imag**2)
        _D = x.shape[0]
        if _purity <= 1 / (_D - 1):
            return True
        _ex = xp.linalg.eigvalsh(x)
        _ex = xp.clip(_ex, 0.0, None)
        if (_ex[-1] - _ex[1]) ** 2 <= 4 * _ex[0] * _ex[3]:
            return True
        return False

    return is_appt, par_trace


@app.cell
def _(combinations, cp, is_appt, np, par_trace):
    def is_TE(psi, dim: int = 2) -> bool:
        xp = cp.get_array_module(psi)
        n = int(np.log2(len(psi)))
        k = n - n // 2
        for _i in combinations(range(n), k):
            per = [x for x in range(n) if x not in _i] + list(_i)
            psi_moved = xp.transpose(psi.reshape([dim] * n), per).flatten()
            _x = par_trace(psi_moved, dim, n, k)
            _x = (_x + _x.conj().T) / 2.0
            if not is_appt(_x):
                return False
        return True

    return (is_TE,)


@app.function
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip("#")
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"


@app.cell
def _(go, is_TE, jl, make_subplots, np, os, pc, pickle, plot_cache, re):
    def compute_sre_exact(psi_np, alpha=2):
        """Compute exact SRE using HadaMAG.jl via JuliaCall."""
        if jl is None:
            return 1e-15, 0.0
        try:
            arr_np = np.asarray(psi_np)
            norm_val = np.linalg.norm(arr_np)
            if norm_val > 1e-12:
                arr_np = arr_np / norm_val
            dim = len(arr_np)
            n_qubits = int(np.log2(dim))

            # Call pre-defined Julia function directly — no seval parser overhead per iteration
            res = jl.jl_compute_sre_exact(arr_np, alpha, n_qubits, dim)
            val = float(res[0])
            return val if val != 0.0 else 1e-15, float(res[1])
        except Exception as e:
            print(f"SRE Exact Calculation Error: {e}")
            return 1e-15, 0.0

    print("compute_sre_exact reloaded OK")

    def cache_plot(func):
        """Decorator: cache plotting results keyed on params + file modification times."""
        def wrapper(pkl_files, labels, step_mes, use_te_filter, central_tendency,
                    selected_metrics, plot_type="Line Chart", fs=None):
            mtimes = []
            if not fs:
                for f in (pkl_files or []):
                    if f and os.path.exists(f):
                        mtimes.append(os.path.getmtime(f))
                    else:
                        mtimes.append(0.0)
            cache_key = (
                tuple(pkl_files) if pkl_files else (),
                tuple(labels) if labels else (),
                step_mes,
                use_te_filter,
                central_tendency,
                tuple(selected_metrics) if selected_metrics else (),
                plot_type,
                tuple(mtimes),
            )
            if cache_key in plot_cache:
                print("[cache] HIT — returning cached plot.")
                return plot_cache[cache_key]
            print("[cache] MISS — computing plot…")
            fig = func(pkl_files, labels, step_mes, use_te_filter, central_tendency,
                       selected_metrics, plot_type, fs)
            if fig is not None:
                plot_cache[cache_key] = fig
            return fig
        return wrapper

    @cache_plot
    def plot_te_filtered_trajectories(
        pkl_files, labels, step_mes, use_te_filter, central_tendency, selected_metrics, plot_type="Line Chart", fs=None
    ):
        """Plot trajectories with optional TE filtering and bar chart modes."""
        def finalize_arxiv_style(fig):
            if fig is None:
                return fig
            fig.update_xaxes(
                showline=True, linewidth=1, linecolor="black", mirror=True,
                ticks="inside", tickwidth=1, tickcolor="black",
                gridcolor="#e0e0e0", gridwidth=0.5, zeroline=False
            )
            fig.update_yaxes(
                showline=True, linewidth=1, linecolor="black", mirror=True,
                ticks="inside", tickwidth=1, tickcolor="black",
                gridcolor="#e0e0e0", gridwidth=0.5, zeroline=False
            )
            return fig

        if not selected_metrics:
            return None

        metric_map = {
            "average_purity": "Average Purity",
            "max_purity": "Max Purity",
            "sre": "SRE"
        }
        metric_titles = [metric_map[m] for m in selected_metrics]
        colors = pc.qualitative.Plotly

        # ------------------------------------------------------------
        # BAR CHART COMPARING INITIAL AND FINAL METRICS
        # ------------------------------------------------------------
        if plot_type == "Bar Chart":
            fig = make_subplots(rows=1, cols=len(selected_metrics), subplot_titles=metric_titles)

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                if use_te_filter:
                    final_states = data["final_states"]
                    te_mask = np.array([is_TE(state) for state in final_states])
                    n_te = te_mask.sum()
                    if n_te == 0:
                        continue
                else:
                    te_mask = np.ones(len(data["final_states"]), dtype=bool)

                base_color = colors[file_idx % len(colors)]

                # Derive SRE initial state gap value
                derived_init_sre = 0.0
                _gap_match = re.search(r'gap(\d+)', label)
                if _gap_match:
                    derived_init_sre = float(_gap_match.group(1))

                data_changed = False
                for i, metric in enumerate(selected_metrics):
                    col = i + 1

                    init_vals = []
                    final_vals = []

                    # Compute SRE for initial/final states if correct_data directory is active
                    is_correct_data = "correct_data" in file

                    for j, keep in enumerate(te_mask):
                        if not keep:
                            continue

                        if metric == "sre" and is_correct_data:
                            final_state = data["final_states"][j]
                            init_sre = data["sre"][j][0]
                            final_sre = data["sre"][j][-1]

                            if init_sre == 0.0 and derived_init_sre != 0.0:
                                init_sre = derived_init_sre
                                data["sre"][j][0] = derived_init_sre
                                data_changed = True
                            if final_sre == 0.0:
                                computed_final, _ = compute_sre_exact(final_state, alpha=2)
                                final_sre = computed_final
                                data["sre"][j][-1] = computed_final
                                data_changed = True
                            init_vals.append(init_sre)
                            final_vals.append(final_sre)
                        else:
                            traj = data[metric][j]
                            init_vals.append(traj[0])
                            final_vals.append(traj[-1])

                    # Average or Median of Initial vs Final
                    if central_tendency == "Average":
                        init_center = np.mean(init_vals)
                        final_center = np.mean(final_vals)
                        init_err = np.std(init_vals)
                        final_err = np.std(final_vals)
                    else:
                        init_center = np.median(init_vals)
                        final_center = np.median(final_vals)
                        init_err = np.nanstd(init_vals)
                        final_err = np.nanstd(final_vals)

                    # Add Initial Bar
                    fig.add_trace(
                        go.Bar(
                            x=[f"{label} Initial"],
                            y=[init_center],
                            error_y=dict(type='data', array=[init_err], visible=True),
                            name=f"{label} (Initial)",
                            legendgroup=label,
                            showlegend=(i == 0),
                            marker_color=base_color,
                            opacity=0.6
                        ),
                        row=1, col=col
                    )

                    # Add Final Bar
                    fig.add_trace(
                        go.Bar(
                            x=[f"{label} Final"],
                            y=[final_center],
                            error_y=dict(type='data', array=[final_err], visible=True),
                            name=f"{label} (Final)",
                            legendgroup=label,
                            showlegend=False,
                            marker_color=base_color,
                            opacity=1.0
                        ),
                        row=1, col=col
                    )

            # Store back to pickle (once per file, after all metrics processed)
            if data_changed and not fs:
                try:
                    with open(file, "wb") as f:
                        pickle.dump(data, f)
                except Exception as e:
                    print(f"Error saving updated SRE back to {file}: {e}")

            fig.update_layout(
                title=f"Comparison: Initial vs Final States ({central_tendency})",
                height=500,
                width=400 * len(selected_metrics),
                template="plotly_white",
                margin=dict(b=120),
                font=dict(family="Computer Modern"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return finalize_arxiv_style(fig)

        # ------------------------------------------------------------
        # STANDARD LINE CHART TRAJECTORIES
        # ------------------------------------------------------------
        fig = make_subplots(rows=1, cols=len(selected_metrics), subplot_titles=metric_titles)

        for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
            if not file or not label:
                continue
            try:
                if fs:
                    with fs.open(file, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(file, "rb") as f:
                        data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # Filter by TE if checkbox is checked
            if use_te_filter:
                final_states = data["final_states"]
                te_mask = np.array([is_TE(state) for state in final_states])
                n_te = te_mask.sum()
                if n_te == 0:
                    continue
                prefix = f"TE-only (n={n_te}) "
            else:
                n_total = len(data["final_states"])
                te_mask = np.ones(n_total, dtype=bool)
                prefix = ""

            base_color = colors[file_idx % len(colors)]
            fill_color = hex_to_rgba(base_color, alpha=0.2)

            # Derive SRE initial state gap value
            derived_init_sre = 0.0
            _gap_match = re.search(r'gap(\d+)', label)
            if _gap_match:
                derived_init_sre = float(_gap_match.group(1))

            data_changed = False
            for i, metric in enumerate(selected_metrics):
                col = i + 1

                filtered_trajs = []
                is_correct_data = "correct_data" in file

                for j, keep in enumerate(te_mask):
                    if not keep:
                        continue
                    if metric == "sre" and is_correct_data:
                        final_state = data["final_states"][j]
                        init_sre = data["sre"][j][0]
                        final_sre = data["sre"][j][-1]

                        if init_sre == 0.0 and derived_init_sre != 0.0:
                            init_sre = derived_init_sre
                            data["sre"][j][0] = derived_init_sre
                            data_changed = True
                        if final_sre == 0.0:
                            computed_final, _ = compute_sre_exact(final_state, alpha=2)
                            final_sre = computed_final
                            data["sre"][j][-1] = computed_final
                            data_changed = True
                        filtered_trajs.append([init_sre, final_sre])
                    else:
                        filtered_trajs.append(data[metric][j])

                # Store back to pickle
                if data_changed and not fs:
                    try:
                        with open(file, "wb") as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        print(f"Error saving updated SRE back to {file}: {e}")

                if not filtered_trajs:
                    continue
                max_len = max(len(t) for t in filtered_trajs)
                arr = np.full((len(filtered_trajs), max_len), np.nan)
                for j, traj in enumerate(filtered_trajs):
                    arr[j, :len(traj)] = traj

                if max_len == 2:
                    steps = np.array([0, 1]) * step_mes
                else:
                    steps = np.arange(max_len) * step_mes

                if central_tendency == "Average":
                    center_line = np.nanmean(arr, axis=0)
                    std = np.nanstd(arr, axis=0)
                    lower_bound = center_line - std
                    upper_bound = center_line + std
                    leg_suffix = f"(Avg ±1 Std)"
                else:
                    center_line = np.nanmedian(arr, axis=0)
                    lower_bound = np.nanpercentile(arr, 25, axis=0)
                    upper_bound = np.nanpercentile(arr, 75, axis=0)
                    leg_suffix = f"(Median & IQR)"

                show_leg = i == 0

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=lower_bound,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=label,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=upper_bound,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=fill_color,
                        showlegend=False,
                        legendgroup=label,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=center_line,
                        mode="lines",
                        line=dict(color=base_color, width=2),
                        name=f"{prefix}{label} {leg_suffix}",
                        legendgroup=label,
                        showlegend=show_leg,
                    ),
                    row=1,
                    col=col,
                )

            # Store back to pickle (once per file, after all metrics processed)
            if data_changed and not fs:
                try:
                    with open(file, "wb") as f:
                        pickle.dump(data, f)
                except Exception as e:
                    print(f"Error saving updated SRE back to {file}: {e}")
        # ------------------------------------------------------------
        # TE vs non-TE SRE BAR CHART COMPARISON
        # ------------------------------------------------------------
        if plot_type == "TE vs non-TE SRE":
            fig = go.Figure()

            x_ticks = []

            init_centers = []
            init_errs = []
            init_labels = []

            all_final_centers = []
            all_final_errs = []
            all_final_labels = []

            te_centers = []
            te_errs = []
            te_labels = []

            non_te_centers = []
            non_te_errs = []
            non_te_labels = []

            # Global lists to aggregate across all gaps
            global_init_sre_vals = []
            global_all_final_sre_vals = []
            global_te_sre_vals = []
            global_non_te_sre_vals = []

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                final_states = data["final_states"]
                te_mask = np.array([is_TE(state) for state in final_states])
                is_correct_data = ("correct_data" in file) or ("more_data" in file)

                init_sre_vals = []
                all_final_sre_vals = []
                te_sre_vals = []
                non_te_sre_vals = []

                # Parse gap value from the label (e.g., 'gap2' -> 2.0 SRE)
                # Since rand_Almost_Stab_state(n, gap) has stabilizer group of size 2^(n - gap)
                # the initial state has SRE exactly equal to the gap value (for alpha=2).
                derived_init_sre = 0.0
                _gap_match = re.search(r'gap(\d+)', label)
                if _gap_match:
                    derived_init_sre = float(_gap_match.group(1))

                data_changed = False
                for j, is_te_state in enumerate(te_mask):
                    final_state = data["final_states"][j]

                    init_sre = data["sre"][j][0]
                    sre_val = data["sre"][j][-1]

                    # Update SRE initial states in file
                    if init_sre == 0.0 and derived_init_sre != 0.0:
                        init_sre = derived_init_sre
                        data["sre"][j][0] = derived_init_sre
                        data_changed = True

                    # Update SRE final states in file
                    if sre_val == 0.0 and is_correct_data:
                        computed, _ = compute_sre_exact(final_state, alpha=2)
                        sre_val = computed
                        data["sre"][j][-1] = computed
                        data_changed = True

                    init_sre_vals.append(init_sre)
                    all_final_sre_vals.append(sre_val)
                    if is_te_state:
                        te_sre_vals.append(sre_val)
                    else:
                        non_te_sre_vals.append(sre_val)

                # Store SRE values back to pickle on local disk to save compute on later runs
                if data_changed and not fs:
                    try:
                        with open(file, "wb") as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        print(f"Error saving updated SRE back to {file}: {e}")

                # Accumulate for global stats
                global_init_sre_vals.extend(init_sre_vals)
                global_all_final_sre_vals.extend(all_final_sre_vals)
                global_te_sre_vals.extend(te_sre_vals)
                global_non_te_sre_vals.extend(non_te_sre_vals)

                # Add x-axis category label (e.g. "gap0")
                x_ticks.append(label)

                # Initial stats
                if init_sre_vals:
                    init_centers.append(np.mean(init_sre_vals) if central_tendency == "Average" else np.median(init_sre_vals))
                    init_errs.append(np.std(init_sre_vals) if len(init_sre_vals) > 1 else 0.0)
                    init_labels.append(f"n={len(init_sre_vals)}")
                else:
                    init_centers.append(0.0)
                    init_errs.append(0.0)
                    init_labels.append("n=0")

                # All Final stats
                if all_final_sre_vals:
                    all_final_centers.append(np.mean(all_final_sre_vals) if central_tendency == "Average" else np.median(all_final_sre_vals))
                    all_final_errs.append(np.std(all_final_sre_vals) if len(all_final_sre_vals) > 1 else 0.0)
                    all_final_labels.append(f"n={len(all_final_sre_vals)}")
                else:
                    all_final_centers.append(0.0)
                    all_final_errs.append(0.0)
                    all_final_labels.append("n=0")

                # TE stats
                if te_sre_vals:
                    te_centers.append(np.mean(te_sre_vals) if central_tendency == "Average" else np.median(te_sre_vals))
                    te_errs.append(np.std(te_sre_vals) if len(te_sre_vals) > 1 else 0.0)
                    te_labels.append(f"n={len(te_sre_vals)}")
                else:
                    te_centers.append(0.0)
                    te_errs.append(0.0)
                    te_labels.append("n=0")

                # Non-TE stats
                if non_te_sre_vals:
                    non_te_centers.append(np.mean(non_te_sre_vals) if central_tendency == "Average" else np.median(non_te_sre_vals))
                    non_te_errs.append(np.std(non_te_sre_vals) if len(non_te_sre_vals) > 1 else 0.0)
                    non_te_labels.append(f"n={len(non_te_sre_vals)}")
                else:
                    non_te_centers.append(0.0)
                    non_te_errs.append(0.0)
                    non_te_labels.append("n=0")

            # Append the global "All Gaps" summary category at the right end of the lists
            x_ticks.append("All Gaps Combined")

            if global_init_sre_vals:
                init_centers.append(np.mean(global_init_sre_vals) if central_tendency == "Average" else np.median(global_init_sre_vals))
                init_errs.append(np.std(global_init_sre_vals) if len(global_init_sre_vals) > 1 else 0.0)
                init_labels.append(f"n={len(global_init_sre_vals)}")
            else:
                init_centers.append(0.0)
                init_errs.append(0.0)
                init_labels.append("n=0")

            if global_all_final_sre_vals:
                all_final_centers.append(np.mean(global_all_final_sre_vals) if central_tendency == "Average" else np.median(global_all_final_sre_vals))
                all_final_errs.append(np.std(global_all_final_sre_vals) if len(global_all_final_sre_vals) > 1 else 0.0)
                all_final_labels.append(f"n={len(global_all_final_sre_vals)}")
            else:
                all_final_centers.append(0.0)
                all_final_errs.append(0.0)
                all_final_labels.append("n=0")

            if global_te_sre_vals:
                te_centers.append(np.mean(global_te_sre_vals) if central_tendency == "Average" else np.median(global_te_sre_vals))
                te_errs.append(np.std(global_te_sre_vals) if len(global_te_sre_vals) > 1 else 0.0)
                te_labels.append(f"n={len(global_te_sre_vals)}")
            else:
                te_centers.append(0.0)
                te_errs.append(0.0)
                te_labels.append("n=0")

            if global_non_te_sre_vals:
                non_te_centers.append(np.mean(global_non_te_sre_vals) if central_tendency == "Average" else np.median(global_non_te_sre_vals))
                non_te_errs.append(np.std(global_non_te_sre_vals) if len(global_non_te_sre_vals) > 1 else 0.0)
                non_te_labels.append(f"n={len(global_non_te_sre_vals)}")
            else:
                non_te_centers.append(0.0)
                non_te_errs.append(0.0)
                non_te_labels.append("n=0")

            # Add Initial trace
            fig.add_trace(
                go.Bar(
                    x=x_ticks,
                    y=init_centers,
                    error_y=dict(type='data', array=init_errs, visible=True),
                    text=init_labels,
                    textposition='outside',
                    textfont=dict(size=14),
                    name="Initial States (Stab/Almost-Stab)",
                    marker_color="mediumseagreen",
                    opacity=0.8
                )
            )

            # Add TE trace
            fig.add_trace(
                go.Bar(
                    x=x_ticks,
                    y=te_centers,
                    error_y=dict(type='data', array=te_errs, visible=True),
                    text=te_labels,
                    textposition='outside',
                    textfont=dict(size=14),
                    name="Final TE States",
                    marker_color="royalblue",
                    opacity=0.9
                )
            )

            # Add non-TE trace
            fig.add_trace(
                go.Bar(
                    x=x_ticks,
                    y=non_te_centers,
                    error_y=dict(type='data', array=non_te_errs, visible=True),
                    text=non_te_labels,
                    textposition='outside',
                    textfont=dict(size=14),
                    name="Final non-TE States",
                    marker_color="crimson",
                    opacity=0.6
                )
            )

            fig.update_layout(
                title=f"SRE (Magic) Comparison of Initial vs Final States ({central_tendency})",
                xaxis_title="Experiment Gap",
                yaxis_title="SRE Value",
                barmode="group",
                height=500,
                width=700,
                template="plotly_white",
                margin=dict(b=120),
                font=dict(family="Computer Modern"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return finalize_arxiv_style(fig)

        # ------------------------------------------------------------
        # HISTOGRAM OF SRE VALUES
        # ------------------------------------------------------------
        if plot_type == "Histogram SRE":
            fig = go.Figure()

            for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
                if not file or not label:
                    continue
                try:
                    if fs:
                        with fs.open(file, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(file, "rb") as f:
                            data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                final_states = data["final_states"]
                te_mask = np.array([is_TE(state) for state in final_states])
                is_correct_data = "correct_data" in file

                final_sre_vals = []
                # Derive SRE initial state gap value
                derived_init_sre = 0.0
                _gap_match = re.search(r'gap(\d+)', label)
                if _gap_match:
                    derived_init_sre = float(_gap_match.group(1))

                data_changed = False
                for j, state in enumerate(final_states):
                    if use_te_filter and not te_mask[j]:
                        continue

                    sre_val = data["sre"][j][-1]
                    if sre_val == 0.0 and is_correct_data:
                        computed, _ = compute_sre_exact(state, alpha=2)
                        sre_val = computed
                        data["sre"][j][-1] = computed
                        data_changed = True

                    final_sre_vals.append(sre_val)

                # Store SRE values back to pickle on local disk to save compute on later runs
                if data_changed and not fs:
                    try:
                        with open(file, "wb") as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        print(f"Error saving updated SRE back to {file}: {e}")

                if not final_sre_vals:
                    continue

                base_color = colors[file_idx % len(colors)]
                fig.add_trace(
                    go.Histogram(
                        x=final_sre_vals,
                        name=f"{label} (n={len(final_sre_vals)})",
                        marker_color=base_color,
                        opacity=0.6,
                        nbinsx=30
                    )
                )

            fig.update_layout(
                title="Distribution of Final SRE (Magic) Values",
                xaxis_title="Final SRE Value",
                yaxis_title="Count",
                barmode="overlay",
                height=500,
                width=800,
                template="plotly_white",
                font=dict(family="Computer Modern"),
                margin=dict(b=120),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return finalize_arxiv_style(fig)

        fig.update_layout(
            title=f"Entanglement Trajectories ({central_tendency})"
            + (" — TE Filtered" if use_te_filter else ""),
            height=500,
            width=400 * len(selected_metrics),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(b=120),
            font=dict(family="Computer Modern"),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
            ),
        )
        fig.update_xaxes(title_text="Steps")
        return finalize_arxiv_style(fig)

    return (plot_te_filtered_trajectories,)


@app.cell
def _(mo):
    mo.md("""
    # Entanglement Trajectory Visualization
    """)
    return


@app.cell
def _(mo):
    # Data source selection UI
    data_source = mo.ui.radio(
        options=["Local", "GitHub"],
        value="Local",
        label="**Data Source:**",
        inline=True
    )

    refresh_button = mo.ui.button(label="🔄 Refresh File List", value=0)

    mo.md(f"### 1. Data Selection\n{mo.hstack([data_source, refresh_button], align='center', gap=2)}")
    return data_source, refresh_button


@app.cell
def _(GithubFileSystem, data_source, mo, os, plot_cache, refresh_button):
    import re

    # We use refresh_button.value as a dependency to trigger re-scanning
    refresh_button.value

    available_files = []
    fs = None

    if data_source.value == "Local":
        available_files = []
        for data_dir in ["correct_data/", "new_data/", "data/"]:
            if os.path.exists(data_dir):
                available_files.extend(
                    [os.path.join(data_dir, _f) for _f in os.listdir(data_dir) if _f.endswith(".pkl")]
                )
        available_files.sort()
    else:
        # GitHub configuration
        org = "invariantfields"
        repo = "Interactive_TE_Plots"
        try:
            fs = GithubFileSystem(org=org, repo=repo)
            # Fetch files from /data folder in the repo
            repo_data_dir = "data"
            available_files = sorted([_f for _f in fs.ls(repo_data_dir) if _f.endswith(".pkl")])
        except Exception as e:
            mo.output.append(mo.md(f"⚠️ Error connecting to GitHub: {e}"))

    # Group files by prefix (everything before 'stps' or 'steps')
    groups = {}
    for _f in available_files:
        _basename = os.path.basename(_f)
        # Parse gap from filename (e.g. "..._stps_4.pkl")
        _match = re.search(r'^(.*)(?:stps|steps|stps_)(\d+)\.pkl$', _basename)
        if _match:
            _prefix = _match.group(1).rstrip("_")
            _gap = int(_match.group(2))
        else:
            # Fallback if no steps/gaps structure
            _prefix = _basename.replace(".pkl", "")
            _gap = 0

        # Check if the folder contains all the data i.e., "x" + str(n) for all n in range(number of qubits + 1)
        # Let's group files by their general prefix (e.g. everything up to '_stps_')
        if _prefix not in groups:
            groups[_prefix] = []
        groups[_prefix].append((_f, _gap))

    sorted_group_names = sorted(groups.keys())

    # Store sorted lists of file paths for each group
    grouped_files = {}
    for _g in groups:
        # Sort by gap value
        sorted_tuples = sorted(groups[_g], key=lambda x: x[1])
        grouped_files[_g] = [item[0] for item in sorted_tuples]

    group_selector = mo.ui.dropdown(
        options=sorted_group_names,
        label="Select Experiment Group",
        value=sorted_group_names[0] if sorted_group_names else None,
    )

    te_filter_checkbox = mo.ui.checkbox(
        value=True, label="🔬 Filter by TE (only trajectories with TE final states)"
    )

    metric_selector = mo.ui.radio(
        options=["Average", "Median"],
        value="Average",
        label="**Central Tendency:** ",
    )

    # Select which metrics to plot
    metric_options = ["average_purity", "max_purity", "sre"]
    metrics_to_plot = mo.ui.multiselect(
        options=metric_options,
        value=metric_options,
        label="**Metrics to Plot:**"
    )

    plot_type_dropdown = mo.ui.dropdown(
        options=["Line Chart", "Bar Chart", "TE vs non-TE SRE", "Histogram SRE"],
        value="Line Chart",
        label="**Plot Type:**"
    )

    step_mes_input = mo.ui.number(
        start=1, stop=1000, step=1, value=1, label="Steps per measurement"
    )

    # Run button to prevent heavy computations on every click
    plot_button = mo.ui.run_button(label="🚀 Generate Plot")
    def clear_cache_callback(_):
        plot_cache.clear()
        mo.status.toast("🧹 Plot cache cleared successfully!")
    clear_cache_button = mo.ui.button(label="🧹 Clear Plot Cache", on_click=clear_cache_callback)

    mo.vstack([
        group_selector,
        mo.md("### 2. Plot Settings"),
        mo.hstack([plot_type_dropdown, metrics_to_plot, te_filter_checkbox, metric_selector, step_mes_input], justify="start", gap=2),
        mo.md("### 3. Execution"),
        mo.hstack([plot_button, clear_cache_button], gap=2)
    ])
    return (
        fs,
        group_selector,
        grouped_files,
        metric_selector,
        metrics_to_plot,
        plot_button,
        plot_type_dropdown,
        re,
        step_mes_input,
        te_filter_checkbox,
    )


@app.cell
def _(
    fs,
    group_selector,
    grouped_files,
    metric_selector,
    metrics_to_plot,
    mo,
    plot_button,
    plot_te_filtered_trajectories,
    plot_type_dropdown,
    re,
    step_mes_input,
    te_filter_checkbox,
):
    mo.stop(not plot_button.value or not group_selector.value, mo.md("Select an experiment group and click **Generate Plot**."))

    selected_group_files = grouped_files[group_selector.value]

    # Deriving labels representing the gaps from the filenames
    labels = []
    for _f in selected_group_files:
        _match = re.search(r'(?:stps|steps|stps_)(\d+)\.pkl$', _f)
        if _match:
            labels.append(f"k = {_match.group(1)}")
        else:
            labels.append(_f.split("/")[-1].replace(".pkl", ""))

    plot = plot_te_filtered_trajectories(
        pkl_files=selected_group_files,
        labels=labels,
        step_mes=step_mes_input.value,
        use_te_filter=te_filter_checkbox.value,
        central_tendency=metric_selector.value,
        selected_metrics=metrics_to_plot.value,
        plot_type=plot_type_dropdown.value,
        fs=fs,
    )
    plot
    return


@app.cell
def _(os, pickle):
    def unpack_pkl_file(archive_path, destination_dir):
        """
        Unpacks a combined .pkl archive back into separate files inside destination_dir.
        """
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        try:
            with open(archive_path, "rb") as f:
                packed_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading archive: {e}")
            return

        # Extract each file content back to disk
        for filename, content in packed_data.items():
            output_path = os.path.join(destination_dir, filename)
            try:
                with open(output_path, "wb") as f:
                    pickle.dump(content, f)
                print(f"Unpacked: {filename}")
            except Exception as e:
                print(f"Error unpacking {filename}: {e}")

        print(f"\nSuccessfully unpacked all files into directory: {destination_dir}")

    return (unpack_pkl_file,)


@app.cell
def _(unpack_pkl_file):
    unpack_pkl_file("co.pkl","correct_data/")
    return


@app.cell
def _(plt, np, os, pickle, re, is_TE, compute_sre_exact):
    def plot_te_filtered_trajectories_matplotlib(
        pkl_files, labels, step_mes, use_te_filter, central_tendency, selected_metrics, plot_type="Line Chart", fs=None
    ):
        if not selected_metrics:
            return None

        import matplotlib
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white"
        })

        metric_map = {
            "average_purity": "Average Purity",
            "max_purity": "Max Purity",
            "sre": "SRE"
        }

        loaded_data = []
        for file, label in zip(pkl_files, labels):
            if not file or not label:
                continue
            try:
                if fs:
                    with fs.open(file, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(file, "rb") as f:
                        data = pickle.load(f)
                loaded_data.append((data, label, file))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not loaded_data:
            return None

        # 1. LINE CHART
        if plot_type == "Line Chart":
            n_plots = len(selected_metrics)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5), sharex=True, squeeze=False)

            for col_idx, metric in enumerate(selected_metrics):
                ax = axes[0, col_idx]
                ax.set_title(metric_map[metric], fontsize=13)

                for data, label, file in loaded_data:
                    if use_te_filter:
                        te_mask = np.array([is_TE(state) for state in data["final_states"]])
                        trajs = [data[metric][j] for j, keep in enumerate(te_mask) if keep]
                    else:
                        trajs = data[metric]

                    if not trajs:
                        continue

                    steps = np.arange(len(trajs[0])) * step_mes
                    center = np.mean(trajs, axis=0) if central_tendency == "Average" else np.median(trajs, axis=0)
                    val_k = label.split("=")[-1].strip()
                    ax.plot(steps, center, label=f"$k={val_k}$", linewidth=1.5)

                ax.set_xlabel(r"$\text{Optimization Step } (t)$", fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(r"$\text{Metric Value}$", fontsize=11)
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.tick_params(direction="in", top=True, right=True, which="both")
                ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
                ax.legend(frameon=True, fontsize=9, loc="best")

            fig.tight_layout()
            return fig

        # 2. BAR CHART — Initial (Magic) vs Final with error bars + All Combined
        elif plot_type == "Bar Chart":
            n_plots = len(selected_metrics)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5.0), squeeze=False, facecolor='white')

            for col_idx, metric in enumerate(selected_metrics):
                ax = axes[0, col_idx]
                ax.set_title(metric_map[metric], fontsize=13)

                x_labels = []
                init_means = []
                init_errs = []
                init_counts = []
                final_means = []
                final_errs = []
                final_counts = []
                global_init_vals = []
                global_final_vals = []

                for data, label, file in loaded_data:
                    is_correct_data = "correct_data" in file or "more_data" in file

                    if use_te_filter:
                        te_mask = np.array([is_TE(state) for state in data["final_states"]])
                    else:
                        te_mask = np.ones(len(data["final_states"]), dtype=bool)

                    init_vals = []
                    final_vals = []

                    derived_init_sre = 0.0
                    _gap_match = re.search(r'gap(\d+)', label)
                    if _gap_match:
                        derived_init_sre = float(_gap_match.group(1))

                    for j, keep in enumerate(te_mask):
                        if not keep:
                            continue
                        if metric == "sre" and is_correct_data:
                            init_sre = data["sre"][j][0]
                            final_sre = data["sre"][j][-1]
                            init_vals.append(init_sre if init_sre != 0.0 else derived_init_sre)
                            final_vals.append(final_sre)
                        else:
                            init_vals.append(data[metric][j][0])
                            final_vals.append(data[metric][j][-1])

                    if not init_vals:
                        continue

                    val_k = label.split("=")[-1].strip()
                    x_labels.append(f"$k={val_k}$")
                    global_init_vals.extend(init_vals)
                    global_final_vals.extend(final_vals)

                    if central_tendency == "Average":
                        init_means.append(np.mean(init_vals))
                        final_means.append(np.mean(final_vals))
                    else:
                        init_means.append(np.median(init_vals))
                        final_means.append(np.median(final_vals))
                    init_errs.append(np.std(init_vals) if len(init_vals) > 1 else 0.0)
                    final_errs.append(np.std(final_vals) if len(final_vals) > 1 else 0.0)
                    init_counts.append(len(init_vals))
                    final_counts.append(len(final_vals))

                # All Combined summary bar
                if global_init_vals:
                    x_labels.append(r"$\mathrm{All\ Combined}$")
                    if central_tendency == "Average":
                        init_means.append(np.mean(global_init_vals))
                        final_means.append(np.mean(global_final_vals))
                    else:
                        init_means.append(np.median(global_init_vals))
                        final_means.append(np.median(global_final_vals))
                    init_errs.append(np.std(global_init_vals) if len(global_init_vals) > 1 else 0.0)
                    final_errs.append(np.std(global_final_vals) if len(global_final_vals) > 1 else 0.0)
                    init_counts.append(len(global_init_vals))
                    final_counts.append(len(global_final_vals))

                x = np.arange(len(x_labels))
                width = 0.35
                ekw = dict(elinewidth=0.8, capthick=0.8)

                b_init = ax.bar(x - width/2, init_means, width, yerr=init_errs, capsize=3,
                       label=r"$\text{Initial State (Magic)}$", color="mediumseagreen",
                       edgecolor="black", linewidth=0.5, error_kw=ekw)
                b_final = ax.bar(x + width/2, final_means, width, yerr=final_errs, capsize=3,
                       label=r"$\text{Final State}$", color="#ff7f0e",
                       edgecolor="black", linewidth=0.5, error_kw=ekw)

                ax.bar_label(b_init, labels=[f"n={n}" for n in init_counts], fontsize=7, padding=3, color="black")
                ax.bar_label(b_final, labels=[f"n={n}" for n in final_counts], fontsize=7, padding=3, color="black")

                ax.set_facecolor('white')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=9)
                ax.set_xlabel(r"$\text{Experiment Gap } (k)$", fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(r"$\text{Value}$", fontsize=11)
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.tick_params(direction="in", top=True, right=True, which="both")
                ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
                ax.legend(frameon=True, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
                ax.margins(y=0.15)

            fig.subplots_adjust(bottom=0.22)
            return fig

        # 3. TE vs non-TE SRE — 3 bar groups: Initial (Magic), TE, non-TE + All Combined
        elif plot_type == "TE vs non-TE SRE":
            fig, ax = plt.subplots(figsize=(8, 5.2), facecolor='white')
            ax.set_facecolor('white')

            x_labels = []
            init_centers = []
            init_errs = []
            init_counts = []
            te_centers = []
            te_errs = []
            te_counts = []
            non_te_centers = []
            non_te_errs = []
            non_te_counts = []
            global_init = []
            global_te = []
            global_non_te = []

            for data, label, file in loaded_data:
                te_mask = np.array([is_TE(state) for state in data["final_states"]])

                derived_init_sre = 0.0
                _gap_match = re.search(r'gap(\d+)', label)
                if _gap_match:
                    derived_init_sre = float(_gap_match.group(1))

                init_vals = []
                te_vals = []
                non_te_vals = []

                for j, is_te_state in enumerate(te_mask):
                    init_sre = data["sre"][j][0]
                    final_sre = data["sre"][j][-1]
                    init_vals.append(init_sre if init_sre != 0.0 else derived_init_sre)
                    if is_te_state:
                        te_vals.append(final_sre)
                    else:
                        non_te_vals.append(final_sre)

                if not te_vals and not non_te_vals:
                    continue

                val_k = label.split("=")[-1].strip()
                x_labels.append(f"$k={val_k}$")
                global_init.extend(init_vals)
                global_te.extend(te_vals)
                global_non_te.extend(non_te_vals)

                def _stat(vals):
                    if not vals:
                        return 0.0, 0.0
                    c = np.mean(vals) if central_tendency == "Average" else np.median(vals)
                    e = np.std(vals) if len(vals) > 1 else 0.0
                    return c, e

                ic, ie = _stat(init_vals)
                tc, te_e = _stat(te_vals)
                nc, ne = _stat(non_te_vals)
                init_centers.append(ic); init_errs.append(ie); init_counts.append(len(init_vals))
                te_centers.append(tc); te_errs.append(te_e); te_counts.append(len(te_vals))
                non_te_centers.append(nc); non_te_errs.append(ne); non_te_counts.append(len(non_te_vals))

            # All Combined
            if global_te or global_non_te:
                x_labels.append(r"$\mathrm{All\ Combined}$")
                ic = np.mean(global_init) if central_tendency == "Average" else np.median(global_init)
                tc = np.mean(global_te) if central_tendency == "Average" else np.median(global_te)
                nc = np.mean(global_non_te) if central_tendency == "Average" else np.median(global_non_te)
                init_centers.append(ic); init_errs.append(np.std(global_init) if global_init else 0.0); init_counts.append(len(global_init))
                te_centers.append(tc); te_errs.append(np.std(global_te) if global_te else 0.0); te_counts.append(len(global_te))
                non_te_centers.append(nc); non_te_errs.append(np.std(global_non_te) if global_non_te else 0.0); non_te_counts.append(len(global_non_te))

            x = np.arange(len(x_labels))
            width = 0.25
            ekw = dict(elinewidth=0.8, capthick=0.8)

            b_init = ax.bar(x - width, init_centers, width, yerr=init_errs, capsize=3,
                   label=r"$\text{Initial (Magic)}$", color="mediumseagreen",
                   edgecolor="black", linewidth=0.5, error_kw=ekw)
            b_te = ax.bar(x, te_centers, width, yerr=te_errs, capsize=3,
                   label=r"$\text{Final TE States}$", color="royalblue",
                   edgecolor="black", linewidth=0.5, error_kw=ekw)
            b_non_te = ax.bar(x + width, non_te_centers, width, yerr=non_te_errs, capsize=3,
                   label=r"$\text{Final non-TE States}$", color="crimson",
                   edgecolor="black", linewidth=0.5, error_kw=ekw)

            ax.bar_label(b_init, labels=[f"n={n}" for n in init_counts], fontsize=7, padding=3, color="black")
            ax.bar_label(b_te, labels=[f"n={n}" for n in te_counts], fontsize=7, padding=3, color="black")
            ax.bar_label(b_non_te, labels=[f"n={n}" for n in non_te_counts], fontsize=7, padding=3, color="black")

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_xlabel(r"$\text{Experiment Gap } (k)$", fontsize=11)
            ax.set_ylabel(r"$\text{SRE } (S_2)$", fontsize=11)
            ax.set_title(r"$\text{SRE (Magic): Initial vs Final TE/non-TE States}$", fontsize=13)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
            ax.legend(frameon=True, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
            ax.margins(y=0.15)

            fig.subplots_adjust(bottom=0.25)
            return fig

        # 4. HISTOGRAM SRE
        elif plot_type == "Histogram SRE":
            fig, ax = plt.subplots(figsize=(6, 4.5))

            for data, label, file in loaded_data:
                is_correct_data = "correct_data" in file
                if not is_correct_data:
                    continue

                if use_te_filter:
                    te_mask = np.array([is_TE(state) for state in data["final_states"]])
                    vals = [data["sre"][j][-1] for j, keep in enumerate(te_mask) if keep]
                else:
                    vals = [data["sre"][j][-1] for j in range(len(data["final_states"]))]

                if not vals:
                    continue

                val_k = label.split("=")[-1].strip()
                ax.hist(vals, bins=25, alpha=0.5, label=f"$k={val_k}$", edgecolor="black", linewidth=0.5)

            ax.set_xlabel(r"$\text{Final SRE } (S_2)$", fontsize=11)
            ax.set_ylabel(r"$\text{Count}$", fontsize=11)
            ax.set_title(r"$\text{Histogram of Final State SRE Values}$", fontsize=13)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(direction="in", top=True, right=True, which="both")
            ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
            ax.legend(frameon=True, fontsize=9, loc="best")

            fig.tight_layout()
            return fig

        return None

    return (plot_te_filtered_trajectories_matplotlib,)


@app.cell
def _(
    fs,
    group_selector,
    grouped_files,
    metric_selector,
    metrics_to_plot,
    mo,
    plot_button,
    plot_te_filtered_trajectories_matplotlib,
    plot_type_dropdown,
    re,
    step_mes_input,
    te_filter_checkbox,
):
    mo.stop(not plot_button.value or not group_selector.value, mo.md("Select an experiment group and click **Generate Plot**."))

    _mpl_files = grouped_files[group_selector.value]
    _mpl_labels = []
    for _f in _mpl_files:
        _m = re.search(r'(?:stps|steps|stps_)(\d+)\.pkl$', _f)
        if _m:
            _mpl_labels.append(f"k = {_m.group(1)}")
        else:
            _mpl_labels.append(_f.split("/")[-1].replace(".pkl", ""))

    _mpl_fig = plot_te_filtered_trajectories_matplotlib(
        pkl_files=_mpl_files,
        labels=_mpl_labels,
        step_mes=step_mes_input.value,
        use_te_filter=te_filter_checkbox.value,
        central_tendency=metric_selector.value,
        selected_metrics=metrics_to_plot.value,
        plot_type=plot_type_dropdown.value,
        fs=fs,
    )

    matplotlib_plot = _mpl_fig
    matplotlib_plot
    return (matplotlib_plot,)


if __name__ == "__main__":
    app.run()
