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
def _(jl):
    if jl is not None:
        jl.seval("using HadaMAG")
        jl.seval("using CUDA")
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
    )


@app.cell
def _(np):
    def par_trace(psi, dim, n, n_parties):
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def is_appt(x: np.ndarray) -> bool:
        # Pure NumPy — GPU overhead dominates for dim=128 (7 qubits)
        _purity = np.sum(x.real**2 + x.imag**2)
        _D = x.shape[0]
        if _purity <= 1 / (_D - 1):
            return True
        _ex = np.linalg.eigvalsh(x)
        _ex = np.clip(_ex, 0.0, None)
        if (_ex[-1] - _ex[1]) ** 2 <= 4 * _ex[0] * _ex[3]:
            return True
        return False

    return is_appt, par_trace


@app.cell
def _(combinations, is_appt, np, par_trace):
    def is_TE(psi: np.ndarray, dim: int = 2) -> bool:
        # Pure NumPy — avoids 17500+ GPU<->CPU transfers per plot for dim=128
        n = int(np.log2(len(psi)))
        k = n - n // 2
        for _i in combinations(range(n), k):
            per = [x for x in range(n) if x not in _i] + list(_i)
            psi_moved = np.transpose(psi.reshape([dim] * n), per).flatten()
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
def _(cp, go, is_TE, jl, make_subplots, np, pc, pickle, re):
    def compute_sre_exact(psi_np, alpha=2):
        """Compute exact SRE using HadaMAG.jl via JuliaCall."""
        if jl is None:
            return 0.0, 0.0

            try:
                arr_np = np.asarray(psi_np)
                norm_val = np.linalg.norm(arr_np)
                if norm_val > 1e-12:
                    arr_np = arr_np / norm_val
                dim = len(arr_np)
                n_qubits = int(np.log2(dim))

                # Call pre-defined Julia function directly — no seval parser overhead per iteration
                res = jl.jl_compute_sre_exact(arr_np, alpha, n_qubits, dim)
                return float(res[0]), float(res[1])
            except Exception as e:
                print(f"SRE Exact Calculation Error: {e}")
                return 0.0, 0.0

    print("compute_sre_exact reloaded OK")

    def plot_te_filtered_trajectories(
        pkl_files, labels, step_mes, use_te_filter, central_tendency, selected_metrics, plot_type="Line Chart", fs=None
    ):
        """Plot trajectories with optional TE filtering and bar chart modes."""
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
                    te_mask = np.array([is_TE(cp.asarray(state)) for state in final_states])
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

                            if init_sre == 0.0:
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

                    # Store back to pickle
                    if data_changed and not fs:
                        try:
                            with open(file, "wb") as f:
                                pickle.dump(data, f)
                        except Exception as e:
                            print(f"Error saving updated SRE back to {file}: {e}")

                    if not init_vals:
                        continue

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

            fig.update_layout(
                title=f"Comparison: Initial vs Final States ({central_tendency})",
                height=500,
                width=400 * len(selected_metrics),
                template="plotly_white",
                margin=dict(b=120),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return fig

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
                te_mask = np.array([is_TE(cp.asarray(state)) for state in final_states])
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

                        if init_sre == 0.0:
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
                te_mask = np.array([is_TE(cp.asarray(state)) for state in final_states])
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
                    if init_sre == 0.0:
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
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return fig

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
                te_mask = np.array([is_TE(cp.asarray(state)) for state in final_states])
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
                margin=dict(b=120),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )
            return fig

        fig.update_layout(
            title=f"Entanglement Trajectories ({central_tendency})"
            + (" — TE Filtered" if use_te_filter else ""),
            height=500,
            width=400 * len(selected_metrics),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(b=120),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
            ),
        )
        fig.update_xaxes(title_text="Steps")
        return fig

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
def _(GithubFileSystem, data_source, mo, os, refresh_button):
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

    mo.vstack([
        group_selector,
        mo.md("### 2. Plot Settings"),
        mo.hstack([plot_type_dropdown, metrics_to_plot, te_filter_checkbox, metric_selector, step_mes_input], justify="start", gap=2),
        mo.md("### 3. Execution"),
        plot_button
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
            labels.append(f"gap{_match.group(1)}")
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
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
