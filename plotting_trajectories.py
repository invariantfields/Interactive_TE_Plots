# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "cupy-cuda12x",
#     "plotly",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


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

    return combinations, cp, go, make_subplots, mo, np, os, pc, pickle


@app.cell
def _(cp, np):
    def par_trace(psi, dim, n, n_parties):
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def is_appt(x: np.array) -> bool:
        _purity = cp.sum(x * x)
        _D = cp.shape(x)[0]
        if _purity <= 1 / (_D - 1):
            return True
        _ex, _ = cp.linalg.eigh(x)
        if (_ex[-1] - _ex[1]) ** 2 <= 4 * _ex[0] * _ex[3]:
            return True
        return False

    return is_appt, par_trace


@app.cell
def _(combinations, cp, is_appt, np, par_trace):
    def is_TE(psi: np.array, dim: int = 2) -> bool:
        n = int(np.log2(len(psi)))
        k = n - n // 2
        for _i in combinations(range(n), k):
            per = list(set(range(n)) - set(_i)) + list(_i)
            psi_moved = cp.moveaxis(
                psi.reshape([dim] * n), list(range(n)), per
            ).flatten()
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
def _(cp, go, is_TE, make_subplots, np, pc, pickle):
    def plot_te_filtered_trajectories(
        pkl_files, labels, step_mes, use_te_filter, central_tendency, selected_metrics
    ):
        """Plot trajectories with optional TE filtering."""
        if not selected_metrics:
            return None

        metric_map = {
            "average_purity": "Average Purity",
            "max_purity": "Max Purity",
            "sre": "SRE"
        }
        metric_titles = [metric_map[m] for m in selected_metrics]

        colors = pc.qualitative.Plotly
        fig = make_subplots(rows=1, cols=len(selected_metrics), subplot_titles=metric_titles)

        for file_idx, (file, label) in enumerate(zip(pkl_files, labels)):
            if not file or not label:
                continue
            try:
                with open(file, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # Filter by TE if checkbox is checked
            if use_te_filter:
                final_states = data["final_states"]
                te_mask = np.array(
                    [is_TE(cp.asarray(state)) for state in final_states]
                )
                n_te = te_mask.sum()
                if n_te == 0:
                    print(f"⚠️ {label}: No TE states, skipping.")
                    continue
                prefix = f"TE-only (n={n_te}) "
            else:
                n_total = len(data["final_states"])
                te_mask = np.ones(n_total, dtype=bool)
                prefix = ""

            base_color = colors[file_idx % len(colors)]
            fill_color = hex_to_rgba(base_color, alpha=0.2)

            for i, metric in enumerate(selected_metrics):
                col = i + 1
                arr = np.array(data[metric])[te_mask]
                steps = np.arange(arr.shape[1]) * step_mes

                if central_tendency == "Average":
                    center_line = np.mean(arr, axis=0)
                    std = np.std(arr, axis=0)
                    lower_bound = center_line - std
                    upper_bound = center_line + std
                    leg_suffix = f"(Avg ±1 Std)"
                else:
                    center_line = np.median(arr, axis=0)
                    lower_bound = np.percentile(arr, 25, axis=0)
                    upper_bound = np.percentile(arr, 75, axis=0)
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
        fig.update_xaxes(title_text="Optimization Steps")
        return fig

    return (plot_te_filtered_trajectories,)


@app.cell
def _(mo):
    mo.md("""
    # Entanglement Trajectory Visualization
    """)
    return


@app.cell
def _(mo, os):
    import re

    # Dynamic file selection from data directory
    data_dir = "data"

    refresh_button = mo.ui.button(label="🔄 Refresh File List", value=0)

    # We use refresh_button.value as a dependency to trigger re-scanning
    refresh_button

    available_files = (
        sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")])
        if os.path.exists(data_dir)
        else []
    )

    # Group files by prefix (everything before 'stps' or 'steps')
    groups = {}
    for _f in available_files:
        _basename = os.path.basename(_f)
        _match = re.search(r'^(.*)(?:stps|steps)\d+\.pkl$', _basename)
        if _match:
            _group_name = _match.group(1).rstrip("_")
        else:
            _group_name = _basename.replace(".pkl", "")

        if _group_name not in groups:
            groups[_group_name] = []
        groups[_group_name].append(_f)

    sorted_group_names = sorted(groups.keys())
    for _g in groups:
        groups[_g] = sorted(groups[_g])

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

    step_mes_input = mo.ui.number(
        start=1, stop=1000, step=1, value=1, label="Steps per measurement"
    )

    # Run button to prevent heavy computations on every click
    plot_button = mo.ui.run_button(label="🚀 Generate Plot")

    mo.vstack([
        mo.md("### 1. Data Selection"),
        mo.hstack([group_selector, refresh_button], align="end"),
        mo.md("### 2. Plot Settings"),
        mo.hstack([metrics_to_plot, te_filter_checkbox, metric_selector, step_mes_input], justify="start", gap=2),
        mo.md("### 3. Execution"),
        plot_button
    ])
    return (
        group_selector,
        groups,
        metric_selector,
        metrics_to_plot,
        plot_button,
        re,
        step_mes_input,
        te_filter_checkbox,
    )


@app.cell
def _(
    group_selector,
    groups,
    metric_selector,
    metrics_to_plot,
    mo,
    plot_button,
    plot_te_filtered_trajectories,
    re,
    step_mes_input,
    te_filter_checkbox,
):
    mo.stop(not plot_button.value or not group_selector.value, mo.md("Select an experiment group and click **Generate Plot**."))

    selected_group_files = groups[group_selector.value]

    # Deriving labels like 'gap0', 'gap1' from the filenames
    labels = []
    for _f in selected_group_files:
        _match = re.search(r'(?:stps|steps)(\d+)\.pkl$', _f)
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
    )
    plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
