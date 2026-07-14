# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import glob
    import os
    import pickle
    from itertools import combinations
    from fractions import Fraction

    return Fraction, combinations, glob, go, mo, np, pickle


@app.cell
def _(mo):
    mo.md(r"""
    # 🔬 Rational Approximations of Entangled States (TE Check)

    This notebook imports optimization trajectory data, extracts the final states, filters out the ones that are **threshold Entangled (TE)**, and applies a rational approximation to their complex coefficients. It then checks if the resulting rationalized state remains TE.# 🔬 Rational Approximations of Entangled States (TE Check)
    """)
    return


@app.cell
def _(combinations, np):
    def par_trace(psi, dim, n, n_parties):
        n_rem = n - n_parties
        psi_mat = psi.reshape(dim**n_rem, dim**n_parties)
        return psi_mat @ psi_mat.conj().T

    def is_appt(x: np.ndarray) -> bool:
        _purity = np.sum(x.real**2 + x.imag**2)
        _D = x.shape[0]
        if _purity <= 1 / (_D - 1):
            return True
        _ex = np.linalg.eigvalsh(x)
        _ex = np.clip(_ex, 0.0, None)
        constraint = _ex[-1] - _ex[1] - 2 * np.sqrt(_ex[0] * _ex[2])
        if constraint <= 0:
            return True
        return False

    def is_TE(psi: np.array, dim: int = 2) -> bool:
        n = int(np.log2(len(psi)))
        k = n - n // 2
        for _i in combinations(range(n), k):
            per = [x for x in range(n) if x not in _i] + list(_i)
            psi_moved = np.transpose(
                psi.reshape([dim] * n), per
            ).flatten()
            _x = par_trace(psi_moved, dim, n, k)
            _x = (_x + _x.conj().T) / 2.0
            if not is_appt(_x):
                return False
        return True

    def prnt_is_te(psi: np.ndarray, dim: int=2): 
        def a_is_appt(x: np.ndarray) -> bool:
            _purity = np.sum(x.real**2 + x.imag**2)
            _D = x.shape[0]
            _ex = np.linalg.eigvalsh(x)
            _ex = np.clip(_ex, 0.0, None)
            constraint = _ex[-1] - _ex[1] - 2 * np.sqrt(_ex[0] * _ex[2])
            print(_ex)
            print(constraint)
            if constraint <= 0:
                return True
            return False
        n = int(np.log2(len(psi)))
        k = n - n // 2
        for _i in combinations(range(n), k):
            per = [x for x in range(n) if x not in _i] + list(_i)
            psi_moved = np.moveaxis(
                psi.reshape([dim] * n), list(range(n)), per
            ).flatten()
            _x = par_trace(psi_moved, dim, n, k)
            _x = (_x + _x.conj().T) / 2.0
            if not a_is_appt(_x):
                return False
        return True

    return (is_TE,)


@app.cell
def _(Fraction, np):
    def approximate_coefficient(val, max_denominator):
        if isinstance(val, (complex, np.complex128)):
            real_frac = Fraction(float(val.real)).limit_denominator(max_denominator)
            imag_frac = Fraction(float(val.imag)).limit_denominator(max_denominator)
            return complex(real_frac) + 1j * complex(imag_frac), real_frac, imag_frac
        else:
            frac = Fraction(float(val)).limit_denominator(max_denominator)
            return float(frac), frac, None

    def approximate_state(psi, max_denominator):
        approx_coeffs = []
        fracs = []
        for c in psi:
            approx_c, r_frac, i_frac = approximate_coefficient(c,10**max_denominator)
            approx_coeffs.append(approx_c)
            fracs.append((r_frac, i_frac))

        approx_coeffs = np.array(approx_coeffs)
        # Re-normalize the state vector after approximation
        norm = np.linalg.norm(approx_coeffs)
        if norm > 0:
            approx_coeffs /= norm
        return approx_coeffs, fracs

    def format_fraction(f):
        if f.denominator == 1:
            return f"{f.numerator}"
        return f"{f.numerator}/{f.denominator}"

    def format_complex_fraction(r_frac, i_frac):
        if i_frac is None:
            return format_fraction(r_frac)
        r_str = format_fraction(r_frac)
        i_str = format_fraction(i_frac)
        if i_frac == 0:
            return r_str
        if r_frac == 0:
            return f"{i_str}j"
        if i_frac > 0:
            return f"{r_str} + {i_str}j"
        else:
            return f"{r_str} - {format_fraction(-i_frac)}j"

    return approximate_state, format_complex_fraction


@app.cell
def _(glob, mo):
    # Find all trajectory pickle files in data/, new_data/, and correct_data/
    files = (
        glob.glob("correct_data/*.pkl")
        + glob.glob("new_data/*.pkl")
        + glob.glob("data/*.pkl")
    )
    files.sort()

    if not files:
        file_selector = None
        file_output = mo.md("⚠️ **No pickle data files found in data/ or new_data/ or correct_data/ directories.**")
    else:
        file_selector = mo.ui.dropdown(
            options={f: f for f in files},
            value=files[0] if files else None,
            label="📁 **Select Trajectory Data File:** ",
        )
        file_output = None
    file_output
    return (file_selector,)


@app.cell
def _(file_selector, mo):
    if file_selector is not None:
        mo.output.replace(file_selector)
    return


@app.cell
def _(file_selector, is_TE, pickle):
    if file_selector is None or file_selector.value is None:
        loaded_data = None
        state_list = []
    else:
        print(file_selector.value)
        with open(file_selector.value, "rb") as f:
            raw_data = pickle.load(f)

        # Extract states
        if isinstance(raw_data, dict) and "final_states" in raw_data:
            loaded_states = raw_data["final_states"]

            # Check which states are TE
            state_list = []
            for idx, state in enumerate(loaded_states):
                is_te_status = is_TE(state)
                state_list.append({
                    "index": idx,
                    "state": state,
                    "is_TE": is_te_status
                })
        else:
            state_list = []
    return (state_list,)


@app.cell
def _(mo, state_list):
    if not state_list:
        denominator_input = None
        state_output = mo.md("⚠️ **No states found in the selected file or file is invalid.**")
    else:
        # Filter for TE indices
        te_indices = [s["index"] for s in state_list if s["is_TE"]]

        if not te_indices:
            denominator_input = None
            state_output = mo.md("⚠️ **No threshold Entangled (TE) states found in this file.**")
        else:
            denominator_input = mo.ui.slider(
                start=1,
                stop=12,
                value=2,
                step=0.1,
                label="🔢 **Max Len Denominator Limit:** ",
            )
            state_output = None
    state_output
    return (denominator_input,)


@app.cell
def _(denominator_input, mo):
    if denominator_input is not None:
        mo.output.replace(denominator_input)
    return


@app.cell
def _(approximate_state, denominator_input, is_TE, mo, np, state_list):
    if denominator_input is None or not state_list:
        batch_results = []
        summary_table = None
        batch_output = None
    else:
        max_denom = denominator_input.value
        batch_results = []

        for s in state_list:
            if s["is_TE"]:
                psi = s["state"]
                _approx_psi, _fracs = approximate_state(psi, max_denom)
                approx_is_te = is_TE(_approx_psi)
                _fidelity = np.abs(np.vdot(psi, _approx_psi)) ** 2
                max_error = np.max(np.abs(psi - _approx_psi))

                batch_results.append({
                    "Index": s["index"],
                    "Original TE": "Yes",
                    "Remaining TE": "✅ Yes" if approx_is_te else "❌ No",
                    "Fidelity": f"{_fidelity:.6f}",
                    "Max Error": f"{max_error:.2e}",
                    "_original_psi": psi,
                    "_approx_psi": _approx_psi,
                    "_fracs": _fracs
                })

        if not batch_results:
            batch_output = mo.md("⚠️ **No TE states found.**")
            summary_table = None
        else:
            num_remaining = sum(1 for r in batch_results if "✅" in r["Remaining TE"])
            total_te = len(batch_results)

            # Summary callout
            if num_remaining == total_te:
                kind = "success"
                msg = f"🎉 **All {total_te} TE states remained threshold entangled** under rational approximation with max denominator digits= {max_denom}!"
            elif num_remaining > 0:
                kind = "warn"
                msg = f"⚠️ **{num_remaining} out of {total_te} TE states remained threshold entangled** under rational approximation with max denominator digits = {max_denom}."
            else:
                kind = "danger"
                msg = f"❌ **None of the {total_te} TE states remained threshold entangled** under rational approximation with max denominator digits= {max_denom}."

            callout = mo.callout(msg, kind=kind)

            # Interactive single selection table
            summary_table = mo.ui.table(
                [
                    {
                        "Index": r["Index"],
                        "Remaining TE": r["Remaining TE"],
                        "Fidelity": r["Fidelity"],
                        "Max Error": r["Max Error"]
                    }
                    for r in batch_results
                ],
                selection="single",
                label="📊 **Batch Test Results (Click any row to view detailed coefficients & plots below):**"
            )

            batch_output = mo.vstack([
                mo.md("### 📊 Batch Test Results for All TE States"),
                callout,
                summary_table
            ], gap=1.5)
    return batch_output, batch_results, summary_table


@app.cell
def _(batch_output, mo):
    if batch_output is not None:
        mo.output.replace(batch_output)
    return


@app.cell
def _(batch_results, format_complex_fraction, mo, np, summary_table):
    if not batch_results or summary_table is None:
        detailed_output = None
        selected_state_data = None
        comp_details = []
    else:
        # Determine which state is selected
        selected_index = None
        if summary_table.value:
            selected_index = summary_table.value[0]["Index"]
        else:
            selected_index = batch_results[0]["Index"]

        # Get the full record for the selected index
        selected_record = next(r for r in batch_results if r["Index"] == selected_index)

        _orig_psi = selected_record["_original_psi"]
        _approx_psi = selected_record["_approx_psi"]
        _fracs = selected_record["_fracs"]
        _fidelity = float(selected_record["Fidelity"])
        _is_te_status = selected_record["Remaining TE"]

        # Build comparison details
        comp_details = []
        for _idx, (orig, approx) in enumerate(zip(_orig_psi, _approx_psi)):
            rf, if_ = _fracs[_idx]
            frac_str = format_complex_fraction(rf, if_)
            comp_details.append({
                "Coefficient Index": _idx,
                "Original": f"{orig.real:+.6f} {orig.imag:+.6f}j",
                "Rational Approx": frac_str,
                "Approximated": f"{approx.real:+.6f} {approx.imag:+.6f}j",
                "Abs Error": f"{np.abs(orig - approx):.6e}"
            })

        detailed_output = mo.vstack([
            mo.md(f"### 🔍 Detailed Analysis for State Index {selected_index}"),
            mo.hstack([
                mo.stat(value=f"{_fidelity:.6f}", label="State Fidelity F(orig, approx)"),
                mo.vstack([
                    mo.md("**Threshold Entanglement Status after Approx:**"),
                    mo.md(f"### {_is_te_status}")
                ], gap=0.5)
            ], justify="start", gap=5),
            mo.md("#### Coefficients Comparison Table"),
            mo.ui.table(comp_details, pagination=True)
        ], gap=1.5)

        selected_state_data = {
            "index": selected_index,
            "original_psi": _orig_psi,
            "approx_psi": _approx_psi
        }
    return detailed_output, selected_state_data


@app.cell
def _(detailed_output, mo):
    if detailed_output is not None:
        mo.output.replace(detailed_output)
    return


@app.cell
def _(go, mo, selected_state_data):
    if selected_state_data is None:
        plot_output = None
        fig = None
    else:
        original_psi = selected_state_data["original_psi"]
        approx_psi = selected_state_data["approx_psi"]
        state_index = selected_state_data["index"]
        indices = list(range(len(original_psi)))

        fig = go.Figure()

        # Original Real Part
        fig.add_trace(go.Bar(
            x=indices,
            y=[c.real for c in original_psi],
            name="Original (Real)",
            marker_color="royalblue",
            opacity=0.85
        ))

        # Approximated Real Part
        fig.add_trace(go.Bar(
            x=indices,
            y=[c.real for c in approx_psi],
            name="Rational (Real)",
            marker_color="lightblue",
            opacity=0.85
        ))

        # Original Imag Part
        fig.add_trace(go.Bar(
            x=indices,
            y=[c.imag for c in original_psi],
            name="Original (Imag)",
            marker_color="crimson",
            opacity=0.85
        ))

        # Approximated Imag Part
        fig.add_trace(go.Bar(
            x=indices,
            y=[c.imag for c in approx_psi],
            name="Rational (Imag)",
            marker_color="pink",
            opacity=0.85
        ))

        fig.update_layout(
            title=f"Comparison of Coefficients for State Index {state_index}: Original vs Rational Approximation",
            xaxis_title="Coefficient Index",
            yaxis_title="Value",
            barmode="group",
            template="plotly_white",
            height=450,
            width=900
        )

        plot_output = mo.vstack([
            mo.md("#### Visualization of Coefficients"),
            fig
        ])
    return (plot_output,)


@app.cell
def _(mo, plot_output):
    if plot_output is not None:
        mo.output.replace(plot_output)
    return


@app.cell
def _(np):
    state_test = np.array([
        (-119/1969) - (20/2411)*1.j,
        (41/7809) + (31/4792)*1.j,
        (47/8597) - (751/5651)*1.j,
        (877/8201) + (250/8699)*1.j,
        (121/9898) + (46/3283)*1.j,
        (306/1859) + (43/869)*1.j,
        (-140/8251) + (341/8299)*1.j,
        (116/3283) + (179/4086)*1.j,
        (-329/4873) + (234/5833)*1.j,
        (311/2977) + (248/4691)*1.j,
        (-353/5312) - (387/9524)*1.j,
        (-479/7580) - (345/9149)*1.j,
        (-103/1283) + (63/2755)*1.j,
        (-187/5442) - (136/3479)*1.j,
        (-519/6685) - (455/4404)*1.j,
        (176/6725) - (151/3924)*1.j,
        (429/6847) - (814/9131)*1.j,
        (-322/5371) + (176/2351)*1.j,
        (-699/7990) - (19/774)*1.j,
        (-351/6809) + (377/8752)*1.j,
        (-89/5829) + (1013/9709)*1.j,
        (-115/9391) - (130/8611)*1.j,
        (199/2295) - (135/3976)*1.j,
        (179/3017) - (97/2499)*1.j,
        (-49/6253) + (5/3659)*1.j,
        (-167/9841) - (21/7144)*1.j,
        (-31/4866) + (32/4137)*1.j,
        (-481/9809) - (86/2069)*1.j,
        (-261/7804) + (5/158)*1.j,
        (-695/6064) + (72/8917)*1.j,
        (926/6603) + (187/3499)*1.j,
        (-247/9537) - (45/707)*1.j,
        (170/8623) + (10/179)*1.j,
        (-119/9852) + (141/3347)*1.j,
        (-242/9969) - (33/5503)*1.j,
        (250/3769) + (127/6002)*1.j,
        (-336/9109) + (36/1135)*1.j,
        (637/9792) - (90/7789)*1.j,
        (383/7308) - (769/7833)*1.j,
        (-371/4133) + (196/7753)*1.j,
        (7/4716) - (71/7916)*1.j,
        (-307/2275) - (161/5207)*1.j,
        (89/7971) - (49/514)*1.j,
        (103/7408) + (129/7324)*1.j,
        (-251/5854) + (753/4711)*1.j,
        (-7/6966),
        (19/5514) + (17/1875)*1.j,
        (-155/8694) + (553/7895)*1.j,
        (380/6399) + (320/9129)*1.j,
        (-661/7925) + (595/9237)*1.j,
        (-314/5793) + (1311/9295)*1.j,
        (149/2132) + (12/995)*1.j,
        (-73/9534) - (697/8125)*1.j,
        (69/9907) + (728/6711)*1.j,
        (29/1545) + (94/2609)*1.j,
        (-109/8932) + (1099/9535)*1.j,
        (302/4613) + (1059/9866)*1.j,
        (107/2147) + (169/3010)*1.j,
        (339/3992) - (279/2729)*1.j,
        (191/3092) - (4/9271)*1.j,
        (31/1106) + (830/9161)*1.j,
        (17/8535) + (422/7285)*1.j,
        (8/1071) + (216/8135)*1.j,
        (-923/9843) - (543/5846)*1.j,
        (92/1515) - (20/153)*1.j,
        (41/1590) + (374/8185)*1.j,
        (54/6049) + (53/2860)*1.j,
        (-192/5333) - (37/667)*1.j,
        (-1031/9592) - (219/9640)*1.j,
        (-7/152) + (202/4995)*1.j,
        (-353/9126) + (547/7493)*1.j,
        (590/3993) - (71/7712)*1.j,
        (401/8612) + (213/5774)*1.j,
        (-61/3123) - (966/7459)*1.j,
        (621/9701) - (67/1684)*1.j,
        (-185/9764) - (134/2285)*1.j,
        (-609/9718) + (1014/9601)*1.j,
        (-397/9844) + (154/7411)*1.j,
        (687/8897) - (34/2483)*1.j,
        (89/3029) + (167/9026)*1.j,
        (589/9000) + (581/8965)*1.j,
        (159/3472) - (169/4234)*1.j,
        (-203/7235) + (361/3405)*1.j,
        (-247/3906) + (956/9851)*1.j,
        (151/8057) + (7/93)*1.j,
        (163/6163) - (46/7443)*1.j,
        (116/2569) + (1/4494)*1.j,
        (200/5943) + (821/8231)*1.j,
        (185/2744) - (154/2707)*1.j,
        (-221/7767) - (100/2659)*1.j,
        (134/2299) + (587/9213)*1.j,
        (-363/3287) - (421/4914)*1.j,
        (-468/9991) + (17/8186)*1.j,
        (5/4091) - (1198/8675)*1.j,
        (-44/2393) - (108/961)*1.j,
        (-503/8008) - (169/8109)*1.j,
        (5/423) + (112/3823)*1.j,
        (-5/889) - (236/2775)*1.j,
        (-103/1014) - (1101/9940)*1.j,
        (62/9909) + (25/3768)*1.j,
        (-193/5127) - (284/8377)*1.j,
        (1009/9691) - (199/3380)*1.j,
        (-1253/9648) - (180/8389)*1.j,
        (-11/4871) - (663/8426)*1.j,
        (-537/8458) - (1163/9894)*1.j,
        (-1/1298) - (249/9637)*1.j,
        (171/6686) - (309/9265)*1.j,
        (466/4091) - (979/8816)*1.j,
        (-187/8112) - (64/7213)*1.j,
        (252/6323) + (370/9161)*1.j,
        (317/6776) - (211/3646)*1.j,
        (-25/298) - (100/6757)*1.j,
        (110/1971) - (256/9983)*1.j,
        (490/7363) - (428/5177)*1.j,
        (-80/4947) + (49/6331)*1.j,
        (-589/9350) - (431/6918)*1.j,
        (-347/6651) + (971/8324)*1.j,
        (1/2888) + (236/2449)*1.j,
        (-62/695) - (201/2896)*1.j,
        (163/6197) + (87/6218)*1.j,
        (65/2413) + (463/8209)*1.j,
        (330/9761) + (5/87)*1.j,
        (941/9852) - (17/1040)*1.j,
        (-827/9891) + (616/9335)*1.j,
        (-341/9872) - (227/4203)*1.j,
        (1028/8849) - (41/6339)*1.j,
        (-609/9847) + (411/8819)*1.j,
        (-160/3259) + (292/7779)*1.j,
    ], dtype=np.complex128)
    return (state_test,)


@app.cell
def _(is_TE, state_test):
    is_TE(state_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
