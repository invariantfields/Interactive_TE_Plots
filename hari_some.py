import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():


    import itertools
    import numpy as np

    from ncpol2sdpa import generate_operators, SdpRelaxation
    from sympy.physics.quantum.dagger import Dagger


    # ------------------------------------------------------------
    # 1. Basic target state data
    # ------------------------------------------------------------

    def psi0_projector_coeffs(p):
        vec = np.zeros(4, dtype=complex)
        vec[0] = np.sqrt(p)       # |00>
        vec[3] = np.sqrt(1 - p)   # |11>

        Pi = np.outer(vec, np.conjugate(vec))
        coeffs = {}

        for oA in range(2):
            for oB in range(2):
                i = 2 * oA + oB
                for opA in range(2):
                    for opB in range(2):
                        j = 2 * opA + opB
                        coeffs[(oA, oB, opA, opB)] = Pi[i, j]

        return coeffs


    def target_value_for_rho1(p, mu):
        overlap = (1 - p) * (1 - 2 * p)
        return 3 + mu * (1 - overlap)


    # ------------------------------------------------------------
    # 2. No-signalling vertices
    # ------------------------------------------------------------

    def local_deterministic_vertices():
        vertices = []

        for fA_vals in itertools.product([0, 1], repeat=2):
            for fB_vals in itertools.product([0, 1], repeat=2):
                P = np.zeros((2, 2, 2, 2))

                for x in range(2):
                    for y in range(2):
                        a = fA_vals[x]
                        b = fB_vals[y]
                        P[x, y, a, b] = 1.0

                vertices.append(P)

        return vertices


    def pr_vertices():
        vertices = []

        for alpha, beta, gamma in itertools.product([0, 1], repeat=3):
            P = np.zeros((2, 2, 2, 2))

            for x in range(2):
                for y in range(2):
                    rhs = (x * y) ^ (alpha * x) ^ (beta * y) ^ gamma

                    for a in range(2):
                        for b in range(2):
                            if (a ^ b) == rhs:
                                P[x, y, a, b] = 0.5

            vertices.append(P)

        return vertices


    def ns_vertices_2222():
        return local_deterministic_vertices() + pr_vertices()


    def witness_coefficients_from_vertex(P, mu):
        c = np.zeros((2, 2))

        for a in range(2):
            for b in range(2):
                c[a, b] = (
                    P[0, 0, a, b]
                    + P[0, 1, a, b]
                    + P[1, 0, a, b]
                    - mu * P[1, 1, a, b]
                )

        return c


    # ------------------------------------------------------------
    # 3. Helper functions
    # ------------------------------------------------------------

    def build_commutation_substitutions(alice_ops, bob_ops):
        subs = {}

        alice_all = []
        bob_all = []

        for Aop in alice_ops:
            alice_all.append(Aop)
            alice_all.append(Dagger(Aop))

        for Bop in bob_ops:
            bob_all.append(Bop)
            bob_all.append(Dagger(Bop))

        for Aop in alice_all:
            for Bop in bob_all:
                subs[Bop * Aop] = Aop * Bop

        return subs


    def unique_by_string(seq):
        out = []
        seen = set()

        for x in seq:
            key = str(x)
            if key not in seen:
                out.append(x)
                seen.add(key)

        return out


    # ------------------------------------------------------------
    # 4. NPA-like relaxation for one NS vertex
    # ------------------------------------------------------------

    def npa_upper_fixed_vertex(
        p,
        mu,
        P_vertex,
        level=1,
        kraus_rank=1,
        solver="cvxpy",
        solverparameters=None,
        verbose=0,
    ):
        Pi = psi0_projector_coeffs(p)
        c = witness_coefficients_from_vertex(P_vertex, mu)

        A = {}
        B = {}

        alice_ops = []
        bob_ops = []

        # Alice channel component operators A[a,r,o]
        for a in range(2):
            for r in range(kraus_rank):
                for o in range(2):
                    name = f"A_{a}_{r}_{o}"
                    op = generate_operators(name, 1, hermitian=False)[0]
                    A[(a, r, o)] = op
                    alice_ops.append(op)

        # Bob channel component operators B[b,s,o]
        for b in range(2):
            for s in range(kraus_rank):
                for o in range(2):
                    name = f"B_{b}_{s}_{o}"
                    op = generate_operators(name, 1, hermitian=False)[0]
                    B[(b, s, o)] = op
                    bob_ops.append(op)

        all_ops = alice_ops + bob_ops

        substitutions = build_commutation_substitutions(alice_ops, bob_ops)

        # Trace-preservation constraints:
        # sum_{r,o} A[a,r,o]^dag A[a,r,o] = I
        # sum_{s,o} B[b,s,o]^dag B[b,s,o] = I
        equalities = []

        for a in range(2):
            expr = 0
            for r in range(kraus_rank):
                for o in range(2):
                    expr += Dagger(A[(a, r, o)]) * A[(a, r, o)]
            equalities.append(expr - 1)

        for b in range(2):
            expr = 0
            for s in range(kraus_rank):
                for o in range(2):
                    expr += Dagger(B[(b, s, o)]) * B[(b, s, o)]
            equalities.append(expr - 1)

        # Build fidelity polynomials f_ab.
        #
        # f_ab = Tr[psi0 (E_a tensor F_b)(omega)]
        #
        # These contain degree-4 monomials:
        # A^dag A B^dag B.
        #
        # At level=1, we manually include them using extra_monomials.
        f = {}
        extra_monomials = []

        for a in range(2):
            for b in range(2):
                expr = 0

                for r in range(kraus_rank):
                    for s in range(kraus_rank):
                        for oA in range(2):
                            for oB in range(2):
                                for opA in range(2):
                                    for opB in range(2):
                                        coeff = Pi[(oA, oB, opA, opB)]

                                        if abs(coeff) > 1e-12:
                                            term = (
                                                Dagger(A[(a, r, opA)])
                                                * A[(a, r, oA)]
                                                * Dagger(B[(b, s, opB)])
                                                * B[(b, s, oB)]
                                            )

                                            expr += float(np.real(coeff)) * term
                                            extra_monomials.append(term)

                f[(a, b)] = expr

        extra_monomials = unique_by_string(extra_monomials)

        objective_expr = 0

        for a in range(2):
            for b in range(2):
                objective_expr += float(c[a, b]) * f[(a, b)]

        # ncpol2sdpa minimizes, so minimize negative.
        objective = -objective_expr

        sdp = SdpRelaxation(all_ops, verbose=verbose)

        sdp.get_relaxation(
            level,
            objective=objective,
            equalities=equalities,
            substitutions=substitutions,
            extramonomials=extra_monomials,
        )

        if solverparameters is None:
            solverparameters = {}

        sdp.solve(
            solver=solver,
            solverparameters=solverparameters,
        )

        max_poly_upper = -float(sdp.primal)
        beta_upper = mu + max_poly_upper

        return beta_upper, {
            "status": sdp.status,
            "primal": sdp.primal,
            "dual": sdp.dual,
            "coefficients": c,
        }


    # ------------------------------------------------------------
    # 5. Full beta upper bound over all NS vertices
    # ------------------------------------------------------------

    def beta_upper_npa(
        p,
        mu,
        level=2,
        kraus_rank=1,
        solver="cvxpy",
        solverparameters=None,
        verbose=0,
    ):
        vertices = ns_vertices_2222()

        best_beta = -1e100
        best_info = None
        all_results = []

        for idx, P in enumerate(vertices):
            print(f"Solving NS vertex {idx + 1}/{len(vertices)}...")

            beta_v, info_v = npa_upper_fixed_vertex(
                p=p,
                mu=mu,
                P_vertex=P,
                level=level,
                kraus_rank=kraus_rank,
                solver=solver,
                solverparameters=solverparameters,
                verbose=verbose,
            )

            result = {
                "vertex_index": idx,
                "beta_upper": beta_v,
                "info": info_v,
            }

            all_results.append(result)

            print(f"  beta_upper(vertex {idx}) = {beta_v}")
            print(f"  status = {info_v['status']}")

            if beta_v > best_beta:
                best_beta = beta_v
                best_info = result

        return best_beta, best_info, all_results


    # ------------------------------------------------------------
    # 6. Optional trivial bound
    # ------------------------------------------------------------

    def beta_trivial_upper(mu):
        """
        Very loose but instant bound using only 0 <= f_ab <= 1.
        """
        vertices = ns_vertices_2222()

        best_beta = -1e100
        best_idx = None

        for idx, P in enumerate(vertices):
            c = witness_coefficients_from_vertex(P, mu)

            beta_v = mu

            for a in range(2):
                for b in range(2):
                    if c[a, b] > 0:
                        beta_v += c[a, b]

            if beta_v > best_beta:
                best_beta = beta_v
                best_idx = idx

        return best_beta, best_idx


    # ------------------------------------------------------------
    # 7. Example run
    # ------------------------------------------------------------

    if __name__ == "__main__":
        p = 0.20
        mu = 1.0

        # Fast weak relaxation.
        # Level 1 is much faster, but looser.
        level = 2

        # Start with 1.
        # If it runs, try kraus_rank = 2.
        kraus_rank = 1

        # Use "cvxpy" only if cvxpy is installed and licensed.
        solver = "cvxpy"


        print("Running trivial bound first...")
        trivial_beta, trivial_idx = beta_trivial_upper(mu)

        print("trivial beta_upper =", trivial_beta)
        print("trivial best vertex =", trivial_idx)

        print("\nRunning level-1 NPA-style bound...")

        beta_up, best, results = beta_upper_npa(
            p=p,
            mu=mu,
            level=level,
            kraus_rank=kraus_rank,
            solver=solver,
            solverparameters={},
            verbose=0,
        )

        T_target = target_value_for_rho1(p, mu)

        print("\n===================================================")
        print("Level-1 NPA-style upper bound calculation")
        print("===================================================")
        print(f"p = {p}")
        print(f"mu = {mu}")
        print(f"level = {level}")
        print(f"kraus_rank = {kraus_rank}")
        print(f"beta_upper = {beta_up}")
        print(f"T_target for rho1 = {T_target}")
        print(f"violation? {T_target > beta_up}")

        if best is not None:
            print("best vertex index:", best["vertex_index"])


    return (
        best,
        beta_trivial_upper,
        beta_upper_npa,
        mu,
        p,
        target_value_for_rho1,
    )


@app.cell
def _(best, beta_trivial_upper, beta_upper_npa, mu, p, target_value_for_rho1):
    _p = 0.20
    _mu = 1.0
    _level = 3
    _kraus_rank = 1
    _solver = "cvxpy"


    print("Running trivial bound first...")
    _trivial_beta, _trivial_idx = beta_trivial_upper(_mu)

    print("trivial beta_upper =", _trivial_beta)
    print("trivial best vertex =", _trivial_idx)

    print("\nRunning level-1 NPA-style bound...")

    _beta_up, _best, _results = beta_upper_npa(
        p=_p,
        mu=_mu,
        level=_level,
        kraus_rank=_kraus_rank,
        solver=_solver,
        solverparameters={},
        verbose=0,
    )

    _T_target = target_value_for_rho1(p, mu)

    print("\n===================================================")
    print("Level-1 NPA-style upper bound calculation")
    print("===================================================")
    print(f"p = {_p}")
    print(f"mu = {_mu}")
    print(f"level = {_level}")
    print(f"kraus_rank = {_kraus_rank}")
    print(f"beta_upper = {_beta_up}")
    print(f"T_target for rho1 = {_T_target}")
    print(f"violation? {_T_target > _beta_up}")

    if best is not None:
        print("best vertex index:", _best["vertex_index"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
