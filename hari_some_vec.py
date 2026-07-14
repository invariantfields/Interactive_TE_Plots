import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import itertools
    import numpy as np
    import sympy as sympy
    from ncpol2sdpa import generate_operators, SdpRelaxation
    from sympy.physics.quantum.dagger import Dagger


    # ------------------------------------------------------------
    # 1. Basic target state data
    # ------------------------------------------------------------

    def psi0_projector_coeffs(p):
        """
        Returns the projector Pi as a 4D tensor (oA, oB, opA, opB).
        Reshaping natively handles the binary string-to-index mapping.
        """
        vec = np.zeros(4, dtype=complex)
        vec[0] = np.sqrt(p)       # |00>
        vec[3] = np.sqrt(1 - p)   # |11>

        Pi_2D = np.outer(vec, np.conjugate(vec))
        return Pi_2D.reshape((2, 2, 2, 2))


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
                        P[x, y, fA_vals[x], fB_vals[y]] = 1.0
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
        """
        Vectorized extraction of witness coefficients. 
        P has shape (2, 2, 2, 2) -> (x, y, a, b).
        This natively returns a (2, 2) array for c[a, b].
        """
        return P[0, 0] + P[0, 1] + P[1, 0] - mu * P[1, 1]


    # ------------------------------------------------------------
    # 3. Helper functions
    # ------------------------------------------------------------

    def build_commutation_substitutions(A_arr, B_arr):
        subs = {}

        # Flatten operator arrays to process all combinations
        A_list = A_arr.flatten().tolist()
        B_list = B_arr.flatten().tolist()

        alice_all = A_list + [Dagger(op) for op in A_list]
        bob_all = B_list + [Dagger(op) for op in B_list]

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
    # 4. NPA-like relaxation for one NS vertex (Vectorized)
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

        # 1. Generate Operators via list comprehension & Reshape
        # This properly creates each operator individually before wrapping them in NumPy matrices
        A_list = [generate_operators(f"A_{a}_{r}_{o}", 1, hermitian=False)[0] 
                  for a in range(2) for r in range(kraus_rank) for o in range(2)]

        B_list = [generate_operators(f"B_{b}_{s}_{o}", 1, hermitian=False)[0] 
                  for b in range(2) for s in range(kraus_rank) for o in range(2)]

        # Cast lists to NumPy object arrays and reshape
        A_arr = np.array(A_list, dtype=object).reshape(2, kraus_rank, 2)
        B_arr = np.array(B_list, dtype=object).reshape(2, kraus_rank, 2)

        A_dag = np.array([Dagger(op) for op in A_list], dtype=object).reshape(2, kraus_rank, 2)
        B_dag = np.array([Dagger(op) for op in B_list], dtype=object).reshape(2, kraus_rank, 2)

        all_ops = A_list + B_list
        substitutions = build_commutation_substitutions(A_arr, B_arr)

        # 2. Vectorized Trace-Preservation Constraints
        # Summing over axis 1 (r) and 2 (o) yields an array of 2 constraints
        eq_A = np.sum(A_dag * A_arr, axis=(1, 2)) - 1
        eq_B = np.sum(B_dag * B_arr, axis=(1, 2)) - 1
        equalities = eq_A.tolist() + eq_B.tolist()

        # 3. Vectorized Polynomial Objective & Extra Monomials
        # Build fundamental pairs: M_A[a, r, opA, oA] = Dagger(A)*A
        M_A = A_dag[:, :, :, np.newaxis] * A_arr[:, :, np.newaxis, :]
        M_B = B_dag[:, :, :, np.newaxis] * B_arr[:, :, np.newaxis, :]

        # Extract ONLY the non-zero coefficients to avoid calculating 0-weight tensor paths
        oA_vals, oB_vals, opA_vals, opB_vals = np.nonzero(np.abs(Pi) > 1e-12)
        coeffs = Pi[oA_vals, oB_vals, opA_vals, opB_vals].real.astype(float)

        # Advance-index out the necessary dimensions. Result shapes: (2, kraus_rank, N_nonzero)
        M_A_nz = M_A[:, :, opA_vals, oA_vals]
        M_B_nz = M_B[:, :, opB_vals, oB_vals]

        # Cross multiply A & B. Result shape: (a, b, r, s, i) -> (2, 2, K, K, N_nonzero)
        monomials = (
            M_A_nz[:, np.newaxis, :, np.newaxis, :] * M_B_nz[np.newaxis, :, np.newaxis, :, :]
        )
        weighted_monomials = monomials * coeffs

        # Expand witness coefficients `c` to broadcast against the 5D tensor, then sum entirely
        c_expanded = c[:, :, np.newaxis, np.newaxis, np.newaxis]

        # Use standard Python sum for the final aggregation to ensure SymPy builds the expression cleanly
        objective_expr = (c_expanded * weighted_monomials).sum()

        extra_monomials = unique_by_string(monomials.flatten().tolist())

        # ncpol2sdpa minimizes, so we minimize the negative objective
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

        sdp.solve(solver=solver, solverparameters=solverparameters)

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
            # Vectorized sum of positive elements only
            beta_v = mu + np.sum(c[c > 0])

            if beta_v > best_beta:
                best_beta = beta_v
                best_idx = idx

        return best_beta, best_idx

    def beta_upper_all_vertices_fast(
        p,
        mu,
        level=1, 
        kraus_rank=1,
        solver="scs",
        solverparameters=None,
        verbose=0,
    ):
        vertices = ns_vertices_2222()
        Pi = psi0_projector_coeffs(p)

        # 1. Generate Operators
        A_list = [generate_operators(f"A_{a}_{r}_{o}", 1, hermitian=False)[0] 
                  for a in range(2) for r in range(kraus_rank) for o in range(2)]
        B_list = [generate_operators(f"B_{b}_{s}_{o}", 1, hermitian=False)[0] 
                  for b in range(2) for s in range(kraus_rank) for o in range(2)]

        A_arr = np.array(A_list, dtype=object).reshape(2, kraus_rank, 2)
        B_arr = np.array(B_list, dtype=object).reshape(2, kraus_rank, 2)
        A_dag = np.array([Dagger(op) for op in A_list], dtype=object).reshape(2, kraus_rank, 2)
        B_dag = np.array([Dagger(op) for op in B_list], dtype=object).reshape(2, kraus_rank, 2)

        all_ops = A_list + B_list
        substitutions = build_commutation_substitutions(A_arr, B_arr)

        # 2. Vectorized Trace-Preservation Constraints
        eq_A = np.sum(A_dag * A_arr, axis=(1, 2)) - 1
        eq_B = np.sum(B_dag * B_arr, axis=(1, 2)) - 1
        equalities = eq_A.tolist() + eq_B.tolist()

        # 3. Base Monomials for Objective
        M_A = A_dag[:, :, :, np.newaxis] * A_arr[:, :, np.newaxis, :]
        M_B = B_dag[:, :, :, np.newaxis] * B_arr[:, :, np.newaxis, :]
        oA_vals, oB_vals, opA_vals, opB_vals = np.nonzero(np.abs(Pi) > 1e-12)
        coeffs = Pi[oA_vals, oB_vals, opA_vals, opB_vals].real.astype(float)
    
        M_A_nz = M_A[:, :, opA_vals, oA_vals]
        M_B_nz = M_B[:, :, opB_vals, oB_vals]
    
        monomials = (M_A_nz[:, np.newaxis, :, np.newaxis, :] * M_B_nz[np.newaxis, :, np.newaxis, :, :])
        weighted_monomials = monomials * coeffs
        extra_monomials = unique_by_string(monomials.flatten().tolist())

        # ==========================================
        # THE BOTTLENECK FIX: COMPILE EXACTLY ONCE
        # ==========================================
        print(f"\n--- SYMBOLIC COMPILATION PHASE ---")
        print(f"Compiling the generic SDP constraints at Level {level}...")
        print(f"(Note: Level 2 with 8+ operators takes a very long time, but it only happens ONCE now.)")
    
        sdp = SdpRelaxation(all_ops, verbose=verbose)
    
        # We pass 0 as the objective to compile the feasible region without calculating gradients yet
        sdp.get_relaxation(
            level,
            objective=0, 
            equalities=equalities,
            substitutions=substitutions,
            extramonomials=extra_monomials,
        )
        print("Compilation complete! Feasible region mapped to SDP.\n")

        # ==========================================
        # NUMERICAL SOLVING PHASE: LIGHTNING FAST
        # ==========================================
        best_beta = -1e100
        best_info = None
        all_results = []
    
        if solverparameters is None:
            solverparameters = {"eps_abs": 1e-5, "eps_rel": 1e-5}

        print("--- SOLVING PHASE ---")
        for idx, P_vertex in enumerate(vertices):
            # Calculate objective strictly for this specific vertex
            c = witness_coefficients_from_vertex(P_vertex, mu)
            c_expanded = c[:, :, np.newaxis, np.newaxis, np.newaxis]
        
            flat_terms = (c_expanded * weighted_monomials).flatten().tolist()
            objective_expr = sympy.Add(*flat_terms)
        
            # SWAP THE OBJECTIVE VECTOR IN THE ALREADY COMPILED SDP
            sdp.set_objective(-objective_expr)
        
            # Fire the GPU solver
            sdp.solve(solver=solver, solverparameters=solverparameters)
        
            max_poly_upper = -float(sdp.primal)
            beta_v = mu + max_poly_upper
        
            result = {
                "vertex_index": idx,
                "status": sdp.status,
                "beta_upper": beta_v
            }
            all_results.append(result)
        
            print(f"  [Vertex {idx+1:02d}/24] beta_upper = {beta_v:.6f} | status = {sdp.status}")

            if beta_v > best_beta:
                best_beta = beta_v
                best_info = result

        return best_beta, best_info, all_results


    if __name__ == "__main__":
        p = 0.20
        mu = 1.0

        # Try this on Level 1 first to test the logic! 
        # It will run entirely in less than a second.
        level = 3 
        kraus_rank = 1
        solver = "scs"

        beta_up, best, results = beta_upper_all_vertices_fast(
            p=p,
            mu=mu,
            level=level,
            kraus_rank=kraus_rank,
            solver=solver
        )
    
        # Optional formatting logic
        if best is not None:
            print("\n===================================================")
            print(f"Overall Best Vertex: {best['vertex_index'] + 1}")
            print(f"Maximum Beta Upper Bound: {beta_up}")
    return (
        beta_trivial_upper,
        beta_upper_npa,
        kraus_rank,
        mu,
        p,
        target_value_for_rho1,
    )


@app.cell
def _(
    beta_trivial_upper,
    beta_upper_npa,
    kraus_rank,
    mu,
    p,
    target_value_for_rho1,
):
    _p = 0.20
    _mu = 1.0
    _level = 3
    _kraus_rank = 1
    _solver = "cvxpy"


    print("Running trivial bound first...")
    trivial_beta, trivial_idx = beta_trivial_upper(_mu)

    print("trivial beta_upper =", trivial_beta)
    print("trivial best vertex =", trivial_idx)

    print("\nRunning level-1 NPA-style bound...")

    beta_up, best, results = beta_upper_npa(
        p=_p,
        mu=_mu,
        level=_level,
        kraus_rank=_kraus_rank,
        solver=_solver,
        solverparameters={},
        verbose=0,
    )

    T_target = target_value_for_rho1(p, mu)

    print("\n===================================================")
    print("Level-1 NPA-style upper bound calculation")
    print("===================================================")
    print(f"p = {p}")
    print(f"mu = {mu}")
    print(f"level = {_level}")
    print(f"kraus_rank = {kraus_rank}")
    print(f"beta_upper = {beta_up}")
    print(f"T_target for rho1 = {T_target}")
    print(f"violation? {T_target > beta_up}")

    if best is not None:
        print("best vertex index:", best["vertex_index"])
    return


if __name__ == "__main__":
    app.run()
