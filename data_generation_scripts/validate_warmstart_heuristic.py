import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import jaxopt
from itertools import combinations

jax.config.update("jax_enable_x64", True)

# Validate Rayleigh Quotient Warm-Start Heuristic along actual 500-step trajectory
n_qubits = 7
num_starts = 100
n_dim = 2**n_qubits
k = n_qubits // 2
k_dim = 2**k

combos = list(combinations(range(n_qubits), k))
perms_list = []
for combo in combos:
    keep = list(combo)
    trace = [i for i in range(n_qubits) if i not in keep]
    perms_list.append(tuple(trace + keep))

# Symplectic random generator
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

PAULI_MAP_NP = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

def pauli_string_to_matrix_np(pauli_str):
    sign = -1 if pauli_str[0] == "-" else 1
    clean_str = pauli_str.lstrip("+-")
    matrices = [PAULI_MAP_NP[char] for char in clean_str]
    mat = matrices[0]
    for m in matrices[1:]:
        mat = np.kron(mat, m)
    return sign * mat

def build_projector_from_generators_np(generators):
    n_qubits = len(generators[0].lstrip("+-"))
    dim = 2**n_qubits
    projector = np.eye(dim, dtype=np.complex128)
    identity = np.eye(dim, dtype=np.complex128)
    for gen in generators:
        g_matrix = pauli_string_to_matrix_np(gen)
        p_g = (identity + g_matrix) / 2.0
        projector = projector @ p_g
    return projector

def haar_random_unitary_np(dim: int) -> np.ndarray:
    z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph

def rand_Almost_Stab_state_np(n_qubits: int, almost_gap: int = 1) -> np.ndarray:
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    psi = haar_random_unitary_np(dim) @ psi

    if n_qubits - almost_gap == 0:
        return psi

    proj = np.kron(
        build_projector_from_generators_np(
            generate_random_generators_symplectic(n_qubits - almost_gap)
        ),
        np.eye(2**almost_gap, dtype=np.complex128),
    )

    projected_psi = proj @ psi
    norm = np.linalg.norm(projected_psi)
    if norm < 1e-12:
        return psi
    return projected_psi / norm

@jax.jit
def get_purity_and_violation_exact(psi_vec):
    psi = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
    psi /= jnp.linalg.norm(psi)
    psi_tensor = psi.reshape((2,) * n_qubits)

    rhos = [
        psi_tensor.transpose(perm).reshape(-1, k_dim).conj().T
        @ psi_tensor.transpose(perm).reshape(-1, k_dim)
        for perm in perms_list
    ]
    batch_rho = jnp.stack(rhos, axis=0) # (35, k_dim, k_dim)

    purities = jnp.sum(jnp.abs(batch_rho)**2, axis=(-2, -1))
    avg_purity = jnp.mean(purities)
    max_purity = jnp.max(purities)

    ex, evecs = jnp.linalg.eigh(batch_rho)
    rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
    viols = jnp.maximum(0.0, ex[:, -1] - rhs)
    total_violation = jnp.sum(viols**2)

    return avg_purity, max_purity, total_violation, ex, evecs, batch_rho

@jax.jit
def objective_fn(params):
    psi = params[:n_dim] + 1j * params[n_dim:]
    psi /= jnp.linalg.norm(psi)
    psi_tensor = psi.reshape((2,) * n_qubits)

    rhos = [
        psi_tensor.transpose(perm).reshape(-1, k_dim).conj().T
        @ psi_tensor.transpose(perm).reshape(-1, k_dim)
        for perm in perms_list
    ]
    batch_rho = jnp.stack(rhos, axis=0)

    ex = jnp.linalg.eigvalsh(batch_rho)
    rhs = ex[:, 1] + 2 * jnp.sqrt(jnp.maximum(ex[:, 0] * ex[:, 2], 1e-15))
    viols = jnp.maximum(0.0, ex[:, -1] - rhs)
    return jnp.sum(viols**2)

def single_opt(p):
    solver = jaxopt.LBFGS(
        fun=objective_fn,
        maxiter=10,
        tol=1e-11,
        history_size=5
    )
    return solver.run(p).params

vmapped_run_step = jax.jit(vmap(single_opt))
vmapped_metrics_exact = jax.jit(vmap(get_purity_and_violation_exact))

print("=========================================================================")
print("VALIDATING WARM-STARTED EIGENVALUE HEURISTIC ALONG TRAJECTORY (100 SEEDS)")
print("=========================================================================")

init_states = [rand_Almost_Stab_state_np(n_qubits, 7) for _ in range(num_starts)]
params_batch = jnp.array([np.concatenate([np.real(s), np.imag(s)]) for s in init_states])

# Step 0: Compute exact initial eigenbases V_0
avg_p, max_p, viol, ex_prev, evecs_prev, rhos_prev = vmapped_metrics_exact(params_batch)
print(f"Step 0   | Mean Violation: {float(jnp.mean(viol)):.2e} | Mean Purity: {float(jnp.mean(avg_p)):.4f}")

for step in range(1, 201):
    params_batch = vmapped_run_step(params_batch)
    
    # Compute exact metrics at current step t
    avg_p, max_p, viol_exact, ex_exact, evecs_exact, rhos_cur = vmapped_metrics_exact(params_batch)
    
    # Compute Rayleigh Quotient Warm-Started eigenvalues: diag(V_prev^\dagger @ rho_cur @ V_prev)
    # rhos_cur shape: (100, 35, 8, 8), evecs_prev shape: (100, 35, 8, 8)
    rho_v = jnp.matmul(rhos_cur, evecs_prev)
    v_rho_v = jnp.matmul(jnp.conj(jnp.swapaxes(evecs_prev, -1, -2)), rho_v)
    ex_warm = jnp.real(jnp.diagonal(v_rho_v, axis1=-2, axis2=-1))
    
    # Update evecs_prev for next step
    evecs_prev = evecs_exact
    
    max_err = float(jnp.max(jnp.abs(ex_exact - ex_warm)))
    mean_err = float(jnp.mean(jnp.abs(ex_exact - ex_warm)))
    
    if step in [1, 5, 10, 20, 50, 100, 150, 200]:
        print(f"Step {step:3d} | Mean Viol: {float(jnp.mean(viol_exact)):.2e} | Max Eigenval Err: {max_err:.2e} | Mean Eigenval Err: {mean_err:.2e}")
