import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import hadamard

jax.config.update("jax_enable_x64", True)

# Pre-computed unnormalized Hadamard matrix for n=7 (128x128)
def create_hadamard_jax(n_qubits: int):
    dim = 2**n_qubits
    H_np = hadamard(dim).astype(np.float64)
    return jnp.array(H_np)

# Pre-computed Pauli Y phases
def get_pauli_y_phases_jax(n_qubits: int):
    dim = 2**n_qubits
    z_grid = np.arange(dim, dtype=np.int32)[:, None]
    x_grid = np.arange(dim, dtype=np.int32)[None, :]
    zx_and = z_grid & x_grid
    num_y_np = np.vectorize(lambda v: bin(v).count('1'))(zx_and)
    phases_np = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)[num_y_np % 4]
    return jnp.array(phases_np)

def get_xor_indices_jax(n_qubits: int):
    dim = 2**n_qubits
    x_idx = np.arange(dim, dtype=np.int32)[:, None]
    b_idx = np.arange(dim, dtype=np.int32)[None, :]
    return jnp.array(x_idx ^ b_idx)

# Pure JAX FWHT SRE Function (Zero CuPy / Zero NVRTC)
@jax.jit
def compute_sre_pure_jax_batch(psi_batch, H_jax, phases_jax, xor_indices_jax):
    norms = jnp.linalg.norm(psi_batch, axis=1, keepdims=True)
    norms = jnp.where(norms > 1e-12, norms, 1.0)
    psi_batch = psi_batch / norms
    
    psi_conj = jnp.conj(psi_batch)[:, :, None]
    psi_xor = psi_batch[:, xor_indices_jax]
    V_complex = psi_conj * psi_xor
    
    Xi_complex = jnp.matmul(H_jax, V_complex)
    expval_pauli = jnp.real(phases_jax[None, :, :] * Xi_complex)
    
    pauli_4_sum = jnp.sum(expval_pauli ** 4, axis=(1, 2))
    dim = H_jax.shape[0]
    sre_batch = -jnp.log2(pauli_4_sum / dim)
    return sre_batch

if __name__ == "__main__":
    n_qubits = 7
    dim = 128
    num_starts = 500

    print("=================================================================")
    print(f" PURE JAX GPU FWHT SRE BENCHMARK")
    print(f" System: {n_qubits} Qubits ({dim} dims) | Batch Size: {num_starts} States")
    print("=================================================================")

    np.random.seed(42)
    z = np.random.randn(num_starts, dim) + 1j * np.random.randn(num_starts, dim)
    psi_batch = jnp.array(z / np.linalg.norm(z, axis=1, keepdims=True))

    H_jax = create_hadamard_jax(n_qubits)
    phases_jax = get_pauli_y_phases_jax(n_qubits)
    xor_indices_jax = get_xor_indices_jax(n_qubits)

    # Warmup JIT
    _ = compute_sre_pure_jax_batch(psi_batch[:10], H_jax, phases_jax, xor_indices_jax).block_until_ready()

    # Benchmark 500 states
    t0 = time.time()
    sre_jax = compute_sre_pure_jax_batch(psi_batch, H_jax, phases_jax, xor_indices_jax).block_until_ready()
    t1 = time.time()

    elapsed = t1 - t0
    print(f"--> Pure JAX GPU Time ({num_starts} states): {elapsed:.4f} seconds ({elapsed/num_starts*1000:.3f} ms / state)")
    print(f"--> Mean SRE: {jnp.mean(sre_jax):.6f}")
    print("=================================================================")
