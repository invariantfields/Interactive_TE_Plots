import numpy as np
import cupy as cp
from juliacall import Main as jl

jl.seval("using Logging; disable_logging(Logging.Error)")
jl.seval("using HadaMAG; using LinearAlgebra: norm")

jl.seval("""
function jl_compute_sre_exact_serial(psi_batch_np, alpha, n_qubits, dim, num_starts)
    results = zeros(Float64, num_starts)
    for i in 1:num_starts
        psi_row = psi_batch_np[i, :]
        psi_jl = Vector{ComplexF64}(psi_row)
        nrm = norm(psi_jl)
        if nrm > 1e-12
            psi_jl ./= nrm
        end
        psi_sv = HadaMAG.StateVec(psi_jl; q=2)
        # Using exact serial backend (not Monte Carlo CUDA sampling)
        sre_result, _ = SRE(psi_sv, alpha; backend=:serial, progress=false)
        results[i] = sre_result
    end
    return results
end
""")

# Native CuPy GPU Implementation
def create_hadamard_gpu(n_qubits: int) -> cp.ndarray:
    H = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64) / cp.sqrt(2.0)
    H_n = H
    for _ in range(n_qubits - 1):
        H_n = cp.kron(H_n, H)
    return H_n

def compute_sre_native_cupy_batch(psi_batch_np: np.ndarray) -> np.ndarray:
    num_starts, dim = psi_batch_np.shape
    n_qubits = int(np.log2(dim))
    H_gpu = create_hadamard_gpu(n_qubits)
    x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
    b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
    xor_map = x_idx ^ b_idx
    
    psi_gpu = cp.array(psi_batch_np, dtype=cp.complex128)
    norms = cp.linalg.norm(psi_gpu, axis=1, keepdims=True)
    norms = cp.where(norms > 1e-12, norms, 1.0)
    psi_gpu = psi_gpu / norms
    
    psi_conj = cp.conj(psi_gpu)[:, :, None]
    psi_xor = psi_gpu[:, xor_map]
    V_batch = cp.real(psi_conj * psi_xor)
    
    Xi_batch = cp.matmul(H_gpu, V_batch)
    xi_4_sum = cp.sum(Xi_batch ** 4, axis=(1, 2))
    
    # S2 = -log2( (1 / 2^N) * sum(Xi^4) )
    sre_batch = -cp.log2(xi_4_sum / dim)
    return cp.asnumpy(sre_batch)

if __name__ == "__main__":
    n_qubits = 7
    dim = 128
    num_states = 10

    np.random.seed(42)
    z = np.random.randn(num_states, dim) + 1j * np.random.randn(num_states, dim)
    psi_batch = z / np.linalg.norm(z, axis=1, keepdims=True)

    print("Computing Julia HadaMAG Exact Serial SRE...")
    sre_julia_exact = np.array(jl.jl_compute_sre_exact_serial(psi_batch, 2, n_qubits, dim, num_states))

    print("Computing Native CuPy GPU SRE...")
    sre_native = compute_sre_native_cupy_batch(psi_batch)

    print("\n=================================================================")
    print(" EXACT BACKEND COMPARISON (10 STATES):")
    print("=================================================================")
    print(f"{'State #':<8} | {'Native CuPy SRE':<20} | {'Julia Exact Serial SRE':<25} | {'Abs Diff':<15}")
    print("-" * 75)
    for i in range(num_states):
        diff = abs(sre_native[i] - sre_julia_exact[i])
        print(f"{i+1:<8} | {sre_native[i]:<20.12f} | {sre_julia_exact[i]:<25.12f} | {diff:<15.2e}")
