import time
import numpy as np
import cupy as cp
import jax
import jax.numpy as jnp
from scipy.linalg import hadamard

# =====================================================================
# 1. Native Python / CuPy GPU SRE Implementation (FWHT Algorithm)
# =====================================================================
def create_hadamard_gpu(n_qubits: int) -> cp.ndarray:
    H = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64)
    H_n = H
    for _ in range(n_qubits - 1):
        H_n = cp.kron(H_n, H)
    return H_n

SRE_GPU_CACHE = {}

def get_gpu_sre_cache(n_qubits: int):
    if n_qubits not in SRE_GPU_CACHE:
        dim = 2**n_qubits
        H_n = create_hadamard_gpu(n_qubits)
        x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
        b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
        xor_indices = x_idx ^ b_idx
        SRE_GPU_CACHE[n_qubits] = (H_n, xor_indices)
    return SRE_GPU_CACHE[n_qubits]

def compute_sre_native_cupy_batch(psi_batch_np: np.ndarray, alpha: int = 2) -> np.ndarray:
    """
    Computes exact S2 (alpha=2) SRE for a batch of state vectors natively on GPU using CuPy.
    psi_batch_np: np.ndarray of shape (num_starts, 2^n_qubits), complex128
    """
    num_starts, dim = psi_batch_np.shape
    n_qubits = int(np.log2(dim))
    H_GPU, XOR_INDICES_GPU = get_gpu_sre_cache(n_qubits)
    
    # Copy batch to GPU
    psi_gpu = cp.array(psi_batch_np, dtype=cp.complex128)
    
    # Normalize states
    norms = cp.linalg.norm(psi_gpu, axis=1, keepdims=True)
    norms = cp.where(norms > 1e-12, norms, 1.0)
    psi_gpu = psi_gpu / norms
    
    # Build V_batch tensor of shape (num_starts, x_dim, b_dim)
    psi_conj = cp.conj(psi_gpu)[:, :, None]
    psi_xor = psi_gpu[:, XOR_INDICES_GPU]
    
    V_batch = cp.real(psi_conj * psi_xor)
    Xi_batch = cp.matmul(H_GPU, V_batch)
    xi_4_sum = cp.sum(Xi_batch ** 4, axis=(1, 2))
    sre_batch = -cp.log2(xi_4_sum / dim)
    
    return cp.asnumpy(sre_batch)

# =====================================================================
# 2. Julia HadaMAG CUDA Reference SRE Implementation
# =====================================================================
def get_julia_handle():
    from juliacall import Main as jl
    jl.seval("using Logging; disable_logging(Logging.Error)")
    jl.seval("using CUDA")
    jl.seval("using HadaMAG")
    jl.seval("using LinearAlgebra: norm")
    jl.seval("""
    function jl_compute_sre_batch(psi_batch_np, alpha, n_qubits, dim, num_starts)
        results = zeros(Float64, num_starts)
        for i in 1:num_starts
            psi_row = psi_batch_np[i, :]
            psi_jl = Vector{ComplexF64}(psi_row)
            nrm = norm(psi_jl)
            if nrm > 1e-12
                psi_jl ./= nrm
            end
            psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, Int(n_qubits), Int(dim))
            sre_result, lost_norm = SRE(psi_sv, alpha, backend= :cuda, progress=false, batch=256, threads=512)
            results[i] = sre_result
        end
        GC.gc()
        return results
    end
    """)
    return jl

if __name__ == "__main__":
    n_qubits = 7
    dim = 2**n_qubits
    num_starts = 50
    
    print("=================================================================")
    print(f" BENCHMARK: Native CuPy GPU SRE vs. Julia HadaMAG CUDA")
    print(f" System: {n_qubits} Qubits ({dim} dims) | Batch Size: {num_starts} States")
    print("=================================================================")
    
    # Generate 2,000 random complex state vectors
    np.random.seed(42)
    z = np.random.randn(num_starts, dim) + 1j * np.random.randn(num_starts, dim)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    psi_batch = z / norms
    
    # -------------------------------------------------------------
    # 1. Benchmark Native CuPy GPU Implementation
    # -------------------------------------------------------------
    print("\nWarming up Native CuPy GPU SRE implementation...")
    sre_native = compute_sre_native_cupy_batch(psi_batch[:10])  # Warmup
    
    print(f"Running Native CuPy GPU SRE for {num_starts} states...")
    t0 = time.time()
    sre_native = compute_sre_native_cupy_batch(psi_batch)
    t1 = time.time()
    native_time = t1 - t0
    print(f"--> Native CuPy GPU Time: {native_time:.4f} seconds ({native_time/num_starts*1000:.3f} ms / state)")
    print(f"--> Mean Native SRE: {np.mean(sre_native):.6f}")
    
    # -------------------------------------------------------------
    # 2. Benchmark Julia HadaMAG CUDA Implementation
    # -------------------------------------------------------------
    print("\nInitializing Julia HadaMAG CUDA backend...")
    jl = get_julia_handle()
    
    print("Warming up Julia HadaMAG...")
    _ = jl.jl_compute_sre_batch(psi_batch[:10], 2, n_qubits, dim, 10)
    
    print(f"Running Julia HadaMAG CUDA SRE for {num_starts} states...")
    t2 = time.time()
    sre_julia = np.array(jl.jl_compute_sre_batch(psi_batch, 2, n_qubits, dim, num_starts))
    t3 = time.time()
    julia_time = t3 - t2
    print(f"--> Julia HadaMAG CUDA Time: {julia_time:.4f} seconds ({julia_time/num_starts*1000:.3f} ms / state)")
    print(f"--> Mean Julia SRE: {np.mean(sre_julia):.6f}")
    
    # -------------------------------------------------------------
    # 3. Accuracy & Speedup Comparison
    # -------------------------------------------------------------
    max_diff = np.max(np.abs(sre_native - sre_julia))
    speedup = julia_time / native_time if native_time > 0 else 0
    
    print("\n=================================================================")
    print(f" RESULTS & ACCURACY SUMMARY:")
    print(f" Max Absolute Difference: {max_diff:.2e}")
    print(f" Native CuPy Time:        {native_time:.4f} seconds")
    print(f" Julia HadaMAG Time:      {julia_time:.4f} seconds")
    print(f" SPEEDUP:                {speedup:.1f}x FASTER!")
    print("=================================================================")
