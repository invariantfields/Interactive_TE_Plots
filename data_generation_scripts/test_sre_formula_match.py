import numpy as np
import cupy as cp
from juliacall import Main as jl

jl.seval("using Logging; disable_logging(Logging.Error)")
jl.seval("using CUDA; using HadaMAG")

jl.seval("""
function jl_sre_single(psi_np, alpha)
    psi_jl = Vector{ComplexF64}(psi_np)
    psi_sv = HadaMAG.StateVec{ComplexF64, 2}(psi_jl, 7, 128)
    res, _ = SRE(psi_sv, alpha, backend= :cuda)
    return res
end
""")

# Test state: Haar random 7-qubit state
np.random.seed(123)
dim = 128
n_qubits = 7
psi = np.random.randn(dim) + 1j * np.random.randn(dim)
psi /= np.linalg.norm(psi)

julia_sre = float(jl.jl_sre_single(psi, 2))

# Native CuPy FWHT
H = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.float64)
H7 = H
for _ in range(6):
    H7 = cp.kron(H7, H)

x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
xor_map = x_idx ^ b_idx

psi_gpu = cp.array(psi)
psi_conj = cp.conj(psi_gpu)[:, None]
psi_xor = psi_gpu[xor_map]
V = cp.real(psi_conj * psi_xor)
Xi = cp.matmul(H7, V)

sum_xi4 = float(cp.sum(Xi ** 4))

print("=== SRE FORMULA DIAGNOSTICS ===")
print(f"Julia HadaMAG SRE (alpha=2): {julia_sre:.10f}")
print(f"Raw Sum Xi^4:                {sum_xi4:.10f}")
print(f"-log2( sum_xi4 / 128 ):     {-np.log2(sum_xi4 / 128):.10f}")
print(f"-log( sum_xi4 / 128 ):      {-np.log(sum_xi4 / 128):.10f}")
print(f"-log2( sum_xi4 / 128^2 ):   {-np.log2(sum_xi4 / (128**2)):.10f}")
print(f"-log2( sum_xi4 ) + 7:        {-np.log2(sum_xi4) + 7:.10f}")
print(f"1/(1-2) * log2( sum_xi4 / 128 ): {1/(1-2) * np.log2(sum_xi4 / 128):.10f}")

# Check standard Leone et al. definition: M_2(psi) = -log2( 1/2^N * sum_sigma Tr(sigma rho)^4 )
# Notice: Tr(sigma rho) = <psi|sigma|psi>. In our V, Xi(a,b) = <psi| sigma_{a,b} |psi>.
# Is there a factor of 1/2^N or 1/(2^N)^2?
print(f"\nTesting variations:")
print(f"S2 = -log2(sum_xi4) + log2(128): {-np.log2(sum_xi4) + 7:.10f}")
print(f"S2 = -log2(sum_xi4 / 128**2):    {-np.log2(sum_xi4 / (128**2)):.10f}")
print(f"S2 = -log2(sum_xi4 / 128):       {-np.log2(sum_xi4 / 128):.10f}")
