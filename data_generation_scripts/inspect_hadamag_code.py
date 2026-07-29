import numpy as np
from juliacall import Main as jl

jl.seval("using InteractiveUtils; using HadaMAG")
jl.seval("""
println("=== HadaMAG SRE Source Code Location ===")
m = @which SRE(HadaMAG.StateVec(ones(ComplexF64, 128), 7, 128), 2)
println(m)
""")
