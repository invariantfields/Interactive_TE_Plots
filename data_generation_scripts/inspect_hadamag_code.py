import numpy as np
from juliacall import Main as jl

jl.seval("using HadaMAG")
jl.seval("""
println("=== HadaMAG SRE Source Code Inspection ===")
try
    # Inspect SRE definition in HadaMAG
    display(@which SRE(HadaMAG.StateVec(ones(ComplexF64, 128), 7, 128), 2))
catch e
    println(e)
end
""")
