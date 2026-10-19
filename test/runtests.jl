using FCANN
using Test
using Random
using DelimitedFiles
using Distributed

include("initTests.jl")

# all numerical gradient checking lives here (both CPU and GPU backends)
include("test_gradients_comprehensive.jl")

# allocation / type-stability checks for the low-level forward and backward kernels
include("test_allocation_stability.jl")

include("CPU_singleCore_tests.jl")

include("GPU_singleCore_tests.jl")

include("CPU_multiCore_tests.jl")
