using FCANN
using Test
using Random
using DelimitedFiles
using Distributed

include("initTests.jl")

# all numerical gradient checking lives here (both CPU and GPU backends)
include("test_gradients_comprehensive.jl")

include("CPU_singleCore_tests.jl")

include("GPU_singleCore_tests.jl")

include("CPU_multiCore_tests.jl")
