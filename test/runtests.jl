using FCANN
using Test

include("initTests.jl")

include("CPU_singleCore_tests.jl")

include("GPU_singleCore_tests.jl")

include("CPU_multiCore_tests.jl")