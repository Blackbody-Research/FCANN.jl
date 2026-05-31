"""
    module FCANN

Fast Convolutional Artificial Neural Network (FCANN) Julia package for training neural networks
with support for multiple cost functions, regularization techniques, and both CPU/GPU backends.

This package provides high-level training functions that automatically handle:
- Weight initialization with specified regularization
- ADAMAX optimization algorithm
- Gradient checking utilities
- Performance benchmarking tools

## Backends
- `:CPU` - Standard CPU-based training (always available)
- `:GPU` - CUDA-accelerated training (requires NVIDIA GPU and CUDA toolkit)

## Usage Example
```julia
using FCANN

# Set backend (CPU or GPU)
setBackend(:CPU)

# Define network architecture
M = 784              # Input dimension
hidden = [128, 64]   # Hidden layer sizes  
O = 10               # Output dimension

# Initialize parameters
params = initializeParams(M, hidden, O)

# Prepare training data
X, Y = preptraining(inputs, targets)

# Train the network
result = ADAMAXTrainNNCPU(X, Y, params; N=100)
```

## Exported Functions
See individual function documentation for detailed usage.
"""
module FCANN

using Statistics
using Random
using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Pkg
using NVIDIALibraries
using RandomMatrices

#set default backend to CPU
global backend = :CPU
global backendList = [:CPU]


include("MASTER_FCN_ABSERR_NFLOATOUT.jl")
include("column_selection_anneal.jl")

"""
    requestCostFunctions() -> Nothing

Display a list of all available cost functions that can be used for training.

The cost functions determine how the neural network measures error between
its predictions and the target values. Each cost function has different
properties regarding sensitivity to outliers and optimization behavior.

## Example
```julia
requestCostFunctions()
# Output:
# Available cost functions are: 
# absErr
# sqErr
# normLogErr
# ...
```
"""
function requestCostFunctions()
    @assert (length(costFuncList) == length(costFuncNames))
    @assert (length(costFuncList) == length(costFuncDerivsList))
    println("Available cost functions are: ")
    [println(n) for n in costFuncNames]
    println("------------------------------")
end

"""
    setBackend(b::Symbol) -> Symbol

Set the computation backend for training and evaluation.

## Arguments
- `b::Symbol` - The backend to use (:CPU or :GPU)

## Returns
The currently selected backend symbol.

## Notes
- GPU backend requires: NVIDIA GPU, CUDA toolkit installed, nvcc in system PATH
- If GPU is not available, the function will print an error message
- Default backend is :CPU

## Example
```julia
setBackend(:GPU)  # Attempt to use GPU (if available)
setBackend(:CPU)  # Use CPU backend
```
"""
function setBackend(b::Symbol)
    if in(b, backendList)
        global backend = b
    else
        println(string("Selected backend: ", b, " is not available."))
    end
    println(string("Backend is set to ", backend))
    return backend
end

"""
    getBackend() -> Symbol

Get the currently selected computation backend.

## Returns
The current backend symbol (:CPU or :GPU).
"""
function getBackend()
    println(string("Backend is set to ", backend))
    backend
end

#normal gradient check with cost functions with target outputs matching size of output layer (or 2x for log cost functions)
"""
    checkNumGrad(lambda::AbstractFloat = 0.0f0; kwargs...) -> Float64

Numerically verify gradients computed by the network using finite differences.

This function compares analytical gradients (computed via backpropagation) with
numerical gradients (computed via finite differences) to verify correct implementation.

## Arguments
- `lambda::AbstractFloat` - Regularization strength (default: 0.0)

## Keyword Arguments
- `M::Integer` - Input dimension (default: 10)
- `hidden::Vector{Integer}` - Hidden layer sizes (default: [5])
- `O::Integer` - Output dimension (default: 3)
- `batchSize::Integer` - Number of training examples (default: 100)
- `N::Integer` - Number of epochs for training (default: 10)
- `costFunc::String` - Cost function name (default: "absErr")
- `hidden_layers::Vector{Integer}` - Alternative specification of hidden layers
- `activation_list::Vector{Bool}` - Activation functions per layer
- `printmsg::Bool` - Print comparison results (default: true)
- `input_orientation::Char` - Data layout ('N' for normal, 'T' for transposed)

## Returns
The maximum absolute difference between numerical and analytical gradients.

## Example
```julia
# Basic gradient check with default parameters
err = checkNumGrad()

# Gradient check with custom architecture
err = checkNumGrad(0.1f0; hidden=[32, 16], costFunc="sqErr")

# Gradient check for specific cost function
err = checkNumGrad(1.0f0, "absErr"; printmsg=true)
```
"""
function checkNumGrad(lambda::AbstractFloat = 0.0f0; kwargs...)
    eval(Symbol("checkNumGrad", backend))(lambda; kwargs...)
end

#gradient check for output index cost function and typically only used for a single example rather than a batch
function checkNumGrad(output_index::Integer, lambda::AbstractFloat = 0.0f0; kwargs...)
    eval(Symbol("checkNumGrad", backend))(lambda, output_index; kwargs...)
end

#gradient check for specialized case of an output value vector and index vector where the loss function is only applied to the target output index
function checkNumGrad(lambda::AbstractFloat, err_name::String; kwargs...)
    eval(Symbol("checkNumGrad", backend))(lambda, err_name; kwargs...)
end

#specialized for checking gradient of cross entropy loss in a batch
function checkNumGrad(lambda::AbstractFloat, input_orientation::Char; kwargs...)
    eval(Symbol("checkNumGrad", backend))(lambda, input_orientation; kwargs...)
end

#gradient check for cross entropy loss with distribution targets
function checkNumGrad(lambda::AbstractFloat, ::Val{:dist}; kwargs...)
    eval(Symbol("checkNumGrad", backend))(lambda, Val(:dist); kwargs...)
end

"""
    benchmarkDevice(;kwargs...) -> Nothing

Benchmark training performance on the currently selected device (CPU or GPU).

This function measures throughput across various network sizes and batch sizes,
outputting results to a CSV file for analysis.

## Keyword Arguments
- `costFunc::String` - Cost function to benchmark (default: "absErr")
- `dropout::AbstractFloat` - Dropout rate (default: 0.0)
- `multi::Bool` - Use multiple workers for parallel training (default: false)
- `numThreads::Integer` - Number of BLAS threads (default: 0, auto-select)
- `minN::Integer` - Minimum number of neurons to benchmark (default: 32)
- `maxN::Integer` - Maximum number of neurons to benchmark (default: 2048)

## Output
Creates a CSV file with columns:
- Neurons, Batch Size, GFLOPS, Time Per Epoch

The filename includes device name and configuration details.

## Example
```julia
# Benchmark default configuration
benchmarkDevice()

# Benchmark GPU performance with dropout
setBackend(:GPU)
benchmarkDevice(costFunc="absErr", dropout=0.5f0)

# Benchmark CPU with multiple threads
setBackend(:CPU)
benchmarkDevice(numThreads=4, multi=true)
```
"""
function benchmarkDevice(;costFunc = "absErr", dropout = 0.0f0, multi=false, numThreads = 0, minN = 32, maxN = 2048)
    batchSizes = [512, 1024, 2048, 4096, 8192]
    Ns = filter(a -> (a >= minN) && (a <= maxN), [32, 64, 128, 256, 512, 1024, 2048])
    (_,_,_, cpuname, gpuname) = testTrain(1, [1], 1, 1024, 1, writeFile = false)
     deviceName = if gpuname == ""
        cpuname
    else
        string(cpuname, "_", gpuname)
    end

    taskStr = multi ? "over $(nprocs()) parallel tasks" : "with a single task"
    dropoutStr = dropout == 0.0 ? "" : " with a dropout rate of $dropout"

    println("Benchmarking device $deviceName with a cost function $costFunc $taskStr $dropoutStr")

    out = [begin
       (GFLOPS, parGFLOPS, t, cpuname, gpuname) = testTrain(N, [N, N], 2, B, 20; multi = multi, writeFile = false, numThreads = numThreads, costFunc = costFunc, dropout = dropout)
       # println(string("Done with neurons = ", N, " batch size = ", B))
       [N B GFLOPS parGFLOPS t] 
    end
    for N in Ns for B in batchSizes]

    header = ["Neurons" "Batch Size" "GFLOPS" "Time Per Epoch"]
    body = mapreduce(a -> a[1, [1 2 (multi ? 4 : 3) 5]], vcat, out)

    trainName = (dropout == 0) ? costFunc : string(dropout, "_dropout_", costFunc)
    threadStr = if (getBackend() == :CPU) 
        (numThreads == 0) ? "" : "_$(numThreads)BLASthreads"
    else
        ""
    end  

    multiStr = if (nprocs() > 1) && multi
        "$(nworkers())xParallel"
    else
        ""
    end

    writedlm(string(deviceName, "_", trainName, "$(threadStr)_$(multiStr)trainingBenchmark.csv"), [header; body], ',')
end

"""
    benchmarkCPUThreads(;kwargs...) -> Nothing

Benchmark CPU training performance across different numbers of BLAS threads.

This function measures how training throughput scales with the number of
parallel threads used for linear algebra operations.

## Keyword Arguments
- `costFunc::String` - Cost function to benchmark (default: "absErr")
- `dropout::AbstractFloat` - Dropout rate (default: 0.0)
- `Ns::Vector{Integer}` - Network sizes to benchmark (default: [16, 32, 64, 128, 256])

## Output
Creates a CSV file with columns:
- Neurons, Batch Size, GFLOPS for each thread count

The filename includes CPU name and configuration.

## Example
```julia
# Benchmark CPU threading performance
benchmarkCPUThreads()

# Benchmark specific network sizes
benchmarkCPUThreads(Ns=[32, 64, 128])
```
"""
function benchmarkCPUThreads(;costFunc = "absErr", dropout = 0.0f0, Ns = [16, 32, 64, 128, 256])
    setBackend(:CPU)
    batchSizes = 2 .^(5:12)
    (_,_,_, cpuname, gpuname) = testTrain(1, [1], 1, 1024, 1, writeFile = false)
     
    dropoutStr = dropout == 0.0 ? "" : " with a dropout rate of $dropout"

    println("Benchmarking device $cpuname with a cost function $costFunc $dropoutStr")

    threads = round.(Int64, 2 .^(0:log2(Sys.CPU_THREADS)))

    out = [begin
       GFLOPS = map(numThreads -> testTrain(N, [N, N], 2, B, 20; writeFile = false, numThreads = numThreads, costFunc = costFunc, dropout = dropout)[1], threads)
       # println(string("Done with neurons = ", N, " batch size = ", B))
       [N B GFLOPS'] 
    end
    for N in Ns for B in batchSizes]

    header = ["Neurons" "Batch Size" ["GFLOPS $a threads" for a in threads']]
    body = reduce(vcat, out)

    trainName = (dropout == 0) ? costFunc : string(dropout, "_dropout_", costFunc)

    writedlm(string(cpuname, "_", trainName, "_BLASthreadBenchmark.csv"), [header; body], ',')
end
         
export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, writeArray, initializeParams, checkNumGrad, predict, requestCostFunctions, setBackend, getBackend, benchmarkDevice, backendList, switch_device, devlist, current_device, benchmarkCPUThreads, readBinInput, calcfeatureimpact, ADAMAXTrainNNCPU, traintrials, preptraining, LossType, OutputIndex, CrossEntropyLoss

"""
    __init__() -> Nothing

Module initialization function that sets up CUDA environment when GPU backend
is available. This function is called automatically when the module is loaded.

This function:
1. Checks for CUDA toolkit availability
2. Initializes CUDA devices if present
3. Compiles and loads CUDA kernels
4. Sets up BLAS handles
5. Verifies GPU gradient computation correctness

If GPU initialization fails, the module falls back to CPU-only mode with an
informative error message.

## Notes
- This function is called automatically on module load
- Users typically don't need to call this directly
"""
function __init__()
    #get cuda toolkit versions if any
    println("Checking for cuda toolkit versions")
    cuda_versions = if check_cuda_presence()
        try
            get_cuda_toolkit_versions()
        catch e
            []
        end
    else
        []
    end

    if isempty(cuda_versions)
        println("No cuda toolkit appears to be installed.  If this sytem has an NVIDIA GPU, install the cuda toolkit and add nvcc to the system path to use the GPU backend.")
        println("Available backends are: CPU")
    elseif cuda_versions[end] > VersionNumber("10.1")
        println("The lastest cuda toolkit installed is $(cuda_versions[end]) which exceeds the latest supported version of 10.1")
        if length(cuda_versions) > 1
            println("Using the latest available version of the cuda toolkit installed by default.  To switch to an earlier cuda version, edit config file above with one of the installed versions: $(cuda_versions)")
        end 
    else
        # println("Using the following cuda settings: $(NVIDIALibraries.get_nvlib_settings()) saved to $(joinpath(pwd(), "nvlib_julia.conf")).")
        try 
            println("Checking nvcc compiler in system path")
            run(`nvcc --version`)
            
            #initialize cuda driver
            println("Attempting to initialize cuda device")
            cuInit(0)

            #get device list and set default device to 0
            println("Getting list of devices")
            deviceNum = cuDeviceGetCount()
            global devlist = [cuDeviceGet(a) for a in 0:deviceNum-1]
            println("Found the following cuda devices: $devlist and setting default device to 0")
            global current_device = devlist[1]

            println("finding device properties")
            global device_multiprocessor_count = Dict(begin
                prop_ref = Ref{cudaDeviceProp}()
                prop_ptr = convert(Ptr{cudaDeviceProp}, Base.pointer_from_objref(prop_ref))
                cudaGetDeviceProperties(prop_ptr, devlist[i])
                prop = unsafe_load(prop_ptr)
                devlist[i] => prop.multiProcessorCount
            end
            for i in eachindex(devlist))

            #set device for kernel and variable loading to default device
            cudaSetDevice(current_device)

            #create primary context handle on default device
            # global ctx = cuDevicePrimaryCtxRetain(current_device)

            println("Creating cublas handle")
            #create cublas handle to reference for calls on the default device
            global cublas_handle = cublasCreate_v2()

            global tmpdir = mktempdir()
            @info "Creating temporary directory for cuda kernel compilation at $tmpdir"

            println("Compiling ptx files of cuda kernels with nvcc")
            global (costpath, adamaxpath) = cu_module_compile(tmpdir)

            @assert (isfile(costpath) && isfile(adamaxpath)) "Compiled .ptx files not found at $costpath and $adamaxpath"

            println("Loading cuda modules from ptx files at $(costpath) and $(adamaxpath)")
            global costfunc_md = cu_module_load(costpath)
            global adamax_md = cu_module_load(adamaxpath)
            # eval(cu_module_load)

            #for cuda version 8 tensor ops are not available so default to regular GEMM algorithm
            global algo = try
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            catch
                CUBLAS_GEMM_DFALT
            end

            println("Creating cuda kernels from loaded modules")
            #create adamax and costfunction kernels in global scope
            fail = true
            tries = 0
            while fail && (tries < 5)
                tries += 1
                try
                    create_costfunc_kernels(costfunc_md)
                    create_adamax_kernels(adamax_md)
                    fail = false
                catch e
                    @info "Failed to load kernels on try $tries with error $e. Retrying..."
                end
            end
            
            println("Creating cost function dictionaries")
            #make error kernels available in global scope
            create_errorfunction_dicts(costfunc_md) 

            println("Verifying correct gradients")
            #verify that cost function works
            gpu_err = checkNumGradGPU(1.0f0; printmsg = false)
            @assert gpu_err < 0.01

            println("Available backends are: CPU, GPU")
            #add GPU to backendList after successful initialization
            push!(backendList, :GPU)
            unique!(backendList)
        catch msg
            println("Could not initialize cuda drivers and compile kernels due to $msg")
            println("Available backends are: CPU")
            try
                cublasDestroy_v2(cublas_handle)
            catch
            end
            global gpu_ready = false
        end
    end

    # else
    #     println("NVIDIALibraries is not currently installed so cuda functions will not be initialized")
    #     println("If you have an Nvidia GPU, install the Cuda Toolkit and the NVIDIALibraries package from https://github.com/Blackbody-Research/NVIDIALibraries.jl to have the GPU backend available")
    #     println("Available backends are: CPU")
    # end

    

    function f()
        if in(:GPU, backendList)
            println("Destroying GPU cublas handle")
            # cuDevicePrimaryCtxRelease(current_device)
            cublasDestroy_v2(cublas_handle)
        end
    end    
    atexit(f)
end

using PrecompileTools

@setup_workload begin
    M = 1
    hidden = [1]
    O = 1
    batchSize = 1024
    N = 150
    # __init__()
    @compile_workload begin
        checkNumGrad(1.0f0, printmsg = false)
        checkNumGrad(1.0f0, resLayers=1, printmsg = false)
        checkNumGrad(0.0f0, costFunc = "sqErr", printmsg = false)
        checkNumGrad(0.0f0, costFunc = "normLogErr", printmsg = false)
        checkNumGrad(0.0f0, costFunc = "cauchyLogErr", printmsg = false)
        checkNumGrad(1, m = 1, printmsg = false)
        checkNumGrad(1, m = 1, loss_type = CrossEntropyLoss(), printmsg = false)
        checkNumGrad(0.0f0, hidden_layers=[10, 10, 10], costFunc="sqErr", activation_list = [true, false, true], printmsg = false)
        checkNumGrad(0f0, "sqErr", printmsg = false)
        checkNumGrad(0f0, "sqErr"; input_orientation = 'T', printmsg = false)
        checkNumGrad(0f0, "absErr", printmsg = false)
        checkNumGrad(0f0, "absErr"; input_orientation = 'T', printmsg = false)
        checkNumGrad(0f0, 'N', printmsg = false)
        checkNumGrad(0f0, 'N'; use_values = true, printmsg = false)
        checkNumGrad(0f0, 'T', printmsg = false)
        checkNumGrad(0f0, 'T'; use_values = true, printmsg = false)
        checkNumGrad(0f0, Val(:dist), printmsg = false)
        checkNumGrad(0f0, Val(:dist); single_example = true, printmsg = false)
        checkNumGrad(1.0f0, Val(:dist), printmsg = false)
        testTrain(M, hidden, O, batchSize, N; writeFile = false, numThreads = 0, printProg = false, print_anything=false)
    end
end

end # module