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

# Opt out of Revise.jl's default package mode (`:eval`), which replays and retracts top-level
# assignment statements.  This module deliberately keeps a little mutable selection state
# (`BACKEND`, `gpu_ready`, `backendList`) that is only sensible when it is set up by `__init__`;
# with `:evalmeth` Revise tracks method definitions only and leaves that state alone.
# See the Revise documentation section "Configuring the revise mode".
const __revise_mode__ = :evalmeth

#----------------------------------------------------------------------------
# Backend selection
#
# The active backend is represented by a singleton type and selected through ordinary multiple
# dispatch (`currentbackend`).  This replaces the former
# `eval(Symbol("someFunction", backend))(...)` dispatch: a runtime `eval` is invisible to type
# inference and precompilation, and - contrary to intuition - it does NOT let a task observe
# methods installed after the task started running (world-age pinning), so it gives no benefit
# for incremental workflows such as Revise.jl while hiding every real dispatch site.
#----------------------------------------------------------------------------
abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct GPUBackend <: AbstractBackend end

const CPUBACKEND = CPUBackend()
const GPUBACKEND = GPUBackend()

# The active backend.  The *binding* is `const` and only the `Ref` contents change, which keeps
# the selection out of reach of tools that retract or replay top-level bindings.
const BACKEND = Ref{AbstractBackend}(CPUBACKEND)

# True once CUDA has been initialized successfully in `__init__`.  This is the single source of
# truth for GPU availability; `backendList` is derived from it by `refreshbackendlist!`.
const gpu_ready = Ref(false)

# Guards the `atexit` hook that releases the cuBLAS handle so that re-running `__init__` (which
# is safe and useful after redefining code in a live session) cannot register it twice.
const atexit_registered = Ref(false)

# The symbols accepted by `setBackend`.  `refreshbackendlist!` keeps this in sync with `gpu_ready`
# so it recovers on its own if a session tool re-evaluates the assignment that created it.
const backendList = Symbol[:CPU]

backendname(::CPUBackend) = :CPU
backendname(::GPUBackend) = :GPU

"""
    currentbackend() -> AbstractBackend

Return the currently active backend object (`CPUBACKEND` or `GPUBACKEND`).  Dispatch on this
type (`::CPUBackend` / `::GPUBackend`) instead of on the `Symbol` returned by `getBackend()`
whenever you need backend-specific behavior.
"""
currentbackend() = BACKEND[]

"""
    backendname() -> Symbol
    backendname(b::AbstractBackend) -> Symbol

Return the `Symbol` name (`:CPU` or `:GPU`) of a backend without printing anything.
"""
backendname() = backendname(currentbackend())

"""
    resolvebackend(b) -> AbstractBackend

Convert a `Symbol` (`:CPU` / `:GPU`) or an `AbstractBackend` into a backend object.
"""
resolvebackend(b::AbstractBackend) = b
resolvebackend(b::Symbol) = b === :GPU ? GPUBACKEND : CPUBACKEND

"""
    availableBackends() -> Vector{Symbol}

Return the backends that are usable in this session: always `:CPU`, plus `:GPU` once CUDA has
been initialized successfully.
"""
availableBackends() = gpu_ready[] ? Symbol[:CPU, :GPU] : Symbol[:CPU]

# Keep the exported `backendList` in agreement with the live GPU state.  `gpu_ready` is the
# source of truth, so a clobbered or stale list repairs itself the next time it is consulted.
function refreshbackendlist!()
    backends = availableBackends()
    empty!(backendList)
    append!(backendList, backends)
    return backendList
end


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
    refreshbackendlist!()
    if b === :CPU || (b === :GPU && gpu_ready[])
        BACKEND[] = resolvebackend(b)
    else
        println(string("Selected backend: ", b, " is not available."))
    end
    println(string("Backend is set to ", backendname()))
    return backendname()
end

"""
    getBackend() -> Symbol

Get the currently selected computation backend.

## Returns
The current backend symbol (:CPU or :GPU).
"""
function getBackend()
    println(string("Backend is set to ", backendname()))
    backendname()
end

#gradient checks for the currently selected backend (see setBackend).  The first positional
#argument selects which of the checkNumGrad methods is used; see the docstring for the full list.
"""
    checkNumGrad(args...; kwargs...) -> Float64

Numerically verify gradients computed by the network using finite differences.

`checkNumGrad` is polymorphic: it dispatches on the first positional argument to one of the
methods below, each of which runs on the currently selected backend (CPU or GPU, see
`setBackend`).  Every method computes both the analytical gradient (backpropagation) and the
numerical gradient (central finite differences) and returns the relative normed difference
between them.

## Methods

- `checkNumGrad(lambda::AbstractFloat = 0.0f0; kwargs...)` - elementwise cost functions whose
  target outputs match the output layer size (or 2x for log-likelihood cost functions such as
  `"normLogErr"` and `"cauchyLogErr"`).
- `checkNumGrad(output_index::Integer, lambda::AbstractFloat = 0.0f0; kwargs...)` - gradient at a
  single output index (`OutputIndex` loss), optionally with entropy-regularized cross
  entropy via `loss_type = CrossEntropyLoss(beta)`.
- `checkNumGrad(lambda::AbstractFloat, err_name::String; kwargs...)` - a cost function applied only
  at a per-example output index (the "index" variants, e.g. `"absErr"`, `"sqErr"`).
- `checkNumGrad(lambda::AbstractFloat, input_orientation::Char; kwargs...)` - cross entropy loss in
  a batch with per-example output indices, optionally with a per-example scalar value
  (`use_values = true`).
- `checkNumGrad(lambda::AbstractFloat, ::Val{:dist}; kwargs...)` - cross entropy loss with per-row
  target probability distributions.

`?checkNumGrad` displays each method's docstring in turn.

## Keyword Arguments

Common keyword arguments accepted by all methods (defaults may vary per method):

- `lambda::Real` - L2 regularization strength (default: 0.0)
- `m::Int` - Number of training examples (default: 1000, or 100 for batch index methods)
- `hidden_layers::Vector{Int}` - Hidden layer sizes (default: [5, 5])
- `resLayers::Int` - Residual connection period; 0 disables (default: 0)
- `input_layer_size::Int` - Number of input features (default: 3)
- `output_layer_size::Int` - Number of outputs (default: 2, or 5 for distribution targets)
- `e::Float32` - Finite-difference step size (default: 1f-3)
- `activation_list::Vector{Bool}` - Whether each hidden layer applies tanh activation
- `printmsg::Bool` - Print the per-parameter numerical vs analytical comparison (default: true)
- `input_orientation::Char` - Data layout: `'N'` = rows are examples, `'T'` = columns are examples

## Returns

The relative normed difference between the numerical and analytical gradients,
`norm(numGrad - funcGrad)/norm(numGrad + funcGrad)`.  Small values (typically `< 0.015`) indicate
the analytical gradient implementation is correct.

## Example
```julia
# Basic gradient check (elementwise absErr cost)
err = checkNumGrad()

# Squared error cost with L2 regularization
err = checkNumGrad(0.1f0; costFunc = "sqErr")

# Cross entropy output index with entropy regularization (beta = 0.1)
err = checkNumGrad(1, 0.0f0; loss_type = CrossEntropyLoss(0.1f0), m = 1)

# Cross entropy with distribution targets
err = checkNumGrad(0.0f0, Val(:dist); output_layer_size = 5)
```
"""
function checkNumGrad(lambda::AbstractFloat = 0.0f0; kwargs...)
    _checkNumGrad(currentbackend())(lambda; kwargs...)
end

"""
    checkNumGrad(output_index::Integer, lambda::AbstractFloat = 0.0f0; kwargs...) -> Float64

Gradient check at a single output index.  The loss is either `OutputIndex()` (the gradient is
just the selected output activation) or, when `loss_type = CrossEntropyLoss(beta)`, the cross
entropy loss at that index with optional entropy regularization `beta`.  Typically used for a
single example (`m = 1`); for a batch (`m > 1`) pass `output_vector = true` (per-example index
array) or `force_matrix = true`.

See `checkNumGrad` for the common keyword arguments.
"""
#gradient check for output index cost function and typically only used for a single example rather than a batch
function checkNumGrad(output_index::Integer, lambda::AbstractFloat = 0.0f0; kwargs...)
    _checkNumGrad(currentbackend())(lambda, output_index; kwargs...)
end

"""
    checkNumGrad(lambda::AbstractFloat, err_name::String; kwargs...) -> Float64

Gradient check for a cost function applied only at a per-example output index (the "index"
variants, e.g. `"absErr"`, `"sqErr"`).  `err_name` is the base cost function name; the loss is
evaluated only at the target output index of each example.  Log-likelihood cost functions are
not supported here.

See `checkNumGrad` for the common keyword arguments.
"""
#gradient check for specialized case of an output value vector and index vector where the loss function is only applied to the target output index
function checkNumGrad(lambda::AbstractFloat, err_name::String; kwargs...)
    _checkNumGrad(currentbackend())(lambda, err_name; kwargs...)
end

"""
    checkNumGrad(lambda::AbstractFloat, input_orientation::Char; kwargs...) -> Float64

Gradient check for cross entropy loss in a batch with a per-example output index.
`input_orientation` selects the data layout: `'N'` for `(m, input_layer_size)` rows-as-examples
or `'T'` for `(input_layer_size, m)` columns-as-examples.  With `use_values = true` a scalar
multiplier per example is applied to the loss.

See `checkNumGrad` for the common keyword arguments.
"""
#specialized for checking gradient of cross entropy loss in a batch
function checkNumGrad(lambda::AbstractFloat, input_orientation::Char; kwargs...)
    _checkNumGrad(currentbackend())(lambda, input_orientation; kwargs...)
end

"""
    checkNumGrad(lambda::AbstractFloat, ::Val{:dist}; kwargs...) -> Float64

Gradient check for cross entropy loss with per-row target probability distributions
(distribution targets).  `single_example = true` uses a single example with a one-row target
distribution.

See `checkNumGrad` for the common keyword arguments.
"""
#gradient check for cross entropy loss with distribution targets
function checkNumGrad(lambda::AbstractFloat, ::Val{:dist}; kwargs...)
    _checkNumGrad(currentbackend())(lambda, Val(:dist); kwargs...)
end

# Backend dispatch for the gradient checks.  Each helper returns the implementation for a backend,
# so a call site reads `_checkNumGrad(currentbackend())(args...; kwargs...)`.  This replaces the
# former `eval(Symbol("checkNumGrad", backend))(...)` pattern: no runtime `eval`, no world-age
# trap, and inference sees a two-element union of concrete functions.
_checkNumGrad(::CPUBackend) = checkNumGradCPU
_checkNumGrad(::GPUBackend) = checkNumGradGPU

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
         
export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, writeArray, initializeParams, checkNumGrad, predict, requestCostFunctions, setBackend, getBackend, benchmarkDevice, backendList, availableBackends, currentbackend, backendname, AbstractBackend, CPUBackend, GPUBackend, switch_device, devlist, current_device, benchmarkCPUThreads, readBinInput, calcfeatureimpact, ADAMAXTrainNNCPU, traintrials, preptraining, LossType, OutputIndex, CrossEntropyLoss

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
function __init__(force::Bool = false)
    if gpu_ready[] && !force
        #already initialized; re-running `__init__` by hand is therefore safe
        println("GPU backend is already initialized.  Available backends are: CPU, GPU")
        refreshbackendlist!()
        return
    end
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
            #record GPU availability after successful initialization; the exported `backendList`
            #is derived from `gpu_ready` by `refreshbackendlist!`
            gpu_ready[] = true
            refreshbackendlist!()
        catch msg
            println("Could not initialize cuda drivers and compile kernels due to $msg")
            println("Available backends are: CPU")
            try
                cublasDestroy_v2(cublas_handle)
            catch
            end
            gpu_ready[] = false
            refreshbackendlist!()
        end
    end

    # else
    #     println("NVIDIALibraries is not currently installed so cuda functions will not be initialized")
    #     println("If you have an Nvidia GPU, install the Cuda Toolkit and the NVIDIALibraries package from https://github.com/Blackbody-Research/NVIDIALibraries.jl to have the GPU backend available")
    #     println("Available backends are: CPU")
    # end

    

    #register the cleanup hook once; re-running `__init__` must not stack handlers
    if !atexit_registered[]
        atexit_registered[] = true
        function f()
            if gpu_ready[]
                println("Destroying GPU cublas handle")
                # cuDevicePrimaryCtxRelease(current_device)
                cublasDestroy_v2(cublas_handle)
            end
        end
        atexit(f)
    end
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