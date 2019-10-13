module FCANN

using Statistics
using Random
using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Pkg

function availableBackends()
    try
        run(`nvcc --version`)
        installList = Pkg.installed()
        if haskey(installList, "NVIDIALibraries")
            println("Available backends are: CPU, GPU")
            return [:CPU, :GPU]
        else
            println("GPU backend not currently available.  Install NVIDIALibraries package to have access to it with instructions here: https://github.com/Blackbody-Research/NVIDIALibraries.jl")
            println("Available backends are: CPU")
            [:CPU]
        end
    catch msg
        println(msg)
        println("Available backends are: CPU.  Install the Cuda Toolkit and NVIDIALibraries package to have access to the GPU backend")
        [:CPU]
    end
end

#set default backend to CPU
global backend = :CPU
global backendList = availableBackends()


include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

function requestCostFunctions()
    @assert (length(costFuncList) == length(costFuncNames))
    @assert (length(costFuncList) == length(costFuncDerivsList))
    println("Available cost functions are: ")
    [println(n) for n in costFuncNames]
    println("------------------------------")
end

function setBackend(b::Symbol)
    list = availableBackends()
    if in(b, list)
        global backend = b
    else
        println(string("Selected backend: ", b, " is not available."))
    end
    println(string("Backend is set to ", backend))
    return backend
end

function getBackend()
    println(string("Backend is set to ", backend))
    backend
end

function checkNumGrad(lambda = 0.0f0; hidden_layers=[5, 5], costFunc = "absErr", input_layer_size = 3, n = 2, m = 100) 
    eval(Symbol("checkNumGrad", backend))(lambda; hidden_layers = hidden_layers, costFunc = costFunc, input_layer_size = input_layer_size, n = n, m = m)
end

function benchmarkDevice(;costFunc = "absErr", dropout = 0.0f0, multi=false, numThreads = 0)
    batchSizes = [512, 1024, 2048, 4096, 8192]
    Ns = [32, 64, 128, 256, 512, 1024, 2048]
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
       [N B GFLOPS parGFLOPS] 
    end
    for N in Ns for B in batchSizes]

    header = ["Neurons" "Batch Size" "GFLOPS"]
    body = mapreduce(a -> a[1, [1 2 (multi ? 4 : 3)]], vcat, out)

    trainName = (dropout == 0) ? costFunc : string(dropout, "_dropout_", costFunc)

    writedlm(string(deviceName, "_", trainName, "_trainingBenchmark.csv"), [header; body], ',')
end
        
export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, writeArray, initializeParams, checkNumGrad, predict, requestCostFunctions,
availableBackends, setBackend, getBackend, benchmarkDevice, backendList

if in(:GPU, backendList)
    export switch_device, devlist, current_device
end

function __init__()
    
    if in(:GPU, backendList)
        #initialize cuda driver api
        cuInit(0)

        #get device list and set default device to 0
        deviceNum = cuDeviceGetCount()
        global devlist = [cuDeviceGet(a) for a in 0:deviceNum-1]
        global current_device = devlist[1]

        #set device for kernel and variable loading to default device
        cudaSetDevice(current_device)

        #create primary context handle on default device
        # global ctx = cuDevicePrimaryCtxRetain(current_device)

        #create cublas handle to reference for calls on the default device
        global cublas_handle = cublasCreate_v2()

        (adamax_md, costfunc_md) = cu_module_load()
        # eval(cu_module_load)

        #create adamax and costfunction kernels in global scope
        create_kernels(adamax_md, adamax_kernel_names)
        create_kernels(costfunc_md, costfunc_kernel_names)
        
        #make error kernels available in global scope
        create_errorfunction_dicts(costfunc_md) 
    end

    function f()
        if in(:GPU, backendList)
            if isfile("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
                println("Removing cuda cost function .ptx files")
                rm("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
            end
            if isfile("ADAMAX_INTRINSIC_KERNELS.ptx")
                println("Removing cuda adamax .ptx files")
                rm("ADAMAX_INTRINSIC_KERNELS.ptx")
            end
            println("Destroying GPU cublas handle")
            # cuDevicePrimaryCtxRelease(current_device)
            cublasDestroy_v2(cublas_handle)
        end
    end    
    atexit(f)
end

end # module
