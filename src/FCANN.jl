module FCANN

using Statistics
using Random
using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Pkg
using NVIDIALibraries

#set default backend to CPU
global backend = :CPU
global backendList = [:CPU]

#get cuda toolkit versions if any
println("Checking for cuda toolkit versions")
cuda_versions = try
    filter(v -> v <= VersionNumber(10), if Sys.islinux()
        map(VersionNumber,
            map((function(name::String)
                return name[6:end]
            end), 
                collect(i for i in readdir("/usr/local/")
                    if occursin("cuda-", i))))
    else
        get_cuda_toolkit_versions()
    end)
catch
    []
end

include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

function requestCostFunctions()
    @assert (length(costFuncList) == length(costFuncNames))
    @assert (length(costFuncList) == length(costFuncDerivsList))
    println("Available cost functions are: ")
    [println(n) for n in costFuncNames]
    println("------------------------------")
end

function setBackend(b::Symbol)
    if in(b, backendList)
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
        
export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, writeArray, initializeParams, checkNumGrad, predict, requestCostFunctions, setBackend, getBackend, benchmarkDevice, backendList, switch_device, devlist, current_device, benchmarkCPUThreads

function __init__()
    if isempty(cuda_versions)
        println("No cuda toolkit appears to be installed.  If this sytem has an NVIDIA GPU, install the cuda toolkit and add nvcc to the system path to use the GPU backend.")
        println("Available backends are: CPU")
    else
        println("CUDA version $(cuda_versions[end]) installed, attempting to load GPU backend functions")
        println("Using the following cuda settings: $(NVIDIALibraries.get_nvlib_settings()) saved to $(joinpath(pwd(), "nvlib_julia.conf")).")
        if length(cuda_versions) > 1
            println("Using the latest available version of the cuda toolkit installed by default.  To switch to an earlier cuda version, edit config file above with one of the installed versions: $(cuda_versions)")
        end 
        
        try 
            

            println("Checking nvcc compiler in system path")
            run(`nvcc --version`)
            
            #initialize cuda driver
            println("Attempting to initialize cuda device")
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

            #verify that cost function works
            checkNumGradGPU()

            println("Available backends are: CPU, GPU")
            #add GPU to backendList after successful initialization
            push!(backendList, :GPU)
        catch msg
            println("Could not initialize cuda drivers and compile kernels due to $msg")
            println("Available backends are: CPU")
        end
    end

    # else
    #     println("NVIDIALibraries is not currently installed so cuda functions will not be initialized")
    #     println("If you have an Nvidia GPU, install the Cuda Toolkit and the NVIDIALibraries package from https://github.com/Blackbody-Research/NVIDIALibraries.jl to have the GPU backend available")
    #     println("Available backends are: CPU")
    # end

    function f()
        if in(:GPU, backendList)
            if myid() == 1
                if isfile("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
                    println("Removing cuda cost function .ptx files")
                    rm("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
                end
                if isfile("ADAMAX_INTRINSIC_KERNELS.ptx")
                    println("Removing cuda adamax .ptx files")
                    rm("ADAMAX_INTRINSIC_KERNELS.ptx")
                end
            end
            println("Destroying GPU cublas handle")
            # cuDevicePrimaryCtxRelease(current_device)
            cublasDestroy_v2(cublas_handle)
        end
    end    
    atexit(f)
end

end # module
