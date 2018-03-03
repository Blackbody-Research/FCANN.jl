module FCANN

function availableBackends()
    if Pkg.installed("CUBLAS") == nothing
        println("Available backends are: CPU")
        [:CPU]
    else
        println("Available backends are: CPU, GPU")
        [:CPU, :GPU]
    end
end

#set default backend to CPU
global backend = :CPU
global backendList = availableBackends()

include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

function requestCostFunctions()
    assert(length(costFuncList) == length(costFuncNames))
    assert(length(costFuncList) == length(costFuncDerivsList))
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

function checkNumGrad(lambda = 0.0f0; hidden_layers = [5, 5], costFunc = "absErr") 
    eval(Symbol("checkNumGrad", backend))(lambda; hidden_layers = hidden_layers, costFunc = costFunc)
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

    if dropout == 0.0
        if multi
            println(string("Benchmarking device ", deviceName, " with cost function ", costFunc, " over ", nprocs(), " parallel tasks"))
        else
            println(string("Benchmarking device ", deviceName, " with cost function ", costFunc, " with a single training task"))
        end
    else
        if multi
            println(string("Benchmarking device ", deviceName, " with cost function ", costFunc, " over ", nprocs(), " parallel tasks with a dropout rate of ", dropout))
        else
            println(string("Benchmarking device ", deviceName, " with cost function ", costFunc, " with a single training task with a dropout rate of ", dropout))
        end
    end

    out = [begin
       (GFLOPS, parGFLOPS, t, cpuname, gpuname) = testTrain(N, [N, N], 2, B, 20; multi = multi, writeFile = false, numThreads = numThreads, costFunc = costFunc, dropout = dropout)
       # println(string("Done with neurons = ", N, " batch size = ", B))
       [N B GFLOPS parGFLOPS] 
    end
    for N in Ns for B in batchSizes]

    header = ["Neurons" "Batch Size" "GFLOPS"]
    body = if multi
        mapreduce(a -> a[1, [1 2 4]], vcat, out)
    else
        mapreduce(a -> a[1, [1 2 3]], vcat, out)
    end

   

    trainName = if dropout == 0
        costFunc
    else
        string(dropout, "_dropout_", costFunc)
    end

    writecsv(string(deviceName, "_", trainName, "_trainingBenchmark.csv"), [header; body])
end
        
export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, 
maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, 
testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, writeArray, initializeParams, checkNumGrad, predict, requestCostFunctions,
availableBackends, setBackend, getBackend, benchmarkDevice

function __init__()
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
            println("Destroying GPU context")
            destroy!(ctx)
        end
    end    
    atexit(f)
end

end # module
