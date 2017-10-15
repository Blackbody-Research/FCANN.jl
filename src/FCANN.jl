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
    func = eval(Symbol("checkNumGrad", backend))
    if backend == :CPU
        func(lambda; hidden_layers = hidden_layers, costFunc = costFunc)
    else
        func(lambda; hidden_layers = hidden_layers)
    end
end

export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, 
maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, 
testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, initializeParams, checkNumGrad, predict, requestCostFunctions,
availableBackends, setBackend, getBackend

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
