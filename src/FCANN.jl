module FCANN
include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

function requestCostFunctions()
    assert(length(costFuncList) == length(costFuncNames))
    assert(length(costFuncList) == length(costFuncDerivsList))
    println("Available cost functions are: ")
    [println(n) for n in costFuncNames]
    println("------------------------------")
end

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

if in(:GPU, backendList)
    using CUBLAS, CUDAdrv
end

function setBackend(b::Symbol)
    list = availableBackends()
    if in(b, list)
        global backend = b
    else
        println(string("Selected backend: ", b, " is not available."))
    end
    println(string("Backend is set to ", backend))
end

export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, 
maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, 
testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, initializeParams, checkNumGrad, predict, requestCostFunctions,
availableBackends, setBackend

end # module
