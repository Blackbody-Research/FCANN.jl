module FCANN
include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

function requestCostFunctions()
    assert(length(costFuncList) == length(costFuncNames))
    assert(length(costFuncList) == length(costFuncDerivsList))
    println("Available cost functions are: ")
    [println(n) for n in costFuncNames]
    println("------------------------------")
end

export archEval, archEvalSample, evalLayers, tuneAlpha, autoTuneParams, autoTuneR, smartTuneR, tuneR, L2Reg, maxNormReg, dropoutReg, advReg, fullTrain, bootstrapTrain, multiTrain, evalMulti, bootstrapTrainAdv, evalBootstrap, testTrain, smartEvalLayers, multiTrainAutoReg, writeParams, readBinParams, initializeParams, checkNumGrad, predict, requestCostFunctions

end # module
