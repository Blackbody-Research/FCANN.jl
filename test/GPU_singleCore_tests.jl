backend = setBackend(:GPU)
if backend != :GPU
    println("Skipping GPU tests")
else

    println("-----------------------Testing GPU Single Core-------------------------------------")

    println("Testing numerical gradient vs backpropagation")
    println("---------------------------------------------")
    println("Lambda = 0, no hidden layers")
    err = checkNumGrad(0.0f0, hidden_layers=Vector{Int64}())
    @test (err < 0.015)

    println("Lambda = 0")
    err = checkNumGrad(0.0f0)
    @test (err < 0.015)

    println("Lambda = 0.1")
    err = checkNumGrad(0.1f0)
    @test(err < 0.015)

    println("Lambda = 1.0")
    err = checkNumGrad(1.0f0)
    @test(err < 0.015)
    println("TEST PASSED")
    println()

    println("Lambda = 1.0, resLayers = 1")
    err = checkNumGrad(1.0f0, resLayers=1)
    @test(err < 0.015)
    println("TEST PASSED")
    println()

    println("Squared Error Cost Function")
    err = checkNumGrad(costFunc = "sqErr")
    @test(err < 0.015)
    println("TEST PASSED")
    println()

    println("Norm Log Likelihood Cost Function")
    err = checkNumGrad(costFunc = "normLogErr")
    @test(err < 0.015)
    println("TEST PASSED")
    println()

    println("Cauchy Log Likelihood Cost Function")
    err = checkNumGrad(costFunc = "cauchyLogErr")
    @test(err < 0.015)
    println("TEST PASSED")
    println()


    #basic test train with 1 input, 1 output, 1 neuron
    println("Testing simple ANN training version 1")
    println("-------------------------------------")
    M = 100
    hidden = [100, 100]
    O = 2
    batchSize = 1024
    N = 150
    testTrain(M, hidden, O, batchSize, N; writeFile = false, numThreads = 0, printProg = true)
    println("TEST PASSED")
    println()

    name = "test"

    println("Training with 0 hidden layers")
    record, T, B = fullTrain(name, 150, 1024, Vector{Int64}(), 0.0f0, Inf, 0.001f0, 0.1f0, 1,writeFiles=false)
    @test(record[end] < record[1])
    println("TEST PASSED")
    println()

    hidden = [10, 10]
    println("Training with ", hidden, " hidden layers")
    record, T, B = fullTrain(name, 10, 1024, hidden, 0.0f0, Inf, 0.0001f0, 0.1f0, 1)
    @test(record[end] < record[1])
    println("TEST PASSED")
    println()

    println("Training with ", hidden, " hidden layers from previous endpoint")
    record, T, B, a, b, bestCost = fullTrain(name, 150, 1024, hidden, 0.0f0, Inf, 0.0001f0, 0.1f0, 2, startID = 1,writeFiles=false)
    @test(bestCost < record[1])
    println("TEST PASSED")
    println()

    M = 10

    filename = string(name, "_10_input_10X2_hidden_2_output_0.0001_alpha_0.1_decay_ADAMAX_absErr")
    for str in ("costRecord", "timeRecord", "performance", "predScatTrain", "predScatTest")
        rm("1_$(str)_$filename.csv")
    end
    rm(string("1_params_", filename, ".bin"))

    println("Testing multiTrain")
    multiTrain(name, Xtrain, ytrain, Xtest, ytest, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 2, 1, sampleCols=[1, 2], dropout = 0.1f0, writefiles=false, reslayers=1)
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 2, 1)
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 2, 2)
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 2, 1, costFunc = "normLogErr")
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 2, 2, costFunc = "normLogErr")
    println("TEST PASSED")
    println()

    println("Testing evalMulti")
    evalMulti(name, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, IDList = [1, 2])
    evalMulti(name, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, IDList = [1, 2], costFunc = "normLogErr")
    println("TEST PASSED")
    println()

    filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.002_alpha_0.1_decay_1.0_maxNorm_ADAMAX_absErr")

    rm(string("1_multiParams_", filename, ".bin"))
    rm(string("1_multiPerformance_", filename, ".csv"))
    rm(string("1_multiPredScatTrain_", filename, ".csv"))
    rm(string("1_multiPredScatTest_", filename, ".csv"))
    rm(string("2_multiParams_", filename, ".bin"))
    rm(string("2_multiPerformance_", filename, ".csv"))
    rm(string("2_multiPredScatTrain_", filename, ".csv"))
    rm(string("2_multiPredScatTest_", filename, ".csv"))
    rm(string("fullMultiParams_", filename, ".bin"))
    rm(string("fullMultiPerformance_", filename, ".csv"))
    rm(string("multiPredScatTrain_", filename, ".csv"))
    rm(string("multiPredScatTest_", filename, ".csv"))

    filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.002_alpha_0.1_decay_1.0_maxNorm_ADAMAX_normLogErr")

    rm(string("1_multiPerformance_", filename, ".csv"))
    rm(string("1_multiPredScatTrain_", filename, ".csv"))
    rm(string("1_multiPredScatTest_", filename, ".csv"))
    rm(string("1_multiParams_", filename, ".bin"))
    rm(string("2_multiPerformance_", filename, ".csv"))
    rm(string("2_multiPredScatTrain_", filename, ".csv"))
    rm(string("2_multiPredScatTest_", filename, ".csv"))
    rm(string("2_multiParams_", filename, ".bin"))
    rm(string("fullMultiPerformance_", filename, ".csv"))
    rm(string("multiPredScatTrain_", filename, ".csv"))
    rm(string("multiPredScatTest_", filename, ".csv"))
    rm(string("fullMultiParams_", filename, ".bin"))  
end