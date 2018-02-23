backend = setBackend(:GPU)
if backend != :GPU
    println("Skipping GPU tests")
else

    println("-----------------------Testing GPU Single Core-------------------------------------")

    println("Testing numerical gradient vs backpropagation")
    println("---------------------------------------------")
    println("Lambda = 0, no hidden layers")
    err = checkNumGrad(0.0f0, hidden_layers=[])
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

    println("Squared Error Cost Function")
    err = checkNumGrad(costFunc = "sqErr")
    @test(err < 0.015)
    println("TEST PASSED")
    println()

    println("Norm Log Likelihood Cost Function")
    err = checkNumGrad(costFunc = "normLog")
    @test(err < 0.015)
    println("TEST PASSED")
    println()

    println("Cauchy Log Likelihood Cost Function")
    err = checkNumGrad(costFunc = "cauchyLog")
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

    println("Training with 0 hidden layers")
    record, T, B = fullTrain(name, 150, 1024, [], 0.0f0, Inf, 0.001f0, 0.1f0, 1,writeFiles=false)
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

    # filename = string(name, "_10_input_", [], "_hidden_2_output_0.0_L2_Inf_maxNorm_0.001_alpha_ADAMAX", backend, "_absErr")
    # rm(string("1_costRecord_", filename, ".csv"))
    # rm(string("1_timeRecord_", filename, ".csv"))
    # rm(string("1_performance_", filename, ".csv"))
    # rm(string("1_params_", filename, ".bin"))

    filename = string(name, "_10_input_", hidden, "_hidden_2_output_0.0_L2_Inf_maxNorm_0.0001_alpha_ADAMAX", backend, "_absErr")
    rm(string("1_costRecord_", filename, ".csv"))
    rm(string("1_timeRecord_", filename, ".csv"))
    rm(string("1_performance_", filename, ".csv"))
    rm(string("1_predictionScatterTrain_", filename, ".csv"))
    rm(string("1_predictionScatterTest_", filename, ".csv"))
    rm(string("1_params_", filename, ".bin"))

    # rm(string("2_costRecord_", filename, ".csv"))
    # rm(string("2_timeRecord_", filename, ".csv"))
    # rm(string("2_performance_", filename, ".csv"))
    # rm(string("2_params_", filename, ".bin"))
    println("Testing multiTrain")
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 6, 1)
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 6, 2)
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 6, 1, costFunc = "normLog")
    multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 6, 2, costFunc = "normLog")
    println("TEST PASSED")
    println()

    println("Testing evalMulti")
    evalMulti(name, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, IDList = [1, 2])
    evalMulti(name, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, IDList = [1, 2], costFunc = "normLog")
    println("TEST PASSED")
    println()

    filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.0_L2_1.0_maxNorm_0.002_alpha_0.1_decayRate_ADAMAX", backend, "_absErr")

    rm(string("1_multiParams_", filename, ".bin"))
    rm(string("1_multiPerformance_", filename, ".csv"))
    rm(string("1_predictionScatterTrain_", filename, ".csv"))
    rm(string("1_predictionScatterTest_", filename, ".csv"))
    rm(string("2_multiParams_", filename, ".bin"))
    rm(string("2_multiPerformance_", filename, ".csv"))
    rm(string("2_predictionScatterTrain_", filename, ".csv"))
    rm(string("2_predictionScatterTest_", filename, ".csv"))
    rm(string("fullMultiParams_", filename, ".bin"))
    rm(string("fullMultiPerformance_", filename, ".csv"))
    rm(string("predictionScatterTrain_", filename, ".csv"))
    rm(string("predictionScatterTest_", filename, ".csv"))

    filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.0_L2_1.0_maxNorm_0.002_alpha_0.1_decayRate_ADAMAX", backend, "_normLog")

    rm(string("1_multiPerformance_", filename, ".csv"))
    rm(string("1_predictionScatterTrain_", filename, ".csv"))
    rm(string("1_predictionScatterTest_", filename, ".csv"))
    rm(string("1_multiParams_", filename, ".bin"))
    rm(string("2_multiPerformance_", filename, ".csv"))
    rm(string("2_predictionScatterTrain_", filename, ".csv"))
    rm(string("2_predictionScatterTest_", filename, ".csv"))
    rm(string("2_multiParams_", filename, ".bin"))
    rm(string("fullMultiPerformance_", filename, ".csv"))
    rm(string("predictionScatterTrain_", filename, ".csv"))
    rm(string("predictionScatterTest_", filename, ".csv"))
    rm(string("fullMultiParams_", filename, ".bin"))

    
end