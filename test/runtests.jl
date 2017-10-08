using FCANN
using Base.Test

#auxiliary function tests
requestCostFunctions()

println("Testing AUX functions for 0 hidden layer network")
println("------------------------------------------------")
println("Testing paramter initialization")
T0, B0 = initializeParams(10, [], 2)
println("TEST PASSED")
println()

println("Testing parameters binary write")
try rm("testParams1") end
writeParams([(T0, B0)], "testParams1")
println("TEST PASSED")
println()

println("Testing binary parameter read")
params = readBinParams("testParams1")
@test(params == [(T0, B0)])
println("TEST PASSED")
println()

println("Testing prediction output")
X = randn(Float32, 10000, 10)
Y = predict(T0, B0, X)
@test(size(Y) == (10000, 2))
println("TEST PASSED")
println()


println("Testing AUX functions for 2 hidden layer network")
println("------------------------------------------------")

println("Testing paramter initialization")
T0, B0 = initializeParams(10, [10 10], 2)
println("TEST PASSED")
println()

println("Testing parameters binary write")
try rm("testParams2") end
writeParams([(T0, B0)], "testParams2")
println("TEST PASSED")
println()

println("Testing binary parameter read")
params = readBinParams("testParams2")
@test (params == [(T0, B0)])
println("TEST PASSED")
println()

println("Testing prediction output")
X = randn(Float32, 10000, 10)
Y = predict(T0, B0, X)
@test (size(Y) == (10000, 2))
println("TEST PASSED")
println()

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
err = checkNumGrad(0.0f0, costFunc = "sqErr")
@test(err < 0.015)
println("TEST PASSED")
println()

#basic test train with 1 input, 1 output, 1 neuron
println("Testing simple ANN training version 1")
println("-------------------------------------")
M = 1
hidden = [1]
O = 1
batchSize = 1024
N = 1000
testTrain(M, hidden, O, batchSize, N; writeFile = false, numThreads = 0, printProg = true)
println("TEST PASSED")
println()

#full train test with data read and output
function writeTestData(name, M, O)
    X = randn(Float32, 100000, M)
    Y = randn(Float32, 100000, O)
    Xtest = randn(Float32, 10000, M)
    Ytest = randn(Float32, 10000, O)
    writecsv(string("Xtrain_", name, ".csv"), X)
    writecsv(string("Xtest_", name, ".csv"), Xtest)
    writecsv(string("ytrain_", name, ".csv"), Y)
    writecsv(string("ytest_", name, ".csv"), Ytest)
end
println("Testing full ANN train with test data")
println("--------------------------------------")
name = "test"
M = 10
O = 2
writeTestData(name, M, O)

println("Training with 0 hidden layers")
record, T, B = fullTrain(name, 1000, 1024, [], 0.0f0, Inf, 0.002f0, 0.1f0, 1)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers")
record, T, B = fullTrain(name, 200, 1024, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 1)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers from previous endpoint")
record, T, B = fullTrain(name, 200, 1024, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 2, startID = 1)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Adding 4 cores to test multitrain algorithms")
addprocs(4)
@everywhere using FCANN
println("--------------------------------------------")
println("Testing archEval")
archEval(name, 1000, 1024, [[1], [2], [3], [2, 2]])
println("TEST PASSED")
println()

rm(string("archEval_", name, "_", M, "_input_", O, "_output_ADAMAX.csv"))

println("Testing evalLayers")
evalLayers(name, 100, 1024, [100, 200, 400, 800], layers=[2, 4, 6])
println("TEST PASSED")
println()

rm(string("evalLayers_", name, "_", M, "_input_", O, "_output_0.002_alpha_ADAMAX.csv"))

println("Testing smartEvalLayers")
smartEvalLayers(name, 100, 1024, [100, 200], layers=[1, 2], tau=0.05f0)
println("TEST PASSED")
println()

rm(string("evalLayers_", name, "_", M, "_input_", O, "_output_100_epochs_smartParams_ADAMAX.csv"))

println("Testing multiTrain")
multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 4, 1)
println("TEST PASSED")
println()

filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.0_L2_1.0_maxNorm_0.002_alpha_0.1_decayRate_ADAMAX")

rm(string("1_multiParams_", filename, ".bin"))
rm(string("1_multiPerformance_", filename, ".csv"))


# Remove generated files
rm("Xtrain_test.csv")
rm("Xtest_test.csv")
rm("ytrain_test.csv")
rm("ytest_test.csv")

filename = string(name, "_", M, "_input_", [], "_hidden_", O, "_output_0.0_L2_Inf_maxNorm_0.002_alpha_ADAMAX.csv")
rm(string("1_costRecord_", filename))
rm(string("1_timeRecord_", filename))
rm(string("1_performance_", filename))
rm(string("1_params_", filename[1:end-4]))

filename = string(name, "_", M, "_input_", [2, 2], "_hidden_", O, "_output_0.0_L2_Inf_maxNorm_0.002_alpha_ADAMAX.csv")
rm(string("1_costRecord_", filename))
rm(string("1_timeRecord_", filename))
rm(string("1_performance_", filename))
rm(string("1_params_", filename[1:end-4]))

rm(string("2_costRecord_", filename))
rm(string("2_timeRecord_", filename))
rm(string("2_performance_", filename))
rm(string("2_params_", filename[1:end-4]))

rm("testParams1")
rm("testParams2")

