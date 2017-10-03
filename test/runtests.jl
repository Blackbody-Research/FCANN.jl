using FCANN
using Base.Test

#auxiliary function tests
println("Testing AUX functions for 0 hidden layer network")
println("------------------------------------------------")
println("Testing paramter initialization")
T0, B0 = initializeParams(10, [], 2)
println("TEST PASSED")
println()

println("Testing parameters binary write")
try rm("testParams") end
writeParams([(T0, B0)], "testParams")
println("TEST PASSED")
println()

println("Testing binary parameter read")
params = readBinParams("testParams")
assert(params == [(T0, B0)])
println("TEST PASSED")
println()

println("Testing prediction output")
X = randn(Float32, 10000, 10)
Y = predict(T0, B0, X)
assert(size(Y) == (10000, 2))
println("TEST PASSED")
println()


println("Testing AUX functions for 2 hidden layer network")
println("------------------------------------------------")

println("Testing paramter initialization")
T0, B0 = initializeParams(10, [10 10], 2)
println("TEST PASSED")
println()

println("Testing parameters binary write")
try rm("testParams") end
writeParams([(T0, B0)], "testParams")
println("TEST PASSED")
println()

println("Testing binary parameter read")
params = readBinParams("testParams")
assert(params == [(T0, B0)])
println("TEST PASSED")
println()

println("Testing prediction output")
X = randn(Float32, 10000, 10)
Y = predict(T0, B0, X)
assert(size(Y) == (10000, 2))
println("TEST PASSED")
println()

println("Testing numerical gradient vs backpropagation")
println("---------------------------------------------")
println("Lambda = 0, no hidden layers")
err = checkNumGrad(0.0f0, hidden_layers=[])
assert(err < 0.015)

println("Lambda = 0")
err = checkNumGrad(0.0f0)
assert(err < 0.015)

println("Lambda = 0.1")
err = checkNumGrad(0.1f0)
assert(err < 0.015)

println("Lambda = 1.0")
err = checkNumGrad(1.0f0)
assert(err < 0.015)
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
assert(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers")
record, T, B = fullTrain(name, 1000, 1024, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 1)
assert(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers from previous endpoint")
record, T, B = fullTrain(name, 1000, 1024, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 2, startID = 1)
assert(record[end] < record[1])
println("TEST PASSED")
println()


# filename = string(name, "_", M, "_input_", hidden, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", alpha, "_alpha_ADAMAX.csv")

# rm("Xtrain_test.csv")
# rm("Xtest_test.csv")
# rm("ytrain_test.csv")
# rm("ytest_test.csv")
# rm("1_costRecord_Test.csv")
# rm("1_timeRecord_test.csv")
# rm("1_performance_test.csv")
# rm("1_params_test.csv")




