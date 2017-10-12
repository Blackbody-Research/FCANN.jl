println("-----------------------Testing Initialization Functions-------------------------------------")
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
T0, B0 = initializeParams(10, [10, 10], 2)
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