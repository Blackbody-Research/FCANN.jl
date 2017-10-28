println("-----------------------Testing CPU Multi Core-------------------------------------")
backend = setBackend(:CPU)
println("Adding 4 cores to test multitrain algorithms")
addprocs(4)
@everywhere using FCANN
println("--------------------------------------------")
println("Testing archEval")
archEval(name, 100, 1024, [[1], [2], [3], [2, 2]])
println("TEST PASSED")
println()



rm(string("archEval_", name, "_", M, "_input_", O, "_output_ADAMAX", backend, "_absErr.csv"))

println("Testing evalLayers")
evalLayers(name, 100, 1024, [100, 200, 400, 800], layers=[2, 4, 6])
println("TEST PASSED")
println()

rm(string("evalLayers_", name, "_", M, "_input_", O, "_output_0.002_alpha_ADAMAX", backend, "_absErr.csv"))

println("Testing smartEvalLayers")
smartEvalLayers(name, 100, 1024, [100, 200], layers=[1, 2], tau=0.05f0)
println("TEST PASSED")
println()

rm(string("evalLayers_", name, "_", M, "_input_", O, "_output_100_epochs_smartParams_ADAMAX", backend, "_absErr.csv"))

println("Testing multiTrain")
multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 4, 1)
println("TEST PASSED")
println()

filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.0_L2_1.0_maxNorm_0.002_alpha_0.1_decayRate_ADAMAX", backend, "_absErr")

rm(string("1_multiParams_", filename, ".bin"))
rm(string("1_multiPerformance_", filename, ".csv"))
rm(string("1_predictionScatterTrain_", filename, ".csv"))
rm(string("1_predictionScatterTest_", filename, ".csv"))


# Remove generated files
rm("Xtrain_test.csv")
rm("Xtest_test.csv")
rm("ytrain_test.csv")
rm("ytest_test.csv")

# filename = string(name, "_10_input_", [], "_hidden_2_output_0.0_L2_Inf_maxNorm_0.002_alpha_ADAMAX", backend, "_absErr")
# rm(string("1_costRecord_", filename, ".csv"))
# rm(string("1_timeRecord_", filename, ".csv"))
# rm(string("1_performance_", filename, ".csv"))
# rm(string("1_params_", filename, ".bin"))

filename = string(name, "_10_input_", [2, 2], "_hidden_2_output_0.0_L2_Inf_maxNorm_0.002_alpha_ADAMAX", backend, "_absErr")
rm(string("1_costRecord_", filename, ".csv"))
rm(string("1_timeRecord_", filename, ".csv"))
rm(string("1_performance_", filename, ".csv"))
rm(string("1_predictionScatterTest_", filename, ".csv"))
rm(string("1_predictionScatterTrain_", filename, ".csv"))
rm(string("1_params_", filename, ".bin"))

# rm(string("2_costRecord_", filename, ".csv"))
# rm(string("2_timeRecord_", filename, ".csv"))
# rm(string("2_performance_", filename, ".csv"))
# rm(string("2_params_", filename, ".bin"))

rm("testParams1")
rm("testParams2")