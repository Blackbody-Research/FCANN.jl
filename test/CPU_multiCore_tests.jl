println("-----------------------Testing CPU Multi Core-------------------------------------")
backend = setBackend(:CPU)
name = "test"
M = 10
O = 2

println("Adding 4 cores to test multitrain algorithms")
addprocs(4)
@everywhere using FCANN
println("--------------------------------------------")
println("Testing archEval")
archEval(name, 100, 1024, [[1], [2], [3], [2, 2]])
println("TEST PASSED")
println()

println("Testing archEval with normLog cost function")
archEval(name, 100, 1024, [[1], [2], [3], [2, 2]], costFunc = "normLogErr")
println("TEST PASSED")
println()

rm(string("archEval_", name, "_", M, "_input_", O, "_output_ADAMAX", backend, "_absErr.csv"))
rm(string("archEval_", name, "_", M, "_input_", O, "_output_ADAMAX", backend, "_normLogErr.csv"))

println("Testing evalLayers")
evalLayers(name, 100, 1024, [100, 200], layers=[2, 4])
println("TEST PASSED")
println()

rm(string("evalLayers_", name, "_", M, "_input_", O, "_output_0.002_alpha_ADAMAX", backend, "_absErr.csv"))

println("Testing smartEvalLayers")
smartEvalLayers(name, 100, 1024, [100, 200], layers=[1, 2], tau=0.05f0)
println("TEST PASSED")
println()

rm(string("evalLayers_", name, "_", M, "_input_", O, "_output_100_epochs_smartParams_ADAMAX", backend, "_absErr.csv"))

println("Testing multiTrain")
for v1 = (1, 2), costFunc = ("absErr", "normLogErr")
	multiTrain(name, 200, 1024, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, 6, v1, costFunc = costFunc)
end
println("TEST PASSED")
println()

println("Testing evalMulti")
evalMulti(name, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, IDList = [1, 2])
evalMulti(name, [2, 2], 0.0f0, 1.0f0, 0.002f0, 0.1f0, IDList = [1, 2], costFunc = "normLogErr")
println("TEST PASSED")
println()

for errFunc = ("absErr", "normLogErr")
	filename = string(name, "_", M, "_input_2X2_hidden_", O, "_output_0.002_alpha_0.1_decay_1.0_maxNorm_ADAMAX_", errFunc)

	for v1 = (1, 2), v2 = ("Params", "Performance", "PredScatTrain", "PredScatTest")
		fend = (v2 == "Params") ? ".bin" : ".csv"
		rmname = string(v1, "_multi", v2, "_", filename, fend)
		println("Removing $rmname")
		rm(rmname)
	end
	rm(string("fullMultiParams_", filename, ".bin"))
	rm(string("fullMultiPerformance_", filename, ".csv"))
	rm(string("multiPredScatTrain_", filename, ".csv"))
	rm(string("multiPredScatTest_", filename, ".csv"))
end

# Remove generated files
for v in ("X", "y"), t = ("train", "test"), e = (".csv", ".bin")
	rm(string(v, t, "_test", e))	
end


# filename = string(name, "_10_input_", [], "_hidden_2_output_0.0_L2_Inf_maxNorm_0.002_alpha_ADAMAX", backend, "_absErr")
# rm(string("1_costRecord_", filename, ".csv"))
# rm(string("1_timeRecord_", filename, ".csv"))
# rm(string("1_performance_", filename, ".csv"))
# rm(string("1_params_", filename, ".bin"))

filename = string(name, "_10_input_2X2_hidden_2_output_0.002_alpha_0.1_decay_ADAMAX_absErr")
for n in ("costRecord", "timeRecord", "performance", "predScatTrain", "predScatTest", "params")
	fend = (n == "params") ? ".bin" : ".csv"
	rm(string("1_", n, "_", filename, fend))
end

# rm(string("2_costRecord_", filename, ".csv"))
# rm(string("2_timeRecord_", filename, ".csv"))
# rm(string("2_performance_", filename, ".csv"))
# rm(string("2_params_", filename, ".bin"))

rm("testParams1")
rm("testParams2")