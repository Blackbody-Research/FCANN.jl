println("-----------------------Testing CPU Single Core-------------------------------------")
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
err = checkNumGrad(0.0f0, costFunc = "sqErr")
@test(err < 0.015)
println("TEST PASSED")
println()

println("Normal Log Likelihood Error Cost Function")
err = checkNumGrad(0.0f0, costFunc = "normLogErr")
@test(err < 0.015)
println("TEST PASSED")
println()

println("Cauchy Log Likelihood Error Cost Function")
err = checkNumGrad(0.0f0, costFunc = "cauchyLogErr")
@test(err < 0.015)
println("TEST PASSED")
println()

println("Skipping activation functions")
err = checkNumGrad(0.0f0, hidden_layers=[10, 10, 10], costFunc="sqErr", activation_list = [true, false, true])
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
N = 150
testTrain(M, hidden, O, batchSize, N; writeFile = false, numThreads = 0, printProg = true)
println("TEST PASSED")
println()

testTrain(M, [10, 10], O, batchSize, N; writeFile = false, numThreads = 0, printProg = true, activation_list = [false, true], reslayers=1, dropout = 0.2f0)
println("TEST PASSED")
println()


#full train test with data read and output
function writeTestData(name, M, O)
    for p1 = ("X", "y"), p2 = ("train", "test")
        l = (p2 == "train") ? 1024 : 1024
        N = (p1 == "X") ? M : O
        writedlm(string(p1, p2, "_", name, ".csv"), rand(Float32, l, N))
    end
end

function writeBinData(name, M, O)
    for p1 = ("X", "y"), p2 = ("train", "test")
        l = (p2 == "train") ? 1024 : 1024
        N = (p1 == "X") ? M : O
        writeArray(rand(Float32, l, N), string(p1, p2, "_", name, ".bin"))
    end
end

println("Testing full ANN train with test data")
println("--------------------------------------")
name = "test"
M = 10
O = 2
Random.seed!(1234)
writeTestData(name, M, O)
writeBinData(name, M, O)

(Xtrain, Xtest, ytrain, ytest) = readBinInput(name)

println("Training with 0 hidden layers")
record, T, B = fullTrain(name, 150, 512, Vector{Int64}(), 0.0f0, Inf, 0.002f0, 0.1f0, 1, writeFiles=false)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 0 hidden layers and binary input read")
record, T, B = fullTrain(name, 150, 512, Vector{Int64}(), 0.0f0, Inf, 0.002f0, 0.1f0, 1, writeFiles=false, binInput = true)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers")
record, T, B = fullTrain(name, 10, 512, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 1)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers and μP")
fullTrain(name, 10, 512, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 1, use_μP = true)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers from previous endpoint")
record, T, B = fullTrain(name, 150, 512, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 2, startID = 1, writeFiles=false)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Training with 2 hidden layers and 1 residual layer")
record, T, B = fullTrain(name, 10, 512, [2, 2], 0.0f0, Inf, 0.002f0, 0.1f0, 1, resLayers=1, writeFiles=false)
@test(record[end] < record[1])
println("TEST PASSED")
println()

println("Calculating feature impacts")
calcfeatureimpact(T, B, Xtest, ytest, num=1)
calcfeatureimpact(T, B, Xtest, ytest, num=2)

println("Training autoencoder")
record, T, B = fullTrain(name, 10, 512, [2], 0.0f0, Inf, 0.002f0, 0.1f0, 1, writeFiles=false, inputdata = (Xtrain, Xtest))

println("Running training trials")
traintrials([(Xtrain, ytrain)], 512, 10, [2, 2], 0.0f0, Inf)
traintrials([(Xtrain, ytrain)], 512, 10, [2, 2], 0.0f0, Inf, use_μP = true, costfunc = "normLogErr", reslayers = 1)
traintrials([(Xtrain, ytrain), (Xtest, ytest)], 512, 10, [2, 2], 0.0f0, Inf)