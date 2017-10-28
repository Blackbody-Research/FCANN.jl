
# function evalBootstrap(name, hidden, lambdaeta, c, alpha, R; IDList = [], adv = false, dropout=0.0f0, costFunc = "absErr")
#   println("reading and converting training data")
#   X, Xtest, Y, Ytest = readInput(name)

#   (N, M) = size(X)
#   O = size(Y, 2)
    
#   h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
#       string(hidden[1], "X", length(hidden))
#   else
#       string(hidden)
#   end

#   filename = if adv
#       string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", c, "_maxNorm_", lambdaeta, "_advNoise_", alpha, "_alpha_AdvADAMAX", backend, "_", costFunc)
#   else
#       if dropout == 0.0f0
#           string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambdaeta, "_L2_", c, "_maxNorm_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
#       else
#           string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambdaeta, "_L2_", c, "_maxNorm_", dropout, "_dropoutRate_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
#       end
#   end

#   function hcatRobust(A, B)
#       if isempty(A)
#           B
#       elseif isempty(B)
#           A
#       else
#           [A B]
#       end
#   end

#   function vcatRobust(A, B)
#       if isempty(A)
#           B
#       elseif isempty(B)
#           A
#       else
#           [A; B]
#       end
#   end 
    
#   println("compiling all multi parameters together")
#   #assembling all network parameters together
#   multiOut = if isempty(IDList)
#       if isfile(string("fullBootstrapParams_", filename, ".csv"))
#           params = map(Float32, readcsv(string("fullBootstrapParams_", filename, ".csv")))
#           map(1:size(params, 2)) do col
#               p = params[:, col]
#               (T, B) = params2Theta(M, hidden, O, p)
#           end
#       else
#           readBinParams(string("fullBootstrapParams_", filename, ".bin"))
#       end
#   else
#       if isfile(string(IDList[1], "_bootstrapParams_", filename, ".csv"))
#           mapreduce(vcatRobust, IDList) do ID
#               try
#                     params = map(Float32, readcsv(string(ID, "_bootstrapParams_", filename, ".csv"))) 
#                   map(1:size(params, 2)) do col
#                       p = params[:, col]
#                       (T, B) = params2Theta(M, hidden, O, p)
#                   end
#                 catch
#                     []
#                 end
#           end
#       else
#           mapreduce(vcatRobust, IDList)  do ID
#               try
#                     readBinParams(string(ID, "_bootstrapParams_", filename, ".bin"))
#                 catch
#                     []
#                 end
#           end
#       end
#   end

#   println("saving combined parameters")
#   writeParams(convert(Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, multiOut), string("fullBootstrapParams_", filename, ".bin"))
#   println(string("calculating combined outputs for ", length(multiOut), " networks"))
    
#   multiOutTrain = pmap(a -> predict(a[1], a[2], X, dropout), multiOut)
#   multiOutTest = pmap(a -> predict(a[1], a[2], Xtest, dropout), multiOut)

#   println("calculating train and test errors as a function of number of networks")
#   header = ["Num Networks" "Training Error" "Training Error Est" "Test Error" "Test Error Est"]
    
#   #calculate average network output
#   fullMultiPerformance = mapreduce(vcat, 1:length(multiOut)) do i
#       combinedOutputTrain = reduce(+, multiOutTrain[1:i])/i
#       errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, multiOutTrain[1:i])/i)    
#       Jtrain = calcError(combinedOutputTrain, Y, costFunc = costFunc)
            
#       combinedOutputTest = reduce(+, multiOutTest[1:i])/i
#       errorEstTest = mean(mapreduce(a -> abs.(a - combinedOutputTest), +, multiOutTest[1:i])/i)   
#       Jtest = calcError(combinedOutputTest, Ytest, costFunc = costFunc)
#       [i Jtrain errorEstTrain Jtest errorEstTest]
#   end
        
#   println("saving results to file")   
#   writecsv(string("fullBootstrapPerformance_", filename, ".csv"), [header; fullMultiPerformance])     
# end

#bootstrapTrain is a function that trains a set of networks with a specified architecture over a random sampling 
#of the training set specified in the name provided in input.  The sampling is performed with replacement N times
#where N is the number of examples in the training set. Training uses the ADAMAX minibatch algorithm and has options for 
#both an L2 and max norm regularization parameter.  It saves the parameters themselves as a matrix where each column
#contains the paramters from each training session converted to a vector. As such the hidden layer structure, input layer size, 
#and output layer size must be specified to reconstruct the network parameters in a usable state.  Each column would be 
#reconstructed in its own network. The training and test set errors calculated using the bootstrap network output which 
#averages the outputs from each individually trained network are saved as well as the error estimates based on the mean 
#abs deviation of the bootstrap network output itself.  one dimensional vector to file. 
#Call this function with: bootstrapTrain(name, numEpochs, batchSize, hidden, lambda, c, alpha, num, ID), where numEpochs is the 
#number of epochs over which to train, batchSize is the number of examples in each minibatch, hidden is a vector of integers 
#to specify the structure of the network network to train.  For example [4, 4] would indicate a network with two hidden 
#layers each with 4 neurons.  Lambda is the L2 regularization hyperparameter, c is the max norm regularization hyper parameter, 
#alpha is the training hyper parameter for the ADAMAX algorithm, num is the number of individual networks to train each with 
#it's own sampling of the training set, and ID is just a string used to specify this bootstrap session. Also printProg can be set 
#with a keyword argument to false to supress output of the training progress to the terminal.  Final results for each bootstrap session
#will still be printed to terminal regardless. 
# function bootstrapTrain(name, numEpochs, batchSize, hidden, lambda, c, alpha, R, num, ID; dropout = 0.0f0, printProg = false, costFunc = "absErr")
#   println("reading and converting training data")
#   X, Xtest, Y, Ytest = readInput(name)

#   (N, M) = size(X)
#   O = size(Y, 2)
    
#   h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
#       string(hidden[1], "X", length(hidden))
#   else
#       string(hidden)
#   end

#   filename = if dropout == 0.0f0
#       string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
#   else
#       string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", dropout, "_dropoutRate_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
#   end

#   bootstrapOut = pmap(1:num) do foo
#       #BLAS.set_num_threads(Sys.CPU_CORES)
#       if (nprocs() > 1) & (backend == :CPU)
#           BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_CORES/min(nprocs(), num)))))
#       end
#       # BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_CORES/min(nprocs(), num)))))
#       T0, B0 = if contains(costFunc, "Log")
#           initializeParams(M, hidden, 2*O)
#       else
#           initializeParams(M, hidden, O)
#       end 
#       bootstrapInd = ceil(Int64, N*rand(N))       
#       (T, B, bestCost, costRecord, timeRecord, GFLOPS) = eval(Symbol("ADAMAXTrainNN", backend))(X[bootstrapInd, :], Y[bootstrapInd, :], batchSize, T0, B0, numEpochs, M, hidden, lambda, c, R = R, alpha=alpha, dropout=dropout, printProgress = printProg, costFunc = costFunc)
#       (T, B)
#   end
#   fileout = convert(Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, bootstrapOut)
#   writeParams(fileout, string(ID, "_bootstrapParams_", filename, ".bin"))
    
#   #calculate average network output
#   bootstrapOutTrain = map(a -> calcOutput(X, Y, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
#   combinedOutputTrain = reduce(+, bootStrapOutTrain)/num
#   errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, bootstrapOutTrain)/num)   
#   Jtrain = calcError(combinedOutputTrain, Y, costFunc = costFunc)
        
#   bootstrapOutTest = map(a -> calcOutput(Xtest, Ytest, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
#   combinedOutputTest = reduce(+, bootstrapOutTest)/num
#   errorEstTest = mean(mapreduce(a -> abs.(a - combinedOutputTest), +, bootstrapOutTest)/num)  
#   Jtest = calcError(combinedOutputTest, Ytest, costFunc = costFunc)
        
#   writecsv(string(ID, "_bootstrapPerformance_", filename, ".csv"), [["Training Error", "Training Error Est", "Test Error", "Test Error Est"] [Jtrain, errorEstTrain, Jtest, errorEstTest]])           
# end

#=
function archEval(name, batchSize, numEpochs, hidden_layers, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
    Xtrain_values = map(Float32, readcsv(string("Xtrain_values_", name, ".csv")))
    Xtest_values = map(Float32, readcsv(string("Xtest_values_", name, ".csv")))
    ytrain = map(Float32, readcsv(string("ytrain_", name, ".csv")))
    ytest = map(Float32, readcsv(string("ytest_", name, ".csv")))

    invY_values = map(Float32, readcsv(string("invY_", name, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    (Xtrain_labels, Xtest_labels) = begin
        try
            (map(Float32, readcsv(string("Xtrain_labels_", name, ".csv"))), float32(readcsv(string("Xtest_labels_", name, ".csv"))))
        catch
            ([], [])
        end
    end
    
    (Xtrain, Xtest) = begin
        if isempty(Xtrain_labels)
            (Xtrain_values, Xtest_values)
        else
            ([Xtrain_labels Xtrain_values], [Xtest_labels Xtest_values])
        end
    end
    
    input_layer_size = size(Xtrain, 2)
    output_layer_size = size(ytrain, 2)
    #println("starting numerical gradient checking")
    #checkNumGrad(0.1f0)
    #println("press any key to continue")
    #readline(STDIN)

    #=
    if useGPU
        out = devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
            device(devlist[1])
            out = CUDArt.CuModule("costFunctionKernelsv2.ptx") do md
                header = ["Num Hidden" "Num Params" "Train Abs Error" "Test Abs Error"]
                out = mapreduce(vcat, hidden_layers) do hidden_layer_size
                    Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
                    (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, 0.0f0, md)
                    nn_params = theta2Params(bestBiases, bestThetas)
                    writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", 0.0, "_lambda_nnParams_OPTIMIZEDGPUABSERR.csv"), nn_params) 
                    trainingOutput = invY(predict(bestThetas, bestBiases, Xtrain))
                    trainingError = errFunc(trainingOutput, invY(ytrain))
                    testOutput = invY(predict(bestThetas, bestBiases, Xtest))
                    testError = errFunc(testOutput, invY(ytest))
                    [length(hidden_layer_size) length(nn_params) trainingError testError]
                end
                line1 = [0 0 errFunc(invY(ytrain), mean(invY(ytrain))) errFunc(invY(ytest), mean(invY(ytest)))]

                
                filename = string(name, input_layer_size, "_input_OPTIMIZEDGPUABSERR_archEval.csv")
                writecsv(filename, [header; line1; out])
            end
        end 
    else
    =#
        header = ["Num Hidden" "Num Params" "Train Abs Error" "Test Abs Error"]
        out = pmap(hidden_layers) do hidden_layer_size
            BLAS.set_num_threads(Sys.CPU_CORES)
            batchSize = round(Int, min(length(ytrain)/10, max(200, (50000 - (input_layer_size*hidden_layer_size[1])) / (input_layer_size))))
            Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
            (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, 0.0f0)
            nn_params = theta2Params(bestBiases, bestThetas)
            writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", 0.0, "_lambda_nnParams_OPTIMIZEDABSERR.csv"), nn_params) 
            trainingOutput = invY(predict(bestThetas, bestBiases, Xtrain))
            trainingError = errFunc(trainingOutput, invY(ytrain))
            testOutput = invY(predict(bestThetas, bestBiases, Xtest))
            testError = errFunc(testOutput, invY(ytest))
            [length(hidden_layer_size) length(nn_params) trainingError testError]
        end
        out = reduce(vcat, out)
        Xtrain = [ones(size(ytrain, 1)) Xtrain]
        Xtest = [ones(size(ytest, 1)) Xtest]
        betas = pinv(Xtrain'*Xtrain)*Xtrain'*ytrain
        line1 = [0 input_layer_size+1 errFunc(invY(Xtrain*betas), invY(ytrain)) errFunc(invY(Xtest*betas), invY(ytest))]
        filename = string(name, input_layer_size, "_input_", output_layer_size, "_output_OPTIMIZEDABSERR_archEval.csv")
        writecsv(filename, [header; line1; out])
    #end
end

#advReg is a function that trains the same network architecture with the same training and test sets using the
#fast gradient sign method of generating adversarial noise.  Different noise hyperparameters (eta) can be specified
#as well as the option to select a max norm regularizer. It saves the mean abs error over the training and test set 
#in a table where each row corresponds to each value of c. The name provided informs which training and test set data to read.  
#Call this function with: advReg(name, N, batchSize, hidden, etaList, a), where N is the 
#number of epochs over which to train, batchSize is the number of examples in each minibatch, hidden 
#is a vector of integers to specify the structure of the network network to train.  For example 
#[4, 4] would indicate a network with two hidden layers each with 4 neurons.  etaList is an array of 32 bit
#floats which list the adversarial noise hyperparameter to use.  alpha is the alpha hyper parameter for the ADAMAX 
#training algorithm. 
function advReg(name, N, batchSize, hidden, etaList, alpha, c = Inf)
    println("reading and converting training data")
    X, Xtest, Y, Ytest = readInput(name)

    M = size(X, 2)
    O = size(Y, 2)

    lambda = 0.0f0
    
    filename = string(name, "_", M, "_input_", hidden, "_hidden_", O, "_output_", c, "_maxNorm_", alpha, "alpha_AdvADAMAX", backend, ".csv")
    
    println(string("training network with ", hidden, " hidden layers "))
    println("initializing network parameters")
    T0, B0 = initializeParams(M, hidden, O)
    
    header = ["Eta Noise" "Train Error" "Test Error"]
    body = @parallel (vcat) for eta = etaList
        #BLAS.set_num_threads(Sys.CPU_CORES)
        BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_CORES/min(nprocs(), length(etaList))))))
        
        srand(1234)
        println("beginning training with ", eta, " adversarial noise")
        T, B, bestCost, record = eval(Symbol("ADAMAXTrainNN", backend))Adv(X, Y, batchSize, T0, B0, N, M, hidden, eta, c, alpha, printProgress = true)
        outTrain = predict(T, B, X)
        outTest = predict(T, B, Xtest)
        Jtrain = mean(abs.(outTrain - Y))
        Jtest = mean(abs.(outTest - Ytest))
        [eta Jtrain Jtest]
    end
    writecsv(string("advReg_", filename), [header; body])
end


function evalBootstrapFull(name, hidden_layer_size, c, suffixes, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
    Xtrain_values = float32(readcsv(string("Xtrain_values_", name, ".csv")))
    Xtest_values = float32(readcsv(string("Xtest_values_", name, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name, ".csv")))
    ytest = float32(readcsv(string("ytest_", name, ".csv")))

    invY_values = float32(readcsv(string("invY_", name, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    (Xtrain_labels, Xtest_labels) = begin
        try
            (float32(readcsv(string("Xtrain_labels_", name, ".csv"))), float32(readcsv(string("Xtest_labels_", name, ".csv"))))
        catch
            ([], [])
        end
    end
    
    (Xtrain, Xtest) = begin
        if isempty(Xtrain_labels)
            (Xtrain_values, Xtest_values)
        else
            ([Xtrain_labels Xtrain_values], [Xtest_labels Xtest_values])
        end
    end
    
    N = length(ytrain)
    
    input_layer_size = size(Xtrain, 2)
    output_layer_size = size(ytrain, 2)
    
    #=
    println("starting numerical gradient checking")
    checkNumGrad(0.1f0)
    println("press any key to continue")
    readline(STDIN)
        
    if useGPU
        bootstrapParams = mapreduce(hcat, suffixes) do suffix
            float32(readcsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_BOOTSTRAPnnParams_OPTIMIZEDGPUABSERR_", suffix, ".csv")))
        end
        
        out = map(1:size(bootstrapParams, 2)) do i
            (Thetas, Biases) = params2Theta(input_layer_size, hidden_layer_size, output_layer_size, bootstrapParams[:, i])
        end
        
        #calculate average network output
        trainingBootstrapOut = invY(mapreduce(a -> predict(a[1], a[2], Xtrain), hcat, out))
        trainingOutput = mean(trainingBootstrapOut, 2)
        trainingErrorEst = mean(abs.(trainingBootstrapOut .- mean(trainingBootstrapOut, 2)), 2)
        trainingError = errFunc(trainingOutput, invY(ytrain))
        testBootstrapOut = invY(mapreduce(a -> predict(a[1], a[2], Xtest), hcat, out))
        testOutput = mean(testBootstrapOut, 2)
        testErrorEst = mean(abs.(testBootstrapOut .- mean(testBootstrapOut, 2)), 2)
        testError = errFunc(testOutput, invY(ytest))

        writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_OPTIMIZEDGPUABSERR_bootstrapPerformance_full.csv"),  [["Training Error", "Training Error Est", "Test Error", "Test Error Est "] [trainingError, mean(trainingErrorEst), testError, mean(testErrorEst)]])
        writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_BOOTSTRAPnnParams_OPTIMIZEDGPUABSERR_full.csv"), bootstrapParams)

    else
    =#
        bootstrapParams = mapreduce(hcat, suffixes) do suffix
            float32(readcsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_BOOTSTRAPnnParams_OPTIMIZEDABSERR_", suffix, ".csv")))
        end
        
        out = map(1:size(bootstrapParams, 2)) do i
            (Thetas, Biases) = params2Theta(input_layer_size, hidden_layer_size, output_layer_size, bootstrapParams[:, i])
        end
        
            #calculate average network output
        trainingBootstrapOut = map(1:output_layer_size) do i
            invY(mapreduce(a -> predict(a[1], a[2], Xtrain)[:, i], hcat, out))
        end     
        trainingOutput = mapreduce(a -> mean(a, 2), hcat, trainingBootstrapOut)
        trainingErrorEst = mapreduce(a -> mean(abs.(a .- mean(a, 2)), 2), hcat, trainingBootstrapOut)
        trainingError = errFunc(trainingOutput, invY(ytrain))
        
        testBootstrapOut = map(1:output_layer_size) do i
            invY(mapreduce(a -> predict(a[1], a[2], Xtest)[:, i], hcat, out))
        end     
        testOutput = mapreduce(a -> mean(a, 2), hcat, testBootstrapOut)
        testErrorEst = mapreduce(a -> mean(abs.(a .- mean(a, 2)), 2), hcat, testBootstrapOut)
        testError = errFunc(testOutput, invY(ytest))

        writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_OPTIMIZEDABSERR_bootstrapPerformance_full.csv"),  [["Training Error", "Training Error Est", "Test Error", "Test Error Est "] [trainingError, mean(trainingErrorEst), testError, mean(testErrorEst)]])
        writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_BOOTSTRAPnnParams_OPTIMIZEDABSERR_full.csv"), bootstrapParams)
    #end
end


function bootstrapTrain(name, batchSize, numEpochs, hidden_layer_size, c, num, suffix, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
Xtrain_values = float32(readcsv(string("Xtrain_values_", name, ".csv")))
    Xtest_values = float32(readcsv(string("Xtest_values_", name, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name, ".csv")))
    ytest = float32(readcsv(string("ytest_", name, ".csv")))

    invY_values = float32(readcsv(string("invY_", name, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    (Xtrain_labels, Xtest_labels) = begin
        try
            (float32(readcsv(string("Xtrain_labels_", name, ".csv"))), float32(readcsv(string("Xtest_labels_", name, ".csv"))))
        catch
            ([], [])
        end
    end
    
    (Xtrain, Xtest) = begin
        if isempty(Xtrain_labels)
            (Xtrain_values, Xtest_values)
        else
            ([Xtrain_labels Xtrain_values], [Xtest_labels Xtest_values])
        end
    end
    
    N = size(ytrain, 1)
    
    input_layer_size = size(Xtrain, 2)
    output_layer_size = size(ytrain, 2)
    
    #=
    println("starting numerical gradient checking")
    checkNumGrad(0.1f0)
    println("press any key to continue")
    readline(STDIN)
    

    if useGPU
        out = devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
            device(devlist[1])
            out2 = CUDArt.CuModule("costFunctionKernelsv2.ptx") do md
                
                #=
                println("starting numerical gradient checking")
                checkNumGrad(0.1f0, md)
                println("press any key to continue")
                readline(STDIN)
                =#

                header = ["Lambda" "Train Abs Error" "Test Abs Error"]
                out3 = @parallel vcat for i = num
                #out3 = mapreduce(vcat, lambdavec) do lambda
                    Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
                    bootstrapInd = int(ceil(N*rand(N))) 
                    (bestThetas, bestBiases, finalCost, costRecord) = trainNNMaxNorm(Xtrain[bootstrapInd, :], ytrain[bootstrapInd], batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, c, md)
                    nn_params = theta2Params(bestBiases, bestThetas)
                    (nn_params, bestTheats, bestBiases)
                end
                
                filedata = mapreduce(a -> a[1], hcat, out3)
                writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", x, "_maxNorm_BOOTSTRAPnnParams_OPTIMIZEDGPUABSERR_", suffix, ".csv"), filedata)

                #calculate average network output
                trainingBootstrapOut = invY(mapreduce(a -> predict(a[2], a[3], Xtrain), hcat, out))
                trainingOutput = mean(trainingBootstrapOut, 2)
                trainingErrorEst = mean(abs.(trainingBootstrapOut .- mean(trainingBootstrapOut, 2)), 2)
                trainingError = errFunc(trainingOutput, invY(ytrain))
                testBootstrapOut = invY(mapreduce(a -> predict(a[2], a[3], Xtest), hcat, out))
                testOutput = mean(testBootstrapOut, 2)
                testErrorEst = mean(abs.(testBootstrapOut .- mean(testBootstrapOut, 2)), 2)
                testError = errFunc(testOutput, invY(ytest))

                writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", c, "_maxNorm_OPTIMIZEDGPUABSERR_bootstrapPerformance_", suffix, ".csv"),  [["Training Error", "Training Error Est", "Test Error", "Test Error Est "] [trainingError, mean(trainingErrorEst), testError, mean(testErrorEst)]])
        
            end
        end 
        gc()
    else
    =#
        batchSize = int(min(length(ytrain)/10, max(200, (50000 - (input_layer_size*hidden_layer_size[1])) / (input_layer_size))))
        #out = @parallel vcat for lambda = lambdavec
        
        #(inputbatchData, outputbatchData) = generateBatches(Xtrain, ytrain[:], batchSize)
        out = pmap(1:num) do foo
            BLAS.set_num_threads(Sys.CPU_CORES)
            Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
            bootstrapInd = int(ceil(N*rand(N)))         
            (bestThetas, bestBiases, finalCost, costRecord) = trainNNMaxNorm(Xtrain[bootstrapInd, :], ytrain[bootstrapInd, :], batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, c)
            nn_params = theta2Params(bestBiases, bestThetas)
            (nn_params, bestThetas, bestBiases)
        end
        filedata = mapreduce(a -> a[1], hcat, out)
        writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_BOOTSTRAPnnParams_OPTIMIZEDABSERR_", suffix, ".csv"), filedata)

        #calculate average network output
        trainingBootstrapOut = map(1:output_layer_size) do i
            invY(mapreduce(a -> predict(a[2], a[3], Xtrain)[:, i], hcat, out))
        end     
        trainingOutput = mapreduce(a -> mean(a, 2), hcat, trainingBootstrapOut)
        trainingErrorEst = mapreduce(a -> mean(abs.(a .- mean(a, 2)), 2), hcat, trainingBootstrapOut)
        trainingError = errFunc(trainingOutput, invY(ytrain))
        
        testBootstrapOut = map(1:output_layer_size) do i
            invY(mapreduce(a -> predict(a[2], a[3], Xtest)[:, i], hcat, out))
        end     
        testOutput = mapreduce(a -> mean(a, 2), hcat, testBootstrapOut)
        testErrorEst = mapreduce(a -> mean(abs.(a .- mean(a, 2)), 2), hcat, testBootstrapOut)
        testError = errFunc(testOutput, invY(ytest))

        writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_OPTIMIZEDABSERR_bootstrapPerformance_", suffix, ".csv"),  [["Training Error", "Training Error Est", "Test Error", "Test Error Est "] [trainingError, mean(trainingErrorEst), testError, mean(testErrorEst)]])
        
    #end
end




function makeErrorSets(name, hidden_layer_size, lambda, useGPU)
    Xtrain_values = float32(readcsv(string("Xtrain_values_", name, ".csv")))
    Xtest_values = float32(readcsv(string("Xtest_values_", name, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name, ".csv")))
    ytest = float32(readcsv(string("ytest_", name, ".csv")))
    
    (Xtrain_labels, Xtest_labels) = begin
        try
            (float32(readcsv(string("Xtrain_labels_", name, ".csv"))), float32(readcsv(string("Xtest_labels_", name, ".csv"))))
        catch
            ([], [])
        end
    end
    
    (Xtrain, Xtest) = begin
        if isempty(Xtrain_labels)
            (Xtrain_values, Xtest_values)
        else
            ([Xtrain_labels Xtrain_values], [Xtest_labels Xtest_values])
        end
    end
    
    input_layer_size = size(Xtrain, 2)
    
    nn_params = begin
        #if useGPU
        #   float32(readcsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", lambda, "_lambda_nnParams_OPTIMIZEDGPUABSERR.csv")))
        #else
            float32(readcsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", lambda, "_lambda_nnParams_OPTIMIZEDABSERR.csv")))
        #end
    end

    Thetas, Biases = params2Theta(input_layer_size, hidden_layer_size, nn_params)
    
    nn_Y_train = predict(Thetas, Biases, Xtrain)
    nn_Y_test = predict(Thetas, Biases, Xtest)
    
    nn_Err_train = abs.(nn_Y_train - ytrain)
    nn_Err_test = abs.(nn_Y_test - ytest)
    
    Ay = mean(nn_Err_train)
    By = std(nn_Err_train)
    
    nn_Err_train = (nn_Err_train - Ay)/By
    nn_Err_test = (nn_Err_test - Ay)/By
    
    name2 = string(name, "_ErrPredictOn_", hidden_layer_size, "_hidden_", lambda, "_lambda_Network")
    
    Xtrain = [Xtrain nn_Y_train]
    ytrain = nn_Err_train
    
    Xtest = [Xtest nn_Y_test]
    ytest = nn_Err_test
    
    writecsv(string("Xtrain_", name2, ".csv"), Xtrain)
    writecsv(string("Xtest_", name2, ".csv"), Xtest)
    writecsv(string("ytrain_", name2, ".csv"), ytrain)
    writecsv(string("ytest_", name2, ".csv"), ytest)
    writecsv(string("invY_", name2, ".csv"), [["mean" "STD"]; [Ay By]])

    return name2
end

function archEvalErr(name, batchSize, numEpochs, hidden_layers, hidden_layer_orig, lambdaorig, useGPUorig, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
    name2 = makeErrorSets(name, hidden_layer_orig, lambdaorig, useGPUorig)
    
    Xtrain = float32(readcsv(string("Xtrain_", name2, ".csv")))
    Xtest = float32(readcsv(string("Xtest_", name2, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name2, ".csv")))
    ytest = float32(readcsv(string("ytest_", name2, ".csv")))

    invY_values = float32(readcsv(string("invY_", name2, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    invY_values2 = float32(readcsv(string("invY_", name, ".csv"))[2, :])
    Ay2 = float(invY_values2[1])
    By2 = float(invY_values2[2])
    invY2(y) = y*By2 + Ay2 
    
    input_layer_size = size(Xtrain, 2)

    #println("starting numerical gradient checking")
    #checkNumGrad(0.1f0)
    #println("press any key to continue")
    #readline(STDIN)

    #=
    if useGPU
        out = devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
            device(devlist[1])
            out = CUDArt.CuModule("costFunctionKernelsv2.ptx") do md
                header = ["Num Hidden" "Num Params" "Train Abs Error" "Test Abs Error"]
                out = mapreduce(vcat, hidden_layers) do hidden_layer_size
                    Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
                    (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, 0.0f0, md)
                    nn_params = theta2Params(bestBiases, bestThetas)
                    writecsv(string(name2, input_layer_size, "_input_", hidden_layer_size, "_hidden_", 0.0, "_lambda_nnParams_OPTIMIZEDGPUABSERR.csv"), nn_params) 
                    trainingOutput = predict(bestThetas, bestBiases, Xtrain)
                    trainingError = By*By2*errFunc(trainingOutput, ytrain)
                    testOutput = predict(bestThetas, bestBiases, Xtest)
                    testError = By*By2*errFunc(testOutput, ytest)
                    [length(hidden_layer_size) length(nn_params) trainingError testError]
                end
                line1 = [0 0 By*By2*errFunc(ytrain, mean(ytrain)) By*By2*errFunc(ytest, mean(ytest))]

                
                filename = string(name2, input_layer_size, "_input_OPTIMIZEDGPUABSERR_archEval.csv")
                writecsv(filename, [header; line1; out])
            end
        end 
    else
    =#
        header = ["Num Hidden" "Num Params" "Train Abs Error" "Test Abs Error"]
        out = pmap(hidden_layers) do hidden_layer_size
            BLAS.set_num_threads(Sys.CPU_CORES)
            batchSize = int(min(length(ytrain)/4, (50000 - (input_layer_size*hidden_layer_size[1])) / (input_layer_size)))
            Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size)
            (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, 0.0f0)
            nn_params = theta2Params(bestBiases, bestThetas)
            writecsv(string(name2, input_layer_size, "_input_", hidden_layer_size, "_hidden_", 0.0, "_lambda_nnParams_OPTIMIZEDABSERR.csv"), nn_params) 
            trainingOutput = predict(bestThetas, bestBiases, Xtrain)
            trainingError = By*By2*errFunc(trainingOutput, ytrain)
            testOutput = predict(bestThetas, bestBiases, Xtest)
            testError = By*By2*errFunc(testOutput, ytest)
            [length(hidden_layer_size) length(nn_params) trainingError testError]
        end
        out = reduce(vcat, out)
        line1 = [0 0 By*By2*errFunc(ytrain, mean(ytrain)) By*By2*errFunc(ytest, mean(ytest))]
        filename = string(name2, input_layer_size, "_input_OPTIMIZEDABSERR_archEval.csv")
        writecsv(filename, [header; line1; out])
    #end
end

function regularizeErr(name, batchSize, numEpochs, hidden_layer_size, lambdavec, hidden_layer_orig, lambdaorig, useGPUorig, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
    name2 = makeErrorSets(name, hidden_layer_orig, lambdaorig, useGPUorig)

    Xtrain = float32(readcsv(string("Xtrain_", name2, ".csv")))
    Xtest = float32(readcsv(string("Xtest_", name2, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name2, ".csv")))
    ytest = float32(readcsv(string("ytest_", name2, ".csv")))

    invY_values = float32(readcsv(string("invY_", name2, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    invY_values2 = float32(readcsv(string("invY_", name, ".csv"))[2, :])
    Ay2 = float(invY_values2[1])
    By2 = float(invY_values2[2])
    invY2(y) = y*By2 + Ay2 
    
    input_layer_size = size(Xtrain, 2)
    
    #=
    println("starting numerical gradient checking")
    checkNumGrad(0.1f0)
    println("press any key to continue")
    readline(STDIN)
    

    if useGPU
        out = devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
            device(devlist[1])
            out2 = CUDArt.CuModule("costFunctionKernelsv2.ptx") do md
                
                #=
                println("starting numerical gradient checking")
                checkNumGrad(0.1f0, md)
                println("press any key to continue")
                readline(STDIN)
                =#

                header = ["Lambda" "Train Abs Error" "Test Abs Error"]
                out3 = @parallel vcat for lambda = lambdavec
                #out3 = mapreduce(vcat, lambdavec) do lambda
                    Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
                    (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda, md)
                    nn_params = theta2Params(bestBiases, bestThetas)
                    writecsv(string(name2, input_layer_size, "_input_", hidden_layer_size, "_hidden_", lambda, "_lambda_nnParams_OPTIMIZEDGPUABSERR.csv"), nn_params) 
                    trainingOutput = predict(bestThetas, bestBiases, Xtrain)
                    trainingError = By*By2*errFunc(trainingOutput, ytrain)
                    testOutput = predict(bestThetas, bestBiases, Xtest)
                    testError = By*By2*errFunc(testOutput, ytest)
                    [lambda trainingError testError]
                end
                
                filename = string(name2, input_layer_size, "_input_", hidden_layer_size, "_hidden_OPTIMIZEDGPUABSERR_regularize.csv")
                writecsv(filename, [header; out3])
            end
        end 
        gc()
    else
    =#
        batchSize = int(min(length(ytrain)/4, (50000 - (input_layer_size*hidden_layer_size[1])) / (input_layer_size)))
        header = ["Lambda" "Train Abs Error" "Test Abs Error"]
        #out = @parallel vcat for lambda = lambdavec
        
        #(inputbatchData, outputbatchData) = generateBatches(Xtrain, ytrain[:], batchSize)
        out = pmap(lambdavec) do lambda
            BLAS.set_num_threads(Sys.CPU_CORES)
            Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
            #(bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda, inputbatchData, outputbatchData)
            (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda)
            nn_params = theta2Params(bestBiases, bestThetas)
            writecsv(string(name2, input_layer_size, "_input_", hidden_layer_size, "_hidden_", lambda, "_lambda_nnParams_OPTIMIZEDABSERR.csv"), nn_params) 
            trainingOutput = predict(bestThetas, bestBiases, Xtrain)
            trainingError = By*By2*errFunc(trainingOutput, ytrain)
            testOutput = predict(bestThetas, bestBiases, Xtest)
            testError = By*By2*errFunc(testOutput, ytest)
            [lambda trainingError testError]
        end
        out = reduce(vcat, out)
        filename = string(name2, input_layer_size, "_input_", hidden_layer_size, "_hidden_OPTIMIZEDABSERR_regularize.csv")
        writecsv(filename, [header; out])
    #end
end
    
    
    
function regularize(name, batchSize, numEpochs, hidden_layer_size, lambdavec, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
    Xtrain_values = float32(readcsv(string("Xtrain_values_", name, ".csv")))
    Xtest_values = float32(readcsv(string("Xtest_values_", name, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name, ".csv")))
    ytest = float32(readcsv(string("ytest_", name, ".csv")))

    invY_values = float32(readcsv(string("invY_", name, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    (Xtrain_labels, Xtest_labels) = begin
        try
            (float32(readcsv(string("Xtrain_labels_", name, ".csv"))), float32(readcsv(string("Xtest_labels_", name, ".csv"))))
        catch
            ([], [])
        end
    end
    
    (Xtrain, Xtest) = begin
        if isempty(Xtrain_labels)
            (Xtrain_values, Xtest_values)
        else
            ([Xtrain_labels Xtrain_values], [Xtest_labels Xtest_values])
        end
    end
    
    input_layer_size = size(Xtrain, 2)
    output_layer_size = size(ytrain, 2)
    #=
    println("starting numerical gradient checking")
    checkNumGrad(0.1f0)
    println("press any key to continue")
    readline(STDIN)
    

    if useGPU
        out = devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
            device(devlist[1])
            out2 = CUDArt.CuModule("costFunctionKernelsv2.ptx") do md
                
                #=
                println("starting numerical gradient checking")
                checkNumGrad(0.1f0, md)
                println("press any key to continue")
                readline(STDIN)
                =#

                header = ["Lambda" "Train Abs Error" "Test Abs Error"]
                out3 = @parallel vcat for lambda = lambdavec
                #out3 = mapreduce(vcat, lambdavec) do lambda
                    Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
                    (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda, md)
                    nn_params = theta2Params(bestBiases, bestThetas)
                    writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", lambda, "_lambda_nnParams_OPTIMIZEDGPUABSERR.csv"), nn_params) 
                    trainingOutput = invY(predict(bestThetas, bestBiases, Xtrain))
                    trainingError = errFunc(trainingOutput, invY(ytrain))
                    testOutput = invY(predict(bestThetas, bestBiases, Xtest))
                    testError = errFunc(testOutput, invY(ytest))
                    [lambda trainingError testError]
                end
                
                filename = string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_OPTIMIZEDGPUABSERR_regularize.csv")
                writecsv(filename, [header; out3])
            end
        end 
        gc()
    else
    =#
        batchSize = int(min(length(ytrain)/10, (50000 - (input_layer_size*hidden_layer_size[1])) / (input_layer_size)))
        header = ["Lambda" "Train Abs Error" "Test Abs Error"]
        #out = @parallel vcat for lambda = lambdavec
        
        #(inputbatchData, outputbatchData) = generateBatches(Xtrain, ytrain[:], batchSize)
        out = pmap(lambdavec) do lambda
            BLAS.set_num_threads(Sys.CPU_CORES)
            Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
            #(bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda, inputbatchData, outputbatchData)
            (bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda)
            nn_params = theta2Params(bestBiases, bestThetas)
            writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", lambda, "_lambda_nnParams_OPTIMIZEDABSERR.csv"), nn_params) 
            trainingOutput = invY(predict(bestThetas, bestBiases, Xtrain))
            trainingError = errFunc(trainingOutput, invY(ytrain))
            testOutput = invY(predict(bestThetas, bestBiases, Xtest))
            testError = errFunc(testOutput, invY(ytest))
            [lambda trainingError testError]
        end
        out = reduce(vcat, out)
        filename = string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_OPTIMIZEDABSERR_regularize.csv")
        writecsv(filename, [header; out])
    #end
end

function maxNormReg(name, batchSize, numEpochs, hidden_layer_size, cvec, useGPU = false, errFunc = (a, b) -> mean(abs.(a-b)))
    Xtrain_values = float32(readcsv(string("Xtrain_values_", name, ".csv")))
    Xtest_values = float32(readcsv(string("Xtest_values_", name, ".csv")))
    ytrain = float32(readcsv(string("ytrain_", name, ".csv")))
    ytest = float32(readcsv(string("ytest_", name, ".csv")))

    invY_values = float32(readcsv(string("invY_", name, ".csv"))[2, :])
    Ay = float(invY_values[1])
    By = float(invY_values[2])
    invY(y) = y*By + Ay 
    
    (Xtrain_labels, Xtest_labels) = begin
        try
            (float32(readcsv(string("Xtrain_labels_", name, ".csv"))), float32(readcsv(string("Xtest_labels_", name, ".csv"))))
        catch
            ([], [])
        end
    end
    
    (Xtrain, Xtest) = begin
        if isempty(Xtrain_labels)
            (Xtrain_values, Xtest_values)
        else
            ([Xtrain_labels Xtrain_values], [Xtest_labels Xtest_values])
        end
    end
    
    input_layer_size = size(Xtrain, 2)
    output_layer_size = size(ytrain, 2)
    #=
    println("starting numerical gradient checking")
    checkNumGrad(0.1f0)
    println("press any key to continue")
    readline(STDIN)
    

    if useGPU
        out = devices(dev->capability(dev)[1] >= 2, nmax=1) do devlist
            device(devlist[1])
            out2 = CUDArt.CuModule("costFunctionKernelsv2.ptx") do md
                
                #=
                println("starting numerical gradient checking")
                checkNumGrad(0.1f0, md)
                println("press any key to continue")
                readline(STDIN)
                =#

                header = ["Max Norm" "Train Abs Error" "Test Abs Error"]
                out3 = @parallel vcat for c = cvec
                #out3 = mapreduce(vcat, lambdavec) do lambda
                    Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
                    (bestThetas, bestBiases, finalCost, costRecord) = trainNNMaxNorm(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, c, md)
                    nn_params = theta2Params(bestBiases, bestThetas)
                    writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxnorm_nnParams_OPTIMIZEDGPUABSERR.csv"), nn_params) 
                    trainingOutput = invY(predict(bestThetas, bestBiases, Xtrain))
                    trainingError = errFunc(trainingOutput, invY(ytrain))
                    testOutput = invY(predict(bestThetas, bestBiases, Xtest))
                    testError = errFunc(testOutput, invY(ytest))
                    [c trainingError testError]
                end
                
                filename = string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_OPTIMIZEDGPUABSERR_maxnormreg.csv")
                writecsv(filename, [header; out3])
            end
        end 
        gc()
    else
    =#
        batchSize = int(min(length(ytrain)/10, max(200, (50000 - (input_layer_size*hidden_layer_size[1])) / (input_layer_size))))
        header = ["Max Norm" "Train Abs Error" "Test Abs Error"]
        #out = @parallel vcat for lambda = lambdavec
        
        #(inputbatchData, outputbatchData) = generateBatches(Xtrain, ytrain[:], batchSize)
        out = pmap(cvec) do c
            BLAS.set_num_threads(Sys.CPU_CORES)
            Thetas, Biases = initializeParams(input_layer_size, hidden_layer_size, output_layer_size)
            #(bestThetas, bestBiases, finalCost, costRecord) = trainNN(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, lambda, inputbatchData, outputbatchData)
            (bestThetas, bestBiases, finalCost, costRecord) = trainNNMaxNorm(Xtrain, ytrain, batchSize, Thetas, Biases, numEpochs, input_layer_size, hidden_layer_size, c)
            nn_params = theta2Params(bestBiases, bestThetas)
            writecsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_nnParams_OPTIMIZEDABSERR.csv"), nn_params) 
            #nn_params = float32(readcsv(string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_", c, "_maxNorm_nnParams_OPTIMIZEDABSERR.csv")))
            #(bestThetas, bestBiases) = params2Theta(input_layer_size, hidden_layer_size, output_layer_size, nn_params)
            trainingOutput = invY(predict(bestThetas, bestBiases, Xtrain))
            trainingError = errFunc(trainingOutput, invY(ytrain))
            testOutput = invY(predict(bestThetas, bestBiases, Xtest))
            testError = errFunc(testOutput, invY(ytest))
            [c trainingError testError]
        end
        out = reduce(vcat, out)
        filename = string(name, input_layer_size, "_input_", hidden_layer_size, "_hidden_", output_layer_size, "_output_OPTIMIZEDABSERR_maxNormReg.csv")
        writecsv(filename, [header; out])
    #end
end



function updateG!(TG, BG, GT, GB)
    for i = 1:length(TG)
        @simd for ii = 1:length(TG[i])
            @inbounds GT[i][ii] = GT[i][ii] + (TG[i][ii]*TG[i][ii])
        end
        @simd for ii = 1:length(BG[i])
            @inbounds GB[i][ii] = GB[i][ii] + (BG[i][ii]*BG[i][ii])
        end
    end
end

function updateParams!(T, B, TG, BG, GT, GB, eta)
    for i = 1:length(T)
        @simd for ii = 1:length(T[i])
            @inbounds T[i][ii] = T[i][ii] - eta*TG[i][ii]/(sqrt(GT[i][ii]))
        end
        @simd for ii = 1:length(B[i])
            @inbounds B[i][ii] = B[i][ii] - eta*BG[i][ii]/(sqrt(GB[i][ii]))
        end
    end
end

function updateParams!(T, B, TG, BG, GT, GB, eta, c)
    for i = 1:length(T)
        @simd for ii = 1:length(T[i])
            @inbounds T[i][ii] = T[i][ii] - eta*TG[i][ii]/(sqrt(GT[i][ii]))
        end
        @simd for ii = 1:length(B[i])
            @inbounds B[i][ii] = B[i][ii] - eta*BG[i][ii]/(sqrt(GB[i][ii]))
        end
    end
    
    #project Thetas onto l2 ball of radius c
    if c < Inf
    for i = 1:length(T)
        #calculate the squared sum of each row
        fs = zeros(Float32, size(T[i], 1))
        for k = 1:size(T[i], 2)
            @simd for j = 1:size(T[i], 1)
                @inbounds fs[j] = fs[j] + T[i][j, k].^2
            end
        end
        @simd for j = 1:length(fs)
            @inbounds fs[j] = sqrt(fs[j])
        end
        
        #rescale each row of T if the magnitude is too high
        for k = 1:size(T[i], 2)
            @simd for j = 1:length(fs)
                #if fs[j] > c
                #   @inbounds T[i][j, k] = c*T[i][j, k]/fs[j]
                #end
                @inbounds T[i][j, k] = min(1.0f0, c/fs[j]) * T[i][j, k]
            end
        end
    end
    end
end

function saveValid!(validT, validB, validGT, validGB, newT, newB, newGT, newGB)
#save last known valid training parameters  
    for i = 1:length(validB)
        @simd for ii = 1:length(validB[i])
            @inbounds validB[i][ii] = newB[i][ii]
            @inbounds validGB[i][ii] = newGB[i][ii]
        end
        @simd for ii = 1:length(validT[i])
            @inbounds validT[i][ii] = newT[i][ii]
            @inbounds validGT[i][ii] = newGT[i][ii]
        end
    end
end

function updateBest!(bestT, bestB, newT, newB)
#update best parameters when a new lowest cost is found 
    for i = 1:length(bestB)
        @simd for ii = 1:length(bestB[i])
            @inbounds bestB[i][ii] = newB[i][ii]
        end
        @simd for ii = 1:length(bestT[i])
            @inbounds bestT[i][ii] = newT[i][ii]
        end
    end
end

function host2GPU(hostvars, GPUvars)
#move data from host to GPU where hostvars is a tuple of Array{Array, 1}
#that are matched with the corresponding cuda datatype in the tuple GPUvars
l1 = length(hostvars)
l2 = length(GPUvars)
if l1 != l2
    error("trying to move incompatible data")
end
for i = 1:l1
    for j = 1:length(hostvars[i])
        GPUvars[i][j] = CudaArray(hostvars[i][j])
    end
end
end

function GPU2Host(hostvars, GPUvars)
#move data from GPU to host where hostvars is a tuple of Array{Array, 1}
#that are matched with the corresponding cuda datatype in the tuple GPUvars
l1 = length(hostvars)
l2 = length(GPUvars)
if l1 != l2
    error("trying to move incompatible data")
end
for i = 1:l1
    for j = 1:length(hostvars[i])
        hostvars[i][j] = to_host(GPUvars[i][j])
    end
end
end


function G2GPU(GT, GB, d_GT, d_GB)
#send training parameters to GPU
    for i = 1:length(T)
        d_GT[i] = CudaArray(GT[i])
        d_GB[i] = CudaArray(GB[i])
    end
end

function params2GPU(T, B, d_T, d_B)
#send training parameters to GPU
    for i = 1:length(T)
        d_T[i] = CudaArray(T[i])
        d_B[i] = CudaArray(B[i])
    end
end


function grads2Host(d_TG, d_BG, TG, BG)
    for i = 1:length(TG)
        TG[i] = to_host(d_TG[i])
        BG[i] = to_host(d_BG[i])
    end
end

function params2Host(d_T, d_B, T, B)
    for i = 1:length(T)
        T[i] = to_host(d_T[i])
        B[i] = to_host(d_B[i])
    end
end

function G2Host(d_GT, d_GB, GT, GB)
    for i = 1:length(GT)
        GT[i] = to_host(d_GT[i])
        GB[i] = to_host(d_GB[i])
    end
end


function checkNumGrad(lambda)
    m = 5
    input_layer_size = 3
    output_layer_size = 2
    X = float32(randn(m, input_layer_size))
    y = float32(randn(m, output_layer_size))
    
    hidden_layers = [5, 5]


    T0, B0 = initializeParams(input_layer_size, hidden_layers, output_layer_size)
    
    num_hidden = length(hidden_layers)
    tanh_grad_z = Array(Matrix{Float32}, num_hidden)
    for i = 1:num_hidden
        tanh_grad_z[i] = Array(Float32, m, hidden_layers[i])
    end

    a = Array(Matrix{Float32}, num_hidden+1)
    for i = 1:num_hidden
        a[i] = Array(Float32, m, hidden_layers[i])
    end
    a[end] = Array(Float32, m, output_layer_size)

    deltas = Array(Matrix{Float32}, num_hidden+1)

    for i = 1:length(deltas)
        deltas[i] = similar(a[i])
    end

    Theta_grads = similar(T0)
    for i = 1:length(Theta_grads)
        Theta_grads[i] = similar(T0[i])
    end


    Bias_grads = similar(B0)
    for i = 1:length(B0)
        Bias_grads[i] = similar(B0[i])
    end

    onesVec = ones(Float32, m)

    e = 0.0001f0
    params = theta2Params(B0, T0)
    l = length(params)
    perturb = zeros(Float32, l)
    numGrad = Array(Float32, l)

    nnCostFunction(T0, B0, input_layer_size, hidden_layers, X, y, lambda, Theta_grads, Bias_grads, tanh_grad_z, a, deltas, onesVec)
    funcGrad = theta2Params(Bias_grads, Theta_grads)

    for i = 1:l
        perturb[i] = e
        
        
        Tplus, Bplus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params+perturb)
        
        Tminus, Bminus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params-perturb)
        
        outminus = nnCostFunctionNOGRAD(Tminus, Bminus, input_layer_size, hidden_layers, X, y, lambda, a)
        outplus = nnCostFunctionNOGRAD(Tplus, Bplus, input_layer_size, hidden_layers, X, y, lambda, a)
        
        perturb[i] = 0.0f0  #restore perturb vector to 0

        numGrad[i] = (outplus - outminus)/(2.0f0*e)
        
    end
    cost = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, X, y, lambda, a)
    deltas = abs.(funcGrad - numGrad)
    
    println(string("Cost is ", cost))
    println([["Num Grads" "Func Grads"];[numGrad funcGrad]])
    newErr = norm(numGrad-funcGrad)/norm(numGrad + funcGrad)
    println(string("Relative differences are ", newErr, ".  Should be small (1e-9)"))
end

#=
function checkNumGrad(lambda, md)
    m = 5
    input_layer_size = 3
    X = float32(randn(m, input_layer_size))
    y = float32(randn(m))
    d_X = CudaArray(X)
    d_y = CudaArray(reshape(y, m, 1))

    fill_cols = CUDArt.CuFunction(md, "fill_cols")
    GPU_sign = CUDArt.CuFunction(md, "sign")
    elMul = CUDArt.CuFunction(md, "elMul")
    cudaTanhGrad = CUDArt.CuFunction(md, "tanhGradient")

    kernels = (fill_cols, GPU_sign, elMul, cudaTanhGrad)

    
    #create variables on GPU to modify inplace throughout training
    K = 32
    threads = (K, K)
    blocks(N, M) = (int(ceil(N/K)), int(ceil(M/K))) #function for launching CUDA kernels later based on matrix sizes


    hidden_layers = [5, 5]
    num_hidden = length(hidden_layers)

    T0, B0 = initializeParams(input_layer_size, hidden_layers, output_layer_size)

    d_Thetas = Array(CudaArray{Float32, 2}, length(T0))
    d_Biases = Array(CudaArray{Float32, 1}, length(T0))

    for i = 1:length(T0)
        d_Thetas[i] = CudaArray(T0[i])
        d_Biases[i] = CudaArray(B0[i])
    end


    Theta_grads = similar(T0)
    for i = 1:length(Theta_grads)
        Theta_grads[i] = similar(T0[i])
    end


    Bias_grads = similar(B0)
    for i = 1:length(B0)
        Bias_grads[i] = similar(B0[i])
    end


    d_Theta_grads = Array(CudaArray{Float32, 2}, length(B0))
    d_Bias_grads = Array(CudaArray{Float32, 1},length(B0))
    for i = 1:length(B0)
        d_Theta_grads[i] = CudaArray(Theta_grads[i])
        d_Bias_grads[i] = CudaArray(Bias_grads[i])
    end

    d_ones = CudaArray(ones(Float32, m))
    d_tanh_grad_z = Array(CudaArray{Float32, 2}, num_hidden)
    for i = 1:num_hidden
        d_tanh_grad_z[i] = CudaArray(Array(Float32, m, hidden_layers[i]))
    end
    d_a = Array(CudaArray{Float32, 2}, num_hidden+1)
    d_deltas = Array(CudaArray{Float32, 2}, num_hidden+1)
    for i = 1:num_hidden
        d_a[i] = CudaArray(Array(Float32, m, hidden_layers[i]))
        d_deltas[i] = CudaArray(Array(Float32, m, hidden_layers[i]))
    end
    d_a[end] = CudaArray(Array(Float32, m, 1))
    d_deltas[end] = CudaArray(Array(Float32, m, 1))



    numLayers = length(T0)

    
    a = Array(Matrix{Float32}, num_hidden+1)
    for i = 1:num_hidden
        a[i] = Array(Float32, m, hidden_layers[i])
    end
    a[end] = Array(Float32, m, 1)


    e = 0.0001f0
    params = theta2Params(B0, T0)
    l = length(params)
    perturb = zeros(Float32, l)
    numGrad = Array(Float32, l)

    nnCostFunction(d_Thetas, d_Biases, input_layer_size, hidden_layers, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, d_X, d_y,lambda, blocks, threads, kernels)

    grads2Host(d_Theta_grads, d_Bias_grads, Theta_grads, Bias_grads)

    funcGrad = theta2Params(Bias_grads, Theta_grads)

    for i = 1:l
        perturb[i] = e
        Tplus, Bplus = params2Theta(input_layer_size, hidden_layers, params+perturb)
        Tminus, Bminus = params2Theta(input_layer_size, hidden_layers, params-perturb)
        
        outminus = nnCostFunctionNOGRAD(Tminus, Bminus, input_layer_size, hidden_layers, X, y, lambda, a)
        outplus = nnCostFunctionNOGRAD(Tplus, Bplus, input_layer_size, hidden_layers, X, y, lambda, a)
        
        perturb[i] = 0.0f0  #restore perturb vector to 0

        numGrad[i] = (outplus - outminus)/(2.0f0*e)
    end

    
    println([["Num Grads" "GPU Grads"];[numGrad funcGrad]])
    GPUErr = norm(numGrad-funcGrad)/norm(numGrad + funcGrad)
    println(string("Relative differences for method are ", GPUErr, ".  Should be small (1e-9)"))
end
=#

function bootstrapTrainAdv(name, numEpochs, batchSize, hidden, eta, c, alpha, num, ID, printProg = true)
    println("reading and converting training data")
    X, Xtest, Y, Ytest = readInput(name)

    (N, M) = size(X)
    O = size(Y, 2)
    
    filename = string(name, "_", M, "_input_", hidden, "_hidden_", O, "_output_", c, "_maxNorm_", eta, "_advNoise_", alpha, "_alpha_AdvADAMAX", backend, ".csv")
        
    bootstrapOut = pmap(1:num) do foo
        #BLAS.set_num_threads(Sys.CPU_CORES)
        if (nprocs() > 1) & (backend == :CPU)
            BLAS.set_num_threads(max(1, ceil(Int64, Sys.CPU_CORES / min(nprocs(), num))))
        end
        T0, B0 = initializeParams(M, hidden, O) 
        bootstrapInd = ceil(Int64, N*rand(N))           
        (T, B, bestCost, costRecord, timeRecord) = eval(Symbol("ADAMAXTrainNN", backend))Adv(X[bootstrapInd, :], Y[bootstrapInd, :], batchSize, T0, B0, numEpochs, M, hidden, eta, c, alpha, printProgress = printProg)
        (theta2Params(B, T), T, B)
    end
    filedata = mapreduce(a -> a[1], hcat, bootstrapOut)
    writecsv(string(ID, "_bootstrapParams_", filename), filedata)
    
    #calculate average network output
    bootstrapOutTrain = map(a -> predict(a[2], a[3], X), bootstrapOut)  
    combinedOutputTrain = reduce(+, bootstrapOutTrain)/length(bootstrapOutTrain)
    errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, bootstrapOutTrain)/length(bootstrapOutTrain)) 
    Jtrain = mean(abs.(combinedOutputTrain - Y))
        
    bootstrapOutTest = map(a -> predict(a[2], a[3], Xtest), bootstrapOut)   
    combinedOutputTest = reduce(+, bootstrapOutTest)/length(bootstrapOutTest)
    errorEstTest = mean(mapreduce(a -> abs.(a - combinedOutputTest), +, bootstrapOutTest)/length(bootstrapOutTest)) 
    Jtest = mean(abs.(combinedOutputTest - Ytest))  
        
    writecsv(string(ID, "_bootstrapPerformance_", filename), [["Training Error", "Training Error Est", "Test Error", "Test Error Est "] [Jtrain, errorEstTrain, Jtest, errorEstTest]])      
end

function restoreValid!(bestThetas, bestBiases, best_GT, best_GB, Thetas, Biases, GT, GB)
    #restore training parameters to the last best known configuration
    for i = 1:length(bestBiases)
        @simd for j = 1:length(bestThetas[i])
            @inbounds Thetas[i][j] = bestThetas[i][j]
            @inbounds GT[i][j] = best_GT[i][j]
        end
        @simd for j = 1:length(bestBiases[i])
            @inbounds Biases[i][j] = bestBiases[i][j]
            @inbounds GB[i][j] = best_GB[i][j]
        end
    end
end

function generateBatches(input_data, output_data, batchsize)
    m = size(output_data, 1)
    if batchsize > m
        error("Your batchsize is larger than the total number of examples.")
    end
    
    numBatches = round(Int, ceil(m/batchsize))
    inputbatchData = Array(Matrix{Float32}, numBatches)
    outputbatchData = Array(Matrix{Float32}, numBatches)
    randInd = [shuffle(collect(1:m)) shuffle(collect(1:m))]
    for i = 1:numBatches
        inputbatchData[i] = input_data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
        outputbatchData[i] = output_data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
    end
    return (inputbatchData, outputbatchData)
end


function trainNN(input_data, output_data, batchsize, T0, B0, N, input_layer_size, hidden_layers, lambda, x...)
#train fully connected neural network with single floating point output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda
#Note that all floating point input variables must be float32 or single precision   
println(string("Training Neural Network with hidden layers ", hidden_layers, ", batch size ", batchsize, ", and lambda ", lambda))

#make sure output data is in vector form
(m, n) = size(input_data)
(m2, n2) = size(output_data)
if m2 != m 
    error("input and output data do not match")
end

#total number of examples in dataset
if batchsize > m
    error("Your batchsize is larger than the total number of examples.")
end

numBatches = round(Int, ceil(m/batchsize))


(inputbatchData, outputbatchData) = begin
    if isempty(x)
        generateBatches(input_data, output_data, batchsize)
    else
        x
    end
end

    
#create memory objects used in cost function
num_hidden = length(hidden_layers)
tanh_grad_zFULL = Array(Matrix{Float32}, num_hidden)
for i = 1:num_hidden
    tanh_grad_zFULL[i] = Array(Float32, m, hidden_layers[i])
end
tanh_grad_zBATCH = Array(Matrix{Float32}, num_hidden)
for i = 1:num_hidden
    tanh_grad_zBATCH[i] = Array(Float32, batchsize, hidden_layers[i])
end


aFULL = Array(Matrix{Float32}, num_hidden+1)
for i = 1:num_hidden
    aFULL[i] = Array(Float32, m, hidden_layers[i])
end
aFULL[end] = Array(Float32, m, n2)

aBATCH = Array(Matrix{Float32}, num_hidden+1)
for i = 1:num_hidden
    aBATCH[i] = Array(Float32, batchsize, hidden_layers[i])
end
aBATCH[end] = Array(Float32, batchsize, n2)

deltasFULL = Array(Matrix{Float32}, num_hidden+1)
deltasBATCH = Array(Matrix{Float32}, num_hidden+1)

for i = 1:length(deltasFULL)
    deltasFULL[i] = similar(aFULL[i])
    deltasBATCH[i] = similar(aBATCH[i])
end

Theta_grads = similar(T0)
for i = 1:length(Theta_grads)
    Theta_grads[i] = similar(T0[i])
end


Bias_grads = similar(B0)
for i = 1:length(B0)
    Bias_grads[i] = similar(B0[i])
end

onesVecFULL = ones(Float32, m)
onesVecBATCH = ones(Float32, batchsize)

numLayers = length(T0)



nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
currentOut = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, input_data, output_data, lambda, aFULL)


epoch = 1
eta = 0.1f0
minETA = 1.0f-2
maxETA = 1.0f0
scale = 0.5f0
bounce = false
lastChange = minETA
Thetas = deepcopy(T0)
Biases = deepcopy(B0)
GT = map(a-> Theta_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update
GB = map(a-> Bias_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update

validGT = deepcopy(GT)
validGB = deepcopy(GB)

validT = deepcopy(T0)
validB= deepcopy(B0)

period = 30
record = Array(Float32, round(Int, ceil(N+1/period)))
record[1] = currentOut

startTime = time()
lastReport = startTime

bestThetas = deepcopy(T0)
bestBiases = deepcopy(B0)
bestCost = currentOut
rollingAvgCost = currentOut

iter = 1

while epoch <= N
#while epoch <= N
    #run through an epoch in batches with randomized order
    for batch = 1:numBatches
        nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
        updateG!(Theta_grads, Bias_grads, GT, GB)
        updateParams!(Thetas, Biases, Theta_grads, Bias_grads, GT, GB, eta)
    end
    epoch += 1  

    if epoch%period == 0
        currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, input_data, output_data, lambda, aFULL)
        record[iter + 1] = currentOut
        if isnan(currentOut)
            println(string("on epoch ", epoch, " setting eta to ", minETA, " and  Restoring valid params "))
            eta = minETA
            restoreValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        elseif currentOut < bestCost
            if (eta < maxETA) && (scale < 0.95f0)
                println(string("on epoch ", epoch, " increasing eta from ", eta, " to ", eta/scale))
                eta = eta/scale
                bounce = true
            end
            bestCost = currentOut
            updateBest!(bestThetas, bestBiases, Thetas, Biases)
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        else
            println(string("on epoch ", epoch, " reducing eta from ", eta, " to ", scale*eta))
            eta = scale*eta
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
            #restoreValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        end
        scale = min(0.95f0, (0.9f0*scale + 0.1f0))
        iter += 1
    end
    
    
    
    if epoch%(30*period) == 0
        println(string("resetting eta to ", minETA))
        eta = minETA
        scale = 0.5f0
    end
    
    #=
    if epoch%(30*period) == 0
        if lastChange == minETA
            println(string("resetting eta to ", maxETA))
            eta = maxETA
            lastChange = maxETA
            scale = 0.5f0
        else
            println(string("resetting eta to ", minETA))
            eta = minETA
            lastChange = minETA
            scale = 0.5f0
        end
    end
    =#

    currentTime = time()
    #print status every 5 seconds
    
    if (currentTime - lastReport) >= 5
        elapsed = currentTime - startTime
        percentComplete = epoch/N
        totalTimeEst = elapsed / percentComplete
        timeRemainingEst = totalTimeEst - elapsed
        lastReport = currentTime
        hoursLeft = floor(timeRemainingEst/(60*60))
        minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
        secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
        println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
        println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
    end
    epoch += 1
end

currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, input_data, output_data, lambda, aFULL)
if currentOut < bestCost
    bestCost = currentOut
    updateBest!(bestThetas, bestBiases, Thetas, Biases)
end
println(string("Cost reduced from ", record[1], "to ", bestCost))       
return bestThetas, bestBiases, bestCost, record
end

function trainNNMaxNorm(input_data, output_data, batchsize, T0, B0, N, input_layer_size, hidden_layers, c, x...)
#train fully connected neural network with single floating point output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda
#Note that all floating point input variables must be float32 or single precision   
println(string("Training Neural Network with hidden layers ", hidden_layers, ", batch size ", batchsize, ", and maxnorm ", c))
lambda = 0.0f0
#make sure output data is in vector form
(m, n) = size(input_data)
(m2, n2) = size(output_data)
if m2 != m 
    error("input and output data do not match")
end

#total number of examples in dataset
if batchsize > m
    error("Your batchsize is larger than the total number of examples.")
end

numBatches = int(ceil(m/batchsize))


(inputbatchData, outputbatchData) = begin
    if isempty(x)
        generateBatches(input_data, output_data, batchsize)
    else
        x
    end
end

    
#create memory objects used in cost function
num_hidden = length(hidden_layers)
tanh_grad_zFULL = Array(Matrix{Float32}, num_hidden)
for i = 1:num_hidden
    tanh_grad_zFULL[i] = Array(Float32, m, hidden_layers[i])
end
tanh_grad_zBATCH = Array(Matrix{Float32}, num_hidden)
for i = 1:num_hidden
    tanh_grad_zBATCH[i] = Array(Float32, batchsize, hidden_layers[i])
end


aFULL = Array(Matrix{Float32}, num_hidden+1)
for i = 1:num_hidden
    aFULL[i] = Array(Float32, m, hidden_layers[i])
end
aFULL[end] = Array(Float32, m, n2)

aBATCH = Array(Matrix{Float32}, num_hidden+1)
for i = 1:num_hidden
    aBATCH[i] = Array(Float32, batchsize, hidden_layers[i])
end
aBATCH[end] = Array(Float32, batchsize, n2)

deltasFULL = Array(Matrix{Float32}, num_hidden+1)
deltasBATCH = Array(Matrix{Float32}, num_hidden+1)

for i = 1:length(deltasFULL)
    deltasFULL[i] = similar(aFULL[i])
    deltasBATCH[i] = similar(aBATCH[i])
end

Theta_grads = similar(T0)
for i = 1:length(Theta_grads)
    Theta_grads[i] = similar(T0[i])
end


Bias_grads = similar(B0)
for i = 1:length(B0)
    Bias_grads[i] = similar(B0[i])
end

onesVecFULL = ones(Float32, m)
onesVecBATCH = ones(Float32, batchsize)

numLayers = length(T0)



nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
currentOut = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, input_data, output_data, lambda, aFULL)


epoch = 1
eta = 0.1f0
minETA = 1.0f-2
maxETA = 1.0f0
scale = 0.5f0
bounce = false
lastChange = minETA
Thetas = deepcopy(T0)
Biases = deepcopy(B0)
GT = map(a-> Theta_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update
GB = map(a-> Bias_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update

validGT = deepcopy(GT)
validGB = deepcopy(GB)

validT = deepcopy(T0)
validB= deepcopy(B0)

period = 30
record = Array(Float32, int(ceil(N+1/period)))
record[1] = currentOut

startTime = time()
lastReport = startTime

bestThetas = deepcopy(T0)
bestBiases = deepcopy(B0)
bestCost = currentOut
rollingAvgCost = currentOut

iter = 1

while epoch <= N
#while epoch <= N
    #run through an epoch in batches with randomized order
    for batch = 1:numBatches
        nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
        updateG!(Theta_grads, Bias_grads, GT, GB)
        updateParams!(Thetas, Biases, Theta_grads, Bias_grads, GT, GB, eta, c)
    end
    epoch += 1  

    if epoch%period == 0
        currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, input_data, output_data, lambda, aFULL)
        record[iter + 1] = currentOut
        if isnan(currentOut)
            println(string("on epoch ", epoch, " setting eta to ", minETA, " and  Restoring valid params "))
            eta = minETA
            restoreValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        elseif currentOut < bestCost
            if (eta < maxETA) && (scale < 0.95f0)
                println(string("on epoch ", epoch, " increasing eta from ", eta, " to ", eta/scale))
                eta = eta/scale
                bounce = true
            end
            bestCost = currentOut
            updateBest!(bestThetas, bestBiases, Thetas, Biases)
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        else
            println(string("on epoch ", epoch, " reducing eta from ", eta, " to ", scale*eta))
            eta = scale*eta
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
            #restoreValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        end
        scale = min(0.95f0, (0.9f0*scale + 0.1f0))
        iter += 1
    end
    
    
    if epoch%(30*period) == 0
        println(string("resetting eta to ", minETA))
        eta = minETA
        scale = 0.5f0
    end
    
    #=
    if epoch%(30*period) == 0
        if lastChange == minETA
            println(string("resetting eta to ", maxETA))
            eta = maxETA
            lastChange = maxETA
            scale = 0.5f0
        else
            println(string("resetting eta to ", minETA))
            eta = minETA
            lastChange = minETA
            scale = 0.5f0
        end
    end
    =#

    currentTime = time()
    #print status every 5 seconds
    
    if (currentTime - lastReport) >= 5
        elapsed = currentTime - startTime
        percentComplete = epoch/N
        totalTimeEst = elapsed / percentComplete
        timeRemainingEst = totalTimeEst - elapsed
        lastReport = currentTime
        hoursLeft = floor(timeRemainingEst/(60*60))
        minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
        secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
        println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
        println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
    end
    epoch += 1
end

currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, input_data, output_data, lambda, aFULL)
if currentOut < bestCost
    bestCost = currentOut
    updateBest!(bestThetas, bestBiases, Thetas, Biases)
end
println(string("Cost reduced from ", record[1], "to ", bestCost))       
return bestThetas, bestBiases, bestCost, record
end

function trainNN(input_data, output_data, batchsize, T0, B0, N, input_layer_size, hidden_layer_size, lambda, md)
println(string("Training Neural Network with hidden layers ", hidden_layer_size, ", batch size ", batchsize, ", and lambda ", lambda))
#train fully connected neural network, using GPU backend, with single floating point output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda
#Note that all floating point input variables must be float32 or single precision   

#make sure output data is in vector form
(m, n) = size(input_data)
output_data = output_data[:] #force output data into vector form
if length(output_data) != m 
    error("input and output data do not match")
end

#total number of examples in dataset
l = length(output_data)


fill_cols = CUDArt.CuFunction(md, "fill_cols")
GPU_sign = CUDArt.CuFunction(md, "sign")
elMul = CUDArt.CuFunction(md, "elMul")
cudaTanhGrad = CUDArt.CuFunction(md, "tanhGradient")
CUDAupdateG = CUDArt.CuFunction(md, "updateG")
CUDAupdateParams = CUDArt.CuFunction(md, "updateParams")

kernels = (fill_cols, GPU_sign, elMul, cudaTanhGrad)

m = batchsize
#create variables on GPU to modify inplace throughout training
K = 32
threads = (K, K)
blocks(N, M) = (int(ceil(N/K)), int(ceil(M/K))) #function for launching CUDA kernels later based on matrix sizes

function updateG!(d_Theta_grads::Array{CudaArray{Float32, 2}, 1}, d_Bias_grads::Array{CudaArray{Float32, 1}, 1}, d_GT::Array{CudaArray{Float32, 2}, 1}, d_GB::Array{CudaArray{Float32, 1}, 1})
    for i = 1:length(d_Bias_grads)
        CUDArt.launch(CUDAupdateG, blocks(size(T0[i], 1), size(T0, 2)), threads, (size(T0[i], 1), size(T0[i], 2), d_Theta_grads[i], d_GT[i]))
        CUDArt.launch(CUDAupdateG, blocks(length(B0[i]), 1), threads, (length(B0[i]), 1, d_Bias_grads[i], d_GB[i]))
    end
end

function updateParams!(d_Thetas::Array{CudaArray{Float32, 2}, 1}, d_Biases::Array{CudaArray{Float32, 1}, 1}, d_Theta_grads::Array{CudaArray{Float32, 2}, 1}, d_Bias_grads::Array{CudaArray{Float32, 1}, 1}, d_GT::Array{CudaArray{Float32, 2}, 1}, d_GB::Array{CudaArray{Float32, 1}, 1}, eta::Float32)
    for i = 1:length(d_Biases)
        CUDArt.launch(CUDAupdateParams, blocks(size(T0[i], 1), size(T0[i], 2)), threads, (size(T0[i], 1), size(T0[i], 2), eta, d_Thetas[i], d_Theta_grads[i], d_GT[i]))
        CUDArt.launch(CUDAupdateParams, blocks(length(B0[i]), 1), threads, (length(B0[i]), 1, eta, d_Biases[i], d_Bias_grads[i], d_GB[i]))
    end
end



num_hidden = length(T0) - 1

d_Theta_grads = Array(CudaArray{Float32, 2}, num_hidden+1)
d_Theta_grads[1] = CudaArray(Float32, size(T0[1]))

if num_hidden > 1
    for i = 2:num_hidden+1
        d_Theta_grads[i] = CudaArray(Float32, size(T0[i]))
        
    end
end
d_Theta_grads[num_hidden + 1] = CudaArray(Float32, size(T0[num_hidden+1]))

d_ones = CudaArray(ones(Float32, m))
d_tanh_grad_z = Array(CudaArray{Float32, 2}, num_hidden)
for i = 1:num_hidden
    d_tanh_grad_z[i] = CudaArray(Array(Float32, m, hidden_layer_size[i]))
end
d_a = Array(CudaArray{Float32, 2}, num_hidden+1)
d_deltas = Array(CudaArray{Float32, 2}, num_hidden+1)
for i = 1:num_hidden
    d_a[i] = CudaArray(Array(Float32, m, hidden_layer_size[i]))
    d_deltas[i] = CudaArray(Array(Float32, m, hidden_layer_size[i]))
end
d_a[end] = CudaArray(Array(Float32, m, 1))
d_deltas[end] = CudaArray(Array(Float32, m, 1))

Theta_grads = similar(T0)
Bias_grads = similar(B0)
d_Bias_grads = Array(CudaArray{Float32, 1}, length(Bias_grads))
for i = 1:length(B0)
    Bias_grads[i] = similar(B0[i])
    d_Bias_grads[i] = CudaArray(Bias_grads[i])
end

aFULL = Array(Matrix{Float32}, num_hidden+1)
for i = 1:num_hidden
    aFULL[i] = Array(Float32, l, hidden_layer_size[i])
end
aFULL[end] = Array(Float32, l, 1)


currentOut = []

#define important variables
numLayers = length(T0)
d_Thetas = Array(CudaArray{Float32, 2}, numLayers)
d_Biases = Array(CudaArray{Float32, 1}, numLayers)

function calcL2Reg(Thetas, lambda)
    accum = 0.0f0
    for i = 1:length(Thetas)
        @simd for j = 1:length(Thetas[i])
            @inbounds accum += Thetas[i][j]*Thetas[i][j]
        end
    end
    accum = accum * lambda/(2.0f0*m)
end





#move initial parameters to GPU
#params2GPU(T0, B0, d_Thetas, d_Biases)
host2GPU((T0, B0), (d_Thetas, d_Biases))



#total number of examples in dataset
l = length(output_data)
if batchsize > l
    println("Your batchsize is larger than the total number of examples.")
    return false
end

uniqueBatches = int(ceil(2*l/batchsize))
batchInputs = Array(CudaArray{Float32, 2}, uniqueBatches)
batchOutputs = Array(CudaArray{Float32, 2}, uniqueBatches)
for i = 1:uniqueBatches
    batchInds = shuffle(collect(1:l))[1:batchsize]
    inputBatchData = input_data[batchInds, :]
    outputBatchData = output_data[batchInds, 1]
    batchInputs[i] = CudaArray(input_data[batchInds, :])
    batchOutputs[i] = CudaArray(reshape(output_data[batchInds], batchsize, 1))
end


nnCostFunction(d_Thetas, d_Biases, input_layer_size, hidden_layer_size, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, batchInputs[end], batchOutputs[end],lambda, blocks, threads, kernels)
currentOut = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layer_size, input_data, output_data, lambda, aFULL)

#grads2Host(d_Theta_grads, d_Bias_grads, Theta_grads, Bias_grads)

GPU2Host((Theta_grads, Bias_grads), (d_Theta_grads, d_Bias_grads))

GT = map(a-> Theta_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update
GB = map(a-> Bias_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update
d_GT = Array(CudaArray{Float32, 2}, numLayers)
d_GB = Array(CudaArray{Float32, 1}, numLayers)
#G2GPU(GT, GB, d_GT, d_GB)
host2GPU((GT, GB), (d_GT, d_GB))


numBatches = int(ceil(l/batchsize))
converged = false
epoch = 1
eta = 1.0f-3
maxETA = 10.0f0
minETA = 1.0f-6
bounce = false
Thetas = deepcopy(T0)
Biases = deepcopy(B0)
period = numBatches*100
scale = 0.5f0
record = Array(Float32, int(ceil((numBatches*N + 1)/period)))
record[1] = currentOut
absDev = 10
converged = false
startTime = time()
lastReport = startTime
lastChange = minETA

bestThetas = deepcopy(T0)
bestBiases = deepcopy(B0)

validT = deepcopy(T0)
validB = deepcopy(B0)
validGT = deepcopy(GT)
validGB = deepcopy(GB)

bestCost = currentOut
rollingAvgCost = currentOut
absDev = 100
L = 10000

batchNum = 1
iter = 1
iter2 = 1
while epoch <= N

    nnCostFunction(d_Thetas, d_Biases, input_layer_size, hidden_layer_size, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, batchInputs[batchNum], batchOutputs[batchNum],lambda, blocks, threads, kernels)

    updateG!(d_Theta_grads, d_Bias_grads, d_GT, d_GB)
    
    updateParams!(d_Thetas, d_Biases, d_Theta_grads, d_Bias_grads, d_GT, d_GB, eta)

    #check cost after a number of epochs specified in period
    if iter%period == 0
        #params2Host(d_Thetas, d_Biases, Thetas, Biases)
        GPU2Host((Thetas, Biases),(d_Thetas, d_Biases))
        currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layer_size, input_data, output_data, lambda, aFULL)
        if isnan(currentOut)
            println(string("on epoch ", epoch, " resetting eta to ", minETA))
            eta = minETA
            #params2GPU(validT, validB, validGT, validGB, d_Thetas, d_Biases, d_GT, d_GB)
            host2GPU((validT, validB, validGT, validGB), (d_Thetas, d_Biases, d_GT, d_GB))
        elseif currentOut < bestCost
            if (eta < maxETA) 
                println(string("on epoch ", epoch, " increasing eta from ", eta, " to ", min(eta/scale, maxETA)))
                eta = min(eta/scale, maxETA)
                
            end
            bestCost = currentOut
            updateBest!(bestThetas, bestBiases, Thetas, Biases)
            GPU2Host((GT, GB), (d_GT, d_GB))
            #G2Host(d_GT, d_GB, GT, GB)
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        else
            if eta > minETA
                println(string("on epoch ", epoch, " reducing eta from ", eta, " to ", max(minETA, scale*eta)))
                eta = max(minETA, scale*eta)
            else
                println(string("on epoch ", epoch, " resetting eta to ", 1.0f0))
                eta = 1.0f0
            end
            GPU2Host((GT, GB), (d_GT, d_GB))
            #G2Host(d_GT, d_GB, GT, GB)
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        end
        record[iter2 + 1] = currentOut
        iter2 += 1
        scale = min(0.95f0, (0.9f0*scale + 0.1f0))
    end

    
    #=
    if iter%(30*period) == 0
        if lastChange == minETA
            println(string("resetting eta to ", maxETA))
            eta = maxETA
            lastChange = maxETA
        else
            println(string("resetting eta to ", minETA))
            eta = minETA
            lastChange = minETA
        end
    end
    =#


    currentTime = time()
    #print status every 5 seconds
    
    if (currentTime - lastReport) >= 5
        elapsed = currentTime - startTime
        percentComplete = epoch/N
        totalTimeEst = elapsed / percentComplete
        timeRemainingEst = totalTimeEst - elapsed
        lastReport = currentTime
        hoursLeft = floor(timeRemainingEst/(60*60))
        minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
        secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
        println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
        println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
    end
    
    iter += 1
    epoch = int(ceil(iter*batchsize/l))
    batchNum += 1
    if batchNum > uniqueBatches
        batchNum = 1
    end
end
params2Host(d_Thetas, d_Biases, Thetas, Biases)
GPU2Host((Thetas, Biases), (d_Thetas, d_Biases))
currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layer_size, input_data, output_data, lambda, aFULL)
if currentOut < bestCost
    bestCost = currentOut
    updateBest!(bestThetas, bestBiases, Thetas, Biases)
end

println(string("Cost reduced from ", record[1], "to ", bestCost))       
return bestThetas, bestBiases, bestCost, record
end


function trainNNMaxNorm(input_data, output_data, batchsize, T0, B0, N, input_layer_size, hidden_layer_size, c, md)
println(string("Training Neural Network with hidden layers ", hidden_layer_size, ", batch size ", batchsize, ", and max norm ", c))
#train fully connected neural network, using GPU backend, with single floating point output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda
#Note that all floating point input variables must be float32 or single precision   

#make sure output data is in vector form
(m, n) = size(input_data)
output_data = output_data[:] #force output data into vector form
if length(output_data) != m 
    error("input and output data do not match")
end

#total number of examples in dataset
l = length(output_data)
lambda = 0.0f0

fill_cols = CUDArt.CuFunction(md, "fill_cols")
GPU_sign = CUDArt.CuFunction(md, "sign")
elMul = CUDArt.CuFunction(md, "elMul")
cudaTanhGrad = CUDArt.CuFunction(md, "tanhGradient")
CUDAupdateG = CUDArt.CuFunction(md, "updateG")
CUDAupdateParams = CUDArt.CuFunction(md, "updateParams")

kernels = (fill_cols, GPU_sign, elMul, cudaTanhGrad)

m = batchsize
#create variables on GPU to modify inplace throughout training
K = 32
threads = (K, K)
blocks(N, M) = (int(ceil(N/K)), int(ceil(M/K))) #function for launching CUDA kernels later based on matrix sizes


num_hidden = length(T0) - 1

d_Theta_grads = Array(CudaArray{Float32, 2}, num_hidden+1)
d_Theta_grads[1] = CudaArray(Float32, size(T0[1]))

if num_hidden > 1
    for i = 2:num_hidden+1
        d_Theta_grads[i] = CudaArray(Float32, size(T0[i]))
        
    end
end
d_Theta_grads[num_hidden + 1] = CudaArray(Float32, size(T0[num_hidden+1]))

d_ones = CudaArray(ones(Float32, m))
d_tanh_grad_z = Array(CudaArray{Float32, 2}, num_hidden)
for i = 1:num_hidden
    d_tanh_grad_z[i] = CudaArray(Array(Float32, m, hidden_layer_size[i]))
end
d_a = Array(CudaArray{Float32, 2}, num_hidden+1)
d_deltas = Array(CudaArray{Float32, 2}, num_hidden+1)
for i = 1:num_hidden
    d_a[i] = CudaArray(Array(Float32, m, hidden_layer_size[i]))
    d_deltas[i] = CudaArray(Array(Float32, m, hidden_layer_size[i]))
end
d_a[end] = CudaArray(Array(Float32, m, 1))
d_deltas[end] = CudaArray(Array(Float32, m, 1))

Theta_grads = similar(T0)
Bias_grads = similar(B0)
d_Bias_grads = Array(CudaArray{Float32, 1}, length(Bias_grads))
for i = 1:length(B0)
    Bias_grads[i] = similar(B0[i])
    d_Bias_grads[i] = CudaArray(Bias_grads[i])
end

aFULL = Array(Matrix{Float32}, num_hidden+1)
for i = 1:num_hidden
    aFULL[i] = Array(Float32, l, hidden_layer_size[i])
end
aFULL[end] = Array(Float32, l, 1)


currentOut = []

#define important variables
numLayers = length(T0)
d_Thetas = Array(CudaArray{Float32, 2}, numLayers)
d_Biases = Array(CudaArray{Float32, 1}, numLayers)

function calcL2Reg(Thetas, lambda)
    accum = 0.0f0
    for i = 1:length(Thetas)
        @simd for j = 1:length(Thetas[i])
            @inbounds accum += Thetas[i][j]*Thetas[i][j]
        end
    end
    accum = accum * lambda/(2.0f0*m)
end





#move initial parameters to GPU
#params2GPU(T0, B0, d_Thetas, d_Biases)
host2GPU((T0, B0), (d_Thetas, d_Biases))



#total number of examples in dataset
l = length(output_data)
if batchsize > l
    println("Your batchsize is larger than the total number of examples.")
    return false
end

uniqueBatches = int(ceil(2*l/batchsize))
batchInputs = Array(CudaArray{Float32, 2}, uniqueBatches)
batchOutputs = Array(CudaArray{Float32, 2}, uniqueBatches)
for i = 1:uniqueBatches
    batchInds = shuffle(collect(1:l))[1:batchsize]
    inputBatchData = input_data[batchInds, :]
    outputBatchData = output_data[batchInds, 1]
    batchInputs[i] = CudaArray(input_data[batchInds, :])
    batchOutputs[i] = CudaArray(reshape(output_data[batchInds], batchsize, 1))
end


nnCostFunction(d_Thetas, d_Biases, input_layer_size, hidden_layer_size, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, batchInputs[end], batchOutputs[end],lambda, blocks, threads, kernels)
currentOut = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layer_size, input_data, output_data, lambda, aFULL)

#grads2Host(d_Theta_grads, d_Bias_grads, Theta_grads, Bias_grads)

GPU2Host((Theta_grads, Bias_grads), (d_Theta_grads, d_Bias_grads))

GT = map(a-> Theta_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update
GB = map(a-> Bias_grads[a].^2 + 1.0f-15, 1:numLayers) #note add small constant to avoid dividing by 0 in update
d_GT = Array(CudaArray{Float32, 2}, numLayers)
d_GB = Array(CudaArray{Float32, 1}, numLayers)
#G2GPU(GT, GB, d_GT, d_GB)
host2GPU((GT, GB), (d_GT, d_GB))


numBatches = int(ceil(l/batchsize))
converged = false
epoch = 1
eta = 1.0f-2
maxETA = 0.2f0
minETA = 1.0f-3
bounce = false
Thetas = deepcopy(T0)
Biases = deepcopy(B0)
period = numBatches*100
scale = 0.5f0
record = Array(Float32, int(ceil((numBatches*N + 1)/period)))
record[1] = currentOut
absDev = 10
converged = false
startTime = time()
lastReport = startTime
lastChange = minETA

bestThetas = deepcopy(T0)
bestBiases = deepcopy(B0)

validT = deepcopy(T0)
validB = deepcopy(B0)
validGT = deepcopy(GT)
validGB = deepcopy(GB)

bestCost = currentOut
rollingAvgCost = currentOut
absDev = 100
L = 10000

batchNum = 1
iter = 1
iter2 = 1
while epoch <= N

    nnCostFunction(d_Thetas, d_Biases, input_layer_size, hidden_layer_size, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, batchInputs[batchNum], batchOutputs[batchNum],lambda, blocks, threads, kernels)
    GPU2Host((Theta_grads, Bias_grads), (d_Theta_grads, d_Bias_grads))

    updateG!(Theta_grads, Bias_grads, GT, GB)
    updateParams!(Thetas, Biases, Theta_grads, Bias_grads, GT, GB, eta, c)
    
    host2GPU((Thetas, Biases), (d_Thetas, d_Biases))    


    #check cost after a number of epochs specified in period
    if iter%period == 0
        #params2Host(d_Thetas, d_Biases, Thetas, Biases)
        GPU2Host((Thetas, Biases),(d_Thetas, d_Biases))
        currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layer_size, input_data, output_data, lambda, aFULL)
        if isnan(currentOut)
            println(string("on epoch ", epoch, " resetting eta to ", minETA))
            eta = minETA
            #params2GPU(validT, validB, validGT, validGB, d_Thetas, d_Biases, d_GT, d_GB)
            host2GPU((validT, validB, validGT, validGB), (d_Thetas, d_Biases, d_GT, d_GB))
        elseif currentOut < bestCost
            if (eta < maxETA) 
                println(string("on epoch ", epoch, " increasing eta from ", eta, " to ", min(eta/scale, maxETA)))
                eta = min(eta/scale, maxETA)
            end
            bestCost = currentOut
            updateBest!(bestThetas, bestBiases, Thetas, Biases)
            #GPU2Host((GT, GB), (d_GT, d_GB))
            #G2Host(d_GT, d_GB, GT, GB)
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        else
            if eta > minETA
                println(string("on epoch ", epoch, " reducing eta from ", eta, " to ", max(minETA, scale*eta)))
                eta = max(minETA, scale*eta)
                bounce = true
            else
                println(string("on epoch ", epoch, " resetting eta to ", 0.2f0))
                eta = 0.2f0
            end
            #GPU2Host((GT, GB), (d_GT, d_GB))
            #G2Host(d_GT, d_GB, GT, GB)
            saveValid!(validT, validB, validGT, validGB, Thetas, Biases, GT, GB)
        end
        record[iter2 + 1] = currentOut
        iter2 += 1
        scale = min(0.95f0, (0.9f0*scale + 0.1f0))
    end

    
    
    if iter%(30*period) == 0
        if lastChange == minETA
            println(string("resetting eta to ", maxETA))
            eta = maxETA
            lastChange = maxETA
            scale = 0.5f0
        else
            println(string("resetting eta to ", minETA))
            eta = minETA
            lastChange = minETA
            scale = 0.5f0
        end
    end
    


    currentTime = time()
    #print status every 5 seconds
    
    if (currentTime - lastReport) >= 5
        elapsed = currentTime - startTime
        percentComplete = epoch/N
        totalTimeEst = elapsed / percentComplete
        timeRemainingEst = totalTimeEst - elapsed
        lastReport = currentTime
        hoursLeft = floor(timeRemainingEst/(60*60))
        minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
        secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
        println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
        println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
    end
    
    iter += 1
    epoch = int(ceil(iter*batchsize/l))
    batchNum += 1
    if batchNum > uniqueBatches
        batchNum = 1
    end
end
#params2Host(d_Thetas, d_Biases, Thetas, Biases)
GPU2Host((Thetas, Biases), (d_Thetas, d_Biases))
currentOut = nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layer_size, input_data, output_data, lambda, aFULL)
if currentOut < bestCost
    bestCost = currentOut
    updateBest!(bestThetas, bestBiases, Thetas, Biases)
end

println(string("Cost reduced from ", record[1], "to ", bestCost))       
return bestThetas, bestBiases, bestCost, record
end
=#

    
    