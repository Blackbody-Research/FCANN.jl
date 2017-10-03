##-------------------REQUIRED FUNCTION DEFINITIONS--------------------------------------
@everywhere include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

function phi(x1, x2)
	#euclidean distance between x1 and x2 cubed
	norm(x1 .- x2, 2)^3
end

#calculate phi matrix using an input of previous hyperparamter vectors
function calcPhiMat(Xs)
    n = length(Xs)
    mapreduce(vcat, 1:n) do i
        map(j -> phi(Xs[i], Xs[j]), 1:n)'
    end            
end

function formP(Xs)
    n = length(Xs)
    mapreduce(vcat, 1:n) do i
        [Xs[i]' 1.0]
    end
end

function S(x, Xs, lambda, b, a)
#surrogate model output for hyperparameter vector input x
#requires knowing current list of n hyperparameter vectors and other
#model parameters lambda (vector of n constants), b (vector of D constants), 
#and a (constant) with n hyperparameter vector inputs
    n = length(Xs)
    D = length(Xs[1])
    mapreduce(+, 1:n) do i
        lambda[i]*phi(x, Xs[i]) + dot(b', x) + a
    end
end

function calcPhiN(D, n, n0, Nmax)
    phi0 = min(20/D, 1)
    phi0*(1 - log(n - n0+1)/log(Nmax - n0))
end

function fillOmegaN(xbest, phiN, varN)
    D = length(xbest)
    m = 100*D

    #generate m new candidate points
    Ys = map(1:m) do _
        #select coordinates to perturb with probability phiN and perterb sampled from a normal
        #distribution with 0 mean and variance of varN
        p = randn(D)*sqrt(varN).*(rand(D) .<= phiN)

        #ensure new x values are above 0
        xnew = map(x -> max(0, x), xbest .+ p)
    end
end

function mapRange(x, ymin, ymax)
#map an input value from 0 to 1 to an output range from ymin to ymax
#using linear scaling
    d = ymax - ymin
    y = x*d + ymin
end

function mapRangeInv(x, ymin, ymax)
#map an input value from ymin to ymax to an output range of 0 to 1
#using linear scaling
    d = ymax - ymin
    y = (x - ymin)*1/d
end


#run training program with hyperparamter vector X, h identifies which parameters are being tuned,
#conv is an array of conversion functions to generate the proper parameters, ranges is an array of 
#tuples containing the minimum and maximum value for each parameter, defaults is an array
#of 6 elements containing the default values to be used for hyperparameters not being tuned
function f(X, Xtest, Y, Ytest, rawParams, pconv, batchSize, OPT_PARAMS)
    printProg = false
    M = size(X, 2)
    O = size(Y, 2)

    #vector of indices that contain tuples => these parameters will be tuned
    h = find(p -> issubtype(typeof(p), Tuple), OPT_PARAMS)

    paramsDict = Dict(zip(h, rawParams))

    #scaledParams = map((a, b) -> mapRange(a, b[1], b[2]), rawParams, OPT_PARAMS[h])

    #vector of remaining indices not being tuned
    ih = setdiff(1:5, h)
    
    #generate full set of training parameters to be used properly converted
    params = map(1:5) do p
        if in(p, h)
            #rescale parameter from 0 to 1 range into specified range
            scaledParam = mapRange(paramsDict[p], OPT_PARAMS[p][1], OPT_PARAMS[p][2])
            pconv[p](scaledParam)
        else
            pconv[p](OPT_PARAMS[p])
        end
    end

    N = params[1] #number of training epochs
    numParams =params[2] #target number of training parameters
    L = params[3] #number of layers
    c = params[4] #max norm reg constant
    ensnum = params[5] #number of ensemble networks

    parameterNames = ["N", "Target_Params", "Layers", "Max_Norm", "Num_Networks"]

    #form hidden layers
    hidden_layers = ones(Int64, L)*ceil(Int64, getHiddenSize(M, O, L, numParams))
    
    println()
    if !isempty(ih)
        println(string("Using the following fixed hyper parameters : ", mapreduce(i -> string(parameterNames[i], " = ", params[i], ", "), (a, b) -> string(a, b), ih)))
    end
    println(string("Setting the following hyper parameters : ", mapreduce(i -> string(parameterNames[i], " = ", params[i], ", "), (a, b) -> string(a, b), h)))
    println()

    if ensnum > 1
    #do multitrain
        
        bootstrapOut = pmap(1:ensnum) do foo
            #BLAS.set_num_threads(Sys.CPU_CORES)
            BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_CORES/min(nprocs(), ensnum)))))
            srand(1234 + foo - 1)
            T0, B0 = initializeParams(M, hidden_layers, O)         
            alpha, R, (T, B, bestCost, record, timeRecord, GFLOPS), success = autoTuneParams(X, Y, batchSize, T0, B0, N, hidden_layers; lambda = 0.0f0, c = Inf, dropout = 0.0f0)
        end
        #calculate average network output
        bootstrapOutTrain = map(a -> predict(a[1], a[2], X), bootstrapOut)  
        combinedOutputTrain = reduce(+, bootstrapOutTrain)/length(bootstrapOutTrain)
        errorEstTrain = mean(mapreduce(a -> abs(a - combinedOutputTrain), +, bootstrapOutTrain)/length(bootstrapOutTrain))  
        Jtrain = mean(abs(combinedOutputTrain - Y))
            
        bootstrapOutTest = map(a -> predict(a[1], a[2], Xtest), bootstrapOut)   
        combinedOutputTest = reduce(+, bootstrapOutTest)/length(bootstrapOutTest)
        errorEstTest = mean(mapreduce(a -> abs(a - combinedOutputTest), +, bootstrapOutTest)/length(bootstrapOutTest))  
        Jtest = mean(abs(combinedOutputTest - Ytest))   

        numParams = length(theta2Params(bootstrapOut[1][2], bootstrapOut[1][1]))
        (Jtest, params, [numParams hidden_layers[1] Jtrain])
    else
    #do single train 
        BLAS.set_num_threads(0)
        srand(1234)
        T0, B0 = initializeParams(M, hidden_layers, O)
        alpha, R, (T, B, bestCost, record, timeRecord, GFLOPS), success = autoTuneParams(X, Y, batchSize, T0, B0, N, hidden_layers; lambda = 0.0f0, c = Inf, dropout = 0.0f0)

        outTrain = predict(T, B, X)
        outTest = predict(T, B, Xtest)

        Jtrain = mean(abs(outTrain - Y))
        Jtest = mean(abs(outTest - Ytest))

        numParams = length(theta2Params(B, T))

        (Jtest, params, [alpha R numParams hidden_layers[1] Jtrain])
    end
end


#-------------------------SET UP VARIABLES------------------------------------

#list of potential hyperparameters to tune given a fixed target number of parameters
#   1. num epochs from 100 to 1000
#   2. target num params from 100 to 1000 (will change with problem)
#   3. num layers from 1 to 10
#   4. max norm constant inverse from 0 (c = Inf) to 2 (c = 0.5...)
#   5. number of ensemble networks to use from 1 to 100

##----------------------------------------------------

function HORDEvalLayers(name, OPT_PARAMS, trialID, Nmax, ISP = [])
    parameterNames = ["N", "Target_Params", "Layers", "Max_Norm", "Num_Networks"]

    #vector of functions to ensure parameters are of the correct type
    pconvert = [a -> round(Int64, a), a -> round(Int64, a), a -> round(Int64, a), a -> Float32(1/a), a -> round(Int64, a)]

    #vector of indices that contain tuples => these parameters will be tuned
    h = find(p -> issubtype(typeof(p), Tuple), OPT_PARAMS)

    #vector of remaining indices not being tuned
    ih = setdiff(1:5, h)

    #scale the ISP to the 0 to 1 X range
    ISP_X = if isempty(ISP)
        []
    else
        map((a, b) -> mapRangeInv(a, b[1], b[2]), ISP[h],  OPT_PARAMS[h])
    end

    #define defaults for parameters not being tuned
    #defaults = [0.002f0, 0.1f0, 500, 200, 4, 0.0f0, 1]

    #ranges = [(0.0, 0.1), (0.0, 0.2), (100, 1000), (100, 1000), (1, 10), (0, 2), (1, 10)]


    #--------Predefined Variables----------------------

    println()
    println(string("On trial ", trialID, " tuning the following hyperparameters: ", mapreduce(a -> string(parameterNames[a], ", "), (a, b) -> string(a, b), h)))
    if !isempty(ih)
        println(string("Keeping the folowing hyperparameters fixed: ", mapreduce(a -> string(parameterNames[a], " = ", pconvert[a](OPT_PARAMS[a]), ", "), (a, b) -> string(a, b), ih)))
    end
    println()

    #target number of parameters in ANN
    #numParams = 1000

    #training batch size
    batchSize = 1024



    #string that contains the fixed values for the HORD training
    ihNames = if isempty(ih)
        ""
    else
        mapreduce(a -> string(pconvert[a](OPT_PARAMS[a]), "_", parameterNames[a], "_"), (a, b) -> string(a, b), ih)
    end

    fixedName = string(batchSize, "_batchSize_", ihNames)
    println(string("Training with batch size = ", batchSize))
    # println()
    # println(string("Training with the following fixed variables : ", fixedName))
    # println()



    #number of hyperparameters to tune
    D = length(h)

    #initial number of configurations to try
    n0 = 2*(D + 1)

    #number of candidate points to consider each step
    m = 100*D

    #weight balance
    w = 0.3

    #variance for weight perterbations
    varN = 0.2

    #number of concecutive failed iterations
    Tfail = 0

    #number of concecutive successful iterations
    Tsucc = 0

    #generate a latin hypercube sample of n0 points using and interval of 0 to 1
    #divided into n0 sections
    paramVec = linspace(0, 1, n0)
    # paramVecs = map(h) do p
    #     #range for this hyperparameter as a tuple
    #     r = ranges[p]

    #     #create vector of possible sample points 
    #     linspace(r[1], r[2], n0)
    # end

    ##----------------------ALGORITHM INITIALIZATION------------------------------------
    #for each coordinate generate a list of n0 unique values to sample from -1 to 1
    println(string("Generating ", n0, " initial parameter vectors"))
    sampleVecs = map(a -> randperm(n0), 1:D)
    Xs = map(1:n0) do i
        #take the ith element of each sample vec so that once an element has been used
        #it will not appear in any other point
        map(1:D) do j
            v = sampleVecs[j]   
            paramVec[v[i]]
        end
    end

    (Xs, n0) = if isempty(ISP_X)
        (Xs, n0)
    else
        println("Appending initial starting point to parameter vectors")
        ([[ISP_X]; Xs], n0+1)
    end

    println("reading and converting training data")
    X, Xtest, Y, Ytest = readInput(name)

    println()
    println("Training initial ", n0, " networks")
    println()
    #generate initial results of Xs parameter samples
    F0 = map(x -> f(X, Xtest, Y, Ytest, x, pconvert, batchSize, OPT_PARAMS), Xs)

    #extract current training errors which we are trying to minimize
    Fs = map(x -> x[1], F0)

    #extract the scaled training hyper parameters
    params = map(x -> x[2], F0)

    #extract the other output variables 
    outputs = map(x -> x[3], F0)

    #initial number of configurations
    n = n0

    ##---------------------ALGORITHM LOOP----------------------------------------------
    #also stop after being stuck on the same parameter set 3 times in a row
    while (n < Nmax) & (Tfail < max(5, D)*3) #& (sum(abs.(2*params[end] - params[end-1] - params[end-2])) > 0)
        println()
        println(string("Updating surrogate model on iteration ", n, " out of ", Nmax))
        println()

        PHI = calcPhiMat(Xs)
        P = formP(Xs)

        MAT1 = [PHI P; P' zeros(D+1, D+1)]
        VEC = [Fs; zeros(D+1)]

        #interpolated paramters
        c = pinv(MAT1) * VEC

        lambda = c[1:n]
        b = c[n+1:end-1]
        a = c[end]

        (fbest, indbest) = findmin(Fs)
        xbest = Xs[indbest]

        println()
        println(string("Current lowest test set error is ", fbest, " from iteration ", indbest, " using the following configuration:"))
        if !isempty(ih)
            println(string("Fixed hyper parameters:", mapreduce(i -> string(parameterNames[i], " = ", params[indbest][i], ", "), (a, b) -> string(a, b), ih)))
        end
        println(string("Tuned hyper parameters:", mapreduce(i -> string(parameterNames[i], " = ", params[indbest][i], ", "), (a, b) -> string(a, b), h)))
        println()

        phiN = calcPhiN(D, n, n0, Nmax)

        #calculate candidate points
        T = fillOmegaN(xbest, phiN, varN)

        #calculate surrogate values for each candidate point
        surrogateValues = map(t -> S(t, Xs, lambda, b, a), T)
        smax = maximum(surrogateValues)
        smin = minimum(surrogateValues)

        #calculate distances from the previously evaluated points for each surrogate point and select the minimum distance
        deltas = map(T) do t
            delts = map(Xs) do x
                norm(t .- x)
            end

            minimum(delts)
        end

        deltaMax = maximum(deltas)
        deltaMin = minimum(deltas)

        #estimated value scores for candidate points
        Vev = if smax == smin
            ones(length(T))
        else
            map(s -> (s - smin)/(smax - smin), surrogateValues) 
        end

        #distance metric scores for candidate points
        Vdm = if deltaMax == deltaMin
            ones(length(T))
        else
            map(d -> (deltaMax - d)/(deltaMax - deltaMin), deltas)
        end

        #final weighted score for candidate points
        W = w*Vev .+ (1-w)*Vdm

        #cyclically permute through weights
        w = if w == 0.3
            0.5
        elseif w == 0.5
            0.8
        elseif w == 0.8
            0.95
        elseif w == 0.95
            0.3
        end

        #select the point that has the lowest score to add as a new configuration
        (score, ind) = findmin(W)
        xnew = T[ind]

        println()
        println("Training with newly selected configuration")
        println()
        #calculate ANN error with new parameter configuration
        (fNew, paramsNew, outputNew) = f(X, Xtest, Y, Ytest, xnew, pconvert, batchSize, OPT_PARAMS)

        #iterate function evaluation counter
        n += 1

        #update Tsucc, Tfail based on results
        (Tsucc, Tfail) = if fNew < fbest
            println()
            println(string("New configuration has a new lowest test set error of ", fNew))
            println()
            (Tsucc + 1, 0)
        else
            println()
            println(string("New configuration has a worse test set error of ", fNew))
            println()
            (0, Tfail + 1)
        end

        #update perturbation variance if needed
        varN = if Tsucc >= 3
            min(0.2, varN*2)
        elseif Tfail >= max(5, D)
            println()
            println(string("Number of consecutive failed iterations = ", Tfail))
            println()
            min(varN/2, 0.005)
        else
            varN
        end

        println(string("Updated perturbation variance is ", varN))

        #update Fs, Xs, parameter vectors, and outputs 
        Fs = [Fs; fNew]
        Xs = [Xs; [xnew]]
        params = [params; [paramsNew]]
        outputs = [outputs; [outputNew]]
    end

    header = [parameterNames[h]' "Optimal Alpha" "Optimal R" "Actual Num Params" "Hidden Layer Size" "Training Error" "Test Error"]
    body = [mapreduce(a -> a[h]', vcat, params) reduce(vcat, outputs) Fs]

    writecsv(string(name, "_", fixedName, "_HORD_smartEvalLayers_results_trial_", trialID, ".csv"), [header; body])

    HORDHistory = [mapreduce(x -> x', vcat, Xs) Fs]
    writecsv(string(name, "_", fixedName, "_HORDHistory_smartEvalLayers_trial_", trialID, ".csv"), HORDHistory)
    println("Done writing results to file")
end



