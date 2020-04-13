#include minibatch stochastic gradient descent ADAMAX algorithm which includes
#function to read train and test sets with a specified name.  Forming the batches
#is also part of the ADAMAX algorithm
using Distributed

include("ADAMAXTRAIN_FCN_NFLOATOUT.jl")

if !isempty(cuda_versions)
	include(joinpath("cuda", "ADAMAXTRAINGPU_FCN_ABSERR_NFLOATOUT.jl"))
end

# dispatch to output calculation for proper backend, the GPU backend version will crash
# with any cost function other than "absErr"
function calcOutput(input_data, output_data, T, B; dropout = 0.0f0, costFunc = "absErr", resLayers = 0, autoencoder=false)
	eval(Symbol("calcOutput", backend))(input_data, output_data, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers)
end

function calcOutput(input_data, T, B; dropout = 0.0f0, costFunc = "absErr", resLayers = 0, autoencoder = true)
	eval(Symbol("calcOutput", backend))(input_data, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers, autoencoder = autoencoder)
end

function calcMultiOut(input_data, output_data, multiParams; dropout = 0.0f0, costFunc = "absErr", resLayers=0)
	eval(Symbol("calcMultiOut", backend))(input_data, output_data, multiParams, dropout = dropout, costFunc = costFunc, resLayers=resLayers)
end

#form file suffix string based on training parameters, ignore those that do not impact training
#order of function inputs are lambda/L2 Reg factor, c/max norm reg factor, training decay rate, dropout rate
function formIndicString(R, L2, c, D)
	s1 = if L2 == 0.0
		""
	else
		string(L2, "_L2_")
	end

	s2 = if c == Inf
		""
	else
		string(round(c, digits=3), "_maxNorm_")
	end

	s3 = if R == 0.0
		""
	else
		string(round(R, digits=3), "_decay_")
	end

	s4 = if D == 0.0
		""
	else
		string(round(D, digits=3), "_dropout_")
	end
	string(s3, s1, s2, s4)
end

function formIndicString(R, L2, c, D, res)
	s1 = if L2 == 0.0
		""
	else
		string(L2, "_L2_")
	end

	s2 = if c == Inf
		""
	else
		string(round(c, digits=3), "_maxNorm_")
	end

	s3 = if R == 0.0
		""
	else
		string(round(R, digits=3), "_decay_")
	end

	s4 = if D == 0.0
		""
	else
		string(round(D, digits=3), "_dropout_")
	end

	s5 = if res == 0
		""
	else
		string(res, "_resLayers_")
	end
	string(s3, s1, s2, s4, s5)
end

"""
	archEval(name, N, batchSize, hiddenList, alpha = 0.002f0)

Train neural networks with architectures specified in 'hiddenList' computing the training and test set
errors for each architecture and saving results to file.

# Arguments
* 'name::String' : name prefix of the training and test sets, also used to name output file
* 'N::Int' : Number of epochs to train where each epoch is a full iteration through the test set
* 'batchSize::Int' : Number of examples in each minibatch
* 'hiddenList::Array{Array{Int, 1}, 1}' : List of network structures each of which is a list of number of neurons per hidden layer
* 'alpha::Float32 = 0.001f0' : ADAMAX hyperparameter 

# Additional Details
Training uses the ADAMAX algorithm which is a variation of stochastic gradient descent with mini batches.
Training and test set errors are calculated using mean absolute error and saved to file along with the 
number of layers and total number of parameters for each network.  The top data line of the file also contains
errors for a simple linear regression using the same input variables.  Each network architecture is 
specified with an array of integers that list the number of neurons in each hidden layer.  For example [4, 4]
would be a network with two hidden layers each with 4 neurons.  The input and output layer sizes are automatically
determined by the size of the training and test set.  

"""
function archEval(name, N, batchSize, hiddenList; alpha = 0.002f0, costFunc = "absErr", binInput = false, reslayers = 0)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end

	M = size(X, 2)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	filename = string(name, "_", M, "_input_", O, "_output_ADAMAX", backend, "_", costFunc, ".csv")

	baseheader = ["Layers" "Num Params" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error")]
	exheader = [string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error")]
	header = (costFunc2 == costFunc) ? baseheader : hcat(baseheader, exheader)
		

	body = reduce(vcat, pmap(hiddenList) do hidden
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(hiddenList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		println(string("training network with ", hidden, " hidden layers"))
		println("initializing network parameters")
		Random.seed!(1234)
		T0, B0 = if occursin("Log", costFunc)
			initializeparams_saxe(M, hidden, 2*O)
		else
			initializeparams_saxe(M, hidden, O)
		end	

		println("beginning training")
		Random.seed!(1234)
		T, B, bestCost, record = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, N, M, hidden, 0.0f0, Inf, alpha=alpha, printProgress = true, costFunc = costFunc, resLayers=reslayers)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, costFunc = costFunc)


		numParams = length(theta2Params(B, T))
		if costFunc2 == costFunc
			[length(hidden) numParams Jtrain Jtest]
		else
			[length(hidden) numParams Jtrain[1] Jtest[1] Jtrain[2] Jtest[2]]
		end
	end)
	
	
	if isfile(string("archEval_", filename))
		f = open(string("archEval_", filename), "a")
		writedlm(f, body)
		close(f)
		string("archEval_", filename)[1:end-4]
	else
		Xtrain_lin = [ones(Float32, size(Y, 1)) X]
		Xtest_lin = [ones(Float32, size(Ytest, 1)) Xtest]
		betas = pinv(Xtrain_lin'*Xtrain_lin)*Xtrain_lin'*Y
		linRegTrainErr = calcError(Xtrain_lin*betas, Y, costFunc = costFunc2)
		linRegTestErr = calcError(Xtest_lin*betas, Ytest, costFunc = costFunc2)
		(naiveTrainErr1, naiveTrainErr2) = if occursin("sq", costFunc2)
			err2 = calcError(fill(mean(Y), length(Y), 1), Y, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([mean(Y) -log(std(Y))], inner = (length(Y), 1)), Y, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		else
			u = median(Y)
			b = mean(abs.(Y .- u))
			err2 = calcError(fill(u, length(Y), 1), Y, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([u -log(b)], inner = (length(Y), 1)), Y, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		end
		(naiveTestErr1, naiveTestErr2) = if occursin("sq", costFunc2)
			err2 = calcError(fill(mean(Y), length(Ytest), 1), Ytest, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([mean(Y) -log(std(Y))], inner = (length(Ytest), 1)), Ytest, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		else
			u = median(Y)
			b = mean(abs.(Y .- u))
			err2 = calcError(fill(u, length(Ytest), 1), Ytest, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([u -log(b)], inner = (length(Ytest), 1)), Ytest, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		end
		line0 = if costFunc2 == costFunc
			[0 0 naiveTrainErr1 naiveTestErr1]
		else
			[0 0 naiveTrainErr1 naiveTestErr1 naiveTrainErr2 naiveTestErr2]
		end

		line1 = if costFunc2 == costFunc
			[0 M+1 linRegTrainErr linRegTestErr]
		else
			[0 M+1 "NA" "NA" linRegTrainErr linRegTestErr]
		end

		writedlm(string("archEval_", filename), [header; line0; line1; body])
		string("archEval_", filename)[1:end-4]

	end
end

function archEvalSample(name, N, batchSize, hiddenList, cols; alpha = 0.002f0, costFunc = "absErr", binInput = false, reslayers=0)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end

	M = length(cols)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	filename = string(name, "_", M, "_input_", O, "_output_ADAMAX", backend, "_", costFunc, ".csv")
	# BLAS.set_num_threads(0)

	header = if costFunc2 == costFunc
		["Layers" "Num Params" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error")]
	else
		["Layers" "Num Params" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error")]
	end

	body = reduce(vcat, pmap(hiddenList) do hidden # @parallel (vcat) for hidden = hiddenList
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(hiddenList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		println(string("training network with ", hidden, " hidden layers"))
		println("initializing network parameters")

		Random.seed!(1234)
		T0, B0 = if occursin("Log", costFunc)
			initializeParams(M, hidden, 2*O)
		else
			initializeParams(M, hidden, O)
		end	
		println("beginning training")
		Random.seed!(1234)
		T, B, bestCost, record = eval(Symbol("ADAMAXTrainNN", backend))(((X[:, cols], Y), (Xtest[:, cols], Ytest)), batchSize, T0, B0, N, M, hidden, 0.0f0, Inf, alpha=alpha, printProgress = true, costFunc = costFunc, resLayers=reslayers)

		GC.gc()
		(outTrain, Jtrain) = calcOutput(X[:, cols], Y, T, B, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest[:, cols], Ytest, T, B, costFunc = costFunc)


		
		numParams = length(theta2Params(B, T))
		if costFunc2 == costFunc
			[length(hidden) numParams Jtrain Jtest]
		else
			[length(hidden) numParams Jtrain[1] Jtest[1] Jtrain[2] Jtest[2]]
		end	
	end)
	

	colNames = replace(if length(cols) > 10
		string(cols[1:3], "_", cols[end-2:end])
	else
		string(cols)
	end, " " => "")
	
	if isfile(string("archEval_", colNames, "_cols_", filename))
		f = open(string("archEval_", colNames, "_cols_", filename), "a")
		writedlm(f, body)

		close(f)
		string("archEval_", colNames, "_cols_", filename)[1:end-4]
	else
		Xtrain_lin = [ones(Float32, size(Y, 1)) X[:, cols]]
		Xtest_lin = [ones(Float32, size(Ytest, 1)) Xtest[:, cols]]
		betas = pinv(Xtrain_lin'*Xtrain_lin)*Xtrain_lin'*Y
		linRegTrainErr = calcError(Xtrain_lin*betas, Y, costFunc = costFunc2)
		linRegTestErr = calcError(Xtest_lin*betas, Ytest, costFunc = costFunc2)
		(naiveTrainErr1, naiveTrainErr2) = if occursin("sq", costFunc2)
			err2 = calcError(fill(mean(Y), length(Y), 1), Y, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([mean(Y) -log(std(Y))], inner = (length(Y), 1)), Y, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		else
			u = median(Y)
			b = mean(abs.(Y .- u))
			err2 = calcError(fill(u, length(Y), 1), Y, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([u -0.5*log(b)], inner = (length(Y), 1)), Y, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		end
		(naiveTestErr1, naiveTestErr2) = if occursin("sq", costFunc2)
			err2 = calcError(fill(mean(Y), length(Ytest), 1), Ytest, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([mean(Y) -log(std(Y))], inner = (length(Ytest), 1)), Ytest, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		else
			u = median(Y)
			b = mean(abs.(Y .- u))
			err2 = calcError(fill(u, length(Ytest), 1), Ytest, costFunc = costFunc2)
			err1 = if costFunc2 != costFunc
				calcError(repeat([u -0.5*log(b)], inner = (length(Ytest), 1)), Ytest, costFunc = costFunc)[1]
			else
				err2
			end
			(err1, err2)
		end
		line0 = if costFunc2 == costFunc
			[0 0 naiveTrainErr1 naiveTestErr1]
		else
			[0 0 naiveTrainErr1 naiveTestErr1 naiveTrainErr2 naiveTestErr2]
		end		

		line1 = if costFunc2 == costFunc
			[0 M+1 linRegTrainErr linRegTestErr]
		else
			[0 M+1 "NA" "NA" linRegTrainErr linRegTestErr]
		end
		writedlm(string("archEval_", colNames, "_cols_", filename), [header; line0; line1; body])
		string("archEval_", colNames, "_cols_", filename)[1:end-4]

	end
end

#train a network with a variable number of layers for a given target number
#of parameters.
function evalLayers(name, N, batchSize, Plist; layers = [2, 4, 6, 8, 10], alpha = .002f0, R = 0.1f0, printProg = false, costFunc = "absErr", binInput = false, reslayers=0)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	filename = string(name, "_", M, "_input_", O, "_output_", alpha, "_alpha_ADAMAX", backend, "_", costFunc, ".csv")

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	#determine number of layers to test in a range
	fatHiddenList = mapreduce(vcat, Plist) do P 
		map(layers) do L
			H = if occursin("Log", costFunc)
				ceil(Int64, getHiddenSize(M, 2*O, L, P))
			else
				ceil(Int64, getHiddenSize(M, O, L, P))
			end
			if H == 0
				(P, Int64.([]))
			else
				(P, H*ones(Int64, L))
			end
		end
	end

	hiddenLayers = map(a -> a[2], fatHiddenList)
	hiddenList = map(H -> fatHiddenList[findall(a -> a == H, hiddenLayers)[1]], unique(hiddenLayers))
	
	# BLAS.set_num_threads(0)

	header = if costFunc2 == costFunc
		["Layers" "Num Params" "Target Num Params" "H" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") "Median GFLOPS"]
	else
		["Layers" "Num Params" "Target Num Params" "H" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error") "Median GFLOPS"]
	end

	body = reduce(vcat, pmap(hiddenList) do hidden # @parallel (vcat) for hidden in hiddenList
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(hiddenList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		println(string("training network with ", hidden[2], " hidden layers"))
		println("initializing network parameters")
		Random.seed!(1234)
		T0, B0 = if occursin("Log", costFunc)
			initializeparams_saxe(M, hidden[2], 2*O)
		else
			initializeparams_saxe(M, hidden[2], O)
		end	

		println("beginning training")
		Random.seed!(1234)
		T, B, bestCost, record, timeRecord, GFLOPS = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, N, M, hidden[2], 0.0f0, Inf, alpha=alpha, R = R, printProgress = printProg, costFunc = costFunc, resLayers=reslayers)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, costFunc = costFunc)

		numParams = length(theta2Params(B, T))
		
		if costFunc2 == costFunc
			[length(hidden[2]) numParams hidden[1] hidden[2][1] Jtrain Jtest median(GFLOPS)]
		else
			[length(hidden[2]) numParams hidden[1] hidden[2][1] Jtrain[1] Jtest[1] Jtrain[2] Jtest[2] median(GFLOPS)]
		end	
	end)

	if isfile(string("evalLayers_", filename))
		f = open(string("evalLayers_", filename), "a")
		writedlm(f, body)
		close(f)
	else
		writedlm(string("evalLayers_", filename), [header; body])
	end
end


#tuneAlpha is a function that trains the same network architecture with the same training and test sets
#over different alpha hyper parameters using the ADAMAX minibatch algorithm. It saves the full cost records
#for each training session in a table where each column represents training under a different hyper parameter.
#The name provided informs which training and test set data to read.  
#Call this function with: tuneAlpha(name, N, batchSize, hidden, alphaList), where N is the 
#number of epochs over which to train, batchSize is the number of examples in each minibatch, hidden 
#is a vector of integers to specify the structure of the network network to train.  For example 
#[4, 4] would indicate a network with two hidden layers each with 4 neurons.  alphaList is an array of 32 bit
#floats which list the training hyperparameters to use.  Optionally lambda and c can be set as keyword values 
#and are the L2 and max norm regularization constants respectively which by default are set to 0 and Inf which 
#results in no regularization.
function tuneAlpha(name, N, batchSize, hidden, alphaList; R = 0.1f0, lambda = 0.0f0, c = Inf, dropout = 0.0f0, costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end
	
	if dropout == 0.0f0
		filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", batchSize, "batchSize_", R, "_decayRate_ADAMAX", backend, "_", costFunc, ".csv")
	else
		filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", dropout, "_dropoutRate_", batchSize, "batchSize_", R, "_decaytRate_ADAMAX", backend, "_", costFunc, ".csv")
	end

	println(string("training network with ", hidden, " hidden layers ", lambda, " L2, and ", c, " maxNorm"))
	println("initializing network parameters")
	Random.seed!(1234)
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end	
	# BLAS.set_num_threads(0)
	header = map(a -> string("alpha ",  a), alphaList')
	body = reduce(hcat, pmap(alphaList) do alpha # @parallel (hcat) for alpha = alphaList
		#BLAS.set_num_threads(Sys.CPU_THREADS)
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(alphaList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		Random.seed!(1234)
		println("beginning training with ", alpha, " alpha")
		T, B, bestCost, record = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha=alpha, R=R, dropout = dropout, printProgress = true, costFunc = costFunc)
		record
	end)
	writedlm(string("alphaCostRecords_", filename), [header; body])
end


function autoTuneParams(inputs, batchSize, T0, B0, N, hidden; tau = 0.01f0, lambda = 0.0f0, c = Inf, dropout = 0.0f0, costFunc = "absErr", resLayers = resLayers, printProg=false, printanything=false)
	M = size(inputs[1][1], 2)
	O = size(inputs[1][2], 2)

	trainFunc(N, a, r) = eval(Symbol("ADAMAXTrainNN", backend))(inputs, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = a, R = r, costFunc = costFunc, resLayers = resLayers, printProgress=printProg, printAnything=printanything)

	Random.seed!(1234)
	# c1 = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, 1, M, hidden, lambda, c, alpha = 0.0f0, costFunc = costFunc)[3]
	c1 = trainFunc(1, 0.0f0, 0.0f0)[3]
	println(string("Baseline cost = ", c1))
	phi = 0.5f0*(1.0f0+sqrt(5.0f0))
 	
 	numParams = if occursin("Log", costFunc)
 		getNumParams(M, hidden, 2*O)
 	else
 		getNumParams(M, hidden, O)
 	end

	function findAlphaInterval(f, c1)
		phi = 0.5f0*(1.0f0+sqrt(5.0f0))
		x = if numParams > 10000000
			0.0001f0
		elseif numParams > 1000000
			0.0005f0
		elseif numParams > 100000
			0.001f0
		elseif numParams > 10000
			0.002f0
		elseif numParams > 1000
			0.005f0
		else
			0.01f0
		end

		x1 = 0.0f0
		x2 = x
		Random.seed!(1234)
		out2 = f(x2)
		c2 = out2[3]

		
		if c2 > c1
			c3 = c2
			x3 = x2

			x2 = (x3+phi*x1)/(phi+1.0f0)
			Random.seed!(1234)
			out2 = f(x2)
			c2 = out2[3]

			while c2 > c1
				c3 = c2
				x3 = x2

				x2 = (x3+phi*x1)/(phi+1.0f0)
				Random.seed!(1234)
				out2 = f(x2)
				c2 = out2[3]
			end
			((x1, c1), (x2, out2), (x3, c3))
		else
			x3 = x*(1.0f0+phi)
			Random.seed!(1234)
			out3 = f(x3)
			c3 = out3[3]

			t = 2
			while c3 < c2
				x1 = x2
				c1 = c2
				x2 = x3
				out2 = out3
				c2 = out2[3]
				x3 = x2 + x*(phi^t)
				Random.seed!(1234)
				out3 = f(x3)
				c3 = out3[3]
				t += 1
			end

			((x1, c1), (x2, out2), (x3, c3))
		end
	end

	function findRInterval(alpha, c1)
		# f = R -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha, R = R, dropout = dropout, costFunc = costFunc)
		f = R -> trainFunc(N, alpha, R)
		phi = 0.5f0*(1.0f0+sqrt(5.0f0))
		x = 0.02f0
		x1 = 0.0f0
		x2 = x
		Random.seed!(1234)
		out2 = f(x2)
		c2 = out2[3]

		if c2 > c1
			c3 = c2
			x3 = x2

			x2 = (x3+phi*x1)/(phi+1.0f0)
			Random.seed!(1234)
			out2 = f(x2)
			c2 = out2[3]

			while c2 > c1
				c3 = c2
				x3 = x2

				x2 = (x3+phi*x1)/(phi+1.0f0)
				Random.seed!(1234)
				out2 = f(x2)
				c2 = out2[3]
			end
			((x1, c1), (x2, out2), (x3, c3))
		else
			x3 = x*(1.0f0+phi)
			Random.seed!(1234)
			out3 = f(x3)
			c3 = out3[3]

			t = 2
			while (c3 < c2) 
				x1 = x2
				c1 = c2
				x2 = x3
				out2 = out3
				c2 = out2[3]
				x3 = x2 + x*(phi^t)
				Random.seed!(1234)
				out3 = f(x3)
				c3 = out3[3]
				t += 1
			end

			((x1, c1), (x2, out2), (x3, c3))
		end
	end



	function findMin(f, tau, p1, p3, p2...)
		x1 = p1[1]
		c1 = p1[2]
		x3 = p3[1]
		c3 = p3[2]
		(x2, out2) = if isempty(p2)
			x2 = (x3+phi*x1)/(phi+1.0f0)
			Random.seed!(1234)
			out2 = f(x2)
			(x2, out2)
		else
			p2[1]
		end
		c2 = out2[3]
		x4 = (phi*x3 + x1)/(1.0f0+phi)
		Random.seed!(1234)
		out4 = f(x4)
		c4 = out4[3]
		println("Current Values Are:")
		println("x       |y      ")
		println(string(round(x1, digits=6), "|", round(c1, digits=6)))
		println(string(round(x2, digits=6), "|", round(c2, digits=6)))
		println(string(round(x4, digits=6), "|", round(c4, digits=6)))
		println(string(round(x3, digits=6), "|", round(c3, digits=6)))
		#println(string("Current x values are ", [x1, x2, x4, x3]))

		while (abs(c4-c2)/(0.5f0*abs(c4+c2)) > tau) & ((2.0f0*abs(x4-x2)/abs(x4+x2)) > tau)
			if c4 < c2
				x1 = x2
				x2 = x4
				c1 = c2
				out2 = out4
				c2 = out2[3]

				x4 = (phi*x3 + x1)/(1.0f0+phi)
				Random.seed!(1234)
				out4 = f(x4)
				c4 = out4[3]
			else
				x3 = x4
				x4 = x2
				c3 = c4
				out4 = out2
				c4 = out4[3]

				x2 = (x3+phi*x1)/(phi+1.0f0)
				Random.seed!(1234)
				out2 = f(x2)
				c2 = out2[3]
			end
			println("Current Values Are:")
			println("x       |y      ")
			println(string(round(x1, digits=6), "|", round(c1, digits=6)))
			println(string(round(x2, digits=6), "|", round(c2, digits=6)))
			println(string(round(x4, digits=6), "|", round(c4, digits=6)))
			println(string(round(x3, digits=6), "|", round(c3, digits=6)))
			#println(string("Current x values are ", [x1, x2, x4, x3]))
		end

		costs = (c1, c2, c3, c4)
		xs = (x1, c2, c3, c4)
		ind = findmin(costs)

		if (ind[2] == 1) | (ind[2] == 3)
			if c2 < c4
				(x2, out2, false)
			else
				(x4, out4, false)
			end
		else
			if c2 < c4
				(x2, out2, true)
			else
				(x4, out4, true)
			end
		end
	end

	# f = alpha -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, 100, M, hidden, lambda, c, alpha = alpha, dropout = dropout, costFunc = costFunc)
	f = alpha -> trainFunc(100, alpha, 0.0f0)

	(p1, p2, p3) = findAlphaInterval(a -> f(a), c1)
	# d1 = 2.0f0*(p1[2] - p2[2][3])/(p1[2]+p2[2][3])
	# d2 = 2.0f0*(p3[2] - p2[2][3])/(p3[2]+p2[2][3])

	# (alpha1, out, status) = if max(d1, d2) < tau
	# 	println()
	# 	println(string("At default R initial best alpha is, ", p2[1], " with a cost of ", p2[2][3]))
	# 	println()
	# 	(p2[1], p2[2], true)
	# else
		println()
		println(string("Starting with alpha interval of ", p1[1], " to ", p3[1], " with a midpoint of ", p2[1]))
		println()
		(alpha1, out, status) = findMin(a -> f(a), tau, p1, p3, p2)
		println()
		println(string("At default R initial best alpha is ", alpha1, " with a cost of ", out[3]))
		println()
	# 	(alpha1, out, status)
	# end
	cost = out[3]

	if N > 100
		println()
		println("Beginning search for decay rate interval")
		println()
		Random.seed!(1234)
		# c1 = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha1, R = 0.0f0, dropout = dropout, costFunc = costFunc)[3]
		c1 = trainFunc(N, alpha1, 0.0f0)[3]
		(p1, p2, p3) = findRInterval(alpha1, c1)
		# d1 = 2.0f0*(p1[2] - p2[2][3])/(p1[2]+p2[2][3])
		# d2 = 2.0f0*(p3[2] - p2[2][3])/(p3[2]+p2[2][3])
		# (R1, out2, status) = if max(d1, d2) < tau
		# 	println()
		# 	println(string("At alpha of ", alpha1, " optimal decay rate found to be ", p2[1], " with a cost of ", p2[2][3]))
		# 	println(string("Checking costs between alpha of ", alpha1*(1.0f0-1.0f0/phi), " and ", 2.0f0*alpha1, " at optimal decay rate of ", p2[1]))
		# 	println()
		# 	(p2[1], p2[2], status)
		# else

			#srand(1234)
			#c1 = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, R = 0.0f0, alpha = alpha1)[3]
			println()
			#println(string("Starting decay rate optimization with window from 0 to 1 and costs of ", c1, " to ", cost))
			println(string("Starting decay rate optimization with window from ", p1[1], " to ", p3[1], " and costs of ", c1, " to ", p3[2]))
			println()
			# f = R -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha1, R = R, dropout = dropout, costFunc = costFunc)
			f = R -> trainFunc(N, alpha1, R)
			(R1, out2, status) = findMin(a -> f(a), tau, p1, p3, p2)
			println()
			println(string("At alpha of ", alpha1, " optimal decay rate found to be ", R1, " with a cost of ", out2[3]))
			println(string("Checking costs between alpha of ", alpha1*(1.0f0-1.0f0/phi), " and ", 2.0f0*alpha1, " at optimal decay rate of ", R1))
			println()
		# 	(R1, out2, status)
		# end

		cost2 = out2[3]
		range = 2.0f0
		alpha1_plus = range*alpha1
		alpha1_minus = alpha1*(1.0f0-(range-1.0f0)/phi)
		Random.seed!(1234)
		# out2_plus = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha1_plus, R = R1, dropout = dropout, costFunc = costFunc)
		out2_plus = trainFunc(N, alpha1_plus, R1)
		Random.seed!(1234)
		# out2_minus = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha1_minus, R = R1, dropout = dropout, costFunc = costFunc)
		out2_minus = trainFunc(N, alpha1_minus, R1)
		cost2_plus = out2_plus[3]
		cost2_minus = out2_minus[3]
		
		if cost2_minus < cost2
			println(string("Alpha of ", alpha1_minus, " has a cost of ", cost2_minus, " which is lower than the midpoint cost of ", cost2, " at alpha = ", alpha1))
			if (cost2_plus > cost2_minus)
				println()
				println(string("Alpha of ", alpha1_minus, " has a cost of ", cost2_minus, " which is lower than the midpoint cost of ", cost2, " at alpha = ", alpha1))
				println(string("Re-optimizing alpha over the window 0.0 to ", alpha1, " with a decay rate of ", R1))
				println()
				# f = alpha -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, R = R1, alpha = alpha, dropout = dropout, costFunc = costFunc)
				f = alpha -> trainFunc(N, alpha, R1)
				(alpha2, out3, status) = findMin(a -> f(a), tau, (0.0f0, c1), (alpha1, cost2))
				cost3 = out3[3]
				println()
				println(string("Cost at alpha = ", alpha2, " and R = ", R1, " is ", cost3))
				(alpha2, R1, out3, status)
			else
				println(string("Alpha of ", alpha1_plus, " has a cost of ", cost2_plus, " which is lower than the midpoint cost of ", cost2, " at alpha = ", alpha1))
				(alpha1_plus, out2_plus) = if alpha1_plus >= 0.1f0
					(alpha1_plus, out2_plus)
				else
					alpha1_minus = alpha1
					out2_minus = out2
					cost2_minus = cost2
					alpha1 = alpha1_plus
					out2 = out2_plus
					cost2 = cost2_plus
					alpha1_plus = 0.1f0
					Random.seed!(1234)
					# out2_plus = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha1_plus, R = R1, dropout = dropout, costFunc = costFunc)
					out2_plus = trainFunc(N, alpha1_plus, R1)
					(alpha1_plus, out2_plus)
				end
				cost2_plus = out2_plus[3]
				if cost2_plus < cost2
					println()
					println(string("Warning alpha could not be re-optimized despite expanding search spread to ", alpha1_minus, " to 0.1"))
					println(string("No interval could be established as ", cost2_plus, " at an alpha of 0.1 is still less than the midpoint cost of ", cost2, " at alpha = ", alpha1))
					println(string("Cost at alpha = ", alpha1_plus, " and R = ", R1, " is ", cost2_plus))
					println()
					(alpha1_plus, R1, out2_plus, false)
				else
					println()
					println(string("The alpha range from ", alpha1_minus, " to ", alpha1_plus, " encloses the current minimum cost of ", cost2, " at alpha = ", alpha1))
					println(string("Re-optimizing alpha over the window ", alpha1_minus, " to ", alpha1_plus, " with a decay rate of ", R1))
					println()
					# f = alpha -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, R = R1, alpha = alpha, dropout = dropout, costFunc = costFunc)
					f = alpha -> trainFunc(N, alpha, R1)
					(alpha2, out3, status) = findMin(a -> f(a), tau, (alpha1_minus, cost2_minus), (alpha1_plus, cost2_plus))
					cost3 = out3[3]
					println()
					println(string("Cost at alpha = ", alpha2, " and R = ", R1, " is ", cost3))
					(alpha2, R1, out3, status)
				end
			end
		elseif cost2_plus < cost2
			if cost2_minus > cost2
				alpha1_minus = alpha1
				out2_minus = out2
				cost2_minus = cost2
				alpha1 = alpha1_plus
				out2 = out2_plus
				cost2 = cost2_plus
				alpha1_plus = phi*0.5f0*alpha1 + alpha1
				Random.seed!(1234)
				# out2_plus = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha1_plus, R = R1, dropout = dropout, costFunc = costFunc)
				out2_plus = trainFunc(N, alpha1_plus, R1)
				cost2_plus = out2_plus[3]
				if cost2_plus < cost2
					println()
					println(string("Warning alpha could not be re-optimized despite expanding search spread to ", alpha1_minus, " to ", alpha1_plus))
					println(string("No interval could be established as ", cost2_plus, " at an alpha of ", alpha1_plus, " is still less than the midpoint cost of ", cost2, " at alpha = ", alpha1))
					println(string("Cost at alpha = ", alpha1_plus, " and R = ", R1, " is ", cost2_plus))
					println()
					(alpha1_plus, R1, out2_plus, false)
				else
					println()
					println(string("Alpha of ", alpha1, " has a cost of ", cost2, " which is lower than the original midpoint cost of ", cost2_minus, " at alpha = ", alpha1_minus))
					println(string("Re-optimizing alpha over the window ", alpha1_minus, " to ", alpha1_plus, " with a decay rate of ", R1))
					println()
					# f = alpha -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, R = R1, alpha = alpha, dropout = dropout, costFunc = costFunc)
					f = alpha -> trainFunc(N, alpha, R1)
					(alpha2, out3, status) = findMin(a -> f(a), tau, (alpha1_minus, cost2_minus), (alpha1_plus, cost2_plus), (alpha1, out2))
					cost3 = out3[3]
					println()
					println(string("Cost at alpha = ", alpha2, " and R = ", R1, " is ", cost3))
					(alpha2, R1, out3, status)
				end
			end
		# elseif max(cost2_plus/cost2, cost2_minus/cost2) < (1+tau)
		# 	println()
		# 	println(string("Re-optimizing alpha is not necessary after checking range of ", alpha1_minus, " to ", alpha1_plus))
		# 	println(string("Cost at alpha = ", alpha1, " and R = ", R1, " is ", cost2))
		# 	(alpha1, R1, out2, status)
		else
			println()
			println(string("Re-optimizing alpha over the window ", alpha1_minus, " to ", alpha1_plus, " with a decay rate of ", R1))
			println()
			# f = alpha -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, R = R1, alpha = alpha, dropout = dropout, costFunc = costFunc)
			f = alpha -> trainFunc(N, alpha, R1)
			(alpha2, out3, status) = findMin(a -> f(a), tau, (alpha1_minus, cost2_minus), (alpha1_plus, cost2_plus), (alpha1, out2))
			cost3 = out3[3]
			println()
			println(string("Cost at alpha = ", alpha2, " and R = ", R1, " is ", cost3))
			(alpha2, R1, out3, status)
		end
	else
		println()
		println(string("Cost at alpha = ", alpha1, " and the default decay rate is ", cost))
		(alpha1, 0.1f0, out, true)
	end
end

function autoTuneR(X, Y, batchSize, T0, B0, N, hidden; alpha = 0.002f0, tau = 0.01f0, lambda = 0.0f0, c = Inf, dropout = 0.0f0, costFunc = "absErr")
	M = size(X, 2)
	O = size(Y, 2)

	function findRInterval(alpha, p1)
		f = R -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha, R = R, dropout = dropout, costFunc = costFunc)
		phi = 0.5f0*(1.0f0+sqrt(5.0f0))
		x = 0.01f0
		x1 = p1[1]
		c1 = p1[2]
		x2 = x
		Random.seed!(1234)
		out2 = f(x2)
		c2 = out2[3]

		if c2 > c1
			c3 = c2
			x3 = x2

			x2 = (x3+phi*x1)/(phi+1.0f0)
			Random.seed!(1234)
			out2 = f(x2)
			c2 = out2[3]

			while c2 > c1
				c3 = c2
				x3 = x2

				x2 = (x3+phi*x1)/(phi+1.0f0)
				Random.seed!(1234)
				out2 = f(x2)
				c2 = out2[3]
			end
			((x1, c1), (x2, out2), (x3, c3))
		else
			x3 = x*(1.0f0+phi)
			Random.seed!(1234)
			out3 = f(x3)
			c3 = out3[3]

			t = 2
			while (c3 < c2) 
				x1 = x2
				c1 = c2
				x2 = x3
				out2 = out3
				c2 = out2[3]
				x3 = x2 + x*(phi^t)
				Random.seed!(1234)
				out3 = f(x3)
				c3 = out3[3]
				t += 1
			end

			((x1, c1), (x2, out2), (x3, c3))
		end
	end


	function findMin(f, tau, p1, p3, p2...)
		x1 = p1[1]
		c1 = p1[2]
		x3 = p3[1]
		c3 = p3[2]
		(x2, out2) = if isempty(p2)
			x2 = (x3+phi*x1)/(phi+1.0f0)
			Random.seed!(1234)
			out2 = f(x2)
			(x2, out2)
		else
			p2[1]
		end
		c2 = out2[3]
		
		# if 2.0f0*abs.(c2 - min(c1, c3))/(c2+min(c1, c3)) < tau
		# 	status = (c2 < min(c1, c3))
		# 	(x2, out2, status)
		# else

			x4 = (phi*x3 + x1)/(1.0f0+phi)
			Random.seed!(1234)
			out4 = f(x4)
			c4 = out4[3]
			println(string("Current x values are ", [x1, x2, x4, x3]))

			while (abs(c4-c2)/(0.5f0*abs(c4+c2)) > tau) & ((2.0f0*abs(x4-x2)/abs(x4+x2)) > tau)
				
				if c4 < c2
					x1 = x2
					x2 = x4
					c1 = c2
					out2 = out4
					c2 = out2[3]

					x4 = (phi*x3 + x1)/(1.0f0+phi)
					Random.seed!(1234)
					out4 = f(x4)
					c4 = out4[3]
				else
					x3 = x4
					x4 = x2
					c3 = c4
					out4 = out2
					c4 = out4[3]

					x2 = (x3+phi*x1)/(phi+1.0f0)
					Random.seed!(1234)
					out2 = f(x2)
					c2 = out2[3]
				end
				println(string("Current x values are ", [x1, x2, x4, x3]))
			end

			costs = (c1, c2, c3, c4)
			xs = (x1, c2, c3, c4)
			ind = findmin(costs)

			if (ind[2] == 1) | (ind[2] == 3)
				if c2 < c4
					(x2, out2, false)
				else
					(x4, out4, false)
				end
			else
				if c2 < c4
					(x2, out2, true)
				else
					(x4, out4, true)
				end
			end
		# end
	end

	if N > 100
		Random.seed!(1234)
		out1 = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, R = 0.0f0, alpha = alpha, costFunc = costFunc)
		c1 = out1[3]
		println(string("Baseline cost = ", c1))
		phi = 0.5f0*(1.0f0+sqrt(5.0f0))

		println()
		println("Beginning search for decay rate interval")
		println()

		(p1, p2, p3) = findRInterval(alpha, (0.0f0, c1))
		if isempty(p3)
			println()
			println(string("Could not establish an interval as the cost at R = ", p2[1], " of ", p2[2], " is still greater than the cost at R = ", p1[1], " of ", p1[2]))
			println(string("At alpha of ", alpha, " defaulting to 0.0 decay rate with a cost of ", out1[2][3]))
			println()
			(0.0f0, out1, false)
		else	
			# d1 = 2.0f0*(p1[2] - p2[2][3])/(p1[2]+p2[2][3])
			# d2 = 2.0f0*(p3[2] - p2[2][3])/(p3[2]+p2[2][3])
			
			# (R, out, status) = if min(d1, d2) < tau
			# 	println()
			# 	println(string("Optimizing alpha not necessary since the midpoint left after establishing a range from ", p1[1], " to ", p3[1], " is within the tolerance"))
			# 	println(string("At alpha of ", alpha, " optimal decay rate found to be ", p2[1], " with a cost of ", p2[2][3]))
			# 	println()
			# 	(p2[1], p2[2], true)
			# else
				println()
				println(string("Starting decay rate optimization with window from ", p1[1], " to ", p3[1], " and costs of ", p1[2], " to ", p3[2]))
				println()
				f = R -> eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha, R = R, dropout = dropout, costFunc = costFunc)
				out = findMin(a -> f(a), tau, p1, p3, p2)
				println()
				println(string("At alpha of ", alpha, " optimal decay rate found to be ", out[1], " with a cost of ", out[2][3]))
				println()
				out
			# end
		end
	else
		error("Cannot optimize decay rate when number of epochs is less than 100")
	end
end


function smartTuneR(name, N, batchSize, hidden, alphaList; tau = 0.01f0, dropout = 0.0f0, lambda = 0.0f0, c = Inf, costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	filename = if dropout == 0.0f0
		string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", batchSize, "batchSize_", N, "_epochs_ADAMAX", backend, "_", costFunc, ".csv")
	else
		string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", dropout, "_dropoutRate_", batchSize, "_batchSize_", N, "_epochs_ADAMAX", backend, "_", costFunc, ".csv")
	end

	println(string("training network with ", hidden, " hidden layers ", lambda, " L2, and ", c, " maxNorm"))
	println("initializing network parameters")
	Random.seed!(1234)
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end	
	
	header = ["Alpha" "Optimal Decay Rate" "Training Error" "Test Error" "Extrapolated Final Training Error" string("Additional Epochs to Reach Final Error with Tolerance ", tau) "GFLOPS" "Time Per Epoch" "Status"]
	
	body = reduce(vcat, pmap(alphaList) do alpha # @parallel (vcat) for alpha in alphaList
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(alphaList))))))
		else
			BLAS.set_num_threads(0)
		end

		Random.seed!(1234)
		println("beginning decay rate optimization with ", alpha, " alpha")
		(R, out, status) = autoTuneR(X, Y, batchSize, T0, B0, N, hidden; alpha = alpha, tau = tau, lambda = lambda, c = c, dropout = dropout, costFunc = costFunc)
		T, B, bestCost, record, timeRecord, GFLOPS = out
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, dropout = dropout, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, dropout = dropout, costFunc = costFunc)
		l = length(record)

		#final 3 points of training cost
		y1 = record[max(1, l-2)]
		y2 = record[max(1, l-1)]
		y3 = record[l]
		
		#extrapolated convergence point of training cost assuming an exponential 
		#asymptote at cc with a decay rate of D. t represents the number of additional periods
		#required for the training cost to be within the tolerance tau
		D = (y3-y2)/(y2-y1)
		a = (y2-y1)/(D-1)
		cc = y1-a
		t = try
			10*(round(Int64, log(tau*cc)/log(a*D)) - 2)
		catch
			"NA"
		end

		(trainErr, testErr) = if costFunc2 == costFunc
			(Jtrain, Jtest)
		else
			(Jtrain[1], Jtest[1])
		end

		#conv = mean((record[max(2, l-9):l]./record[max(1, l-10):l-1])-1)
		
		[alpha R trainErr testErr cc t median(GFLOPS) median(timeRecord[2:end]-timeRecord[1:end-1]) status]
	end)

	if isfile(string("smartDecayRates_", filename))
		f = open(string("smartDecayRates_", filename), "a")
		writedlm(f, body)
		close(f)
	else
		writedlm(string("smartDecayRates_", filename), [header; body])
	end
end

function tuneR(name, N, batchSize, hidden, RList; alpha = 0.002f0, lambda = 0.0f0, c = Inf, dropout = 0.0f0, costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end
	
	if dropout == 0.0f0
		filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", batchSize, "batchSize_", alpha, "_alpha_ADAMAX", backend, "_", costFunc, ".csv")
	else
		filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambda, "_L2_", c, "_maxNorm_", batchSize, "batchSize_", dropout, "_dropoutRate_ADAMAX", backend, "_", costFunc, ".csv")
	end

	println(string("training network with ", hidden, " hidden layers ", lambda, " L2, and ", c, " maxNorm"))
	println("initializing network parameters")
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end	
	
	header = map(a -> string("R ",  a), RList')
	body = reduce(hcat, pmap(RList) do R # @parallel (hcat) for R in RList
		#BLAS.set_num_threads(Sys.CPU_THREADS)
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(RList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		Random.seed!(1234)
		println("beginning training with ", alpha, " alpha")
		T, B, bestCost, record = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha=alpha, R=R, dropout = dropout, printProgress = true, costFunc = costFunc)
		record
	end)
	writedlm(string("decayRateCostRecords_", filename), [header; body])
end

#L2Reg is a function that trains the same network architecture with the same training and test sets
#over different L2 regularization hyper parameters (lambda) using the ADAMAX minibatch algorithm. It saves the 
#mean abs error over the training and test set in a table where each row corresponds to each value of lambda.
#The name provided informs which training and test set data to read.  
#Call this function with: L2Reg(name, N, batchSize, hidden, lambdaList, a), where N is the 
#number of epochs over which to train, batchSize is the number of examples in each minibatch, hidden 
#is a vector of integers to specify the structure of the network network to train.  For example 
#[4, 4] would indicate a network with two hidden layers each with 4 neurons.  lambdaList is an array of 32 bit
#floats which list the training L2 Reg hyperparameters to use.  alpha is the alpha hyper parameter for the ADAMAX 
#training algorithm. 
function L2Reg(name, N, batchSize, hidden, lambdaList, alpha, c = 0.0f0; costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	c = Inf
	
	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", c, "_maxNorm_", alpha, "_alpha_ADAMAX", backend, "_", costFunc, ".csv")
	
	println(string("training network with ", hidden, " hidden layers"))
	println("initializing network parameters")
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end

	header = if costFunc2 == costFunc
		["Lambda" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") "Median GFLOPS" "Median Time Per Epoch"]
	else
		["Lambda" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error") "Median GFLOPS" "Median Time Per Epoch"]
	end
	
	body = reduce(vcat, pmap(lambdaList) do lambda # @parallel (vcat) for lambda = lambdaList
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), length(lambdaList))))))
		else
			BLAS.set_num_threads(0)
		end
		Random.seed!(1234)
		println("beginning training with ", lambda, " lambda")
		(T, B, bestCost, record, timeRecord, GFLOPS) = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, lambda, c, alpha, printProgress = true, costFunc = costFunc)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, costFunc = costFunc)
		
		if costFunc2 == costFunc
			[lambda Jtrain Jtest median(GFLOPS) median(timeRecord[2:end] .- timeRecord[1:end-1])]
		else
			[lambda Jtrain[1] Jtest[1] Jtrain[2] Jtest[2] median(GFLOPS) median(timeRecord[2:end] .- timeRecord[1:end-1])]
		end
	end)
	writedlm(string("L2Reg_", filename), [header; body])
end

#maxNormReg is a function that trains the same network architecture with the same training and test sets
#over different max norm regularization hyper parameters (c) using the ADAMAX minibatch algorithm. It saves the 
#mean abs error over the training and test set in a table where each row corresponds to each value of c.
#The name provided informs which training and test set data to read.  
#Call this function with: maxNormReg(name, N, batchSize, hidden, cList, a), where N is the 
#number of epochs over which to train, batchSize is the number of examples in each minibatch, hidden 
#is a vector of integers to specify the structure of the network network to train.  For example 
#[4, 4] would indicate a network with two hidden layers each with 4 neurons.  cList is an array of 32 bit
#floats which list the training max norm hyperparameters to use.  alpha is the alpha hyper parameter for the ADAMAX 
#training algorithm. 
function maxNormReg(name, N, batchSize, hidden, cList, alpha, R; dropout = 0.0f0, costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	lambda = 0.0f0
	
	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	if dropout == 0.0f0
		filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc, ".csv")
	else
		filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", dropout, "_dropoutRate_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc, ".csv")
	end

	println(string("training network with ", hidden, " hidden layers "))
	println("initializing network parameters")
	Random.seed!(1234)
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end	
	
	header = if costFunc2 == costFunc
		["Max Norm" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") "Median GFLOPS" "Median Time Per Epoch"]
	else
		["Max Norm" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error") "Median GFLOPS" "Median Time Per Epoch"]
	end
	
	body = reduce(vcat, pmap(cList) do c # @parallel (vcat) for c = cList
		#BLAS.set_num_threads(Sys.CPU_THREADS)
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), length(cList))))))
		else
			BLAS.set_num_threads(0)
		end
		#BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(cList))))))
		
		Random.seed!(1234)
		println("beginning training with ", c, " max norm")
		T, B, bestCost, record, timeRecord, GFLOPS = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, 0.0f0, c, alpha=alpha, R=R, dropout=dropout, printProgress = true, costFunc = costFunc)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, dropout = dropout, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, dropout = dropout, costFunc = costFunc)
		
		if costFunc2 == costFunc
			[c Jtrain Jtest median(GFLOPS) median(timeRecord[2:end] .- timeRecord[1:end-1])]
		else 
			[c Jtrain[1] Jtest[1] Jtrain[2] Jtest[2] median(GFLOPS) median(timeRecord[2:end] .- timeRecord[1:end-1])]
		end			
	end)
	if isfile(string("maxNormReg_", filename))
		f = open(string("maxNormReg_", filename), "a")
		writedlm(f, body)
		close(f)
	else
		writedlm(string("maxNormReg_", filename), [header; body])
	end
end

function dropoutReg(name, N, batchSize, hidden, dropouts, c, alpha, R; costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	M = size(X, 2)
	O = size(Y, 2)

	lambda = 0.0f0
	
	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", c, "_maxNorm_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc, ".csv")

	println(string("training network with ", hidden, " hidden layers "))
	println("initializing network parameters")
	Random.seed!(1234)
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end	

	header = if costFunc2 == costFunc
		["Dropout Rate" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") "Median GFLOPS" "Median Time Per Epoch"]
	else
		["Dropout Rate" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error") "Median GFLOPS" "Median Time Per Epoch"]
	end

	body = reduce(vcat, pmap(dropouts) do dropout # @parallel (vcat) for dropout in dropouts
		#BLAS.set_num_threads(Sys.CPU_THREADS)
		#BLAS.set_num_threads(min(5, max(1, floor(Int, Sys.CPU_THREADS/min(nprocs(), length(dropouts))))))
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), length(dropouts))))))
		else
			BLAS.set_num_threads(0)
		end

		Random.seed!(1234)
		println("beginning training with ", dropout, " dropout rate")
		T, B, bestCost, record, timeRecord, GFLOPS = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, N, M, hidden, 0.0f0, c, alpha=alpha, R=R, dropout=dropout, printProgress = true, costFunc = costFunc)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, dropout = dropout, costFunc = costFunc)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, dropout = dropout, costFunc = costFunc)
		
		if costFunc2 == costFunc
			[dropout Jtrain Jtest median(GFLOPS) median(timeRecord[2:end] .- timeRecord[1:end-1])]
		else
			[dropout Jtrain[1] Jtest[1] Jtrain[2] Jtest[2] median(GFLOPS) median(timeRecord[2:end] .- timeRecord[1:end-1])]
		end			
	end)
	if isfile(string("dropoutReg_", filename))
		f = open(string("dropoutReg_", filename), "a")
		writedlm(f, body)
		close(f)
	else
		writedlm(string("dropoutReg_", filename), [header; body])
	end
end

#fullTrain is a function that trains a single network with a specified architecture over the training and test sets
#determined by the name provided in the input.  Training uses the ADAMAX minibatch algorithm and has options for 
#both an L2 and max norm regularization parameter.  It saves the full cost record, performance based on the mean
#abs error over the training and test sets, and the parameters themselves as a one dimensional vector to file. 
#As such the hidden layer structure, input layer size, and output layer size must be specified to reconstruct
#the network parameters in a usable state.  
#Call this function with: fullTrain(name, N, batchSize, hidden, lambda, c, alpha, ID), where N is the 
#number of epochs over which to train, batchSize is the number of examples in each minibatch, hidden 
#is a vector of integers to specify the structure of the network network to train.  For example 
#[4, 4] would indicate a network with two hidden layers each with 4 neurons.  Lambda is the L2 regularization
#hyperparameter, c is the max norm regularization hyper parameter, alpha is the training hyper parameter for the 
#ADAMAX algorithm, and ID is just a string used to specify this training session.  Additonally a startID can be
#specified with a keyword argument which will use the training results from a previous session with the specified
#start ID instead of random initializations.  Also printProg can be set to false to supress output of the training
#progress to the terminal.  Final results will still be printed to terminal regardless. 
function fullTrain(name, N, batchSize, hidden, lambda, c, alpha, R, ID; startID = [], sampleCols = [], dropout = 0.0f0, printanything=true, printProg = true, costFunc = "absErr", writeFiles = true, binInput = false, resLayers = 0, swa=false, blasthreads=0, inputdata = (), initparams = (), ignorebest=false, prepdata = (), prepactivations = ())
	printanything && println("reading and converting training data")
	
	if isempty(inputdata)
		if binInput
			inputdata = readBinInput(name)
		else
			inputdata = readInput(name)
		end
	end

	if length(inputdata) == 2
		autoencoder = true
		(Xraw, Xtestraw) = inputdata
	else
		autoencoder = false
		(Xraw, Xtestraw, Y, Ytest) = inputdata
	end

	X, Xtest = if isempty(sampleCols)
		(Xraw, Xtestraw)
	else
		(Xraw[:, sampleCols], Xtestraw[:, sampleCols])
	end

	colNames = if isempty(sampleCols)
		""
	elseif length(sampleCols) > 10
		replace(string(sampleCols[1:3], "_", sampleCols[end-2:end], "_"), " " => "")
	else
		replace(string(sampleCols, "_"), " " => "")
	end

	(numRows, M) = size(X)
	if autoencoder
		O = size(X, 2)
	else
		O = size(Y, 2)
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end
	
	h_name = if isempty(hidden)
		0
	elseif hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	filename = 	string(colNames, name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", alpha, "_alpha_", formIndicString(R, lambda, c, dropout, resLayers), "ADAMAX_", costFunc)
	
	printanything && println(string("training network with ", hidden, " hidden layers ", lambda, " L2, and ", c, " maxNorm"))
	
	(T0, B0) = if isempty(initparams) 
		if isempty(startID)
			printanything && println("initializing network parameters")
			Random.seed!(1234)
			if occursin("Log", costFunc)
				initializeparams_saxe(M, hidden, 2*O)
			else
				initializeparams_saxe(M, hidden, O)
			end	
		else
			printanything && println("reading previous session parameters")
			readBinParams(string(startID, "_params_", filename, ".bin"))[1]
		end
	else
		initparams
	end
	
	#BLAS.set_num_threads(Sys.CPU_THREADS)	
	#BLAS.set_num_threads(5)	
	BLAS.set_num_threads(blasthreads)

	if autoencoder
		traindata = (X,)
		testdata = (Xtest,)
	else
		traindata = (X, Y)
		testdata = (Xtest, Ytest)
	end

	Random.seed!(1234)
	T, B, bestCost, record, timeRecord, gflops, bestCostTest, costRecordTest, lastepoch, bestresultepoch = eval(Symbol("ADAMAXTrainNN", backend))((traindata, testdata), batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha, R = R, printProgress = printProg, dropout = dropout, costFunc = costFunc, resLayers = resLayers, swa=swa, printAnything=printanything, ignorebest=ignorebest, prepdata = prepdata, prepactivations=prepactivations)
	# GC.gc()
	(outTrain, Jtrain) = calcOutput(traindata..., T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers, autoencoder=autoencoder)
	# GC.gc()
	(outTest, Jtest) = calcOutput(testdata..., T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers, autoencoder=autoencoder)

	if writeFiles
		if occursin("Log", costFunc)
			header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range ", s), 1:O), 1, O)]

			writedlm(string(ID, "_predScatTrain_", filename, ".csv"), [header; [Y outTrain[:, 1:O] exp.(-outTrain[:, O+1:2*O])]])
			writedlm(string(ID, "_predScatTest_", filename, ".csv"), [header; [Ytest outTest[:, 1:O] exp.(-outTest[:, O+1:2*O])]])
		else
			header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O)]
			writedlm(string(ID, "_predScatTrain_", filename, ".csv"), [header; [Y outTrain]])
			writedlm(string(ID, "_predScatTest_", filename, ".csv"), [header; [Ytest outTest]])
		end

		writedlm(string(ID, "_costRecord_", filename, ".csv"), record)
		writedlm(string(ID, "_timeRecord_", filename, ".csv"), timeRecord)
		
		if costFunc2 == costFunc
			writedlm(string(ID, "_performance_", filename, ".csv"), [[string("Train ", costFunc, " Error"), string("Test ", costFunc, " Error")] [Jtrain, Jtest]])
		else
			writedlm(string(ID, "_performance_", filename, ".csv"), [[string("Train ", costFunc, " Error"), string("Test ", costFunc, " Error"), string("Train ", costFunc2, " Error"), string("Test ", costFunc2, " Error")] [Jtrain[1], Jtest[1], Jtrain[2], Jtest[2]]])
		end

		writeParams([(T, B)], string(ID, "_params_", filename, ".bin"))
	end

	(record, T, B, Jtrain, outTrain, bestCost, Jtest, outTest, bestCostTest, bestresultepoch)
end

function fullTrain(name, X, Y, N, batchSize, hidden, lambda, c, alpha, R, ID; startID = [], dropout = 0.0f0, printProg = true, costFunc = "absErr", writeFiles = true, resLayers = resLayers, swa = false, printanything=true, initparams=(), ignorebest=false)

	M = size(X, 2)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	h_name = if isempty(hidden)
		0
	elseif hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	trainSym = Symbol("ADAMAXTrainNN", backend)

	filename = string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", alpha, "_alpha_", formIndicString(R, lambda, c, dropout, resLayers), swa ? "ADAMAXSWA_" : "ADAMAX_", costFunc)
	
	printanything && println(string("training network with ", hidden, " hidden layers ", lambda, " L2, and ", c, " maxNorm"))
	
	(T0, B0) = if isempty(initparams) 
		if isempty(startID)
			printanything && println("initializing network parameters")
			Random.seed!(1234)
			if occursin("Log", costFunc)
				initializeparams_saxe(M, hidden, 2*O)
			else
				initializeparams_saxe(M, hidden, O)
			end	
		else
			printanything && println("reading previous session parameters")
			readBinParams(string(startID, "_params_", filename, ".bin"))[1]
		end
	else
		initparams
	end
	
	#BLAS.set_num_threads(Sys.CPU_THREADS)	
	# BLAS.set_num_threads(0)	
	Random.seed!(1234)
	T, B, bestCost, record, timeRecord = eval(trainSym)(((X, Y),), batchSize, T0, B0, N, M, hidden, lambda, c, alpha = alpha, R = R, printProgress = printProg, dropout = dropout, costFunc = costFunc, resLayers = resLayers, swa = swa, printAnything=printanything, ignorebest=ignorebest)
	GC.gc()
	(outTrain, Jtrain) = calcOutput(X, Y, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers)
	

	if writeFiles
		if occursin("Log", costFunc)
			header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range ", s), 1:O), 1, O)]

			writedlm(string(ID, "_predScat_", filename, ".csv"), [header; [Y outTrain[:, 1:O] exp.(-outTrain[:, O+1:2*O])]])
		else
			header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O)]
			writedlm(string(ID, "_predScat_", filename, ".csv"), [header; [Y outTrain]])

		end
	
		writedlm(string(ID, "_costRecord_", filename, ".csv"), record)
		writedlm(string(ID, "_timeRecord_", filename, ".csv"), timeRecord)
		writedlm(string(ID, "_performance_", filename, ".csv"), ["Cost", Jtrain])
		
		if costFunc2 == costFunc
			writedlm(string(ID, "_performance_", filename, ".csv"), [[string("Train ", costFunc, " Error")] [Jtrain]])
		else
			writedlm(string(ID, "_performance_", filename, ".csv"), [[string("Train ", costFunc, " Error"), string("Train ", costFunc2, " Error")] [Jtrain[1], Jtrain[2]]])
		end

		writeParams([(T, B)], string(ID, "_params_", filename, ".bin"))
	end
	
	(record, T, B, Jtrain, outTrain, bestCost)
end

function multiTrain(name, numEpochs, batchSize, hidden, lambda, c, alpha, R, num, ID; sampleCols = [], dropout = 0.0f0, printProg = false, costFunc = "absErr", binInput = false, writefiles = true, printanything=true, swa = false, reslayers=0, toltest=Inf)
	println("reading and converting training data")
	Xraw, Xtestraw, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end

	X, Xtest = if isempty(sampleCols)
		(Xraw, Xtestraw)
	else
		(Xraw[:, sampleCols], Xtestraw[:, sampleCols])
	end

	colNames = if isempty(sampleCols)
		""
	elseif length(sampleCols) > 10
		replace(string(sampleCols[1:3], "_", sampleCols[end-2:end], "_"), " " => "")
	else
		replace(string(sampleCols, "_"), " " => "")
	end

	(N, M) = size(X)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end
	
	h_name = if isempty(hidden)
		0
	elseif hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	filename = 	string(colNames, name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", round(alpha, digits = 3), "_alpha_", formIndicString(R, lambda, c, dropout), "ADAMAX_", costFunc)

	bootstrapOut = pmap(1:num) do foo
		#BLAS.set_num_threads(Sys.CPU_THREADS)
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), num)))))
		else
			BLAS.set_num_threads(0)
		end

		if ID == 1
            Random.seed!(1234 + foo - 1)
        else
            Random.seed!(ID)
            Random.seed!(1234+rand(UInt32)+foo)
        end
  
		T0, B0 = if occursin("Log", costFunc)
			initializeparams_saxe(M, hidden, 2*O)
		else
			initializeparams_saxe(M, hidden, O)
		end		
		if ID == 1
            Random.seed!(1234 + foo - 1)
        else
               Random.seed!(ID)
            Random.seed!(1234+rand(UInt32)+foo)
        end	
		(T, B, bestCost, costRecord, timeRecord, GFLOPS) = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, numEpochs, M, hidden, lambda, c, R = R, alpha=alpha, dropout=dropout, printProgress = printProg, costFunc = costFunc, printAnything=printanything, resLayers=reslayers, swa = swa, tol=toltest)
		(T, B)
	end
	GC.gc()
	fileout = convert(Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, bootstrapOut)
	writefiles && writeParams(fileout, string(ID, "_multiParams_", filename, ".bin"))
	
	println("Calculating training set outputs")
	(bootstrapOutTrain, outTrain, errTrain, errEstTrain) = calcMultiOut(X, Y, bootstrapOut, dropout = dropout, costFunc = costFunc)#pmap(a -> calcOutput(X, Y, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
    GC.gc()
    println("Calculating test set outputs")
    (bootstrapOutTest, outTest, errTest, errEstTest) = calcMultiOut(Xtest, Ytest, bootstrapOut, dropout = dropout, costFunc = costFunc) #pmap(a -> calcOutput(Xtest, Ytest, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
	
	header = if occursin("Log", costFunc)
		[reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value Error Est ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range Error Est ", s), 1:O), 1, O)]
	else
		header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Error Est ", s), 1:O), 1, O)]
	end
	
	println("Writing prediction scatters to file")
	writefiles && writedlm(string(ID, "_multiPredScatTrain_", filename, ".csv"), [header; [Y outTrain errEstTrain]])
	writefiles && writedlm(string(ID, "_multiPredScatTest_", filename, ".csv"), [header; [Ytest outTest errEstTest]])

	header = if costFunc2 == costFunc
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est"]
	else
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est" string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error")]
	end

	println("Calculating performance vs number of networks")
	d_y = if backend == :GPU
		cuda_allocate(Y)
	else
		[]
	end

	d_ytest = if backend == :GPU
		cuda_allocate(Ytest)
	else
		[]
	end
    	
	bootstrapOutTrain2 = if occursin("Log", costFunc)
		println("Calculating training non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTrain]
	else
		[]
	end

	bootstrapOutTest2 = if occursin("Log", costFunc)
		println("Calculating test non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTest]
	else
		[]
	end

    fullMultiPerformance = mapreduce(vcat, 1:length(bootstrapOutTrain)) do i
        #calculate average network output     
        line = if occursin("Log", costFunc)
            # combinedOutputTrainPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTrain[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTrain2[1:i]]))]  
            combinedOutputTrainPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTrain[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTrain2[1:i])/i)]
            combinedOutputTrain = [combinedOutputTrainPre[:, 1:O] log.(1 ./combinedOutputTrainPre[:, O+1:2*O])]
            # errorEstTrain = mean(reduce(+, [abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]) for a in bootstrapOutTrain2])/i)

            errorEstTrain = mean(mapreduce(a -> abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]), +, bootstrapOutTrain2[1:i])/i)
      
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end
		
			# combinedOutputTestPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTest[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTest2[1:i]]))] 
            combinedOutputTestPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTest[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTest2[1:i])/i)]
            combinedOutputTest = [combinedOutputTestPre[:, 1:O] log.(1 ./combinedOutputTestPre[:, O+1:2*O])]
            # errorEstTest = mean(reduce(+, [abs.(a .- [combinedOutputTest[:, 1:O] combinedOutputTestPre[:, O+1:2*O]]) for a in bootstrapOutTest2])/i)
            errorEstTest = mean(mapreduce(a -> abs.(a .- [combinedOutputTest[:, 1:O] combinedOutputTestPre[:, O+1:2*O]]), +, bootstrapOutTest2[1:i])/i)
            
            Jtest= if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTest), d_ytest, costFunc = costFunc)
            else
            	calcError(combinedOutputTest, Ytest, costFunc = costFunc)
            end

            [i Jtrain[1] errorEstTrain Jtest[1] errorEstTest Jtrain[2] Jtest[2]]
        else
            combinedOutputTrain = reduce(+, bootstrapOutTrain[1:i])/i
            errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, bootstrapOutTrain[1:i])/i)  
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end

			combinedOutputTest = reduce(+, bootstrapOutTest[1:i])/i
            errorEstTest = mean(mapreduce(a -> abs.(a - combinedOutputTest), +, bootstrapOutTest[1:i])/i)  
            Jtest= if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTest), d_ytest, costFunc = costFunc)
            else
            	calcError(combinedOutputTest, Ytest, costFunc = costFunc)
            end

            [i Jtrain errorEstTrain Jtest errorEstTest]
        end
        println(string("Done with networks 1 to ", i, " out of ", length(bootstrapOutTrain)))
        line
    end 


    println("saving results to file")
	writefiles && writedlm(string(ID, "_multiPerformance_", filename, ".csv"), [header; fullMultiPerformance])			
	return (fullMultiPerformance, bootstrapOut)
end

function multiTrain(name, Xraw::U, Y::U, Xtestraw::U, Ytest::U, numEpochs, batchSize, hidden, lambda, c, alpha, R, num, ID; sampleCols = [], dropout = 0.0f0, printProg = false, printanything = true, costFunc = "absErr", writefiles = true, reslayers = 0, toltest = Inf, swa = false, multiparams=(), blasthreads=0, ignorebest=false, lockepochs=false, minepoch=0, trainsample=1.0) where U <: Matrix{Float32}

	X, Xtest = if isempty(sampleCols)
		(Xraw, Xtestraw)
	else
		(Xraw[:, sampleCols], Xtestraw[:, sampleCols])
	end

	colNames = if isempty(sampleCols)
		""
	elseif length(sampleCols) > 10
		replace(string(sampleCols[1:3], "_", sampleCols[end-2:end], "_"), " " => "")
	else
		replace(string(sampleCols, "_"), " " => "")
	end

	(N, M) = size(X)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	swastr = swa ? "SWA" : ""
	
	h_name = if isempty(hidden)
		0
	elseif hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	filename = 	string(colNames, name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", round(alpha, digits = 3), "_alpha_", formIndicString(R, lambda, c, dropout), "ADAMAX$(swastr)_", costFunc)

	if lockepochs
		if ID == 1
            Random.seed!(1234)
        else
            Random.seed!(ID)
            Random.seed!(1234+rand(UInt32)+1)
        end
  
		T0, B0 = if isempty(multiparams)
			if occursin("Log", costFunc)
				initializeparams_saxe(M, hidden, 2*O, reslayers)
			else
				initializeparams_saxe(M, hidden, O, reslayers)
			end		
		else
			multiparams[1]
		end

		if ID == 1
            Random.seed!(1234)
        else
            Random.seed!(ID)
            Random.seed!(1234+rand(UInt32)+1)
        end	

        println("Prepping batch and test data")
		(inputbatchData, outputbatchData) = generateBatches(X, Y, batchSize)
		if backend == :GPU
			testbatches = generateBatches(Xtest, Ytest, batchSize)
			batchInputs = device_allocate(inputbatchData)
			batchOutputs = device_allocate(outputbatchData)
			# d_testinput = cuda_allocate(Xtest)
			# d_testoutput = cuda_allocate(Ytest)
			batchInputsTest = device_allocate(testbatches[1])
			batchOutputsTest = device_allocate(testbatches[2])
			# prepdata = (batchInputs, batchOutputs, d_testinput, d_testoutput)
			prepdata = (batchInputs, batchOutputs, batchInputsTest, batchOutputsTest)
		else
			prepdata = (inputbatchData, outputbatchData)
		end

		println("Prepping activation data")
		if backend == :GPU
			d_Thetas = FCANN.device_allocate(T0) 

			tanh_grad_zBATCH = FCANN.form_tanh_grads(hidden, batchSize)
			d_tanh_grad_zBATCH = FCANN.device_allocate(tanh_grad_zBATCH)
				
			#initialize activations and deltas on device
			d_aBATCH = FCANN.form_activations(d_Thetas, batchSize)
			d_deltasBATCH = FCANN.form_activations(d_Thetas, batchSize)
			prepactivations = (d_tanh_grad_zBATCH, d_aBATCH, d_deltasBATCH)
		else
			tanh_grad_zBATCH = form_tanh_grads(hidden_layers, batchSize)
			aBATCH = form_activations(T0, batchSize)
			deltasBATCH = form_activations(T0, batchSize)
			prepactivations = (tanh_grad_zBATCH, aBATCH, daltasBATCH)
		end

		(T, B, bestCost, costRecord, timeRecord, GFLOPS, bestCostTest, costRecordTest, lastepoch, bestresultepoch) = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, numEpochs, M, hidden, lambda, c, R = R, alpha=alpha, dropout=dropout, printProgress = printProg, printAnything = printanything, costFunc = costFunc, resLayers = reslayers, tol = toltest, swa=swa, ignorebest=ignorebest, minepoch=minepoch, prepdata=prepdata, prepactivations=prepactivations, trainsample=trainsample)
		multiOut1 = (T, B, bestCost, costRecord, timeRecord, GFLOPS, bestCostTest, costRecordTest, lastepoch, bestresultepoch)
		multiOut2 = pmap(2:num) do foo
			printanything && printstyled("Training network $foo out of $num", color=:yellow, bold=true)
			if (nprocs() > 1) & (backend == :CPU)
				BLAS.set_num_threads(min(round(Int64, min(8, Sys.CPU_THREADS/4)), max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), num)))))
			else
				BLAS.set_num_threads(blasthreads)
			end

			if ID == 1
	            Random.seed!(1234 + foo - 1)
	        else
	            Random.seed!(ID)
	            Random.seed!(1234+rand(UInt32)+foo)
	        end
	  
			T0, B0 = if isempty(multiparams)
				if occursin("Log", costFunc)
					initializeparams_saxe(M, hidden, 2*O, reslayers)
				else
					initializeparams_saxe(M, hidden, O, reslayers)
				end		
			else
				multiparams[foo]
			end

			if ID == 1
	            Random.seed!(1234 + foo - 1)
	        else
	            Random.seed!(ID)
	            Random.seed!(1234+rand(UInt32)+foo)
	        end	
			(T, B, bestCost, costRecord, timeRecord, GFLOPS, bestCostTest, costRecordTest, lastepoch, bestresultepoch) = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, bestresultepoch, M, hidden, lambda, c, R = R, alpha=alpha, dropout=dropout, printProgress = printProg, printAnything = printanything, costFunc = costFunc, resLayers = reslayers, tol = toltest, swa=swa, ignorebest=true, prepdata=prepdata, prepactivations=prepactivations, trainsample=trainsample)
		end
		multiOut = [multiOut1; multiOut2]
	else


		T0, B0 = if isempty(multiparams)
			if occursin("Log", costFunc)
				initializeparams_saxe(M, hidden, 2*O, reslayers)
			else
				initializeparams_saxe(M, hidden, O, reslayers)
			end		
		else
			multiparams[1]
		end
	

        println("Prepping batch and test data")
		(inputbatchData, outputbatchData) = generateBatches(X, Y, batchSize)
		if backend == :GPU
			testbatches = generateBatches(Xtest, Ytest, batchSize)
			batchInputs = device_allocate(inputbatchData)
			batchOutputs = device_allocate(outputbatchData)
			# d_testinput = cuda_allocate(Xtest)
			# d_testoutput = cuda_allocate(Ytest)
			batchInputsTest = device_allocate(testbatches[1])
			batchOutputsTest = device_allocate(testbatches[2])
			# prepdata = (batchInputs, batchOutputs, d_testinput, d_testoutput)
			prepdata = (batchInputs, batchOutputs, batchInputsTest, batchOutputsTest)
		else
			prepdata = (inputbatchData, outputbatchData)
		end
		println("Prepping activation data")
		if backend == :GPU
			d_Thetas = FCANN.device_allocate(T0) 

			tanh_grad_zBATCH = FCANN.form_tanh_grads(hidden, batchSize)
			d_tanh_grad_zBATCH = FCANN.device_allocate(tanh_grad_zBATCH)
				
			#initialize activations and deltas on device
			d_aBATCH = FCANN.form_activations(d_Thetas, batchSize)
			d_deltasBATCH = FCANN.form_activations(d_Thetas, batchSize)
			prepactivations = (d_tanh_grad_zBATCH, d_aBATCH, d_deltasBATCH)
		else
			tanh_grad_zBATCH = form_tanh_grads(hidden, batchSize)
			aBATCH = form_activations(T0, batchSize)
			deltasBATCH = form_activations(T0, batchSize)
			prepactivations = (tanh_grad_zBATCH, aBATCH, deltasBATCH)
		end

		multiOut = map(1:num) do foo
			#BLAS.set_num_threads(Sys.CPU_THREADS)
			printanything && printstyled("Training network $foo out of $num", color=:yellow, bold=true)
			if (nprocs() > 1) & (backend == :CPU)
				BLAS.set_num_threads(min(round(Int64, min(8, Sys.CPU_THREADS/4)), max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), num)))))
			else
				BLAS.set_num_threads(blasthreads)
			end

			if ID == 1
	            Random.seed!(1234 + foo - 1)
	        else
	            Random.seed!(ID)
	            Random.seed!(1234+rand(UInt32)+foo)
	        end
	  
			T0, B0 = if isempty(multiparams)
				if occursin("Log", costFunc)
					initializeparams_saxe(M, hidden, 2*O, reslayers)
				else
					initializeparams_saxe(M, hidden, O, reslayers)
				end		
			else
				multiparams[foo]
			end

			if ID == 1
	            Random.seed!(1234 + foo - 1)
	        else
	            Random.seed!(ID)
	            Random.seed!(1234+rand(UInt32)+foo)
	        end	
			(T, B, bestCost, costRecord, timeRecord, GFLOPS, bestCostTest, costRecordTest, lastepoch, bestresultepoch) = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, numEpochs, M, hidden, lambda, c, R = R, alpha=alpha, dropout=dropout, printProgress = printProg, printAnything = printanything, costFunc = costFunc, resLayers = reslayers, tol = toltest, swa=swa, ignorebest=ignorebest, minepoch=minepoch, prepdata=prepdata, prepactivations=prepactivations, trainsample=trainsample)
		end
	end
	GC.gc()
	bootstrapOut = [(a[1], a[2]) for a in multiOut]
	resultepochs = [a[10] for a in multiOut]
	fileout = convert(Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, bootstrapOut)
	writefiles && writeParams(fileout, string(ID, "_multiParams_", filename, ".bin"))
	
	printanything && println("Calculating training set outputs")
	(bootstrapOutTrain, outTrain, errTrain, errEstTrain) = calcMultiOut(X, Y, bootstrapOut, dropout = dropout, costFunc = costFunc, resLayers=reslayers)#pmap(a -> calcOutput(X, Y, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
    GC.gc()
    printanything && println("Calculating test set outputs")
    (bootstrapOutTest, outTest, errTest, errEstTest) = calcMultiOut(Xtest, Ytest, bootstrapOut, dropout = dropout, costFunc = costFunc, resLayers=reslayers) #pmap(a -> calcOutput(Xtest, Ytest, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
	
	header = if occursin("Log", costFunc)
		[reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value Error Est ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range Error Est ", s), 1:O), 1, O)]
	else
		header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Error Est ", s), 1:O), 1, O)]
	end
	
	printanything && println("Writing prediction scatters to file")
	writefiles && writedlm(string(ID, "_multiPredScatTrain_", filename, ".csv"), [header; [Y outTrain errEstTrain]])
	writefiles && writedlm(string(ID, "_multiPredScatTest_", filename, ".csv"), [header; [Ytest outTest errEstTest]])

	header = if costFunc2 == costFunc
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est"]
	else
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est" string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error")]
	end

	printanything && println("Calculating performance vs number of networks")
	d_y = if backend == :GPU
		cuda_allocate(Y)
	else
		[]
	end

	d_ytest = if backend == :GPU
		cuda_allocate(Ytest)
	else
		[]
	end
    	
	bootstrapOutTrain2 = if occursin("Log", costFunc)
		println("Calculating training non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTrain]
	else
		[]
	end

	bootstrapOutTest2 = if occursin("Log", costFunc)
		println("Calculating test non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTest]
	else
		[]
	end

    fullMultiErrors = map(1:length(bootstrapOutTrain)) do i
        #calculate average network output     
        line = if occursin("Log", costFunc)
            # combinedOutputTrainPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTrain[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTrain2[1:i]]))]  
            combinedOutputTrainPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTrain[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTrain2[1:i])/i)]
            combinedOutputTrain = [combinedOutputTrainPre[:, 1:O] log.(1 ./combinedOutputTrainPre[:, O+1:2*O])]
            # errorEstTrain = mean(reduce(+, [abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]) for a in bootstrapOutTrain2])/i)

            errorEstTrain = mean(mapreduce(a -> abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]), +, bootstrapOutTrain2[1:i])/i)
      
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end
		
			# combinedOutputTestPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTest[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTest2[1:i]]))] 
            combinedOutputTestPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTest[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTest2[1:i])/i)]
            combinedOutputTest = [combinedOutputTestPre[:, 1:O] log.(1 ./combinedOutputTestPre[:, O+1:2*O])]
            # errorEstTest = mean(reduce(+, [abs.(a .- [combinedOutputTest[:, 1:O] combinedOutputTestPre[:, O+1:2*O]]) for a in bootstrapOutTest2])/i)
            errorEstTest = mean(mapreduce(a -> abs.(a .- [combinedOutputTest[:, 1:O] combinedOutputTestPre[:, O+1:2*O]]), +, bootstrapOutTest2[1:i])/i)
            
            Jtest= if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTest), d_ytest, costFunc = costFunc)
            else
            	calcError(combinedOutputTest, Ytest, costFunc = costFunc)
            end

            ([i Jtrain[1] errorEstTrain Jtest[1] errorEstTest Jtrain[2] Jtest[2]], combinedOutputTrain, combinedOutputTest)
        else
            combinedOutputTrain = reduce(+, bootstrapOutTrain[1:i])/i
            errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, bootstrapOutTrain[1:i])/i)  
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end

			combinedOutputTest = reduce(+, bootstrapOutTest[1:i])/i
            errorEstTest = mean(mapreduce(a -> abs.(a - combinedOutputTest), +, bootstrapOutTest[1:i])/i)  
            Jtest= if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTest), d_ytest, costFunc = costFunc)
            else
            	calcError(combinedOutputTest, Ytest, costFunc = costFunc)
            end

            ([i Jtrain errorEstTrain Jtest errorEstTest], combinedOutputTrain, combinedOutputTest)
        end
        printanything && println(string("Done with networks 1 to ", i, " out of ", length(bootstrapOutTrain)))
        line
    end 

    fullMultiPerformance = mapreduce(a -> a[1], vcat, fullMultiErrors)

    testerrs = fullMultiPerformance[:, 4]
    (besterr, bestind) = findmin(testerrs)
    println("Completed training networks with parameters: $numEpochs epochs, $(length(hidden)) layers of size $(isempty(hidden) ? 0 : hidden[1]), $c maxnorm, $R decay rate, $alpha alpha, $dropout dropout, and $reslayers reslayer size")
    println("Best test error is $besterr using $bestind networks")

    printanything && println("saving results to file")
	writefiles && writedlm(string(ID, "_multiPerformance_", filename, ".csv"), [header; fullMultiPerformance])			
	return (fullMultiPerformance, bootstrapOut, resultepochs, fullMultiErrors[end][2], fullMultiErrors[end][3])
end

function multiTrain(name, Xraw::U, Y::U, numEpochs, batchSize, hidden, lambda, c, alpha, R, num, ID; sampleCols = [], dropout = 0.0f0, printProg = false, printanything = true, costFunc = "absErr", writefiles = true, reslayers = 0, toltest = Inf, swa = false, multiparams=(), blasthreads=0, ignorebest=false) where U <: Matrix{Float32}

	X = if isempty(sampleCols)
		Xraw
	else
		Xraw[:, sampleCols]
	end

	colNames = if isempty(sampleCols)
		""
	elseif length(sampleCols) > 10
		replace(string(sampleCols[1:3], "_", sampleCols[end-2:end], "_"), " " => "")
	else
		replace(string(sampleCols, "_"), " " => "")
	end

	(N, M) = size(X)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	swastr = swa ? "SWA" : ""
	
	h_name = if isempty(hidden)
		0
	elseif hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	filename = 	string(colNames, name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", round(alpha, digits = 3), "_alpha_", formIndicString(R, lambda, c, dropout), "ADAMAX$(swastr)_", costFunc)

	multiout = pmap(1:num) do foo
		printanything && printstyled("Training network $foo out of $num", color=:yellow, bold=true)
		#BLAS.set_num_threads(Sys.CPU_THREADS)
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(round(Int64, min(8, Sys.CPU_THREADS)/4), max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), num)))))
		else
			BLAS.set_num_threads(blasthreads)
		end

		if ID == 1
            Random.seed!(1234 + foo - 1)
        else
            Random.seed!(ID)
            Random.seed!(1234+rand(UInt32)+foo)
        end
  
		T0, B0 = if isempty(multiparams)
			if occursin("Log", costFunc)
				initializeparams_saxe(M, hidden, 2*O, reslayers)
			else
				initializeparams_saxe(M, hidden, O, reslayers)
			end		
		else
			multiparams[foo]
		end

		if ID == 1
            Random.seed!(1234 + foo - 1)
        else
            Random.seed!(ID)
            Random.seed!(1234+rand(UInt32)+foo)
        end	
		(T, B, bestCost, costRecord, timeRecord, GFLOPS, bestepoch) = eval(Symbol("ADAMAXTrainNN", backend))(((X, Y),), batchSize, T0, B0, numEpochs, M, hidden, lambda, c, R = R, alpha=alpha, dropout=dropout, printProgress = printProg, printAnything = printanything, costFunc = costFunc, resLayers = reslayers, tol = toltest, ignorebest=ignorebest, swa=swa)
	end
	bestepochs = [a[7] for a in multiout]
	bootstrapOut = [(a[1], a[2]) for a in multiout]
	GC.gc()
	fileout = convert(Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, bootstrapOut)
	writefiles && writeParams(fileout, string(ID, "_multiParams_", filename, ".bin"))
	
	printanything && println("Calculating training set outputs")
	(bootstrapOutTrain, outTrain, errTrain, errEstTrain) = calcMultiOut(X, Y, bootstrapOut, dropout = dropout, costFunc = costFunc, resLayers=reslayers)#pmap(a -> calcOutput(X, Y, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
    GC.gc()
	
	header = if occursin("Log", costFunc)
		[reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value Error Est ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range Error Est ", s), 1:O), 1, O)]
	else
		header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Error Est ", s), 1:O), 1, O)]
	end
	
	printanything && println("Writing prediction scatters to file")
	writefiles && writedlm(string(ID, "_multiPredScatTrain_", filename, ".csv"), [header; [Y outTrain errEstTrain]])

	header = if costFunc2 == costFunc
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est"]
	else
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Train ", costFunc2, " Error")]
	end

	printanything && println("Calculating performance vs number of networks")
	d_y = if backend == :GPU
		cuda_allocate(Y)
	else
		[]
	end
    	
	bootstrapOutTrain2 = if occursin("Log", costFunc)
		println("Calculating training non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTrain]
	else
		[]
	end

    fullMultiPerformance = mapreduce(vcat, 1:length(bootstrapOutTrain)) do i
        #calculate average network output     
        line = if occursin("Log", costFunc)
            # combinedOutputTrainPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTrain[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTrain2[1:i]]))]  
            combinedOutputTrainPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTrain[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTrain2[1:i])/i)]
            combinedOutputTrain = [combinedOutputTrainPre[:, 1:O] log.(1 ./combinedOutputTrainPre[:, O+1:2*O])]
            # errorEstTrain = mean(reduce(+, [abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]) for a in bootstrapOutTrain2])/i)

            errorEstTrain = mean(mapreduce(a -> abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]), +, bootstrapOutTrain2[1:i])/i)
      
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end
		

            [i Jtrain[1] errorEstTrain Jtrain[2]]
        else
            combinedOutputTrain = reduce(+, bootstrapOutTrain[1:i])/i
            errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, bootstrapOutTrain[1:i])/i)  
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end

            [i Jtrain errorEstTrain]
        end
        printanything && println(string("Done with networks 1 to ", i, " out of ", length(bootstrapOutTrain)))
        line
    end 
    errs = fullMultiPerformance[:, 2]
    (besterr, bestind) = findmin(errs)
    println("Completed training networks with parameters: $numEpochs epochs, $(length(hidden)) layers of size $(isempty(hidden) ? 0 : hidden[1]), $c maxnorm, $R decay rate, $alpha alpha, $dropout dropout, and $reslayers reslayer size")
    println("Best error is $besterr using $bestind networks")

    printanything && println("saving results to file")
	writefiles && writedlm(string(ID, "_multiPerformance_", filename, ".csv"), [header; fullMultiPerformance])			
	return (fullMultiPerformance, bootstrapOut, bestepochs)
end

function evalMulti(name, hidden, lambdaeta, c, alpha, R; sampleCols = [], IDList = [], adv = false, dropout=0.0f0, costFunc = "absErr", binInput = false)
	println("reading and converting training data")
	Xraw, Xtestraw, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end

	X, Xtest = if isempty(sampleCols)
		(Xraw, Xtestraw)
	else
		(Xraw[:, sampleCols], Xtestraw[:, sampleCols])
	end

	# meanY = Float32.(readcsv(string("invY_values_", name, ".csv"))[2:end, 1])
	# varY = Float32.(readcsv(string("invY_values_", name, ".csv"))[2:end, 2])
	
	(N, M) = size(X)
	O = size(Y, 2)
	
	h_name = if isempty(hidden)
		0
	elseif hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	colNames = if isempty(sampleCols)
		""
	elseif length(sampleCols) > 10
		replace(string(sampleCols[1:3], "_", sampleCols[end-2:end], "_"), " " => "")
	else
		replace(string(sampleCols, "_"), " " => "")
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	filename = if adv
		string(colNames, name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", c, "_maxNorm_", lambdaeta, "_advNoise_", alpha, "_alpha_AdvADAMAX", backend, "_", costFunc)
	else
		string(colNames, name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", alpha, "_alpha_", formIndicString(R, lambdaeta, c, dropout), "ADAMAX_", costFunc)
		# if dropout == 0.0f0
		# 	string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambdaeta, "_L2_", c, "_maxNorm_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
		# else
		# 	string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", lambdaeta, "_L2_", c, "_maxNorm_", dropout, "_dropoutRate_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
		# end
	end
	
	function hcatRobust(A, B)
		if isempty(A)
			B
		elseif isempty(B)
			A
		else
			[A B]
		end
	end

	function vcatRobust(A, B)
		if isempty(A)
			B
		elseif isempty(B)
			A
		else
			[A; B]
		end
	end	
	
	println("compiling all multi parameters together")
	#combine bootstram params from each ID into one large matrix
	multiOut = if isempty(IDList)
		if isfile(string("fullMultiParams_", filename, ".csv"))
			params = map(Float32, readdlm(string("fullMultiParams_", filename, ".csv")))
			map(1:size(params, 2)) do col
				p = params[:, col]
				(T, B) = if occursin("Log", costFunc)
					params2Theta(M, hidden, 2*O, p)
				else
					params2Theta(M, hidden, O, p)
				end					
			end
		else
			readBinParams(string("fullMultiParams_", filename, ".bin"))
		end
	else
		if isfile(string(IDList[1], "_multiParams_", filename, ".csv"))
			mapreduce(vcatRobust, IDList) do ID
				try
                    params = map(Float32, readdlm(string(ID, "_multiParams_", filename, ".csv"))) 
        			map(1:size(params, 2)) do col
        				p = params[:, col]
        				(T, B) = if occursin("Log", costFunc)
      						params2Theta(M, hidden, 2*O, p)
 						else
 	   						params2Theta(M, hidden, O, p)
 						end
        			end
                catch
                    []
                end
			end
		else
			mapreduce(vcatRobust, IDList)  do ID
				try
                    readBinParams(string(ID, "_multiParams_", filename, ".bin"))
                catch
                    []
                end
			end
		end
	end
	
	println("saving combined parameters")
	writeParams(convert(Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, multiOut), string("fullMultiParams_", filename, ".bin"))
	println(string("calculating combined outputs for ", length(multiOut), " networks"))
	GC.gc()
	# BLAS.set_num_threads(0)
	# num = length(multiOut)
 #  	if (nprocs() > 1) & (FCANN.backend == :CPU)
 #        BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), num)))))
 #    end

 	println("Calculating training set outputs")
	(bootstrapOutTrain, outTrain, errTrain, errEstTrain) = calcMultiOut(X, Y, multiOut, dropout = dropout, costFunc = costFunc)#pmap(a -> calcOutput(X, Y, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)
    GC.gc()
    println("Calculating test set outputs")
    (bootstrapOutTest, outTest, errTest, errEstTest) = calcMultiOut(Xtest, Ytest, multiOut, dropout = dropout, costFunc = costFunc) #pmap(a -> calcOutput(Xtest, Ytest, a[1], a[2], dropout = dropout, costFunc = costFunc)[1], bootstrapOut)

    header = if  occursin("Log", costFunc)
		[reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value Error Est ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Range Error Est ", s), 1:O), 1, O)]
	else
		header = [reshape(map(s -> string("Output ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Value ", s), 1:O), 1, O) reshape(map(s -> string("Prediction Error Est ", s), 1:O), 1, O)]
	end

	println("Writing prediction scatters to file")
	writedlm(string("multiPredScatTrain_", filename, ".csv"), [header; [Y outTrain errEstTrain]])
	writedlm(string("multiPredScatTest_", filename, ".csv"), [header; [Ytest outTest errEstTest]])
	
	header = if costFunc2 == costFunc
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est"]
	else
		["Num Networks" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est" string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error")]
	end
    	
	#calculate average network output
	println("Calculating performance vs number of networks")
	d_y = if backend == :GPU
		cuda_allocate(Y)
	else
		[]
	end

	d_ytest = if backend == :GPU
		cuda_allocate(Ytest)
	else
		[]
	end

	bootstrapOutTrain2 = if  occursin("Log", costFunc)
		println("Calculating training non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTrain]
	else
		[]
	end

	bootstrapOutTest2 = if occursin("Log", costFunc)
		println("Calculating test non inverted ranges")
		[[a[:, 1:O] exp.(-a[:, O+1:2*O])] for a in bootstrapOutTest]
	else
		[]
	end

    fullMultiPerformance = mapreduce(vcat, 1:length(bootstrapOutTrain)) do i
        #calculate average network output     
        line = if occursin("Log", costFunc)
            # combinedOutputTrainPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTrain[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTrain2[1:i]]))]  
            combinedOutputTrainPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTrain[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTrain2[1:i])/i)]
            combinedOutputTrain = [combinedOutputTrainPre[:, 1:O] log.(1 ./combinedOutputTrainPre[:, O+1:2*O])]
            # errorEstTrain = mean(reduce(+, [abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]) for a in bootstrapOutTrain2])/i)

            errorEstTrain = mean(mapreduce(a -> abs.(a .- [combinedOutputTrain[:, 1:O] combinedOutputTrainPre[:, O+1:2*O]]), +, bootstrapOutTrain2[1:i])/i)
      
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end
		
			# combinedOutputTestPre = [reduce(+, [a[:, 1:O] for a in bootstrapOutTest[1:i]])/i sqrt.(reduce(+, [a[:, O+1:2*O].^2 for a in bootstrapOutTest2[1:i]]))] 
            combinedOutputTestPre = [mapreduce(a -> a[:, 1:O], +, bootstrapOutTest[1:i])/i sqrt.(mapreduce(a -> a[:, O+1:2*O].^2, +, bootstrapOutTest2[1:i])/i)]
            combinedOutputTest = [combinedOutputTestPre[:, 1:O] log.(1 ./combinedOutputTestPre[:, O+1:2*O])]
            # errorEstTest = mean(reduce(+, [abs.(a .- [combinedOutputTest[:, 1:O] combinedOutputTestPre[:, O+1:2*O]]) for a in bootstrapOutTest2])/i)
            errorEstTest = mean(mapreduce(a -> abs.(a .- [combinedOutputTest[:, 1:O] combinedOutputTestPre[:, O+1:2*O]]), +, bootstrapOutTest2[1:i])/i)
            
            Jtest= if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTest), d_ytest, costFunc = costFunc)
            else
            	calcError(combinedOutputTest, Ytest, costFunc = costFunc)
            end

            [i Jtrain[1] errorEstTrain Jtest[1] errorEstTest Jtrain[2] Jtest[2]]
        else
            combinedOutputTrain = reduce(+, bootstrapOutTrain[1:i])/i
            errorEstTrain = mean(mapreduce(a -> abs.(a - combinedOutputTrain), +, bootstrapOutTrain[1:i])/i)  
            Jtrain = if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTrain), d_y, costFunc = costFunc)
            else
            	calcError(combinedOutputTrain, Y, costFunc = costFunc)
            end

			combinedOutputTest = reduce(+, bootstrapOutTest[1:i])/i
            errorEstTest = mean(mapreduce(a -> abs.(a - combinedOutputTest), +, bootstrapOutTest[1:i])/i)  
            Jtest= if backend == :GPU
            	calcError(cuda_allocate(combinedOutputTest), d_ytest, costFunc = costFunc)
            else
            	calcError(combinedOutputTest, Ytest, costFunc = costFunc)
            end

            [i Jtrain errorEstTrain Jtest errorEstTest]
        end
        println(string("Done with networks 1 to ", i, " out of ", length(bootstrapOutTrain)))
        line
    end 
		
	println("saving results to file")	
	writedlm(string("fullMultiPerformance_", filename, ".csv"), [header; fullMultiPerformance])		
	return fullMultiPerformance
end

function testTrain(M::Int64, hidden::Array{Int64, 1}, O::Int64, batchSize::Int64, N::Int64; multi = false, writeFile = true, numThreads = 0, printProg = false, costFunc = "absErr", dropout = 0.0f0, reslayers=0, swa=false)
	#generate training set with m examples
	m = 102400
	X = randn(Float32, m, M)
	params1 = randn(Float32, M, O)
	params2 = randn(Float32, M, O)
	Y = (X*params1) .+ (X*params2) .^2 .+ randn(Float32, m, O)

	Xtest = randn(Float32, 1024, M)
	ytest = (Xtest*params1) .+ (Xtest*params2) .^2

	#if multi is true, run training tasks across workers
	num = if multi & (backend == :CPU)
		nprocs() - 1
	else
		1
	end

	swastr = swa ? "SWA" : ""

	numBatches = ceil(Int, m/batchSize)
	
	#number of total perations per batch
	(fops, bops, pops) = calcOps(M, hidden, O, batchSize)
	total_ops = fops + bops + pops
	
	T0, B0 = if occursin("Log", costFunc)
		initializeParams(M, hidden, 2*O)
	else
		initializeParams(M, hidden, O)
	end	
	out = pmap(1:num) do _
		BLAS.set_num_threads(numThreads)
		eval(Symbol("ADAMAXTrainNN", backend))(((X, Y), (Xtest, ytest)), batchSize, T0, B0, N, M, hidden, 0.0f0, Inf; printProgress = printProg, costFunc = costFunc, dropout = dropout, resLayers=reslayers, swa=swa, ignorebest=true)
	end
	slowestInd = argmax(map(a -> a[end][end], out))
	(bestThetas, bestBiases, finalCost, costRecord, timeRecord) = out[slowestInd]
	train_time = timeRecord[end]
	timePerBatch = train_time/N/numBatches
	
	f = IOBuffer()
	Sys.cpu_summary(f)
	cpu_info = String(take!(f))
	cpu_name = strip(split(cpu_info, ':')[1])

	gpu_name = if backend == :GPU
		cuDeviceGetName(current_device)
	else
		""
	end
	
	if backend == :CPU
		println(string("Completed benchmark with ", M, " input ", hidden, " hidden ", O, " output, and ", batchSize, " batchSize on a ", cpu_name))
	else
		println(string("Completed benchmark with ", M, " input ", hidden, " hidden ", O, " output, and ", batchSize, " batchSize on a ", gpu_name))
	end
	
	println("Time to train on ", backend, " took ", train_time, " seconds for ", N, " epochs")
	println("Average time of ", timePerBatch/batchSize/1e-9, " ns per example")
	println("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize)
	if num > 1
		println("Approximate GFLOPS = ", total_ops/timePerBatch/1e9, " per network and = ", num*total_ops/timePerBatch/1e9, " total")
	else
		println("Approximate GFLOPS = ", total_ops/timePerBatch/1e9)
	end

	filename = if backend == :GPU
		string(M, "_input_", hidden, "_hidden_", O, "_output_", batchSize, "_batchSize_", replace(cpu_name, ' ' => '_'), "_", replace(gpu_name, ' ' => '_'), "_", dropout, "_dropout_", costFunc, "_timingBenchmark.csv")
	elseif num > 1
		string(M, "_input_", hidden, "_hidden_", O, "_output_", batchSize, "_batchSize_", replace(cpu_name, ' ' => '_'), "_", dropout, "_dropout_", costFunc, "_", num, "_parallelTasks_", numThreads, "_BLASThreads_timingBenchmark.csv")
	else
		string(M, "_input_", hidden, "_hidden_", O, "_output_", batchSize, "_batchSize_", replace(cpu_name, ' ' => '_'), "_", dropout, "_dropout_", costFunc, "_", numThreads, "_BLASThreads_timingBenchmark.csv")
	end

	time_per_epoch = timeRecord[2:end] .- timeRecord[1:end-1]
	GFLOPS_per_epoch = total_ops *numBatches ./ time_per_epoch / 1e9
	header = ["Epoch" "Time" "GFLOPS per Task" "Total GFLOPS"]
	
	if writeFile
		writedlm(filename, [header; [1:N timeRecord[2:end] GFLOPS_per_epoch num*GFLOPS_per_epoch]])
	end
	
	return (median(GFLOPS_per_epoch), median(GFLOPS_per_epoch*num), median(time_per_epoch), replace(cpu_name, ' ' => '_'), replace(gpu_name, ' ' => '_'), out)
end

#train a network with a variable number of layers for a given target number
#of parameters.
function smartEvalLayers(name, N, batchSize, Plist; tau = 0.01f0, layers = [2, 4, 6, 8, 10, 12, 14, 16], dropout = 0.0f0, costFunc = "absErr", binInput = false, resLayers = 0)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end

	M = size(X, 2)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	dropoutStr = (dropout == 0.0f0) ? "" : string(dropout, "_dropoutRate_")
	resStr = (resLayers == 0) ? "" : string(resLayers, "_resLayers_")

	filename = string(name, "_", M, "_input_", O, "_output_", resStr, dropoutStr, N, "_epochs_smartParams_ADAMAX", backend, "_", costFunc, ".csv")

	#determine number of layers to test in a range
	fatHiddenList = mapreduce(vcat, Plist) do P 
		map(layers) do l
			L = resLayers == 0 ? l : l*resLayers + 1
			H = if occursin("Log", costFunc)
				ceil(Int64, getHiddenSize(M, 2*O, L, P))
			else
				ceil(Int64, getHiddenSize(M, O, L, P))
			end
			if H == 0
				(P, Int64.([]))
			else
				(P, H*ones(Int64, L))
			end
		end
	end

	hiddenLayers = map(a -> a[2], fatHiddenList)
	hiddenList = map(H -> fatHiddenList[findall(a -> a == H, hiddenLayers)[1]], unique(hiddenLayers))

	header = if costFunc2 == costFunc
		["Layers" "Num Params" "Target Num Params" "H" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") "Alpha" "Decay Rate" "Time Per Epoch" "Median GFLOPS" "Opt Success"]
	else
		["Layers" "Num Params" "Target Num Params" "H" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error") "Alpha" "Decay Rate" "Time Per Epoch" "Median GFLOPS" "Opt Success"]
	end

	body = reduce(vcat, pmap(hiddenList) do hidden # @parallel (vcat) for hidden in hiddenList
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), length(hiddenList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		# if (nprocs() > 1) & (backend == :CPU)
		# 	BLAS.set_num_threads(max(1, ceil(Int64, Sys.CPU_THREADS / min(nprocs(), length(hiddenList)))))
		# end
		println(string("training network with ", hidden[2], " hidden layers"))
		println("initializing network parameters")
		Random.seed!(1234)
		T0, B0 = if occursin("Log", costFunc)
			initializeparams_saxe(M, hidden[2], 2*O)
		else
			initializeparams_saxe(M, hidden[2], O)
		end	

		println("beginning training")
		Random.seed!(1234)
		alpha, R, (T, B, bestCost, record, timeRecord, GFLOPS), success = autoTuneParams(((X, Y),), batchSize, T0, B0, N, hidden[2], tau = tau, dropout = dropout, costFunc = costFunc, resLayers = resLayers)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers)

		numParams = length(theta2Params(B, T))

		Hsize = if isempty(hidden[2])
			0
		else
			hidden[2][1]
		end

		if costFunc == costFunc2
			[length(hidden[2]) numParams hidden[1] Hsize Jtrain Jtest alpha R median(timeRecord[2:end] - timeRecord[1:end-1]) median(GFLOPS) success]	
		else
			[length(hidden[2]) numParams hidden[1] Hsize Jtrain[1] Jtest[1] Jtrain[2] Jtest[2] alpha R median(timeRecord[2:end] - timeRecord[1:end-1]) median(GFLOPS) success]	
		end
	end)
	if isfile(string("evalLayers_", filename))
		f = open(string("evalLayers_", filename), "a")
		writedlm(f, body)
		close(f)
	else
		writedlm(string("evalLayers_", filename), [header; body])
	end
	return body
end

#train a network with a variable number of layers for a given target number
#of parameters.
function smartEvalLayers(name, (X, Y, Xtest, Ytest), N, batchSize, Plist; tau = 0.01f0, layers = [2, 4, 6, 8, 10, 12, 14, 16], dropout = 0.0f0, costFunc = "absErr", resLayers = 0)

	M = size(X, 2)
	O = size(Y, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	dropoutStr = (dropout == 0.0f0) ? "" : string(dropout, "_dropoutRate_")
	resStr = (resLayers == 0) ? "" : string(resLayers, "_resLayers_")

	filename = string(name, "_", M, "_input_", O, "_output_", resStr, dropoutStr, N, "_epochs_smartParams_ADAMAX", backend, "_", costFunc, ".csv")

	#determine number of layers to test in a range
	fatHiddenList = mapreduce(vcat, Plist) do P 
		map(layers) do L
			H = if occursin("Log", costFunc)
				ceil(Int64, getHiddenSize(M, 2*O, L, P))
			else
				ceil(Int64, getHiddenSize(M, O, L, P))
			end
			if H == 0
				(P, Int64.([]))
			else
				(P, H*ones(Int64, L))
			end
		end
	end

	hiddenLayers = map(a -> a[2], fatHiddenList)
	hiddenList = map(H -> fatHiddenList[findall(a -> a == H, hiddenLayers)[1]], unique(hiddenLayers))

	header = if costFunc2 == costFunc
		["Layers" "Num Params" "Target Num Params" "H" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") "Alpha" "Decay Rate" "Time Per Epoch" "Median GFLOPS" "Opt Success"]
	else
		["Layers" "Num Params" "Target Num Params" "H" string("Train ", costFunc, " Error") string("Test ", costFunc, " Error") string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error") "Alpha" "Decay Rate" "Time Per Epoch" "Median GFLOPS" "Opt Success"]
	end

	body = reduce(vcat, pmap(hiddenList) do hidden # @parallel (vcat) for hidden in hiddenList
		if (nprocs() > 1) & (backend == :CPU)
			BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), length(hiddenList))))))
		else
			BLAS.set_num_threads(0)
		end
		
		# if (nprocs() > 1) & (backend == :CPU)
		# 	BLAS.set_num_threads(max(1, ceil(Int64, Sys.CPU_THREADS / min(nprocs(), length(hiddenList)))))
		# end
		println(string("training network with ", hidden[2], " hidden layers"))
		println("initializing network parameters")
		Random.seed!(1234)
		T0, B0 = if occursin("Log", costFunc)
			initializeparams_saxe(M, hidden[2], 2*O)
		else
			initializeparams_saxe(M, hidden[2], O)
		end	

		println("beginning training")
		Random.seed!(1234)
		alpha, R, (T, B, bestCost, record, timeRecord, GFLOPS), success = autoTuneParams(((X, Y), (Xtest, Ytest)), batchSize, T0, B0, N, hidden[2], tau = tau, dropout = dropout, costFunc = costFunc, resLayers = resLayers)
		GC.gc()
		(outTrain, Jtrain) = calcOutput(X, Y, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers)
		GC.gc()
		(outTest, Jtest) = calcOutput(Xtest, Ytest, T, B, dropout = dropout, costFunc = costFunc, resLayers = resLayers)

		numParams = length(theta2Params(B, T))

		Hsize = if isempty(hidden[2])
			0
		else
			hidden[2][1]
		end

		if costFunc == costFunc2
			[length(hidden[2]) numParams hidden[1] Hsize Jtrain Jtest alpha R median(timeRecord[2:end] - timeRecord[1:end-1]) median(GFLOPS) success]	
		else
			[length(hidden[2]) numParams hidden[1] Hsize Jtrain[1] Jtest[1] Jtrain[2] Jtest[2] alpha R median(timeRecord[2:end] - timeRecord[1:end-1]) median(GFLOPS) success]	
		end
	end)
	if isfile(string("evalLayers_", filename))
		f = open(string("evalLayers_", filename), "a")
		writedlm(f, body, ',')
		close(f)
	else
		writedlm(string("evalLayers_", filename), [header; body], ',')
	end
	return body
end

function multiTrainAutoReg(name, numEpochs, batchSize, hidden, alpha, R; tau = 0.01f0, c0 = 1.0f0, num = 16, dropout = 0.0f0, printProg = false, costFunc="absErr", binInput = false)
	println("reading and converting training data")
	X, Xtest, Y, Ytest = if binInput
		readBinInput(name)
	else
		readInput(name)
	end
	(N, M) = size(X)
	O = size(Y, 2)
	
	h_name = if hidden == hidden[1]*ones(Int64, length(hidden))
		string(hidden[1], "X", length(hidden))
	else
		string(hidden)
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	filename = if dropout == 0.0f0
		string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
	else
		string(name, "_", M, "_input_", h_name, "_hidden_", O, "_output_", dropout, "_dropoutRate_", alpha, "_alpha_", R, "_decayRate_ADAMAX", backend, "_", costFunc)
	end

	header = if costFunc2 == costFunc
		["Max Norm" string("Train ", costFunc, " Error") "Training Error Est" string("Test ", costFunc, " Error") "Test Error Est" "Median Time Per Epoch" "Median GLFOPS"]
	else
		["Max Norm" string("Train ", costFunc, " Error")  "Training Error Est"  string("Test ", costFunc, " Error") "Test Error Est" string("Train ", costFunc2, " Error") string("Test ", costFunc2, " Error")  "Median Time Per Epoch" "Median GLFOPS"]
	end

	phi = 0.5f0*(1.0f0+sqrt(5.0f0))
	
	function runMultiTrain(c)	
		bootstrapOut = pmap(1:num) do foo
			Random.seed!(1234+foo-1)	
			T0, B0 = if occursin("Log", costFunc)
				initializeParams(M, hidden, 2*O)
			else
				initializeParams(M, hidden, O)
			end	
			if (nprocs() > 1) & (backend == :CPU)
				BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs(), num)))))
			else
				BLAS.set_num_threads(0)
			end

			Random.seed!(1234+foo-1)
			(T, B, bestCost, costRecord, timeRecord, GFLOPS) = eval(Symbol("ADAMAXTrainNN", backend))(X, Y, batchSize, T0, B0, numEpochs, M, hidden, 0.0f0, c, alpha=alpha, R = R, dropout=dropout, printProgress = printProg, costFunc = costFunc)
			(T, B, median(timeRecord[2:end] - timeRecord[1:end-1]), median(GFLOPS))
		end
		GC.gc()
		(bootstrapOutTrain, combinedOutputTrain, Jtrain, errorEstTrain) = calcMultiOut(X, Y, bootstrapOut, dropout = dropout, costFunc = costFunc)

		(bootstrapOutTest, combinedOutputTest, Jtest, errorEstTest) = calcMultiOut(Xtest, Ytest, bootstrapOut, dropout = dropout, costFunc = costFunc)

		if costFunc2 == costFunc
			(Jtest, [c Jtrain errorEstTrain Jtest errorEstTest maximum(map(a -> a[3], bootstrapOut)) minimum(map(a -> a[4], bootstrapOut))])
		else
			(Jtest[1], [c Jtrain[1] mean(errorEstTrain) Jtest[1] mean(errorEstTest) Jtrain[2] Jtest[2] maximum(map(a -> a[3], bootstrapOut)) minimum(map(a -> a[4], bootstrapOut))])
		end
	end

	println()
	println(string("Training network with maxnorm of ", c0))
	println()

	c1 = c0
	(J1, out1) = runMultiTrain(c1)
	
	println()
	println(string("Starting cost at c = ", c1, " is ", J1))
	println()

	println()
	println("Training network with maxnorm of 1.35")
	println()

	c2 = 1.35f0
	(J2, out2) = runMultiTrain(c2)

	(p1, p2, p3) = if J2 < J1
		println()
		println(string("Training error at c = ", c2, " is ", J2, " and lower than ", J1, " so search direction will be increasing in c"))
		println()

		c3 = (c2 - c1)*phi+c2
		(J3, out3) = runMultiTrain(c3)
		println()
		println(string("Training error at c = ", c3, " is ", J3))
		println()
		while J3 < J2
			c1 = c2
			out1 = out2
			J1 = J2
			c2 = c3
			out2 = out3
			J2 = J3
			c3 = phi*(c2-c1) + c2
			(J3, out3) = runMultiTrain(c3)
			println()
			println(string("Training error at c = ", c3, " is ", J3))
			println()
		end
		((c1, J1, out1), (c2, J2, out2), (c3, J3, out3))
	else
		c3 = c2
		out3 = out2
		J3 = J2

		c2 = c1
		out2 = out1
		J2 = J1

		c1 = c2 - (c3-c2)/phi
		(J1, out1) = runMultiTrain(c1)
		
		if J1 > J2
			println()
			println(string("Interval found around original max norm of c = ", c0, " from c = ", c1, " to ", c3, " with costs of ", J1, " and ", J3))
			println()
			((c1, J1, out1), (c2, J2, out2), (c3, J3, out3))
		else

			println()
			println(string("Training error at c = ", c1, " is ", J1, " and lower than ", J2, " so search direction will be decreasing in c"))
			println()
			c1 = c2 - (c3 - c2)/phi
			(J1, out1) = runMultiTrain(c1)
			println()
			println(string("Training error at c = ", c1, " is ", J1))
			println()
			while (J1 < J2) & ((2.0f0*(c2 - c1)/(c2+c1)) > tau)
				c3 = c2
				out3 = out2
				J3 = J2
				c2 = c1
				out2 = out1
				J2 = J1
				c1 = c2 - (c3-c2)/phi
				(J1, out1) = runMultiTrain(c1)
				println()
				println(string("Training error at c = ", c1, " is ", J1))
				println()
			end
			((c1, J1, out1), (c2, J2, out2), (c3, J3, out3))
		end
	end

	c4 = (phi*c3 + c1)/(1.0f0+phi)
	(J4, out4) = runMultiTrain(c4)
	p4 = (c4, J4, out4)

	println()
	println(string("Established an initial max norm interval of ", [c1, c2, c4, c3], " with test errors of ", [J1, J2, J4, J3]))
	println()

	body = vcat(p1[3], p2[3], p3[3], p4[3])

	function goldenSearch(p1, p2, p4, p3, body, tau)
		x1 = p1[1]
		x2 = p2[1]
		x3 = p3[1]
		x4 = p4[1]

		y1 = p1[2]
		y2 = p2[2]
		y3 = p3[2]
		y4 = p4[2]

		if (2.0f0*abs(y4-y2)/abs(y4+y2) < tau) | (2.0f0*abs(x4-x2)/abs(x4+x2) < tau)
			return body
		elseif y2 < y4
			p3 = p4
			x3 = x4
			p4 = p2
			x4 = x2
			x2 = (x3+phi*x1)/(phi+1.0f0)
			(J2, out2) = runMultiTrain(x2)
			p2 = (x2, J2, out2)
			goldenSearch(p1, p2, p4, p3, vcat(body, p2[3]), tau)
		else
			p1 = p2
			x1 = x2
			p2 = p4
			x2 = x4
			x4 = (phi*x3 + x1)/(1.0f0+phi)
			(J4, out4) = runMultiTrain(x4)
			p4 = (x4, J4, out4)
			goldenSearch(p1, p2, p4, p3, vcat(body, p4[3]), tau)
		end
	end

	body = goldenSearch(p1, p2, p4, p3, body, tau)

	if isfile(string("multiMaxNormAutoReg_", filename, ".csv"))
		f = open(string("multiMaxNormAutoReg_", filename, ".csv"), "a")
		writedlm(f, body)
		close(f)
	else
		writedlm(string("multiMaxNormAutoReg_", filename, ".csv"), [header; body])		
	end
	return body
end
