#include neural network functions
include("FCN_NFLOATOUT_COSTFUNCTIONS.jl")

include("FCN_NFLOATOUT_AUXFUNCTIONS.jl")

function calcOps(M, H, O, B)
	
	numParams = if length(H) > 0
		numParams1 = H[1]*M + H[1] + H[end]*O+O
		numParams2 = if length(H) > 1
			sum(H[2:end].*H[1:end-1] + H[1:end-1])
		else
			0
		end
		
		numParams1+numParams2
	else
		M*O + O
	end
	
	(fops, bops, pops) = if length(H) > 0 
		fops1 = B*((sum(H)+O) + H[1]*2*M + H[1]*26 + O*2*H[end])
		fops2 = if length(H) > 1
			B*(2*sum(H[2:end].*H[1:end-1])+sum(H[2:end])*26)
		else
			0
		end
		
		fops = fops1+fops2
		
		bops1 = B*O*2
		
		
		
		bops2 = B*H[end]*(2*O + 1)
		bops3 = if length(H) > 1
			B*sum(H[1:end-1].*(2*H[2:end]+1))
		else
			0
		end
		
		bops4 = (2*B+2)*H[1]*(M+1)
		bops5 = if length(H)> 1
			(2*B+2)*sum(H[2:end].*(H[1:end-1] + 1))
		else
			0
		end
		bops6 = (2*B+2)*O*(H[end]+1)
		
		bops = bops1+bops2+bops3+bops4+bops5
		
		pops = 9*numParams 
		
		(fops, bops, pops)
	else
		#fix this later, order of magnitude correct
		((O*M+O), O*M + O, 0)
	end
end

function checkNumGrad(lambda; hidden_layers=[5, 5], costFunc="absErr")
	srand(1234)
	m = 100
	input_layer_size = 3
	output_layer_size = 2
	X = map(Float32, randn(m, input_layer_size))
	y = map(Float32, randn(m, output_layer_size))

	hidden_layers = [5, 5]
	num_hidden = length(hidden_layers)

	T0, B0 = initializeParams(input_layer_size, hidden_layers, output_layer_size)


	Theta_grads = similar(T0)
	for i = 1:length(Theta_grads)
		Theta_grads[i] = similar(T0[i])
	end

	Bias_grads = similar(B0)
	for i = 1:length(B0)
		Bias_grads[i] = similar(B0[i])
	end

	onesVec = ones(Float32, m)

	numLayers = length(T0)
	
	a = Array{Matrix{Float32}}(num_hidden+1)
	if num_hidden > 0
		for i = 1:num_hidden
			a[i] = Array{Float32}(m, hidden_layers[i])
		end
	end
	a[end] = Array{Float32}(m, output_layer_size)

	tanh_grad_z = deepcopy(a)
	deltas = deepcopy(a)

	e = 0.01f0
	params = theta2Params(B0, T0)
	l = length(params)
	perturb = zeros(Float32, l)
	numGrad = Array{Float32}(l)

	nnCostFunction(T0, B0, input_layer_size, hidden_layers, X, y, lambda, Theta_grads, Bias_grads, tanh_grad_z, a, deltas, onesVec, costFunc=costFunc)
	
	funcGrad = theta2Params(Bias_grads, Theta_grads)

	for i = 1:l
		perturb[i] = e
		Tplus, Bplus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params+perturb)
		Tminus, Bminus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params-perturb)
		
	
		outminus = nnCostFunctionNOGRAD(Tminus, Bminus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc)
		outplus = nnCostFunctionNOGRAD(Tplus, Bplus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc)
		
		perturb[i] = 0.0f0  #restore perturb vector to 0

		numGrad[i] = (outplus - outminus)/(2.0f0*e)
	end

	
	println("Num Grads  Func Grads")
	for i = 1:length(numGrad)
		@printf "%0.6f  %0.6f \n" numGrad[i] funcGrad[i]
	end
	err = norm(numGrad-funcGrad)/norm(numGrad + funcGrad)
	println(string("Relative differences for method are ", err, ".  Should be small (1e-9)"))
	return err
end

function readInput(name)
	X = map(Float32, readcsv(string("Xtrain_", name, ".csv")))
	Xtest = map(Float32, readcsv(string("Xtest_", name, ".csv")))
	Y = map(Float32, readcsv(string("ytrain_", name, ".csv")))
	Ytest = map(Float32, readcsv(string("ytest_", name, ".csv")))
	(X, Xtest, Y, Ytest)
end

function readInputMRQ(name, reduction = (true, true, 0, 1:19))
#read inputs with option to simplify input data for comparison purposes
	X = map(Float32, readcsv(string("Xtrain_", name, ".csv")))
	Xtest = map(Float32, readcsv(string("Xtest_", name, ".csv")))
	Y = map(Float32, readcsv(string("ytrain_", name, ".csv")))
	Ytest = map(Float32, readcsv(string("ytest_", name, ".csv")))
	
	X_quarter_labels = X[:, 1:4]
	X_sector_labels = X[:, 5:18]
	X_values = X[:, 19:end]
	
	Xtest_quarter_labels = Xtest[:, 1:4]
	Xtest_sector_labels = Xtest[:, 5:18]
	Xtest_values = Xtest[:, 19:end]
	
	T = round(Int64, size(X_values, 2)/19)
	
	assert(reduction[3] <= T)
	
	X_values_reduction = mapreduce(hcat, reduction[4]) do i
		X_values[:, (T*(i-1)+1):(T*(i-1)+reduction[3])]
	end
	
	Xtest_values_reduction = mapreduce(hcat, reduction[4]) do i
		Xtest_values[:, (T*(i-1)+1):(T*(i-1)+reduction[3])]
	end
	
	X_final = if reduction[1]
		if reduction[2]
			[X_quarter_labels X_sector_labels X_values_reduction]
		else
			[X_quarter_labels X_values_reduction]
		end
	elseif reduction[2]
		[X_sector_labels X_values_reduction]
	else
		X_values_reduction
	end
	
	Xtest_final = if reduction[1]
		if reduction[2]
			[Xtest_quarter_labels Xtest_sector_labels Xtest_values_reduction]
		else
			[Xtest_quarter_labels Xtest_values_reduction]
		end
	elseif reduction[2]
		[Xtest_sector_labels Xtest_values_reduction]
	else
		Xtest_values_reduction
	end
	
	(X_final, Xtest_final, Y, Ytest)
	
end

function updateM!(beta1, mT, mB, TG, BG)
	for i = 1:length(TG)
		@simd for ii = 1:length(TG[i])
			@inbounds mT[i][ii] = beta1*mT[i][ii] + (1.0f0 - beta1)*TG[i][ii]
		end
		
		@simd for ii = 1:length(BG[i])
			@inbounds mB[i][ii] = beta1*mB[i][ii] + (1.0f0 - beta1)*BG[i][ii]
		end
	end
end

function updateV!(beta2, vT, vB, TG, BG)
	for i = 1:length(TG)
		@simd for ii = 1:length(TG[i])
			@inbounds vT[i][ii] = max(beta2*vT[i][ii], abs(TG[i][ii]), 1.0f-16)
		end
		@simd for ii = 1:length(BG[i])
			@inbounds vB[i][ii] = max(beta2*vB[i][ii], abs(BG[i][ii]), 1.0f-16)
		end
	end
end

function updateParams!(alpha, beta1, T, B, mT, mB, vT, vB, t)
	for i = 1:length(T)
		@simd for ii = 1:length(T[i])
			@inbounds T[i][ii] = T[i][ii] - (alpha/(1.0f0 - beta1^t))*mT[i][ii]/vT[i][ii]
		end
		@simd for ii = 1:length(B[i])
			@inbounds B[i][ii] = B[i][ii] - (alpha/(1.0f0 - beta1^t))*mB[i][ii]/vB[i][ii]
		end
	end
end

function scaleParams!(T, B, c)
	#project Thetas onto l2 ball of radius c
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
				#	@inbounds T[i][j, k] = c*T[i][j, k]/fs[j]
				#end
				@inbounds T[i][j, k] = min(1.0f0, c/fs[j]) * T[i][j, k]
			end
		end
	end
end

function apply!(f, A, B)
#apply a function f(x, y) that takes two inputs and produces a single output to each pair 
#of elements in A and B where A and B are collections of Arrays and saves the answer in B
	for i = 1:length(A)
		@simd for ii = 1:length(A[i])
			@inbounds B[i][ii] = f(A[i][ii], B[i][ii])
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

function generateBatches(input_data, output_data, batchsize)
	m = size(output_data, 1)
	if batchsize > m
		error("Your batchsize is larger than the total number of examples.")
	end
	
	numBatches = round(Int, ceil(m/batchsize))
	inputbatchData = Array{Matrix{Float32}}(numBatches)
	outputbatchData = Array{Matrix{Float32}}(numBatches)
	randInd = [shuffle(collect(1:m)) shuffle(collect(1:m))]
	for i = 1:numBatches
		inputbatchData[i] = input_data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
		outputbatchData[i] = output_data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
	end
	return (inputbatchData, outputbatchData)
end

function ADAMAXTrainNN(input_data, output_data, batchSize, T0, B0, N, input_layer_size, hidden_layers, lambda, c; alpha=0.002f0, R = 0.1f0, printProgress = false, dropout = 0.0f0, costFunc = "absErr")
#train fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  An optional dropout factor is set to 0 by default but can be set to a 32 bit float between 0 and 1.
#Note that all floating point input variables must be float32 or single precision   
	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")
	(m, n) = size(input_data)
	(m2, n2) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
		
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	tanh_grad_zBATCH = Array{Matrix{Float32}}(num_hidden)
	for i = 1:num_hidden
		tanh_grad_zBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	
	aBATCH = Array{Matrix{Float32}}(num_hidden+1)
	for i = 1:num_hidden
		aBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	aBATCH[end] = Array{Float32}(batchSize, n2)
	deltasBATCH = Array{Matrix{Float32}}(num_hidden+1)

	for i = 1:length(deltasBATCH)
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

	onesVecBATCH = ones(Float32, batchSize)

	numLayers = length(T0)



	nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout, costFunc=costFunc)
	currentOut = 0.0f0 
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout, costFunc=costFunc)
	end
	currentOut = currentOut/numBatches

	println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	mT = 0.0f0*deepcopy(T0)
	mB = 0.0f0*deepcopy(B0)

	vT = 0.0f0*deepcopy(T0)
	vB = 0.0f0*deepcopy(B0)

	Thetas = deepcopy(T0)
	Biases = deepcopy(B0)

	period = 10
	costRecord = Array{Float32}(ceil(Int, N/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Array{Float64}(N+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	eta = alpha
	F = (1.0f0 - R)

	t = 1
	while epoch <= N
	#while epoch <= N
		#run through an epoch in batches with randomized order
		for batch = 1:numBatches
			nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout, costFunc=costFunc)
			updateM!(beta1, mT, mB, Theta_grads, Bias_grads)
			updateV!(beta2, vT, vB, Theta_grads, Bias_grads)		
			updateParams!(eta, beta1, Thetas, Biases, mT, mB, vT, vB, t)
			if c < Inf 
				scaleParams!(Thetas, Biases, c)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime

		if epoch%period == 0
			currentOut = 0.0f0
			for i = 1:numBatches
				currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout, costFunc=costFunc)
			end
			currentOut = currentOut/numBatches
			
			costRecord[iter + 1] = currentOut
			
			if currentOut < bestCost
				updateBest!(bestThetas, bestBiases, Thetas, Biases)
				bestCost = currentOut
			end
			
			if epoch/period > 1
				if costRecord[iter+1] > costRecord[iter]
					updateBest!(Thetas, Biases, bestThetas, bestBiases)
					#alpha = max(1.0f-6, alpha*0.9f0)
				else
					#alpha = min(1.0f-1, alpha/0.99f0)
				end
			end

			if epoch > 100
				#println(string("eta = ", eta))
				eta = eta*F
			end
			
			iter += 1
		end

	
		currentTime = time()
		#print status every 5 seconds
		
		if ((currentTime - lastReport) >= 5) & printProgress
			startEpoch = max(0, epoch-10)
			#find average time per epoch over the last 10 epochs
			epochTime = (timeRecord[epoch + 1] - timeRecord[startEpoch + 1]) / (epoch-startEpoch)
			remainingEpochs = N - epoch

			timeRemainingEst = remainingEpochs*epochTime

			#elapsed = currentTime - startTime
			#percentComplete = epoch/N
			#totalTimeEst = elapsed / percentComplete
			#timeRemainingEst = totalTimeEst - elapsed
			lastReport = currentTime
			hoursLeft = floor(timeRemainingEst/(60*60))
			minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
			println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout, costFunc=costFunc)
	end
	currentOut = currentOut/numBatches

	if currentOut < bestCost
		bestCost = currentOut
		updateBest!(bestThetas, bestBiases, Thetas, Biases)
	end
	
	time_per_epoch = timeRecord[2:end] .- timeRecord[1:end-1]
    train_time = timeRecord[end]
    timePerBatch = train_time/N/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9

	if dropout == 0.0f0
		println(string("Completed training with the following parameters: input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha, ", decay rate = ", R))
	else
		println(string("Completed training with dropout factor of ", dropout))
		println(string("Other training parameters: input size  = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha, ", decay rate = ", R))
	end
	println(string("Cost reduced from ", costRecord[1], "to ", bestCost, " after ", timeRecord[N], " seconds"))		
	println(string("Median time of ", 1e9*median(time_per_epoch)/m, " ns per example"))
    println(string("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize))
    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
	return bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch
end	

function ADAMAXTrainNNAdv(input_data, output_data, batchSize, T0, B0, N, input_layer_size, hidden_layers, eta, c, alpha; printProgress = false)
#train fully connected neural network with floating point vector output and adversarial noise.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, adversarial noise parameter eta, max norm parameter c, and
#a training rate alpha
#Note that all floating point input variables must be float32 or single precision   
	
	lambda = 0.0f0

	(m, n) = size(input_data)
	(m2, n2) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))


	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
		
	advX = similar(inputbatchData[1])
	
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	tanh_grad_zBATCH = Array(Matrix{Float32}, num_hidden)
	for i = 1:num_hidden
		tanh_grad_zBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	
	aBATCH = Array(Matrix{Float32}, num_hidden+1)
	for i = 1:num_hidden
		aBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	aBATCH[end] = Array{Float32}(batchSize, n2)
	deltasBATCH = Array(Matrix{Float32}, num_hidden+1)

	for i = 1:length(deltasBATCH)
		deltasBATCH[i] = similar(aBATCH[i])
	end

	Theta_grads = similar(T0)
	advTheta_grads = similar(T0)
	for i = 1:length(Theta_grads)
		Theta_grads[i] = similar(T0[i])
		advTheta_grads[i] = similar(T0[i])
	end


	Bias_grads = similar(B0)
	advBias_grads = similar(B0)
	for i = 1:length(B0)
		Bias_grads[i] = similar(B0[i])
		advBias_grads[i] = similar(B0[i])
	end

	onesVecBATCH = ones(Float32, batchSize)

	numLayers = length(T0)


	nnCostFunctionAdv(T0, B0, input_layer_size, hidden_layers, advX, inputbatchData[end], outputbatchData[end], eta, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
	nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
	currentOut = 0.0f0 
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH)
	end
	currentOut = currentOut/numBatches

	println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	mT = 0.0f0*deepcopy(T0)
	mB = 0.0f0*deepcopy(B0)

	vT = 0.0f0*deepcopy(T0)
	vB = 0.0f0*deepcopy(B0)

	Thetas = deepcopy(T0)
	Biases = deepcopy(B0)

	period = 10
	costRecord = Array{Float32}(ceil(Int, N/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Array(Float64, N+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	t = 1
	while epoch <= N
	#while epoch <= N
		#run through an epoch in batches with randomized order
		for batch = 1:numBatches
			#compute current parameter gradients and adversarial noise
			nnCostFunctionAdv(Thetas, Biases, input_layer_size, hidden_layers, advX, inputbatchData[batch], outputbatchData[batch], eta, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
			
			#compute gradients at adversarial examples
			nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, advX, outputbatchData[batch], lambda, advTheta_grads, advBias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH)
			
			#save average between the two gradients in the original gradient arrays
			apply!((a, b) -> (a+b)/2, advTheta_grads, Theta_grads)
			apply!((a, b) -> (a+b)/2, advBias_grads, Bias_grads)
			
			#update parameters with modified gradient
			updateM!(beta1, mT, mB, Theta_grads, Bias_grads)
			updateV!(beta2, vT, vB, Theta_grads, Bias_grads)
			updateParams!(alpha, beta1, Thetas, Biases, mT, mB, vT, vB, t)
			
			if c < Inf 
				scaleParams!(Thetas, Biases, c)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime

		if epoch%period == 0
			currentOut = 0.0f0
			for i = 1:numBatches
				currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH)
			end
			currentOut = currentOut/numBatches
			
			costRecord[iter + 1] = currentOut
			
			if currentOut < bestCost
				updateBest!(bestThetas, bestBiases, Thetas, Biases)
				bestCost = currentOut
			end
			
			if epoch/period > 1
				if costRecord[iter+1] > costRecord[iter]
					updateBest!(Thetas, Biases, bestThetas, bestBiases)
					alpha = max(1.0f-6, alpha*0.9f0)
				else
					alpha = min(1.0f-1, alpha/0.99f0)
				end
			end
			
			iter += 1
		end

		currentTime = time()
		#print status every 5 seconds
		
		if ((currentTime - lastReport) >= 5) & printProgress
			startEpoch = max(0, epoch-10)
			#find average time per epoch over the last 10 epochs
			epochTime = (timeRecord[epoch + 1] - timeRecord[startEpoch + 1]) / (epoch-startEpoch)
			remainingEpochs = N - epoch

			timeRemainingEst = remainingEpochs*epochTime

			#elapsed = currentTime - startTime
			#percentComplete = epoch/N
			#totalTimeEst = elapsed / percentComplete
			#timeRemainingEst = totalTimeEst - elapsed
			lastReport = currentTime
			hoursLeft = floor(timeRemainingEst/(60*60))
			minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
			println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH)
	end
	currentOut = currentOut/numBatches

	if currentOut < bestCost
		bestCost = currentOut
		updateBest!(bestThetas, bestBiases, Thetas, Biases)
	end
	println(string("Completed training with the following parameters: input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	println(string("Cost reduced from ", costRecord[1], "to ", bestCost, " after ", timeRecord[N], " seconds"))		
	return bestThetas, bestBiases, bestCost, costRecord, timeRecord
end	

function ADAMAXTrainPowDecay(input_data, output_data, batchSize, T0, B0, N, input_layer_size, hidden_layers, lambda, c, alpha; R = 1.0f0, printProgress = false, dropout = 0.0f0)
#train fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  An optional dropout factor is set to 0 by default but can be set to a 32 bit float between 0 and 1.
#Note that all floating point input variables must be float32 or single precision   
	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")
	(m, n) = size(input_data)
	(m2, n2) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
		
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	tanh_grad_zBATCH = Array(Matrix{Float32}, num_hidden)
	for i = 1:num_hidden
		tanh_grad_zBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	
	aBATCH = Array(Matrix{Float32}, num_hidden+1)
	for i = 1:num_hidden
		aBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	aBATCH[end] = Array{Float32}(batchSize, n2)
	deltasBATCH = Array(Matrix{Float32}, num_hidden+1)

	for i = 1:length(deltasBATCH)
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

	onesVecBATCH = ones(Float32, batchSize)

	numLayers = length(T0)



	nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout)
	currentOut = 0.0f0 
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
	end
	currentOut = currentOut/numBatches

	println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	mT = 0.0f0*deepcopy(T0)
	mB = 0.0f0*deepcopy(B0)

	vT = 0.0f0*deepcopy(T0)
	vB = 0.0f0*deepcopy(B0)

	Thetas = deepcopy(T0)
	Biases = deepcopy(B0)

	period = 10
	costRecord = Array{Float32}(ceil(Int, N/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Array(Float64, N+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	eta = alpha
	t = 1
	while epoch <= N
	#while epoch <= N
		#run through an epoch in batches with randomized order
		for batch = 1:numBatches
			nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout)
			updateM!(beta1, mT, mB, Theta_grads, Bias_grads)
			updateV!(beta2, vT, vB, Theta_grads, Bias_grads)		
			updateParams!(eta, beta1, Thetas, Biases, mT, mB, vT, vB, t)
			if c < Inf 
				scaleParams!(Thetas, Biases, c)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime

		if epoch%period == 0
			currentOut = 0.0f0
			for i = 1:numBatches
				currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
			end
			currentOut = currentOut/numBatches
			
			costRecord[iter + 1] = currentOut
			
			if currentOut < bestCost
				updateBest!(bestThetas, bestBiases, Thetas, Biases)
				bestCost = currentOut
			end
			
			if epoch/period > 1
				if costRecord[iter+1] > costRecord[iter]
					updateBest!(Thetas, Biases, bestThetas, bestBiases)
					#alpha = max(1.0f-6, alpha*0.9f0)
				else
					#alpha = min(1.0f-1, alpha/0.99f0)
				end
			end
			
			iter += 1
		end

		if epoch > 100
			#println(string("eta = ", eta))
			eta = alpha/sqrt(R*epoch)
		end
		currentTime = time()
		#print status every 5 seconds
		
		if ((currentTime - lastReport) >= 5) & printProgress
			startEpoch = max(0, epoch-10)
			#find average time per epoch over the last 10 epochs
			epochTime = (timeRecord[epoch + 1] - timeRecord[startEpoch + 1]) / (epoch-startEpoch)
			remainingEpochs = N - epoch

			timeRemainingEst = remainingEpochs*epochTime

			#elapsed = currentTime - startTime
			#percentComplete = epoch/N
			#totalTimeEst = elapsed / percentComplete
			#timeRemainingEst = totalTimeEst - elapsed
			lastReport = currentTime
			hoursLeft = floor(timeRemainingEst/(60*60))
			minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
			println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
	end
	currentOut = currentOut/numBatches

	if currentOut < bestCost
		bestCost = currentOut
		updateBest!(bestThetas, bestBiases, Thetas, Biases)
	end
	
	time_per_epoch = timeRecord[2:end] .- timeRecord[1:end-1]
    train_time = timeRecord[end]
    timePerBatch = train_time/N/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9

	if dropout == 0.0f0
		println(string("Completed training with the following parameters: input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	else
		println(string("Completed training with dropout factor of ", dropout))
		println(string("Other training parameters: input size  = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	end
	println(string("Cost reduced from ", costRecord[1], "to ", bestCost, " after ", timeRecord[N], " seconds"))		
	println(string("Median time of ", 1e9*median(time_per_epoch)/m, " ns per example"))
    println(string("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize))
    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
	return bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch
end	

function ADAMAXTrainRecDecay(input_data, output_data, batchSize, T0, B0, N, input_layer_size, hidden_layers, lambda, c, alpha; R = 1.0f0, printProgress = false, dropout = 0.0f0)
#train fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  An optional dropout factor is set to 0 by default but can be set to a 32 bit float between 0 and 1.
#Note that all floating point input variables must be float32 or single precision   
	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")
	(m, n) = size(input_data)
	(m2, n2) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
		
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	tanh_grad_zBATCH = Array(Matrix{Float32}, num_hidden)
	for i = 1:num_hidden
		tanh_grad_zBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	
	aBATCH = Array(Matrix{Float32}, num_hidden+1)
	for i = 1:num_hidden
		aBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	aBATCH[end] = Array{Float32}(batchSize, n2)
	deltasBATCH = Array(Matrix{Float32}, num_hidden+1)

	for i = 1:length(deltasBATCH)
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

	onesVecBATCH = ones(Float32, batchSize)

	numLayers = length(T0)



	nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout)
	currentOut = 0.0f0 
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
	end
	currentOut = currentOut/numBatches

	println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	mT = 0.0f0*deepcopy(T0)
	mB = 0.0f0*deepcopy(B0)

	vT = 0.0f0*deepcopy(T0)
	vB = 0.0f0*deepcopy(B0)

	Thetas = deepcopy(T0)
	Biases = deepcopy(B0)

	period = 10
	costRecord = Array{Float32}(ceil(Int, N/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Array(Float64, N+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	eta = alpha
	t = 1
	while epoch <= N
	#while epoch <= N
		#run through an epoch in batches with randomized order
		for batch = 1:numBatches
			nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout)
			updateM!(beta1, mT, mB, Theta_grads, Bias_grads)
			updateV!(beta2, vT, vB, Theta_grads, Bias_grads)		
			updateParams!(eta, beta1, Thetas, Biases, mT, mB, vT, vB, t)
			if c < Inf 
				scaleParams!(Thetas, Biases, c)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime

		if epoch%period == 0
			currentOut = 0.0f0
			for i = 1:numBatches
				currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
			end
			currentOut = currentOut/numBatches
			
			costRecord[iter + 1] = currentOut
			
			if currentOut < bestCost
				updateBest!(bestThetas, bestBiases, Thetas, Biases)
				bestCost = currentOut
			end
			
			if epoch/period > 1
				if costRecord[iter+1] > costRecord[iter]
					updateBest!(Thetas, Biases, bestThetas, bestBiases)
					#alpha = max(1.0f-6, alpha*0.9f0)
				else
					#alpha = min(1.0f-1, alpha/0.99f0)
				end
			end
			
			iter += 1
		end

		if epoch > 100
			#println(string("eta = ", eta))
			eta = alpha/(1.0f0 + R*epoch)
		end
		currentTime = time()
		#print status every 5 seconds
		
		if ((currentTime - lastReport) >= 5) & printProgress
			startEpoch = max(0, epoch-10)
			#find average time per epoch over the last 10 epochs
			epochTime = (timeRecord[epoch + 1] - timeRecord[startEpoch + 1]) / (epoch-startEpoch)
			remainingEpochs = N - epoch

			timeRemainingEst = remainingEpochs*epochTime

			#elapsed = currentTime - startTime
			#percentComplete = epoch/N
			#totalTimeEst = elapsed / percentComplete
			#timeRemainingEst = totalTimeEst - elapsed
			lastReport = currentTime
			hoursLeft = floor(timeRemainingEst/(60*60))
			minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
			println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
	end
	currentOut = currentOut/numBatches

	if currentOut < bestCost
		bestCost = currentOut
		updateBest!(bestThetas, bestBiases, Thetas, Biases)
	end
	
	time_per_epoch = timeRecord[2:end] .- timeRecord[1:end-1]
    train_time = timeRecord[end]
    timePerBatch = train_time/N/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9

	if dropout == 0.0f0
		println(string("Completed training with the following parameters: input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	else
		println(string("Completed training with dropout factor of ", dropout))
		println(string("Other training parameters: input size  = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	end
	println(string("Cost reduced from ", costRecord[1], "to ", bestCost, " after ", timeRecord[N], " seconds"))		
	println(string("Median time of ", 1e9*median(time_per_epoch)/m, " ns per example"))
    println(string("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize))
    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
	return bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch
end	

function ADAMAXTrainLinDecay(input_data, output_data, batchSize, T0, B0, N, input_layer_size, hidden_layers, lambda, c, alpha; R = 10000, printProgress = false, dropout = 0.0f0)
#train fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  An optional dropout factor is set to 0 by default but can be set to a 32 bit float between 0 and 1.
#Note that all floating point input variables must be float32 or single precision   
	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")
	(m, n) = size(input_data)
	(m2, n2) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
		
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	tanh_grad_zBATCH = Array(Matrix{Float32}, num_hidden)
	for i = 1:num_hidden
		tanh_grad_zBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	
	aBATCH = Array(Matrix{Float32}, num_hidden+1)
	for i = 1:num_hidden
		aBATCH[i] = Array{Float32}(batchSize, hidden_layers[i])
	end
	aBATCH[end] = Array{Float32}(batchSize, n2)
	deltasBATCH = Array(Matrix{Float32}, num_hidden+1)

	for i = 1:length(deltasBATCH)
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

	onesVecBATCH = ones(Float32, batchSize)

	numLayers = length(T0)



	nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[end], outputbatchData[end], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout)
	currentOut = 0.0f0 
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
	end
	currentOut = currentOut/numBatches

	println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	mT = 0.0f0*deepcopy(T0)
	mB = 0.0f0*deepcopy(B0)

	vT = 0.0f0*deepcopy(T0)
	vB = 0.0f0*deepcopy(B0)

	Thetas = deepcopy(T0)
	Biases = deepcopy(B0)

	period = 10
	costRecord = Array{Float32}(ceil(Int, N/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Array(Float64, N+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	F = alpha/(R-100)
	t = 1
	while epoch <= N
	#while epoch <= N
		#run through an epoch in batches with randomized order
		for batch = 1:numBatches
			nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout)
			updateM!(beta1, mT, mB, Theta_grads, Bias_grads)
			updateV!(beta2, vT, vB, Theta_grads, Bias_grads)		
			updateParams!(alpha, beta1, Thetas, Biases, mT, mB, vT, vB, t)
			if c < Inf 
				scaleParams!(Thetas, Biases, c)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime

		if epoch%period == 0
			currentOut = 0.0f0
			for i in randperm(numBatches)
				currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
			end
			currentOut = currentOut/numBatches
			
			costRecord[iter + 1] = currentOut
			
			if currentOut < bestCost
				updateBest!(bestThetas, bestBiases, Thetas, Biases)
				bestCost = currentOut
			end
			
			if epoch/period > 1
				if costRecord[iter+1] > costRecord[iter]
					updateBest!(Thetas, Biases, bestThetas, bestBiases)
					#alpha = max(1.0f-6, alpha*0.9f0)
				else
					#alpha = min(1.0f-1, alpha/0.99f0)
				end
			end
			
			iter += 1
		end

		if epoch > 100
			#println(string("eta = ", eta))
			alpha = max(1.0f-8, alpha - F)
		end

		currentTime = time()
		#print status every 5 seconds
		
		if ((currentTime - lastReport) >= 5) & printProgress
			startEpoch = max(0, epoch-10)
			#find average time per epoch over the last 10 epochs
			epochTime = (timeRecord[epoch + 1] - timeRecord[startEpoch + 1]) / (epoch-startEpoch)
			remainingEpochs = N - epoch

			timeRemainingEst = remainingEpochs*epochTime

			#elapsed = currentTime - startTime
			#percentComplete = epoch/N
			#totalTimeEst = elapsed / percentComplete
			#timeRemainingEst = totalTimeEst - elapsed
			lastReport = currentTime
			hoursLeft = floor(timeRemainingEst/(60*60))
			minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
			println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, 8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], lambda, aBATCH, dropout)
	end
	currentOut = currentOut/numBatches

	if currentOut < bestCost
		bestCost = currentOut
		updateBest!(bestThetas, bestBiases, Thetas, Biases)
	end
	
	time_per_epoch = timeRecord[2:end] .- timeRecord[1:end-1]
    train_time = timeRecord[end]
    timePerBatch = train_time/N/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9

	if dropout == 0.0f0
		println(string("Completed training with the following parameters: input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	else
		println(string("Completed training with dropout factor of ", dropout))
		println(string("Other training parameters: input size  = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha))
	end
	println(string("Cost reduced from ", costRecord[1], "to ", bestCost, " after ", timeRecord[N], " seconds"))		
	println(string("Median time of ", 1e9*median(time_per_epoch)/m, " ns per example"))
    println(string("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize))
    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
	return bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch
end	