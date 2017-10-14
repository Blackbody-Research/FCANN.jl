#include neural network functions
include("FCN_ABSERR_NFLOATOUT_CUDACOSTFUNCTION.jl")
#include("FCN_NFLOATOUT_COSTFUNCTIONS.jl")
#include("FCN_NFLOATOUT_AUXFUNCTIONS.jl")

filepath = joinpath(@__DIR__, "ADAMAX_INTRINSIC_KERNELS.cu")

adamax_md = if isfile("ADAMAX_INTRINSIC_KERNELS.ptx")
	CuModuleFile("ADAMAX_INTRINSIC_KERNELS.ptx")
else
	run(`nvcc -ptx $filepath`)
	CuModuleFile("ADAMAX_INTRINSIC_KERNELS.ptx")
end

##TO DO: redo threads, blockers, kernels, and launches to work in 1D
#set up NN kernels
updateParams = CuFunction(adamax_md, "updateParams")
elSq = CuFunction(adamax_md, "elSq")
elSq2 = CuFunction(adamax_md, "elSq2")
scaleParams = CuFunction(adamax_md, "scaleParams")
updateEst = CuFunction(adamax_md, "updateEst")

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
			copy!(GPUvars[i][j], hostvars[i][j])
			#GPUvars[i][j] = CuArray(hostvars[i][j])
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
			#hostvars[i][j] = to_host(GPUvars[i][j])
			copy!(hostvars[i][j], GPUvars[i][j])
		end
	end
end

function checkNumGradGPU(lambda; hidden_layers=[5, 5])
	srand(1234)
	m = 100
	input_layer_size = 3
	output_layer_size = 2
	X = randn(Float32, m, input_layer_size)
	y = randn(Float32, m, output_layer_size)
	d_X = CuArray(X)
	d_y = CuArray(y)

	num_hidden = length(hidden_layers)

	T0, B0 = initializeParams(input_layer_size, hidden_layers, output_layer_size)

	d_Thetas = Array{CuArray{Float32, 2}}(length(T0))
	d_Biases = Array{CuArray{Float32, 1}}(length(T0))

	for i = 1:length(T0)
		d_Thetas[i] = CuArray(T0[i])
		d_Biases[i] = CuArray(B0[i])
	end


	Theta_grads = similar(T0)
	for i = 1:length(Theta_grads)
		Theta_grads[i] = similar(T0[i])
	end


	Bias_grads = similar(B0)
	for i = 1:length(B0)
		Bias_grads[i] = similar(B0[i])
	end


	d_Theta_grads = Array{CuArray{Float32, 2}}(length(B0))
	d_Bias_grads = Array{CuArray{Float32, 1}}(length(B0))
	for i = 1:length(B0)
		d_Theta_grads[i] = CuArray(Theta_grads[i])
		d_Bias_grads[i] = CuArray(Bias_grads[i])
	end

	d_ones = CuArray(ones(Float32, m))
	d_tanh_grad_z = Array{CuArray{Float32, 2}}(num_hidden)
	if num_hidden > 0
		for i = 1:num_hidden
			d_tanh_grad_z[i] = CuArray{Float32}(m, hidden_layers[i])
		end
	end
	d_a = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_deltas = Array{CuArray{Float32, 2}}(num_hidden+1)
	if num_hidden > 0
		for i = 1:num_hidden
			d_a[i] = CuArray{Float32}(m, hidden_layers[i])
			d_deltas[i] = CuArray{Float32}(m, hidden_layers[i])
		end
	end
	d_a[end] = CuArray{Float32}(m, output_layer_size)
	d_deltas[end] = CuArray{Float32}(m, output_layer_size)



	numLayers = length(T0)

	
	a = Array{Matrix{Float32}}(num_hidden+1)
	if num_hidden > 0
		for i = 1:num_hidden
			a[i] = Array{Float32}(m, hidden_layers[i])
		end
	end
	a[end] = Array{Float32}(m, output_layer_size)


	e = 0.001f0
	params = theta2Params(B0, T0)
	l = length(params)
	perturb = zeros(Float32, l)
	numGrad = Array{Float32}(l)

	nnCostFunction(d_Thetas, d_Biases, input_layer_size, output_layer_size, hidden_layers, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, d_X, d_y,lambda)
	costGPU = nnCostFunctionNOGRAD(d_Thetas, d_Biases, input_layer_size, output_layer_size, hidden_layers, m, d_a, d_X, d_y,lambda)
	costCPU = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, X, y, lambda, a)
	
	GPU2Host((Theta_grads, Bias_grads), (d_Theta_grads, d_Bias_grads))

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

	println("GPU Cost  CPU Cost" )
	println(string(costGPU, "  ", costCPU))

	println("___________________")
	
	println("Num Grads  GPU Grads")
	for i = 1:length(numGrad)
		@printf "%0.6f  %0.6f \n" numGrad[i] funcGrad[i]
	end
	GPUErr = norm(numGrad-funcGrad)/norm(numGrad + funcGrad)
	println(string("Relative differences for method are ", GPUErr, ".  Should be small (1e-9)"))
	return GPUErr
end

function calcOutputGPU(input_data, output_data, T, B; dropout = 0.0f0)
#calculate network output given input data and a set of network parameters.
#calculation is performed on the GPU and then returned to system memory
	num_hidden = length(T) - 1
	hidden_layers = if num_hidden > 0
		map(p -> length(p), B[1:num_hidden])
	else
		Int64.([])
	end

	(m, input_layer_size) = size(input_data)
	output_layer_size = size(output_data, 2)

	#transfer parameters to GPU
	d_Thetas = Array{CuArray{Float32, 2}}(length(T))
	d_Biases = Array{CuArray{Float32, 1}}(length(T))

	for i = 1:length(T)
		d_Thetas[i] = CuArray(T[i])
		d_Biases[i] = CuArray(B[i])
	end

	d_X = CuArray(input_data)

	d_out = predict(d_Thetas, d_Biases, d_X, m, input_layer_size, output_layer_size, hidden_layers, dropout)

	out = Array(d_out)

	err = mean(abs(out .- output_data))

	return (out, err)
end

function calcOutputGPU(input_data, T, B; dropout = 0.0f0)
#calculate network output given input data and a set of network parameters.
#calculation is performed on the GPU and then returned to system memory
	num_hidden = length(T) - 1
	hidden_layers = if num_hidden > 0
		map(p -> length(p), B[1:num_hidden])
	else
		Int64.([])
	end
	(m, input_layer_size) = size(input_data)
	output_layer_size = size(T[end], 1)

	#transfer parameters to GPU
	d_Thetas = Array{CuArray{Float32, 2}}(length(T))
	d_Biases = Array{CuArray{Float32, 1}}(length(T))

	for i = 1:length(T)
		d_Thetas[i] = CuArray(T[i])
		d_Biases[i] = CuArray(B[i])
	end

	d_X = CuArray(input_data)

	d_out = predict(d_Thetas, d_Biases, d_X, m, input_layer_size, output_layer_size, hidden_layers, dropout)

	out = Array(d_out)

	return out
end

function ADAMAXTrainNNGPU(input_data, output_data, batchSize, T0, B0, numEpochs, input_layer_size, hidden_layers, lambda, c; alpha=0.001f0, R=0.1f0, printProgress = false, dropout = 0.0f0)
#train on a GPU fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  The final required input "md" is the context for the GPU hardware being used.
#Note that all floating point input variables must be float32 or single precision   
	
	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")	(m, n) = size(input_data)
	(m, n) = size(input_data)
	(m2, n2) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	println()
	print_with_color(:green, STDOUT, "Beginning training on GPU with the following parameters:", bold=true)
	println()
	println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha, ", decay rate = ", R))
	println("-------------------------------------------------------------------")
	
	function scaleThetas!(d_Thetas, d_Theta_grads, d_onesVecParams, d_normVecParams, c)
		for i = 1:length(d_Thetas)
			#get rows and columns of input matrix
			(N, M) = size(T0[i])
			
			#square each element of Theta
			run_kernel(elSq2, N, M, d_Thetas[i], d_Theta_grads[i])
			#CUDArt.launch(elSq, blocks(N, M), threads, (N, M, d_Thetas[i]))
			
			#generate vector of row sums using blas operations
			CUBLAS.gemv!('N', 1.0f0, d_Theta_grads[i], d_onesVecParams[i], 0.0f0, d_normVecParams[i])
			
			#scale each row so the squared sum is less than or equal to c^2
			run_kernel(scaleParams, N, M, c, d_Thetas[i], d_normVecParams[i])
			#CUDArt.launch(scaleParams, blocks(N, M), threads, (N, M, c, d_Thetas[i], d_normVecParams[i]))
		end
		synchronize()
		#device_synchronize()	
	end
	
	function updateParams!(alpha, beta1, beta2, t, d_Thetas, d_Theta_grads, d_Biases, d_Bias_grads, d_mT, d_mB, d_vT, d_vB)
		for i = 1:length(d_Thetas)
			(N, M) = size(T0[i])
			
			#launch kernel to update Thetas
			run_kernel(updateParams, N, M, alpha, beta1, beta2, t, d_Thetas[i], d_Theta_grads[i], d_mT[i], d_vT[i])
			#CUDArt.launch(updateParams, blocks(N, M), threads, (N, M, alpha, beta1, beta2, t, d_Thetas[i], d_Theta_grads[i], d_mT[i], d_vT[i]))
			
			N = length(B0[i])
			M = 1
			#launch kernel to update Biases
			run_kernel(updateParams, N, M, alpha, beta1, beta2, t, d_Biases[i], d_Bias_grads[i], d_mB[i], d_vB[i])
			#CUDArt.launch(updateParams, blocks(N, M), threads, (N, M, alpha, beta1, beta2, t, d_Biases[i], d_Bias_grads[i], d_mB[i], d_vB[i]))
		end
		synchronize()
		#device_synchronize()	
	end

	function updateEst!(beta2, t, d_Thetas, d_Biases, d_Theta_avg, d_Bias_avg, d_Theta_est, d_Bias_est)
		for i = 1:length(d_Thetas)
			(N, M) = size(T0[i])
			scale = 1.0f0/(1.0f0 - beta2^t)
			
			#launch kernel to update Thetas
			run_kernel(updateEst, N, M, beta2, scale, d_Thetas[i], d_Theta_avg[i], d_Theta_est[i])
			# CUDArt.launch(updateEst, blocks(N, M), threads, (N, M, beta2, scale, d_Thetas[i], d_Theta_avg[i], d_Theta_est[i]))
			
			N = length(B0[i])
			M = 1
			#launch kernel to update Biases
			run_kernel(updateEst, N, M, beta2, scale, d_Biases[i], d_Bias_avg[i], d_Bias_est[i])
			# CUDArt.launch(updateEst, blocks(N, M), threads, (N, M, beta2, scale, d_Biases[i], d_Bias_avg[i], d_Bias_est[i]))
		end
		synchronize()
		#device_synchronize()	
	end
	
	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
	
	batchInputs = Array{CuArray{Float32, 2}}(numBatches)
	batchOutputs = Array{CuArray{Float32, 2}}(numBatches)
	for i = 1:numBatches
		batchInputs[i] = CuArray(inputbatchData[i])
		batchOutputs[i] = CuArray(outputbatchData[i])
	end

		
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	#create vectors of ones equal to the number of columns in each theta matrix
	d_onesVecParams = map(a -> CuArray(ones(Float32, a)), [n; hidden_layers])
	#create a vector to store the squared sum of each row in the theta matricies
	d_normVecParams = map(a -> CuArray(zeros(Float32, a)), [hidden_layers; n2])
	
	
	d_tanh_grad_zBATCH = Array{CuArray{Float32, 2}}(num_hidden)
	for i = 1:num_hidden
		d_tanh_grad_zBATCH[i] = CuArray{Float32}(batchSize, hidden_layers[i])
	end
	
	d_aBATCH = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_deltasBATCH = Array{CuArray{Float32, 2}}(num_hidden+1)
	for i = 1:num_hidden
		d_aBATCH[i] = CuArray{Float32}(batchSize, hidden_layers[i])
		d_deltasBATCH[i] = CuArray{Float32}(batchSize, hidden_layers[i]) 
	end
	d_aBATCH[end] = CuArray{Float32}(batchSize, n2)
	d_deltasBATCH[end] = CuArray{Float32}(batchSize, n2)
	
	d_T0 = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_Thetas = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_Theta_grads = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_Theta_avg = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_Theta_est = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_mT = Array{CuArray{Float32, 2}}(num_hidden+1)
	d_vT = Array{CuArray{Float32, 2}}(num_hidden+1)
	for i = 1:num_hidden+1
		d_T0[i] = CuArray(T0[i])
		d_Thetas[i] = CuArray(T0[i])
		d_Theta_avg[i] = CuArray(0.0f0*T0[i])
		d_Theta_est[i] = CuArray(0.0f0*T0[i])
		d_Theta_grads[i] = similar(d_T0[i])
		d_mT[i] = CuArray(0.0f0*T0[i])
		d_vT[i] = CuArray(0.0f0*T0[i])
	end
	
	d_B0 = Array{CuArray{Float32, 1}}(num_hidden+1)
	d_Biases = Array{CuArray{Float32, 1}}(num_hidden+1)
	d_Bias_grads = Array{CuArray{Float32, 1}}(num_hidden+1)
	d_Bias_est = Array{CuArray{Float32, 1}}(num_hidden+1)
	d_Bias_avg = Array{CuArray{Float32, 1}}(num_hidden+1)
	d_mB = Array{CuArray{Float32, 1}}(num_hidden+1)
	d_vB = Array{CuArray{Float32, 1}}(num_hidden+1)
	for i = 1:num_hidden+1
		d_B0[i] = CuArray(B0[i])
		d_Biases[i] = CuArray(B0[i])
		d_Bias_grads[i] = similar(d_B0[i])
		d_Bias_avg[i] = CuArray(0.0f0*B0[i]) 
		d_Bias_est[i] = CuArray(0.0f0*B0[i])
		d_mB[i] = CuArray(0.0f0*B0[i])
		d_vB[i] = CuArray(0.0f0*B0[i])
	end

	d_onesVecBATCH = CuArray(ones(Float32, batchSize))
	
	numLayers = length(T0)

	nnCostFunction(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_onesVecBATCH, d_aBATCH, d_tanh_grad_zBATCH, d_deltasBATCH, d_Theta_grads, d_Bias_grads, batchInputs[end], batchOutputs[end],lambda)
	
	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], lambda)
	end
	currentOut = currentOut/numBatches
	
	print_with_color(:red, STDOUT, string("Initial cost is ", currentOut), bold=true)
	println()
	#println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	period = 10
	costRecord = Array{Float32}(ceil(Int, numEpochs/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Array{Float32}(numEpochs+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	eta = alpha
	F = (1.0f0-R)
	t = 1.0f0
	while epoch <= numEpochs
		#run through an epoch in batches with randomized order
		for batch in randperm(numBatches)
			nnCostFunction(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_onesVecBATCH, d_aBATCH, d_tanh_grad_zBATCH, d_deltasBATCH, d_Theta_grads, d_Bias_grads, batchInputs[batch], batchOutputs[batch],lambda,dropout)
			updateParams!(eta, beta1, beta2, t, d_Thetas, d_Theta_grads, d_Biases, d_Bias_grads, d_mT, d_mB, d_vT, d_vB)
			if c < Inf 
				scaleThetas!(d_Thetas[1:end-1], d_Theta_grads[1:end-1], d_onesVecParams, d_normVecParams, c)
			end
			updateEst!(beta2, t, d_Thetas, d_Biases, d_Theta_avg, d_Bias_avg, d_Theta_est, d_Bias_est)
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime
		
		if epoch%period == 0
			currentOut = 0.0f0
			for i = 1:numBatches
				currentOut += nnCostFunctionNOGRAD(d_Theta_est, d_Bias_est, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], lambda, dropout)
			end
			currentOut = currentOut/numBatches
			
			costRecord[iter + 1] = currentOut
			
			if currentOut < bestCost
				GPU2Host((bestThetas, bestBiases), (d_Theta_est, d_Bias_est))
				bestCost = currentOut
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
			remainingEpochs = numEpochs - epoch

			timeRemainingEst = remainingEpochs*epochTime

			#elapsed = currentTime - startTime
			#percentComplete = epoch/N
			#totalTimeEst = elapsed / percentComplete
			#timeRemainingEst = totalTimeEst - elapsed
			lastReport = currentTime
			hoursLeft = floor(timeRemainingEst/(60*60))
			minutesLeft = floor(timeRemainingEst/60 - hoursLeft*60)
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, 1)
			println(string("On epoch ", epoch, " out of ", numEpochs, " best cost is ", round(bestCost, 8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(d_Theta_est, d_Bias_est, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], lambda, dropout)
	end
	currentOut = currentOut/numBatches

	if currentOut < bestCost
		bestCost = currentOut
		GPU2Host((bestThetas, bestBiases), (d_Theta_est, d_Bias_est))
	end

	time_per_epoch = timeRecord[2:end] .- timeRecord[1:end-1]
    train_time = timeRecord[end]
    timePerBatch = train_time/numEpochs/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9

    println("-------------------------------------------------------------------")
	print_with_color(:green, STDOUT, "Completed training on GPU with the following parameters: ", bold = true)
	println()
	#println(string("Completed training on GPU with the following parameters:"))
	println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize))
	println(string("L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha, " decay rate = ", R))
	print_with_color(:red, STDOUT, string("Training Results: Cost reduced from ", costRecord[1], "to ", bestCost, " after ", round(Int64, timeRecord[numEpochs]), " seconds"), bold=true)
	println()	
	#println(string("Cost reduced from ", costRecord[1], " to ", bestCost, " after ", round(Int64, timeRecord[numEpochs]), " seconds and ", numEpochs, " epochs"))		
	println(string("Median time of ", round(Int64, 1e9*median(time_per_epoch)/m), " ns per example"))
    println(string("Total operations per example = ", round(Int64, fops/batchSize), " foward prop ops + ", round(Int64, bops/batchSize), " backprop ops + ", round(Int64, pops/batchSize), " update ops = ", round(Int64, total_ops/batchSize)))
    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
   	println("-------------------------------------------------------------------")
    return bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch
end	