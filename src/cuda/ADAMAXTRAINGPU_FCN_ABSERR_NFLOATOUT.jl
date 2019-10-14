#include neural network functions
include("FCN_ABSERR_NFLOATOUT_CUDACOSTFUNCTION.jl")
#include("FCN_NFLOATOUT_COSTFUNCTIONS.jl")
#include("FCN_NFLOATOUT_AUXFUNCTIONS.jl")

# filepath = joinpath(@__DIR__, "ADAMAX_INTRINSIC_KERNELS.cu")

# adamax_md = if isfile("ADAMAX_INTRINSIC_KERNELS.ptx")
# 	cuModuleLoad("ADAMAX_INTRINSIC_KERNELS.ptx")
# else
# 	run(`nvcc -ptx $filepath`)
# 	cuModuleLoad("ADAMAX_INTRINSIC_KERNELS.ptx")
# end

##TO DO: redo threads, blockers, kernels, and launches to work in 1D
#set up NN kernels
# updateParams = cuModuleGetFunction(adamax_md, "updateParams")
# elSq = cuModuleGetFunction(adamax_md, "elSq")
# elSq2 = cuModuleGetFunction(adamax_md, "elSq2")
# scaleParams = cuModuleGetFunction(adamax_md, "scaleParams")
# updateEst = cuModuleGetFunction(adamax_md, "updateEst")

adamax_kernel_names = ("updateParams", "elSq", "elSq2", "scaleParams", "updateEst")

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
			memcpy!(GPUvars[i][j], hostvars[i][j])
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
			memcpy!(hostvars[i][j], GPUvars[i][j])
		end
	end
end

function calcError(modelOut::NVIDIALibraries.DeviceArray.CUDAArray, dataOut::NVIDIALibraries.DeviceArray.CUDAArray; costFunc = "absErr")
	(m, n) = dataOut.size
	(o, p) = modelOut.size
	
	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	#array to store error values per example
	if costFunc2 == costFunc
		delt = device_copy(modelOut)

		if (m == o) && (n == p)
			run_kernel(costFuncKs[costFunc], m, n, delt, dataOut)
		else
			error("output layer does not match data")
		end

		err = sum(host_allocate(delt))/m
	else
		delt1 = device_copy(modelOut)
		delt2 = device_copy(modelOut)
		if (m == o) && (p == 2*n)
			run_kernel(costFuncKs[costFunc], m, n, delt1, dataOut)
			run_kernel(costFuncKs[costFunc2], m, n, delt2, dataOut)
		else
			error("output layer does not match data")
		end
		err1 = sum(host_allocate(delt1))/m
		err2 = sum(host_allocate(delt2)[:, 1:n])/m #needed b/c only the first n columns of delt2 contain valid errors
		(err1, err2)
	end

end

function calcOutputGPU(input_data, output_data, T, B; dropout = 0.0f0, costFunc = "absErr")
#calculate network output given input data and a set of network parameters.
#calculation is performed on the GPU and then returned to system memory
	# if costFunc != "absErr"
	# 	error("Only the absErr cost function exists for the GPU backend")
	# end

	#Setup some useful variables
	m = size(input_data, 1)
	n = size(output_data, 2)
	
	num_hidden = length(T) - 1
	hidden_layers = if num_hidden > 0
		map(p -> length(p), B[1:num_hidden])
	else
		Int64.([])
	end

	(m, input_layer_size) = size(input_data)
	output_layer_size = if occursin("Log", costFunc)
		2*n
	else
		n
	end

	#transfer parameters to GPU
	d_Thetas = device_allocate(T) 
	d_Biases = device_allocate(B) 
	d_y = cuda_allocate(output_data)
	GC.gc()

	free = zeros(UInt64, 1)
	total = zeros(UInt64, 1)
	cudaMemGetInfo(free, total)
	newMem = free[1] - (100*2^20)
	maxB = min(2^17, getMaxGPUBatchSize(T, B, newMem))
	if maxB == 0
		println("Not enough GPU memory for calculation, returning nothing")
		return nothing
	else
		(d_out, out) = if maxB > m 
			d_X = cuda_allocate(input_data)
			
			predict(d_Thetas, d_Biases, d_X, input_layer_size, output_layer_size, hidden_layers)
		else
			if maxB == 2^17
				println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
			else
				println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of GPU memory"))
			end
			numBatches = ceil(Int64, m/maxB)
			if numBatches == 2
				(d_out1, out1) = predict(d_Thetas, d_Biases, cuda_allocate(input_data[1:maxB, :]), input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				(d_out2, out2) = predict(d_Thetas, d_Biases, cuda_allocate(input_data[maxB+1:m, :]), input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				out3 = [out1; out2]
				(cuda_allocate(out3), out3)
			else
				batchInputs = [view(input_data, (i-1)*maxB+1:i*maxB, :) for i = 1:numBatches-1]
				(d_out1, out1) = predictBatches(d_Thetas, d_Biases, batchInputs, input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				(d_out2, out2) = predict(d_Thetas, d_Biases, cuda_allocate(input_data[(numBatches-1)*maxB+1:m, :]), input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				out3 = [out1; out2]
				(cuda_allocate(out3), out3)
			end
		end
		errs = calcError(d_out, d_y, costFunc = costFunc)
		GC.gc()
		return (out, errs)
	end
end

function calcMultiOutGPU(input_data, output_data, multiParams; dropout = 0.0f0, costFunc = "absErr")
#calculate network output given input data and a set of network parameters.
#calculation is performed on the GPU and then returned to system memory
	#Setup some useful variables
	#Setup some useful variables
	m = size(input_data, 1)
	n = size(output_data, 2)
	
	num_hidden = length(multiParams[1][1]) - 1
	hidden_layers = if num_hidden > 0
		map(p -> length(p), multiParams[1][2][1:num_hidden])
	else
		Int64.([])
	end

	(m, input_layer_size) = size(input_data)
	output_layer_size = if occursin("Log", costFunc)
		2*n
	else
		n
	end

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	#transfer multiParams to GPU
	multiParamsGPU = [begin
		T = params[1]
		B = params[2]
		d_T = [cuda_allocate(M) for M in T]
		d_B = [cuda_allocate(V) for V in B] 
		(d_T, d_B)
	end
	for params in multiParams]	

	
	d_y = cuda_allocate(output_data)
	GC.gc()
	free = zeros(UInt64, 1)
	total = zeros(UInt64, 1)
	cudaMemGetInfo(free, total)
	newMem = free[1] - (100*2^20)
	maxB = min(2^17, getMaxGPUBatchSize(multiParams[1][1], multiParams[1][2], newMem))

	if maxB == 0
		println("Not enough GPU memory for calculation, returning nothing")
		return nothing
	else 
		multiOut = if maxB > m
			d_X = cuda_allocate(input_data)
			predictMulti(multiParamsGPU, d_X, input_layer_size, output_layer_size, hidden_layers)
		else
			if maxB == 2^17
				println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
			else
				println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of GPU memory"))
			end
			numBatches = ceil(Int64, m/maxB)
			(out1, out2) = if numBatches == 2
				out1 = predictMulti(multiParamsGPU, cuda_allocate(input_data[1:maxB, :]), input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				out2 = predictMulti(multiParamsGPU, cuda_allocate(input_data[maxB+1:m, :]), input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				(out1, out2)
			else
				batchInputs = [view(input_data, (i-1)*maxB+1:i*maxB, :) for i = 1:numBatches-1]
				out1 = predictMultiBatches(multiParamsGPU, batchInputs, input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				out2 = predictMulti(multiParamsGPU, cuda_allocate(input_data[(numBatches-1)*maxB+1:m, :]), input_layer_size, output_layer_size, hidden_layers)
				GC.gc()
				(out1, out2)
			end
			map((a, b) -> [a; b], out1, out2)
		end
		GC.gc()
	
		out = if occursin("Log", costFunc)
			out1 = mapreduce(a -> a[:, 1:n], +, multiOut)/length(multiOut)
			out2 = log.(1 ./sqrt.(mapreduce(a -> exp.(-2*a[:, n+1:2*n]), +, multiOut)/length(multiOut)))
			[out1 out2]
		else
			mapreduce(a -> a[:, 1:n], +, multiOut)/length(multiOut)
		end

		outErrEst = if occursin("Log", costFunc)
			mapreduce(a -> abs.([a[:, 1:n] exp.(-a[:, n+1:2*n])] .- [out[:, 1:n] exp.(-out[:, n+1:2*n])]), +, multiOut)/length(multiOut)
		else
			mapreduce(a -> abs.(a .- out), +, multiOut)/length(multiOut)
		end

		errs = calcError(out, output_data, costFunc = costFunc)
		
		return (multiOut, out, errs, outErrEst)
	end
end

function checkNumGradGPU(lambda; hidden_layers=[5, 5], costFunc = "absErr", input_layer_size = 3, n = 2, m = 100)

	Random.seed!(1234)

	#if using log likelihood cost function then need to double output layer size
	#relative to output example size
	output_layer_size = if occursin("Log", costFunc)
		2*n
	else
		n
	end

	X = randn(Float32, m, input_layer_size)
	y = randn(Float32, m, n)
	d_X = cuda_allocate(X)
	d_y = cuda_allocate(y)

	num_hidden = length(hidden_layers)

	T0, B0 = initializeParams(input_layer_size, hidden_layers, output_layer_size)

	d_Thetas = device_allocate(T0) 
	d_Biases = device_allocate(B0) 

	Theta_grads = deepcopy(T0) 
	TGCPU = deepcopy(T0) 

	Bias_grads = deepcopy(B0) 
	BGCPU = deepcopy(B0)

	d_Theta_grads = device_allocate(Theta_grads)
	d_Bias_grads = device_allocate(Bias_grads)

	onesVec = ones(Float32, m)
	d_ones = cuda_allocate(onesVec)
	tanh_grad_z = form_tanh_grads(hidden_layers, m)
	d_tanh_grad_z = device_allocate(tanh_grad_z)

	d_a = form_activations(d_Thetas, m)
	d_deltas = form_activations(d_Thetas, m)
	a = form_activations(T0, m)
	deltas = form_activations(T0, m)

	numLayers = length(T0)

	e = 0.001f0
	params = theta2Params(B0, T0)
	l = length(params)
	perturb = zeros(Float32, l)
	numGrad = Array{Float32}(undef, l)

	nnCostFunction(d_Thetas, d_Biases, input_layer_size, output_layer_size, hidden_layers, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, d_X, d_y,lambda, costFunc = costFunc)
	nnCostFunction(T0, B0, input_layer_size, hidden_layers, X, y, lambda, TGCPU, BGCPU, tanh_grad_z, a, deltas, onesVec, costFunc = costFunc)
	costGPU = nnCostFunctionNOGRAD(d_Thetas, d_Biases, input_layer_size, output_layer_size, hidden_layers, m, d_a, d_X, d_y,lambda, costFunc = costFunc)
	costCPU = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc)
	
	GPU2Host((Theta_grads, Bias_grads), (d_Theta_grads, d_Bias_grads))

	funcGrad = theta2Params(Bias_grads, Theta_grads)
	funcGradCPU = theta2Params(BGCPU, TGCPU)

	for i = 1:l
		perturb[i] = e
		Tplus, Bplus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params+perturb)
		Tminus, Bminus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params-perturb)
		
		outminus = nnCostFunctionNOGRAD(Tminus, Bminus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc)
		outplus = nnCostFunctionNOGRAD(Tplus, Bplus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc)
		
		perturb[i] = 0.0f0  #restore perturb vector to 0

		numGrad[i] = (outplus - outminus)/(2.0f0*e)
	end

	println("GPU Cost  CPU Cost" )
	println(string(costGPU, "  ", costCPU))

	println("___________________")
	
	println("Num Grads  GPU Grads CPU Grads")
	for i = 1:length(numGrad)
		@printf "%0.6f  %0.6f %0.6f \n" numGrad[i] funcGrad[i] funcGradCPU[i]
	end
	GPUErr = norm(numGrad .- funcGrad)/norm(numGrad .+ funcGrad)
	GPUCPUErr = norm(funcGradCPU .- funcGrad)/norm(funcGradCPU .+ funcGrad)
	println(string("Relative differences for method are ", GPUErr, ".  Should be small (1e-9)"))
	println(string("Relative differences with CPU are ", GPUCPUErr, ".  Should be small (1e-9)"))

	return GPUErr
end

# checkNumGradGPU(0.0f0)

function ADAMAXTrainNNGPU(input_data, output_data, batchSize, T0, B0, numEpochs, input_layer_size, hidden_layers, lambda, c; alpha=0.001f0, R=0.1f0, printProgress = false, printAnything = true, dropout = 0.0f0, costFunc="absErr")
#train on a GPU fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  The final required input "md" is the context for the GPU hardware being used.
#Note that all floating point input variables must be float32 or single precision   
	
	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")	(m, n) = size(input_data)
	(m, n) = size(input_data)
	(m2, output_layer_size) = size(output_data)
	if m2 != m 
		error("input and output data do not match")
	end

	n2 = if occursin("Log", costFunc)
		2*output_layer_size
	else
		output_layer_size
	end

	#check that parameters are appropriate for input and output data given selected cost function
	if size(T0[1], 2) != n 
		error("parameters incompatible with input data")
	end
	
	if occursin("Log", costFunc) 
		if length(B0[end]) != 2*output_layer_size
			error("parameters incompatible with output data for log likelihood cost function")
		end
	elseif length(B0[end]) != output_layer_size
		error("parameters incompatible with output data for sq/absErr cost function")
	end


	println()
	printstyled(color = :green, stdout, "Beginning training on GPU with the following parameters:", bold=true)
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
			cublasSgemv(cublas_handle, 'N', 1.0f0, d_Theta_grads[i], d_onesVecParams[i], 0.0f0, d_normVecParams[i])
			
			#scale each row so the squared sum is less than or equal to c^2
			run_kernel(scaleParams, N, M, c, d_Thetas[i], d_normVecParams[i])
			#CUDArt.launch(scaleParams, blocks(N, M), threads, (N, M, c, d_Thetas[i], d_normVecParams[i]))
		end
		cuCtxSynchronize()
	end
	
	function updateParams!(alpha, beta1, beta2, t, d_Thetas, d_Theta_grads, d_Biases, d_Bias_grads, d_mT, d_mB, d_vT, d_vB)
		for i = 1:length(d_Thetas)
			(N, M) = size(T0[i])
			
			#launch kernel to update Thetas
			run_kernel(updateParams, N, M, alpha, beta1, beta2, t, d_Thetas[i], d_Theta_grads[i], d_mT[i], d_vT[i])
			N = length(B0[i])
			M = 1
			#launch kernel to update Biases
			run_kernel(updateParams, N, M, alpha, beta1, beta2, t, d_Biases[i], d_Bias_grads[i], d_mB[i], d_vB[i])
			#CUDArt.launch(updateParams, blocks(N, M), threads, (N, M, alpha, beta1, beta2, t, d_Biases[i], d_Bias_grads[i], d_mB[i], d_vB[i]))
		end
		cuCtxSynchronize()
	end

	function updateEst!(beta2, t, d_Thetas, d_Biases, d_Theta_avg, d_Bias_avg, d_Theta_est, d_Bias_est)
		for i = 1:length(d_Thetas)
			(N, M) = size(T0[i])
			scale = 1.0f0/(1.0f0 - beta2^t)
			
			#launch kernel to update Thetas
			run_kernel(updateEst, N, M, beta2, scale, d_Thetas[i], d_Theta_avg[i], d_Theta_est[i])
			
			N = length(B0[i])
			M = 1
			#launch kernel to update Biases
			run_kernel(updateEst, N, M, beta2, scale, d_Biases[i], d_Bias_avg[i], d_Bias_est[i])
		end
		cuCtxSynchronize()	
	end
	
	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	if printAnything
		println()
		printstyled(color = :green, stdout, "Beginning training with the following parameters:", bold=true)
		println()
		println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", num epochs = ", numEpochs, ", training alpha = ", alpha, ", decay rate = ", R, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", dropout rate = ", dropout))
		println("-------------------------------------------------------------------")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
	
	batchInputs = device_allocate(inputbatchData)
	batchOutputs = device_allocate(outputbatchData)
		
	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	#initialize parameter and gradient variables
	d_T0 = device_allocate(T0)
	d_Thetas = device_allocate(T0)
	d_Theta_grads = device_allocate(T0)
	d_Theta_avg = device_allocate(T0, 0.0f0)
	d_Theta_est = device_allocate(T0, 0.0f0)
	d_mT = device_allocate(T0, 0.0f0)
	d_vT = device_allocate(T0, 0.0f0)

	d_B0 = device_allocate(B0)
	d_Biases = device_allocate(B0)
	d_Bias_grads = device_allocate(B0)
	d_Bias_avg = device_allocate(B0, 0.0f0)
	d_Bias_est = device_allocate(B0, 0.0f0)
	d_mB = device_allocate(B0, 0.0f0)
	d_vB = device_allocate(B0, 0.0f0)


	#create vectors of ones equal to the number of columns in each theta matrix
	d_onesVecParams = map(a -> cuda_allocate(ones(Float32, a)), [n; hidden_layers])
	#create a vector to store the squared sum of each row in the theta matricies
	d_normVecParams = map(a -> cuda_allocate(zeros(Float32, a)), [hidden_layers; n2])
	
	#initialize activation gradients on device
	tanh_grad_zBATCH = form_tanh_grads(hidden_layers, batchSize)
	d_tanh_grad_zBATCH = device_allocate(tanh_grad_zBATCH)
	
	#initialize activations and deltas on device
	d_aBATCH = form_activations(d_Thetas, batchSize)
	d_deltasBATCH = form_activations(d_Thetas, batchSize)
	
	d_onesVecBATCH = cuda_allocate(ones(Float32, batchSize))
	
	numLayers = length(T0)

	nnCostFunction(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_onesVecBATCH, d_aBATCH, d_tanh_grad_zBATCH, d_deltasBATCH, d_Theta_grads, d_Bias_grads, batchInputs[end], batchOutputs[end],lambda, dropout, costFunc = costFunc)
	
	currentOut = 0.0f0

	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], lambda, 0.0f0, costFunc = costFunc)
	end
	currentOut = currentOut/numBatches
	
	printstyled(stdout, string("Initial cost is ", currentOut), color = :red, bold=true)
	println()
	#println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	period = 10
	costRecord = Vector{Float32}(undef, ceil(Int, numEpochs/period)+1)
	costRecord[1] = currentOut

	startTime = time()
	lastReport = startTime

	timeRecord = Vector{Float32}(undef, numEpochs+1)
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
			nnCostFunction(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_onesVecBATCH, d_aBATCH, d_tanh_grad_zBATCH, d_deltasBATCH, d_Theta_grads, d_Bias_grads, batchInputs[batch], batchOutputs[batch],lambda,dropout, costFunc = costFunc)
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
				currentOut += nnCostFunctionNOGRAD(d_Theta_est, d_Bias_est, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], lambda, 0.0f0, costFunc = costFunc)
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
			secondsLeft = round(timeRemainingEst - minutesLeft*60 - hoursLeft*60*60, digits=1)
			println(string("On epoch ", epoch, " out of ", numEpochs, " best cost is ", round(bestCost, digits=8)))
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end

	currentOut = 0.0f0
	for i = 1:numBatches
		currentOut += nnCostFunctionNOGRAD(d_Theta_est, d_Bias_est, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], lambda, 0.0f0, costFunc = costFunc)
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


   if printAnything
		println("-------------------------------------------------------------------")
		printstyled(color = :green, stdout, "Completed training on GPU with the following parameters: ", bold = true)
		println()
		println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", num epochs = ", numEpochs, ", training alpha = ", alpha, ", decay rate = ", R, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", dropout rate = ", dropout))
	
		printstyled(color = :red, stdout, string("Training Results: Cost reduced from ", costRecord[1], "to ", bestCost, " after ", round(Int64, timeRecord[numEpochs]), " seconds and ", numEpochs, " epochs"), bold=true)
		println()	
		println(string("Median time of ", 1e9*median(time_per_epoch)/m, " ns per example"))
	    println(string("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize))
	    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
	    println("-------------------------------------------------------------------")
	end

    return bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch
end	