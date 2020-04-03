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

# function copy_params!(dest::Vector{CUDAArray}, src::Vector{CUDAArray})
function copy_params!(dest, src)
	@assert (length(dest) == length(src))
	for i in eachindex(dest)
		memcpy!(dest[i], src[i])
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
		deallocate!(delt)
		return err
	else
		delt2 = device_copy(modelOut)
		if (m == o) && (p == 2*n)
			delt = device_copy(modelOut)
			run_kernel(costFuncKs[costFunc], m, n, delt, dataOut)
			err1 = sum(host_allocate(delt))/m
			deallocate!(delt)
			delt = device_copy(modelOut)
			run_kernel(costFuncKs[costFunc2], m, n, delt, dataOut)
			err2 = sum(host_allocate(delt)[:, 1:n])/m #needed b/c only the first n columns of delt2 contain valid errors
			deallocate!(delt)
		else
			error("output layer does not match data")
		end
		return (err1, err2)
	end

end

function calcOutputGPU!(d_X, d_y, d_T, d_B, d_a; costFunc = "absErr", resLayers=0)
	predict!(d_T, d_B, d_X, d_a, resLayers)
	errs = calcError(d_a[end], d_y, costFunc=costFunc)
end

function calcOutputGPU!(d_X, d_T, d_B, d_a; costFunc = "absErr", resLayers=0)
	predict!(d_T, d_B, d_X, d_a, resLayers)
	errs = calcError(d_a[end], d_y, costFunc=costFunc)
end

function calcOutputGPU(input_data, output_data, T, B; dropout = 0.0f0, costFunc = "absErr", resLayers = 0)
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

	free = zeros(UInt64, 1)
	total = zeros(UInt64, 1)
	cudaMemGetInfo(free, total)
	newMem = free[1] - (100*2^20)
	maxB = min(2^17, getMaxGPUBatchSize(T, B, newMem))
	if maxB == 0
		println("Not enough GPU memory for calculation, returning nothing")
		return nothing
	else
		(out, errs) = if maxB > m 
			d_X = cuda_allocate(input_data)
			d_a = form_activations(d_Thetas, m)
			predict!(d_Thetas, d_Biases, d_X, d_a, resLayers)
			errs = calcError(d_a[end], cuda_allocate(output_data), costFunc = costFunc)
			(host_allocate(d_a[end]), errs)
			# predict(d_Thetas, d_Biases, d_X, input_layer_size, output_layer_size, hidden_layers, resLayers)
		else
			if maxB == 2^17
				println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
			else
				println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of GPU memory"))
			end
			numBatches = ceil(Int64, m/maxB)
			if numBatches == 2
				d_X = cuda_allocate(input_data[1:maxB, :])
				(d_out1, out1) = predict(d_Thetas, d_Biases, d_X, input_layer_size, output_layer_size, hidden_layers, resLayers)
				deallocate!(d_X)
				d_X = cuda_allocate(input_data[maxB+1:m, :])
				(d_out2, out2) = predict(d_Thetas, d_Biases, d_X, input_layer_size, output_layer_size, hidden_layers, resLayers)
				deallocate!(d_X)
				out3 = [out1; out2]
				d_y = cuda_allocate(output_data[1:maxB, :])
				errs1 = calcError(d_out1, d_y, costFunc = costFunc)
				deallocate!(d_y)
				d_y = cuda_allocate(output_data[maxB+1:m, :])
				errs2 = calcError(d_out2, d_y, costFunc = costFunc)
				deallocate!(d_y)
				if length(errs1) == 1
					errs = (maxB*errs1 + (m-maxB)*errs2) / m
				else
					tmp1 = (maxB*errs1[1] + (m-maxB)*errs2[1]) / m
					tmp2 = (maxB*errs1[2] + (m-maxB)*errs2[2]) / m
					errs = (tmp1, tmp2)
				end
				deallocate!(d_out1)
				deallocate!(d_out2)
				(out3, errs)
			else
				batchinds = [(i-1)*maxB+1:i*maxB for i in 1:numBatches-1]
				# batchInputs = [view(input_data, (i-1)*maxB+1:i*maxB, :) for i = 1:numBatches-1]
				# batchOutputs = [view(output_data, (i-1)*maxB+1:i*maxB, :) for i = 1:numBatches-1]
				l = length(d_Thetas)
				num_hidden = l - 1

				d_a = form_activations(d_Thetas, maxB)
				out = Matrix{Float32}(undef, m, output_layer_size)
				if occursin("Log", costFunc)
					cumerr = [0.0f0, 0.0f0]
				else
					cumerr = 0.0f0
				end 
				for inds in batchinds
					d_X = cuda_allocate(input_data[inds, :])
					forwardNOGRAD!(d_a, d_Thetas, d_Biases, hidden_layers, d_X, resLayers)
					d_y = cuda_allocate(output_data[inds, :])
					deallocate!(d_X)
					err = calcError(d_a[end], d_y, costFunc=costFunc)
					deallocate!(d_y)
					out[inds, :] .= host_allocate(d_a[end])
					if length(err) == 1
						cumerr += err
					else
						cumerr[1] += err[1]
						cumerr[2] += err[2]
					end
				end

				clear_gpu_data(d_a)

				finalinds = (numBatches-1)*maxB+1:m
				(d_out2, out2) = predict(d_Thetas, d_Biases, cuda_allocate(input_data[finalinds, :]), input_layer_size, output_layer_size, hidden_layers, resLayers)
				errs2 = calcError(d_out2, cuda_allocate(output_data[finalinds, :]), costFunc = costFunc)
				out[finalinds, :] .= out2

				if length(errs2) == 1
					err = (size(out2, 1)*errs2 + maxB*cumerr) / m
				else
					err = ((size(out2, 1)*errs2[1] + maxB*cumerr[1]) / m, (size(out2, 1)*errs2[2] + maxB*cumerr[2]) / m)
				end
				deallocate!(d_out2)
				(out, err)
			end
		end
	end
	clear_gpu_data(d_Thetas)
	clear_gpu_data(d_Biases)
	return (out, errs)
end

function errshufflecol(d_T::Vector{CUDAArray}, d_B::Vector{CUDAArray}, input_data::Matrix{Float32}, d_input_data::CUDAArray, d_output_data::CUDAArray, d_a::Vector{CUDAArray}, v::Vector, d_v::CUDAArray, ind; rng=1, reslayers=0, costFunc = "sqErr")


	#fill v with column to shuffle
	v .= view(input_data, :, ind)
	shuffle!(MersenneTwister(rng), v)
	memcpy!(d_v, v)
	run_kernel_1D(swap_matrix_col, length(v), ind, d_input_data, d_v)

	hidden_layers = [a.size[1] for a in d_B][1:end-1]

	errs = nnCostFunctionNOGRAD(d_T, d_B, size(input_data, 2), 1, hidden_layers::Vector, size(input_data, 1), d_a, d_input_data, d_output_data, 0.0f0, costFunc = costFunc, resLayers = reslayers)
	
	#restore input_data_copy to initial state
	run_kernel_1D(swap_matrix_col, length(v), ind, d_input_data, d_v)

	return errs
end

function calcfeatureimpact(d_T::Vector{CUDAArray}, d_B::Vector{CUDAArray}, input_data::Matrix{Float32}, output_data::Matrix{Float32}; reslayers=0, costFunc="sqErr", num=10, fixedshuffle=[])
	(m, n) = size(input_data)
	d_a = form_activations(d_T, m)
	v = Vector{Float32}(undef, m)
	d_v = cuda_allocate(v)
	d_input_data = cuda_allocate(input_data)
	d_output_data = cuda_allocate(output_data)

	hidden_layers = [a.size[1] for a in d_B][1:end-1]
	# println(hidden_layers)

	NNerr = nnCostFunctionNOGRAD(d_T, d_B, n, 1, hidden_layers, m, d_a, d_input_data, d_output_data, 0.0f0, costFunc = costFunc, resLayers= reslayers)
	# println("NNerr = $NNerr")

	shuffleind = setdiff(1:n, fixedshuffle)
	shuffle_errs = Vector{Float64}(undef, length(shuffleind))
	shuffle_cols = Vector{Vector{Int64}}(undef, length(shuffleind))
	if num == 1
		for ind in fixedshuffle
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(1), v)
			memcpy!(d_v, v)
			run_kernel(swap_matrix_col, length(v), ind, d_input_data, d_v)
		end
	end
	println()
	println()
	tlast = time()
	for (i, c) in enumerate(shuffleind)
		if num == 1
			err = errshufflecol(d_T, d_B, input_data, d_input_data, d_output_data, d_a, v, d_v, c, rng=1, reslayers=reslayers, costFunc=costFunc)
			# println("new err = $err")
		else
			err = maximum([errshufflecol(d_T, d_B, input_data, d_input_data, d_output_data, d_a, v, d_v, c, rng=j, reslayers=reslayers, costFunc=costFunc) for j in 1:num])
		end
		shuffle_errs[i] = err
		shuffle_cols[i] = [c; fixedshuffle]
		if (i == 1) || (time()-tlast > 2)
			print("\33[2K\033[A\r")
			print("\33[2K\033[A\r")
			println("Finished evaluating shuffled column $c: number $i of $(length(shuffleind))")
			println("with an error change of $(100*(err - NNerr)/abs(NNerr))%")
			tlast = time()
		end
	end

	if num == 1
		for ind in fixedshuffle
			view(input_copy, :, ind) .= view(input_data, :, ind)
		end
	end

	if !isempty(fixedshuffle)
		fixederr = maximum([errshufflecols(T, B, input_data, output_data, input_copy, a, v, fixedshuffle, rng=j, reslayers=reslayers, costFunc=costFunc)[1] for j in 1:num])
	else
		fixederr = NNerr
	end
	featureimpact = 100 .*(shuffle_errs .- NNerr) ./ abs(NNerr)
	sortinds = reverse(sortperm(featureimpact))
	(NNerr, zip(shuffle_cols[sortinds], featureimpact[sortinds]), fixederr)
end

function calcMultiOutGPU(input_data, output_data, multiParams; dropout = 0.0f0, costFunc = "absErr", resLayers=0)
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
			predictMulti(multiParamsGPU, d_X, input_layer_size, output_layer_size, hidden_layers, resLayers)
		else
			if maxB == 2^17
				println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
			else
				println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of GPU memory"))
			end
			numBatches = ceil(Int64, m/maxB)
			(out1, out2) = if numBatches == 2
				out1 = predictMulti(multiParamsGPU, cuda_allocate(input_data[1:maxB, :]), input_layer_size, output_layer_size, hidden_layers, resLayers)
				GC.gc()
				out2 = predictMulti(multiParamsGPU, cuda_allocate(input_data[maxB+1:m, :]), input_layer_size, output_layer_size, hidden_layers, resLayers)
				GC.gc()
				(out1, out2)
			else
				batchInputs = [view(input_data, (i-1)*maxB+1:i*maxB, :) for i = 1:numBatches-1]
				out1 = predictMultiBatches(multiParamsGPU, batchInputs, input_layer_size, output_layer_size, hidden_layers, resLayers)
				GC.gc()
				out2 = predictMulti(multiParamsGPU, cuda_allocate(input_data[(numBatches-1)*maxB+1:m, :]), input_layer_size, output_layer_size, hidden_layers, resLayers)
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

function checkNumGradGPU(lambda; hidden_layers=[5, 5], costFunc = "absErr", input_layer_size = 3, n = 2, m = 100, resLayers=0)

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

	nnCostFunction(d_Thetas, d_Biases, input_layer_size, output_layer_size, hidden_layers, m, d_ones, d_a, d_tanh_grad_z, d_deltas, d_Theta_grads, d_Bias_grads, d_X, d_y,lambda, costFunc = costFunc, resLayers=resLayers)
	nnCostFunction(T0, B0, input_layer_size, hidden_layers, X, y, lambda, TGCPU, BGCPU, tanh_grad_z, a, deltas, onesVec, costFunc = costFunc, resLayers=resLayers)
	costGPU = nnCostFunctionNOGRAD(d_Thetas, d_Biases, input_layer_size, output_layer_size, hidden_layers, m, d_a, d_X, d_y,lambda, costFunc = costFunc, resLayers=resLayers)
	costCPU = nnCostFunctionNOGRAD(T0, B0, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc, resLayers=resLayers)
	
	GPU2Host((Theta_grads, Bias_grads), (d_Theta_grads, d_Bias_grads))

	funcGrad = theta2Params(Bias_grads, Theta_grads)
	funcGradCPU = theta2Params(BGCPU, TGCPU)

	for i = 1:l
		perturb[i] = e
		Tplus, Bplus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params+perturb)
		Tminus, Bminus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params-perturb)
		
		outminus = nnCostFunctionNOGRAD(Tminus, Bminus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc, resLayers=resLayers)
		outplus = nnCostFunctionNOGRAD(Tplus, Bplus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc, resLayers=resLayers)
		
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

function scaleThetas!(d_Thetas::Vector{CUDAArray}, d_Theta_grads::Vector{CUDAArray}, d_onesVecParams::Vector{CUDAArray}, d_normVecParams::Vector{CUDAArray}, c)
	for i = 1:length(d_Thetas)
		#get rows and columns of input matrix
		(N, M) = d_Thetas[i].size
		
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

function updateParams!(alpha, beta1, beta2, t, d_Thetas::Vector{CUDAArray}, d_Theta_grads::Vector{CUDAArray}, d_Biases::Vector{CUDAArray}, d_Bias_grads::Vector{CUDAArray}, d_mT, d_mB, d_vT, d_vB)
	for i = 1:length(d_Thetas)
		(N, M) = d_Thetas[i].size
		
		#launch kernel to update Thetas
		run_kernel(updateParams, N, M, alpha, beta1, beta2, t, d_Thetas[i], d_Theta_grads[i], d_mT[i], d_vT[i])
		N = d_Biases[i].size[1]
		M = 1
		#launch kernel to update Biases
		run_kernel(updateParams, N, M, alpha, beta1, beta2, t, d_Biases[i], d_Bias_grads[i], d_mB[i], d_vB[i])
		#CUDArt.launch(updateParams, blocks(N, M), threads, (N, M, alpha, beta1, beta2, t, d_Biases[i], d_Bias_grads[i], d_mB[i], d_vB[i]))
	end
	cuCtxSynchronize()
end

function updateParams!(alpha, d_T::Vector{CUDAArray}, d_B::Vector{CUDAArray}, d_TG::Vector{CUDAArray}, d_BG::Vector{CUDAArray})
	for i in eachindex(d_T)
		cublasSaxpy(cublas_handle, -alpha, d_TG[i], d_T[i])
		cublasSaxpy(cublas_handle, -alpha, d_BG[i], d_B[i])
	end
	cuCtxSynchronize()	
end

function updateEst!(beta2, t, d_Thetas::Vector{CUDAArray}, d_Biases::Vector{CUDAArray}, d_Theta_avg::Vector{CUDAArray}, d_Bias_avg::Vector{CUDAArray}, d_Theta_est::Vector{CUDAArray}, d_Bias_est::Vector{CUDAArray})
	for i = 1:length(d_Thetas)
		(N, M) = d_Thetas[i].size
		scale = 1.0f0/(1.0f0 - beta2^t)
		
		#launch kernel to update Thetas
		run_kernel(updateEst, N, M, beta2, scale, d_Thetas[i], d_Theta_avg[i], d_Theta_est[i])
		
		N = d_Biases[i].size[1]
		M = 1
		#launch kernel to update Biases
		run_kernel(updateEst, N, M, beta2, scale, d_Biases[i], d_Bias_avg[i], d_Bias_est[i])
	end
	cuCtxSynchronize()	
end

function updateAvg!(nModels, d_T::Vector{CUDAArray}, d_B::Vector{CUDAArray}, d_Tavg::Vector{CUDAArray}, d_Bavg::Vector{CUDAArray})
	a = Float32(1/nModels)
	b = Float32(nModels/(nModels+1))
	for i = eachindex(d_T)
		cublasSaxpy(cublas_handle, a, d_T[i], d_Tavg[i])
		cublasSscal(cublas_handle, b, d_Tavg[i])
		cublasSaxpy(cublas_handle, a, d_B[i], d_Bavg[i])
		cublasSscal(cublas_handle, b, d_Bavg[i])
	end
	cuCtxSynchronize()
end

function ADAMAXTrainNNGPU(data, batchSize, T0, B0, numEpochs, input_layer_size, hidden_layers, lambda, c; alpha=0.001f0, R=0.1f0, printProgress = false, printAnything = true, dropout = 0.0f0, costFunc="absErr", resLayers = 0, tol=Inf, patience=3, swa=false, ignorebest=false, minepoch=0, prepdata=(), prepactivations=())
#train on a GPU fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  The final required input "md" is the context for the GPU hardware being used.
#Note that all floating point input variables must be float32 or single precision   
	input_data = data[1][1]
	output_data = data[1][2]


	testset = (length(data) > 1)

	if testset
		input_test = data[2][1]
		output_test = data[2][2]
		(mtest, ntest) = size(input_test)
	end

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

	if printAnything
		println()
		printstyled(color = :green, stdout, "Beginning training on GPU with the following parameters:", bold=true)
		println()
		println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", training alpha = ", alpha, ", decay rate = ", R, ", residual size = ", resLayers))
		println("-------------------------------------------------------------------")
	end
	
	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

	if isempty(prepdata)
		(inputbatchData, outputbatchData) = generateBatches(input_data, output_data, batchSize)
		
		batchInputs = device_allocate(inputbatchData)
		batchOutputs = device_allocate(outputbatchData)

		if testset
			d_testinput = cuda_allocate(input_test)
			d_testoutput = cuda_allocate(output_test)
		end
	else
		batchInputs = prepdata[1]
		batchOutputs = prepdata[2]
		if testset
			d_testinput = prepdata[3]
			d_testoutput = prepdata[4]
		end
	end

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
	if isempty(prepactivations)
		tanh_grad_zBATCH = form_tanh_grads(hidden_layers, batchSize)
		d_tanh_grad_zBATCH = device_allocate(tanh_grad_zBATCH)
		
		#initialize activations and deltas on device
		d_aBATCH = form_activations(d_Thetas, batchSize)
		d_deltasBATCH = form_activations(d_Thetas, batchSize)
	else
		d_tanh_grad_zBATCH = prepactivations[1]
		d_aBATCH = prepactivations[2]
		d_deltasBATCH = prepactivations[3]
	end
		
	d_onesVecBATCH = cuda_allocate(ones(Float32, batchSize))
	
	numLayers = length(T0)

	testset && (d_aTEST = form_activations(d_Thetas, mtest))

	nnCostFunction(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_onesVecBATCH, d_aBATCH, d_tanh_grad_zBATCH, d_deltasBATCH, d_Theta_grads, d_Bias_grads, batchInputs[end], batchOutputs[end],lambda, dropout, costFunc = costFunc, resLayers=resLayers)
	
	function calcout_batches(d_T, d_B)
		currentOut = 0.0f0

		for i = 1:numBatches
			currentOut += nnCostFunctionNOGRAD(d_T, d_B, input_layer_size, n2, hidden_layers, batchSize, d_aBATCH, batchInputs[i], batchOutputs[i], 0.0f0, 0.0f0, costFunc = costFunc, resLayers=resLayers)
		end
		currentOut = currentOut/numBatches
	end

	if testset
		calcout_test(d_T, d_B) = nnCostFunctionNOGRAD(d_T, d_B, input_layer_size, n2, hidden_layers, mtest, d_aTEST, d_testinput, d_testoutput, 0.0f0, 0.0f0, costFunc=costFunc, resLayers=resLayers)
	end

	currentOut = calcout_batches(d_Thetas, d_Biases)
	testset && (testout = calcout_test(d_Thetas, d_Biases))
	
	if printAnything
		printstyled(stdout, string("Initial cost is ", currentOut), color = :red, bold=true)
		println()
	end
	#println(string("Initial cost is ", currentOut))

	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	period = 10
	costRecord = Vector{Float32}(undef, ceil(Int, numEpochs/period)+1)
	costRecord[1] = currentOut
	if testset
		costRecordTest = Vector{Float32}(undef, ceil(Int, numEpochs/period)+1)
		costRecordTest[1] = testout
	end

	startTime = time()
	lastReport = startTime

	timeRecord = Vector{Float32}(undef, numEpochs+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	testset && (bestCostTest = testout)
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	eta = alpha
	F = (1.0f0-R)
	G = alpha*F
	t = 1.0f0
	tfail = 0
	tolpass=true
	bestresultepoch = 0

	while (epoch <= minepoch) || ((epoch <= numEpochs) && (tfail <= patience) && tolpass)
		#run through an epoch in batches with randomized order
		for batch in randperm(numBatches)
			if swa || (eta > 0)
				nnCostFunction(d_Thetas, d_Biases, input_layer_size, n2, hidden_layers, batchSize, d_onesVecBATCH, d_aBATCH, d_tanh_grad_zBATCH, d_deltasBATCH, d_Theta_grads, d_Bias_grads, batchInputs[batch], batchOutputs[batch],lambda,dropout, costFunc = costFunc, resLayers=resLayers)
				if swa && (epoch > 100)
					updateParams!(G, d_Thetas, d_Biases, d_Theta_grads, d_Bias_grads)
				else
					updateParams!(eta, beta1, beta2, t, d_Thetas, d_Theta_grads, d_Biases, d_Bias_grads, d_mT, d_mB, d_vT, d_vB)
				end
				if c < Inf 
					scaleThetas!(d_Thetas[1:end-1], d_Theta_grads[1:end-1], d_onesVecParams, d_normVecParams, c)
				end
			end
			#use recent time average of parameter changes for estimate
			if !swa || (epoch <= 100)
				updateEst!(beta2, t, d_Thetas, d_Biases, d_Theta_avg, d_Bias_avg, d_Theta_est, d_Bias_est)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime
		
		if epoch%period == 0
			currentOut = calcout_batches(d_Theta_est, d_Bias_est)
			costRecord[iter + 1] = currentOut
			
			if testset
				testout = calcout_test(d_Theta_est, d_Bias_est)
				costRecordTest[iter+1] = testout
			end
			
			if (epoch <= minepoch) || ignorebest || (testset && (testout < bestCostTest)) || (!testset && (currentOut < bestCost))
				GPU2Host((bestThetas, bestBiases), (d_Theta_est, d_Bias_est))
				bestCost = currentOut
				testset && (bestCostTest = testout)
				bestresultepoch = epoch
				tfail = 0
			else
				tfail += 1
			end
			
			if epoch > 100
				#println(string("eta = ", eta))
				eta = eta*F

				if (testset ? testout : currentOut) > tol
					tolpass = false
				end
			end
			iter += 1
		end

		currentTime = time()
		#print status every 5 seconds
		
		if ((currentTime - lastReport) >= 5) & printProgress & printAnything
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
			if testset
				println(string("On epoch ", epoch, " out of ", numEpochs, " best train and test cost is ", (round(bestCost, sigdigits=5), round(bestCostTest, sigdigits=5))))
			else
				println(string("On epoch ", epoch, " out of ", numEpochs, " best cost is ", round(bestCost, digits=8)))
			end
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end

		epoch += 1
	end
	lastepoch = epoch - 1
	currentOut = calcout_batches(d_Theta_est, d_Bias_est)
	testset && (testout = calcout_test(d_Theta_est, d_Bias_est))

	if ignorebest || (testset && (testout < bestCostTest)) || (!testset && (currentOut < bestCost))
		bestCost = currentOut
		testset && (bestCostTest = testout)
		bestresultepoch = lastepoch
		GPU2Host((bestThetas, bestBiases), (d_Theta_est, d_Bias_est))
	end

	time_per_epoch = timeRecord[2:lastepoch+1] .- timeRecord[1:lastepoch]
    train_time = timeRecord[lastepoch]
    timePerBatch = train_time/numEpochs/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9


   if printAnything
		println("-------------------------------------------------------------------")
		printstyled(color = :green, stdout, "Completed training on GPU with the following parameters: ", bold = true)
		println()
		println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", num epochs = ", numEpochs, ", training alpha = ", alpha, ", decay rate = ", R, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", dropout rate = ", dropout))
	
		printstyled(stdout, string("Training Results: Cost reduced from ", testset ? costRecordTest[1] : costRecord[1], "to ", testset ? bestCostTest : bestCost, " after ", round(Int64, timeRecord[lastepoch+1]), " seconds and ", lastepoch, " epochs"), bold=true, color=:red)
		println()	
		println(string("Median time of ", 1e9*median(time_per_epoch)/m, " ns per example"))
	    println(string("Total operations per example = ", fops/batchSize, " foward prop ops + ", bops/batchSize, " backprop ops + ", pops/batchSize, " update ops = ", total_ops/batchSize))
	    println(string("Approximate GFLOPS = ", median(GFLOPS_per_epoch)))
	    println("-------------------------------------------------------------------")
	end

	testresults = if testset
		(bestCostTest, costRecordTest, lastepoch, bestresultepoch)
	else
		(bestresultepoch,)
	end

    return (bestThetas, bestBiases, bestCost, costRecord, timeRecord, GFLOPS_per_epoch, testresults...)
end