#include neural network functions
include("FCN_NFLOATOUT_COSTFUNCTIONS.jl")
include("FCN_NFLOATOUT_AUXFUNCTIONS.jl")

function calcOps(M, H, O, B)
	
	numParams = if length(H) > 0
		numParams1 = H[1]*M + H[1] + H[end]*O+O
		numParams2 = if length(H) > 1
			sum(H[2:end].*H[1:end-1] .+ H[1:end-1])
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
			B*(2*sum(H[2:end] .* H[1:end-1])+sum(H[2:end])*26)
		else
			0
		end
		
		fops = fops1+fops2
		
		bops1 = B*O*2
		
		
		
		bops2 = B*H[end]*(2*O + 1)
		bops3 = if length(H) > 1
			B*sum(H[1:end-1] .* (2*H[2:end] .+ 1))
		else
			0
		end
		
		bops4 = (2*B+2)*H[1]*(M+1)
		bops5 = if length(H)> 1
			(2*B+2)*sum(H[2:end] .* (H[1:end-1] .+ 1))
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

function calcError(modelOut::Array{Float32, 2}, dataOut::Array{Float32, 2}; costFunc = "absErr")
	#Setup some useful variables
	(m, n) = size(dataOut)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	#array to store error values per example
	if costFunc2 == costFunc
		delt = similar(dataOut)

		if (length(modelOut) == length(dataOut))
			@simd for i = 1:m*n
				@inbounds delt[i] = costFuncs[costFunc](modelOut[i], dataOut[i])
			end
		else
			error("output layer does not match data")
		end

		err = sum(delt)/m
	else
		delt1 = similar(dataOut)
		delt2 = similar(dataOut)
		if (length(modelOut) == 2*length(dataOut))
			@simd for i = 1:m*n
				@inbounds delt1[i] = costFuncs[costFunc](modelOut[i], modelOut[i+(m*n)], dataOut[i])
				@inbounds delt2[i] = costFuncs[costFunc2](modelOut[i], dataOut[i])
			end
		else
			error("output layer does not match data")
		end
		err1 = sum(delt1)/m
		err2 = sum(delt2)/m
		(err1, err2)
	end
end

function calcOutputCPU(input_data, T, B; layerout = length(T), resLayers = 0, autoencoder = false, costFunc = "absErr", dropout = 0.0f0, activation_list = fill(true, length(T)-1))
#calculate network output given input data and a set of network parameters.
	#Setup some useful variables
	m = size(input_data, 1)

	membuffer = min(1E9, Sys.total_memory()*0.01)
	#leave 1 GB of memory left over except on apple systems where free memory is underreported
	newMem = if Sys.isapple()
		Int64(Sys.free_memory())
	else
		Int64(Sys.free_memory()) - membuffer
	end

	maxB = min(2^17, getMaxBatchSize(T, B, newMem))
	BLAS.set_num_threads(0)
	
	if maxB == 0
		println("Not enough memory for calculation, returning nothing")
		return nothing
	else
		out = if maxB > m
			predict(T, B, input_data, resLayers, layerout=layerout, activation_list=activation_list)
		else
			if maxB == 2^17
				println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
			else
				println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of memory"))
			end
			numBatches = ceil(Int64, m/maxB)
			batchInputs = [input_data[(i-1)*maxB+1:i*maxB, :] for i = 1:numBatches-1]
			out1 = predictBatches(T, B, batchInputs, resLayers, layerout=layerout)
			out2 = predict(T, B, input_data[(numBatches-1)*maxB+1:m, :], resLayers, layerout=layerout)
			[out1; out2]
		end
		if autoencoder
			errs = calcError(out, input_data, costFunc = costFunc)
			return (out, errs)
		else
			return out
		end 
	end
end

function calcOutputCPU(input_data, output_data, T, B; dropout = 0.0f0, costFunc = "absErr", resLayers = 0, activation_list = fill(true, length(T)-1))
#calculate network output given input data and a set of network parameters.
#calculation is performed on the GPU and then returned to system memory
	#Setup some useful variables
	m = size(input_data, 1)
	n = size(output_data, 2)

	membuffer = min(1E9, Sys.total_memory()*0.01)

	#leave 1 GB of memory left over except on apple systems where free memory is underreported
	newMem = if Sys.isapple()
		Int64(Sys.free_memory())
	else
		Int64(Sys.free_memory()) - membuffer
	end

	maxB = min(2^17, getMaxBatchSize(T, B, newMem))
	BLAS.set_num_threads(0)
	
	if maxB == 0
		println("Not enough memory for calculation, returning nothing")
		return nothing
	else
		out = if maxB > m
			predict(T, B, input_data, resLayers, activation_list=activation_list)
		else
			if maxB == 2^17
				println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
			else
				println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of memory"))
			end
			numBatches = ceil(Int64, m/maxB)
			batchInputs = [input_data[(i-1)*maxB+1:i*maxB, :] for i = 1:numBatches-1]
			out1 = predictBatches(T, B, batchInputs, resLayers, activation_list=activation_list)
			out2 = predict(T, B, input_data[(numBatches-1)*maxB+1:m, :], resLayers, activation_list=activation_list)
			[out1; out2]
		end

		errs = calcError(out, output_data, costFunc = costFunc)
		GC.gc()
		return (out, errs)
	end
end

function calcOutputCPU!(input_data, output_data, T, B, a; costFunc = "absErr", resLayers = 0, activation_list = fill(true, length(T)-1))
#calculate network output given input data and a set of network parameters.
#calculation is performed on the GPU and then returned to system memory
	#Setup some useful variables
	m = size(input_data, 1)
	n = size(output_data, 2)
			
	predict!(T, B, input_data, a, resLayers, activation_list=activation_list)
	errs = calcError(a[end], output_data, costFunc = costFunc)
	return errs
end

function calcMultiOutCPU(input_data, output_data, multiParams; dropout = 0.0f0, costFunc = "absErr", resLayers = 0, activation_list = fill(true, length(multiParams[1][1])-1))
#calculate network output given input data and a set of network parameters.
	#Setup some useful variables
	m = size(input_data, 1)
	n = size(output_data, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	membuffer = min(1E9, Sys.total_memory()*0.01)

	#if copying the input data will result in needing to break up the data into smaller batches to preserve system memory, then it isn't worth it
	#account for copying input data memory into other workers while leaving 1 GB left over 
	#except on apple systems where free memory is underreported
	function availMem(w)
		if Sys.isapple()
			Int64(Sys.free_memory()) - (w * sizeof(input_data))
		else
			Int64(Sys.free_memory()) - (w * sizeof(input_data)) - membuffer
		end
	end

	calcMaxB(w) = getMaxBatchSize(multiParams[1][1], multiParams[1][2], availMem(w))
	#create a vector of added workers that will still allow for large batch sizes
	validProcCount = if nprocs() < 3
		[]
	else
		findall(a -> a > m, calcMaxB.(2:nprocs()-1)) 
	end


	multiOut = if isempty(validProcCount) 
		if nprocs() > 2
			println(string("Performing single threaded prediction because the limited available memory would require breaking input data into batches when copied to workers"))
		else
			println("Performing multi prediction on a single thread")
		end
		BLAS.set_num_threads(0)
		newMem = if Sys.isapple()
			Int64(Sys.free_memory())
		else
			Int64(Sys.free_memory()) - membuffer
		end
		maxB = min(2^17, getMaxBatchSize(multiParams[1][1], multiParams[1][2], newMem))
		if maxB == 0
			println("Not enough memory for calculation, returning nothing")
			return nothing
		else
			if maxB > m	
				predictMulti(multiParams, input_data, resLayers, activation_list=activation_list)
			else
				if maxB == 2^17
					println(string("Breaking up ", m, " input examples into batches of the maximum size : ", maxB))
				else
					println(string("Breaking up ", m, " input examples into batches of size ", maxB, " to fit in ", newMem/(1024^3), " gigabytes of memory"))
				end
				numBatches = ceil(Int64, m/maxB)
				batchInputs = [input_data[(i-1)*maxB+1:i*maxB, :] for i = 1:numBatches-1]
				out1 = predictMultiBatches(multiParams, batchInputs, resLayers, activation_list=activation_list)
				out2 = predictMulti(multiParams, input_data[(numBatches-1)*maxB+1:m, :], resLayers, activation_list=activation_list)
				map((a, b) -> [a; b], out1, out2)
			end
		end
	else
		#value is the total number of parallel tasks that should be created to maximize batch size
		numTasks = min(nprocs()-1, validProcCount[end] + 1)
		# println(string("Running multi prediction using ", numTasks, " parallel tasks"))
		BLAS.set_num_threads(min(5, max(1, ceil(Int, Sys.CPU_THREADS/min(nprocs()-1, numTasks)))))
		if length(multiParams) > numTasks
			partitionInds = rem.(1:length(multiParams), numTasks) .+ 1
			multiParamsPartition = [multiParams[findall(i -> i == n, partitionInds)] for n in 1:numTasks]
			reduce(vcat, pmap(a -> predictMulti(a, input_data, resLayers, activation_list=activation_list), multiParamsPartition))
		else
			pmap(a -> predict(a[1], a[2], input_data, resLayers, activation_list=activation_list), multiParams) 
		end
	end


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
	GC.gc()
	return (multiOut, out, errs, outErrEst)
end

function calcMultiOutCPU!(input_data, output_data, multiParams, a, multiOut; dropout = 0.0f0, costFunc = "absErr", resLayers = 0, activation_list = fill(true, length(multiParams[1][1])-1))
	#calculate network output given input data and a set of network parameters.
	#Setup some useful variables
	m = size(input_data, 1)
	n = size(output_data, 2)

	costFunc2 = if occursin("sq", costFunc) | occursin("norm", costFunc)
		"sqErr"
	else
		"absErr"
	end

	predictMulti!(multiParams, input_data, a, multiOut, resLayers, activation_list=activation_list)

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
	GC.gc()
	return (out, errs, outErrEst)
end

function errshufflecols(T::Vector{Matrix{Float32}}, B::Vector{Vector{Float32}}, input_data::Matrix{Float32}, output_data::Matrix{Float32}, input_data_copy::Matrix{Float32}, a::Vector{Matrix{Float32}}, v, inds; rng=1, reslayers=0, costFunc = "sqErr", activation_list = fill(true, length(T)-1))

	for ind in inds
		if ind != 0
			#fill v with column to shuffle
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(rng), v)
			view(input_data_copy, :, ind) .= v
		end
	end

	errs = calcOutputCPU!(input_data_copy, output_data, T, B, a, costFunc=costFunc, resLayers = reslayers, activation_list=activation_list)
	
	#restore input_data_copy to initial state
	for ind in inds
		if ind != 0
			view(input_data_copy, :, ind) .= view(input_data, :, ind)
		end
	end

	return errs
end

function errshufflecols(multiparams, input_data::Matrix{Float32}, output_data::Matrix{Float32}, input_data_copy::Matrix{Float32}, a::Vector{Matrix{Float32}}, multiout::Vector{Matrix{Float32}}, v, inds; rng=1, reslayers=0, costFunc = "sqErr", activation_list = fill(true, length(multiparams[1][1])-1))

	for ind in inds
		if ind != 0
			#fill v with column to shuffle
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(rng), v)
			view(input_data_copy, :, ind) .= v
		end
	end

	(_, errs, _) = calcMultiOutCPU!(input_data_copy, output_data, multiparams, a, multiout, costFunc=costFunc, resLayers = reslayers, activation_list=activation_list)
	
	#restore input_data_copy to initial state
	for ind in inds
		if ind != 0
			view(input_data_copy, :, ind) .= view(input_data, :, ind)
		end
	end

	return errs
end

function calcfeatureimpact(T::Vector{Matrix{Float32}}, B::Vector{Vector{Float32}}, input_data::Matrix{Float32}, output_data::Matrix{Float32}; reslayers=0, costFunc="sqErr", num=10, fixedshuffle=[], activation_list = fill(true, length(T)-1))
	(m, n) = size(input_data)
	a = form_activations(T, m)
	v = Vector{Float32}(undef, m)
	input_copy = copy(input_data)

	NNerr = calcOutputCPU!(input_data, output_data, T, B, a, costFunc=costFunc, resLayers=reslayers, activation_list=activation_list)[1]
	shuffleind = setdiff(1:n, fixedshuffle)
	shuffle_errs = Vector{Float64}(undef, length(shuffleind))
	shuffle_cols = Vector{Vector{Int64}}(undef, length(shuffleind))
	if num == 1
		for ind in fixedshuffle
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(1), v)
			view(input_copy, :, ind) .= v
		end
	end
	println()
	println()
	tlast = time()
	for (i, c) in enumerate(shuffleind)
		if num == 1
			err = errshufflecols(T, B, input_data, output_data, input_copy, a, v, [c], rng=1, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list)[1]
		else
			err = maximum([errshufflecols(T, B, input_data, output_data, input_copy, a, v, [c; fixedshuffle], rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])[1]
		end
		shuffle_errs[i] = err
		shuffle_cols[i] = [c; fixedshuffle]
		if (i == 1) || (time()-tlast > 2)
			print("\33[2K\033[A\r")
			print("\33[2K\033[A\r")
			println("Finished evaluating shuffled column $c: number $i of $(length(shuffleind))")
			println("with an error change of $(100*(err/NNerr -1))%")
			tlast = time()
		end
	end

	if num == 1
		for ind in fixedshuffle
			view(input_copy, :, ind) .= view(input_data, :, ind)
		end
	end

	if !isempty(fixedshuffle)
		fixederr = maximum([errshufflecols(T, B, input_data, output_data, input_copy, a, v, fixedshuffle, rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list)[1] for j in 1:num])
	else
		fixederr = NNerr
	end
	featureimpact = 100 .*(shuffle_errs ./ NNerr .- 1)
	sortinds = reverse(sortperm(featureimpact))
	(NNerr, zip(shuffle_cols[sortinds], featureimpact[sortinds]), fixederr)
end

function calcfeatureimpact(T::Vector{Matrix{Float32}}, B::Vector{Vector{Float32}}, input_data::Matrix{Float32}, output_data::Matrix{Float32}, candidatecols::Vector{Int64}; reslayers=0, costFunc="sqErr", num=10, activation_list = fill(true, length(T)-1))
	(m, n) = size(input_data)
	a = form_activations(T, m)
	v = Vector{Float32}(undef, m)
	input_copy = copy(input_data)

	NNerr = calcOutputCPU!(input_data, output_data, T, B, a, costFunc=costFunc, resLayers=reslayers, activation_list=activation_list)
	# shuffleind = setdiff(1:n, fixedshuffle)
	shuffle_errs = Vector{Float64}(undef, length(candidatecols))
	shuffle_cols = Vector{Vector{Int64}}(undef, length(candidatecols))
	if num == 1
		for ind in candidatecols
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(1), v)
			view(input_copy, :, ind) .= v
		end
	end
	println()
	println()
	tlast = time()
	for (i, c) in enumerate(candidatecols)
		shufflecols = setdiff(candidatecols, c)
		if num == 1
			v .= view(input_copy, :, c)
			view(input_copy, :, c) .= view(input_data, :, c)
			err = calcOutputCPU!(input_copy, output_data, T, B, a, costFunc=costFunc, resLayers=reslayers, activation_list=activation_list)
			view(input_copy, :, c) .= v
		else
			err = maximum([errshufflecols(T, B, input_data, output_data, input_copy, a, v, shufflecols, rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])
		end
		shuffle_errs[i] = err
		shuffle_cols[i] = shufflecols
		if (i == 1) || (time()-tlast > 2)
			print("\33[2K\033[A\r")
			print("\33[2K\033[A\r")
			println("Finished evaluating unshuffled column $c: number $i of $(length(candidatecols))")
			println("with an error change of $(100*(err/NNerr -1))%")
			tlast = time()
		end
	end
	if !isempty(candidatecols)
		fixederr = maximum([errshufflecols(T, B, input_data, output_data, input_copy, a, v, candidatecols, rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])
	else
		fixederr = NNerr
	end
	featureimpact = 100 .*(shuffle_errs ./ NNerr .- 1)
	sortinds = reverse(sortperm(featureimpact))
	(NNerr, zip(shuffle_cols[sortinds], featureimpact[sortinds]), fixederr)
end


function calcfeatureimpact(multiparams::Vector{U}, input_data::Matrix{Float32}, output_data::Matrix{Float32}; reslayers=0, costFunc="sqErr", num=10, fixedshuffle=[], activation_list = fill(true, length(multiparams[1][1])-1)) where U <: Tuple
	(m, n) = size(input_data)
	a = form_activations(multiparams[1][1], m)
	v = Vector{Float32}(undef, m)
	input_copy = copy(input_data)

	multiout = [copy(a[end]) for i in eachindex(multiparams)]
	(_, NNerr, _) = calcMultiOutCPU!(input_data, output_data, multiparams, a, multiout, costFunc= costFunc, resLayers=reslayers, activation_list=activation_list)

	shuffleind = setdiff(1:n, fixedshuffle)
	shuffle_errs = Vector{Float64}(undef, length(shuffleind))
	shuffle_cols = Vector{Vector{Int64}}(undef, length(shuffleind))
	if num == 1
		for ind in fixedshuffle
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(1), v)
			view(input_copy, :, ind) .= v
		end
	end
	println()
	println()
	tlast = time()
	for (i, c) in enumerate(shuffleind)
		if num == 1
			err = errshufflecols(multiparams, input_data, output_data, input_copy, a, multiout, v, [c], rng=1, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list)
		else
			err = maximum([errshufflecols(multiparams, input_data, output_data, input_copy, a, multiout, v, [c; fixedshuffle], rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])
		end
		shuffle_errs[i] = err
		shuffle_cols[i] = [c; fixedshuffle]
		if (i == 1) || (time()-tlast > 2)
			print("\33[2K\033[A\r")
			print("\33[2K\033[A\r")
			println("Finished evaluating shuffled column $c: number $i of $(length(shuffleind))")
			println("with an error change of $(100*(err/NNerr -1))%")
			tlast = time()
		end
	end
	if num == 1
		for ind in fixedshuffle
			view(input_copy, :, ind) .= view(input_data, :, ind)
		end
	end
	if !isempty(fixedshuffle)
		fixederr = maximum([errshufflecols(multiparams, input_data, output_data, input_copy, a, multiout, v, fixedshuffle, rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])
	else
		fixederr=NNerr
	end
	featureimpact = 100 .*(shuffle_errs ./ NNerr .- 1)
	sortinds = reverse(sortperm(featureimpact))
	(NNerr, zip(shuffle_cols[sortinds], featureimpact[sortinds]), fixederr)
end

function calcfeatureimpact(multiparams::Vector{U}, input_data::Matrix{Float32}, output_data::Matrix{Float32}, candidatecols::Vector{Int64}; reslayers=0, costFunc="sqErr", num=10, activation_list = fill(true, length(multiparams[1][1])-1)) where U <: Tuple
	(m, n) = size(input_data)
	a = form_activations(multiparams[1][1], m)
	v = Vector{Float32}(undef, m)
	input_copy = copy(input_data)

	multiout = [copy(a[end]) for i in eachindex(multiparams)]
	(_, NNerr, _) = calcMultiOutCPU!(input_data, output_data, multiparams, a, multiout, costFunc= costFunc, resLayers=reslayers, activation_list=activation_list)

	# shuffleind = setdiff(1:n, fixedshuffle)
	shuffle_errs = Vector{Float64}(undef, length(candidatecols))
	shuffle_cols = Vector{Vector{Int64}}(undef, length(candidatecols))
	if num == 1
		for ind in candidatecols
			v .= view(input_data, :, ind)
			shuffle!(MersenneTwister(1), v)
			view(input_copy, :, ind) .= v
		end
	end
	println()
	println()
	tlast = time()
	for (i, c) in enumerate(candidatecols)
		shufflecols = setdiff(candidatecols, c)
		if num == 1
			v .= view(input_copy, :, c)
			view(input_copy, :, c) .= view(input_data, :, c)
			_, err, _ = calcMultiOutCPU!(input_copy, output_data, multiparams, a, multiout, costFunc=costFunc, resLayers=reslayers, activation_list=activation_list)
			view(input_copy, :, c) .= v
		else
			err = maximum([errshufflecols(multiparams, input_data, output_data, input_copy, a, multiout, v, shufflecols, rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])
		end
		shuffle_errs[i] = Float64(err)
		shuffle_cols[i] = shufflecols
		if (i == 1) || (time()-tlast > 2)
			print("\33[2K\033[A\r")
			print("\33[2K\033[A\r")
			println("Finished evaluating unshuffled column $c: number $i of $(length(candidatecols))")
			println("with an error change of $(100*(err/NNerr -1))%")
			tlast = time()
		end
	end
	if !isempty(candidatecols)
		fixederr = maximum([errshufflecols(multiparams, input_data, output_data, input_copy, a, multiout, v, candidatecols, rng=j, reslayers=reslayers, costFunc=costFunc, activation_list=activation_list) for j in 1:num])
	else
		fixederr = NNerr
	end

	featureimpact = 100 .*(shuffle_errs ./ NNerr .- 1)
	sortinds = reverse(sortperm(featureimpact))
	(NNerr, zip(shuffle_cols[sortinds], featureimpact[sortinds]), fixederr)
end


function checkNumGradCPU(lambda; hidden_layers=[5, 5], costFunc="absErr", resLayers = 0, m = 1000, input_layer_size = 3, n = 2, e = 1f-3, activation_list = fill(true, length(hidden_layers)))
	Random.seed!(1234)
	#if using log likelihood cost function then need to double output layer size
	#relative to output example size
	output_layer_size = if occursin("Log", costFunc)
		2*n
	else
		n
	end
	X = map(Float32, randn(m, input_layer_size))
	y = map(Float32, randn(m, n))

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
	
	a = Array{Matrix{Float32}}(undef, num_hidden+1)
	if num_hidden > 0
		for i = 1:num_hidden
			a[i] = Array{Float32}(undef, m, hidden_layers[i])
		end
	end
	a[end] = Array{Float32}(undef, m, output_layer_size)

	tanh_grad_z = deepcopy(a)
	deltas = deepcopy(a)

	# e = 1f-3
	params = theta2Params(B0, T0)
	l = length(params)
	perturb = zeros(Float32, l)
	numGrad = Array{Float32}(undef, l)

	nnCostFunction(T0, B0, input_layer_size, hidden_layers, X, y, lambda, Theta_grads, Bias_grads, tanh_grad_z, a, deltas, onesVec, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
	
	funcGrad = theta2Params(Bias_grads, Theta_grads)

	for i = 1:l
		perturb[i] = e
		Tplus, Bplus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params+perturb)
		Tminus, Bminus = params2Theta(input_layer_size, hidden_layers, output_layer_size, params-perturb)
		
	
		outminus = nnCostFunctionNOGRAD(Tminus, Bminus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc, resLayers = resLayers, activation_list=activation_list)
		outplus = nnCostFunctionNOGRAD(Tplus, Bplus, input_layer_size, hidden_layers, X, y, lambda, a, costFunc = costFunc, resLayers = resLayers, activation_list=activation_list)
		
		perturb[i] = 0.0f0  #restore perturb vector to 0

		numGrad[i] = (outplus - outminus)/(2.0f0*e)
	end

	
	println("Num Grads  Func Grads")
	for i = 1:length(numGrad)
		@printf "%0.6f  %0.6f \n" numGrad[i] funcGrad[i]
	end
	err = norm(numGrad .- funcGrad)/norm(numGrad .+ funcGrad)
	println(string("Relative differences for method are ", err, ".  Should be small (1e-9)"))
	return err
end


function updateM_old!(beta1, mT, mB, TG, BG)
	b2 = 1.0f0 - beta1
	for i = 1:length(TG)
		@simd for ii = 1:length(TG[i])
			@inbounds mT[i][ii] = beta1*mT[i][ii] + b2*TG[i][ii]
		end
		
		@simd for ii = 1:length(BG[i])
			@inbounds mB[i][ii] = beta1*mB[i][ii] + b2*BG[i][ii]
		end
	end
end

function updateM!(beta1, mT, mB, TG, BG)
	b2 = 1.0f0 - beta1
	for i in eachindex(TG)
		axpby!(b2, TG[i], beta1, mT[i])
		axpby!(b2, BG[i], beta1, mB[i])
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
	scale = alpha/(1.0f0 - beta1^t)
	for i = 1:length(T)
		@simd for ii = 1:length(T[i])
			@inbounds T[i][ii] = T[i][ii] - scale*mT[i][ii]/vT[i][ii]
		end
		@simd for ii = 1:length(B[i])
			@inbounds B[i][ii] = B[i][ii] - scale*mB[i][ii]/vB[i][ii]
		end
	end
end

function updateParams_old!(alpha, T, B, TG, BG)
	for i = 1:length(T)
		@simd for ii = 1:length(T[i])
			@inbounds T[i][ii] = T[i][ii] - alpha*TG[i][ii]
		end
		@simd for ii = 1:length(B[i])
			@inbounds B[i][ii] = B[i][ii] - alpha*BG[i][ii]
		end
	end
end

function updateParams!(alpha, T, B, TG, BG)
	for i in eachindex(T)
		axpy!(-alpha, TG[i], T[i])
		axpy!(-alpha, BG[i], B[i])
	end
end

function updateEst_old!(beta2, t, T, B, T_avg, B_avg, T_est, B_est)
	for i = 1:length(T)
		(N, M) = size(T[i])
		scale = 1.0f0/(1.0f0 - beta2^t)
		b2 = 1.0f0 - beta2

		@simd for ii = 1:length(T[i])
			@inbounds T_avg[i][ii] = beta2*T_avg[i][ii] + b2*T[i][ii]
			@inbounds T_est[i][ii] = scale*T_avg[i][ii]
		end
		@simd for ii = 1:length(B[i])
			@inbounds B_avg[i][ii] = beta2*B_avg[i][ii] + b2*B[i][ii]
			@inbounds B_est[i][ii] = scale*B_avg[i][ii]
		end
	end
end

function updateEst!(beta2, t, T, B, T_avg, B_avg, T_est, B_est)
	scale = 1.0f0/(1.0f0 - beta2^t)
	b2 = 1.0f0 - beta2
	for i = 1:length(T)
		axpby!(b2, T[i], beta2, T_avg[i])
		T_est[i] .= scale .* T_avg[i] 
		axpby!(b2, B[i], beta2, B_avg[i])
		B_est[i] .= scale .* B_avg[i] 
	end
end

function updateAvg!(nModels, T, B, T_avg, B_avg)
	F = Float32(1/(nModels+1))
	a = Float32(nModels*F) 
	for i = eachindex(T)
		axpby!(F, T[i], a, T_avg[i])
		axpby!(F, B[i], a, B_avg[i])
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

function generateBatches(data, batchsize)
	m = size(data, 1)
	# if batchsize > m
	# 	error("Your batchsize is larger than the total number of examples.")
	# end
	
	numBatches = round(Int, ceil(m/batchsize))
	batchData = Array{Matrix{Float32}}(undef, numBatches)
	
	randInd = repeat(shuffle(collect(1:m)), ceil(Int, batchsize/m)+1)
	
	for i = 1:numBatches
		batchData[i] = data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
	end
	return batchData
end

function generatebatchinds(data, batchsize)
	m = size(data, 1)
	# if batchsize > m
	# 	error("Your batchsize is larger than the total number of examples.")
	# end
	
	numBatches = round(Int, ceil(m/batchsize))
	batchinds = Vector{AbstractVector{Int64}}(undef, numBatches)
	
	randInd = repeat(shuffle(collect(1:m)), ceil(Int, batchsize/m)+1)
	
	for i = 1:numBatches
		batchinds[i] = randInd[(i-1)*batchsize + 1:i*batchsize]
	end
	return batchinds
end

function generatebatches(data, batchinds)
	numbatches = length(batchinds)
	batchdata = Vector{Matrix{Float32}}(undef, numbatches)
	
	for (i, ind) in enumerate(batchinds)
		batchdata[i] = data[ind, :]
	end
	return batchdata
end

function generateBatches(input_data, output_data, batchsize)
	m = size(output_data, 1)
	# if batchsize > m
	# 	error("Your batchsize is larger than the total number of examples.")
	# end
	
	numBatches = round(Int, ceil(m/batchsize))
	inputbatchData = Array{Matrix{Float32}}(undef, numBatches)
	outputbatchData = Array{Matrix{Float32}}(undef, numBatches)
	
	randInd = repeat(shuffle(collect(1:m)), ceil(Int, batchsize/m)+1)
	
	for i = 1:numBatches
		inputbatchData[i] = input_data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
		outputbatchData[i] = output_data[randInd[(i-1)*batchsize + 1:i*batchsize], :]
	end
	return (inputbatchData, outputbatchData)
end

function ADAMAXTrainNNCPU(data, batchSize, T0, B0, N, input_layer_size, hidden_layers, lambda, c; alpha=0.002f0, R = 0.1f0, printProgress = false, printAnything=true, dropout = 0.0f0, costFunc = "absErr", resLayers = 0, tol=Inf, patience=3, swa=false, ignorebest=false, minepoch=0, prepdata = (), prepactivations=(), trainsample=1.0, activation_list = fill(true, length(hidden_layers)))
#train fully connected neural network with floating point vector output.  Requires the following inputs: training data, training output, batchsize
#initial Thetas, initial Biases, max epochs to train, input_layer_size, vector of hidden layer sizes, l2 regularization parameter lambda, max norm parameter c, and
#a training rate alpha.  An optional dropout factor is set to 0 by default but can be set to a 32 bit float between 0 and 1.
#Note that all floating point input variables must be float32 or single precision   
	
	#if the input data only contains one matrix then consider input and output
	#data to be the same and train an autoencoder
	autoencoder = (length(data[1]) == 1)

	input_data = data[1][1]

	output_data = autoencoder ? input_data : data[1][2]
	
	testset = (length(data) > 1)

	if testset
		input_test = data[2][1]
		output_test = autoencoder ? input_test : data[2][2]
		(mtest, ntest) = size(input_test)
	end

	@assert ((dropout >= 0.0f0) & (dropout < 1.0f0)) string("Dropout rate of ", dropout, " is not between 0 and 1")
	(m, n) = size(input_data)
	(m2, output_layer_size) = size(output_data)
	
	n2 = if occursin("Log", costFunc)
		2*output_layer_size
	else
		output_layer_size
	end

	if m2 != m 
		error("input and output data do not match")
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

	#total number of examples in dataset
	if batchSize > m
		error("Your batchsize is larger than the total number of examples.")
	end

	if printAnything
		println()
		printstyled(IOContext(stdout, :color => true), "Beginning training with the following parameters:", bold=true, color=:green)
		println()
		println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", num epochs = ", N, ", training alpha = ", alpha, ", decay rate = ", R, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", dropout rate = ", dropout, ", residual layer size = ", resLayers))
		println("-------------------------------------------------------------------")
	end

	numBatches = round(Int, ceil(m/batchSize))
	(fops, bops, pops) = calcOps(n, hidden_layers, n2, batchSize)
    total_ops = fops + bops + pops

    if trainsample == 1.0
    	batchset = 1:numBatches
    else
    	batchset = rand(1:numBatches, round(Int64, numBatches*trainsample))
    end


    if isempty(prepdata)
    	batchinds = generatebatchinds(input_data, batchSize)
    	inputbatchData = generatebatches(input_data, batchinds)
		if !autoencoder
			outputbatchData = generatebatches(output_data, batchinds)
		end
	else
		inputbatchData = prepdata[1]
		if !autoencoder
			outputbatchData = prepdata[2]
		end
	end

	#create memory objects used in cost function
	num_hidden = length(hidden_layers)

	if isempty(prepactivations)
		tanh_grad_zBATCH = form_tanh_grads(hidden_layers, batchSize)
		aBATCH = form_activations(T0, batchSize)
		deltasBATCH = form_activations(T0, batchSize)
	else
		tanh_grad_zBATCH = prepactivations[1]
		aBATCH = prepactivations[2]
		deltasBATCH = prepactivations[3]
	end

	Theta_grads = deepcopy(T0) 
	Bias_grads = deepcopy(B0)
	onesVecBATCH = ones(Float32, batchSize)
	numLayers = length(T0)

	if testset
		aTEST = form_activations(T0, mtest)
	end

	if autoencoder
		nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[batchset[end]], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
	else
		nnCostFunction(T0, B0, input_layer_size, hidden_layers, inputbatchData[batchset[end]], outputbatchData[batchset[end]], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
	end
	
	function calcout_batches(T, B)
		currentOut = 0.0f0 
		for i = batchset
			if autoencoder
				currentOut += nnCostFunctionNOGRAD(T, B, input_layer_size, hidden_layers, inputbatchData[i], 0.0f0, aBATCH, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
			else
				currentOut += nnCostFunctionNOGRAD(T, B, input_layer_size, hidden_layers, inputbatchData[i], outputbatchData[i], 0.0f0, aBATCH, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
			end
		end
		currentOut = currentOut/length(batchset)
	end

	currentOut = calcout_batches(T0, B0)
	
	if testset
		function calcout_test(T, B)
			if autoencoder
				nnCostFunctionNOGRAD(T, B, input_layer_size, hidden_layers, input_test, 0.0f0, aTEST, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
			else
				nnCostFunctionNOGRAD(T, B, input_layer_size, hidden_layers, input_test, output_test, 0.0f0, aTEST, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
			end
		end
	end

	testset && (testout = calcout_test(T0, B0))

	if printAnything
		printstyled(stdout, string("Initial cost is ", currentOut), bold=true, color=:red)
		println()
		#println(string("Initial cost is ", currentOut))
	end


	#step rate and decay term for rms prop
	beta1 = 0.9f0
	beta2 = 0.999f0

	mT = 0.0f0*deepcopy(T0)
	mB = 0.0f0*deepcopy(B0)

	vT = 0.0f0*deepcopy(T0)
	vB = 0.0f0*deepcopy(B0)

	T_avg = 0.0f0*deepcopy(T0)
	B_avg = 0.0f0*deepcopy(B0)

	T_est = 0.0f0*deepcopy(T0)
	B_est = 0.0f0*deepcopy(B0)

	Thetas = deepcopy(T0)
	Biases = deepcopy(B0)

	period = 10
	costRecord = Array{Float32}(undef, ceil(Int, N/period)+1)
	costRecord[1] = currentOut
	if testset
		costRecordTest = Array{Float32}(undef, ceil(Int, N/period)+1)
		costRecordTest[1] = testout
	end

	startTime = time()
	lastReport = startTime

	timeRecord = Array{Float64}(undef, N+1)
	timeRecord[1] = 0.0

	bestThetas = deepcopy(T0)
	bestBiases = deepcopy(B0)
	bestCost = currentOut
	testset && (bestCostTest = testout)
	rollingAvgCost = currentOut

	iter = 1
	epoch = 1
	eta = alpha
	F = (1.0f0 - R)
	G = alpha*F
	tfail = 0
	tolpass = true
	bestresultepoch = 0
	nModels = 0

	t = 1
	while (epoch <= minepoch) || ((epoch <= N) && (tfail <= patience) && tolpass)
	#while epoch <= N
		#run through an epoch in batches with randomized order
		for batch in shuffle(batchset)
			if eta > 0
				if autoencoder
					nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
				else
					nnCostFunction(Thetas, Biases, input_layer_size, hidden_layers, inputbatchData[batch], outputbatchData[batch], lambda, Theta_grads, Bias_grads, tanh_grad_zBATCH, aBATCH, deltasBATCH, onesVecBATCH, dropout, costFunc=costFunc, resLayers = resLayers, activation_list=activation_list)
				end
				if swa && (epoch > 100)
					updateParams!(G, Thetas, Biases, Theta_grads, Bias_grads)
				else
					updateM!(beta1, mT, mB, Theta_grads, Bias_grads)
					updateV!(beta2, vT, vB, Theta_grads, Bias_grads)		
					updateParams!(eta, beta1, Thetas, Biases, mT, mB, vT, vB, t)
				end
				if c < Inf 
					scaleParams!(Thetas[1:end-1], Biases[1:end-1], c)
				end
			end
			#use recent time average of parameter changes for estimate
			if !swa || (epoch <= 100)
				updateEst!(beta2, t, Thetas, Biases, T_avg, B_avg, T_est, B_est)
			end
			t += 1
		end
		timeRecord[epoch + 1] = time() - startTime

		if swa && (epoch == 100)
			#after 100 epochs reset params to the estimate and start doing SWA
			updateBest!(Thetas, Biases, T_est, B_est)
		end

		if swa && (epoch > 100)
			nModels += 1
			updateAvg!(nModels, Thetas, Biases, T_est, B_est)
		end


		if epoch%period == 0
			currentOut = calcout_batches(T_est, B_est)
			costRecord[iter + 1] = currentOut

			if testset
				testout = calcout_test(T_est, B_est)
				costRecordTest[iter + 1] = testout
			end
			
			if (epoch <= minepoch) || ignorebest || (testset && (testout < bestCostTest)) || (!testset && (currentOut < bestCost))
				updateBest!(bestThetas, bestBiases, T_est, B_est)
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
			remainingEpochs = N - epoch

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
				println(string("On epoch ", epoch, " out of ", N, " best train and test cost is ", (round(bestCost, sigdigits=5), round(bestCostTest, sigdigits=5))))
			else
				println(string("On epoch ", epoch, " out of ", N, " best cost is ", round(bestCost, digits=8)))
			end
			println(string("Estimated remaining time = ", hoursLeft, " hours, ", minutesLeft, " minutes, ", secondsLeft, " seconds."))
		end
		epoch += 1
	end
	lastepoch = epoch - 1
	currentOut = calcout_batches(T_est, B_est)
	testset && (testout = calcout_test(T_est, B_est))



	if ignorebest || (testset && (testout < bestCostTest)) || (!testset && (currentOut < bestCost))
		bestCost = currentOut
		testset && (bestCostTest = testout)
		bestresultepoch = lastepoch
		updateBest!(bestThetas, bestBiases, T_est, B_est)
	end
	
	time_per_epoch = timeRecord[2:lastepoch+1] .- timeRecord[1:lastepoch]
    train_time = timeRecord[lastepoch]
    timePerBatch = train_time/lastepoch/numBatches
    GFLOPS_per_epoch = total_ops * numBatches ./ time_per_epoch / 1e9

    if printAnything
		println("-------------------------------------------------------------------")
		printstyled(IOContext(stdout, :color => true), "Completed training on CPU with the following parameters: ", bold = true, color=:green)

		println()
		println(string("input size = ", n, ", hidden layers = ", hidden_layers, ", output size = ", n2, ", batch size = ", batchSize, ", num epochs = ", N, ", training alpha = ", alpha, ", decay rate = ", R, ", L2 Reg Constant = ", lambda, ", max norm reg constant = ", c, ", dropout rate = ", dropout, ", residual layer size = ", resLayers))
	
		printstyled(IOContext(stdout, :color => true), string("Training Results: Cost reduced from ", testset ? costRecordTest[1] : costRecord[1], "to ", testset ? bestCostTest : bestCost, " after ", round(Int64, timeRecord[lastepoch+1]), " seconds and ", lastepoch, " epochs"), bold=true, color=:red)
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
