using CUDAdrv

devlist = collect(devices())

#assign each worker a device ensuring it is a value in devlist
#allows parallel workers to operate on multiple devices if they exist
dev = myid()%length(devlist)

ctx = CuContext(CuDevice(dev))

using CUBLAS

filepath = joinpath(@__DIR__, "NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.cu")

cost_md = if isfile("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
	CuModuleFile("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
else
	run(`nvcc -ptx $filepath`)
	CuModuleFile("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
end

##TO DO: redo threads, blockers, kernels, and launches to work in 1D
#set up NN kernels
fill_cols = CuFunction(cost_md, "fill_cols")
finish_delt = CuFunction(cost_md, "finish_delta")
elMul = CuFunction(cost_md, "elMul")
cudaTanhGrad = CuFunction(cost_md, "tanhGradient")
cudaTanhGradDropout = CuFunction(cost_md, "tanhGradientDropout")
cudaTanh = CuFunction(cost_md, "tanhActivation")

kernels = (fill_cols, finish_delt, elMul, cudaTanhGrad, cudaTanhGradDropout)
kernelsNOGRAD = (fill_cols, finish_delt, elMul, cudaTanh)

#----------regular kernels----------
#kernel 1 = fill_cols
#kernel 2 = finish_delt
#kernel 3 = elMul
#kernel 4 = cudaTanhGrad
#kernel 5 = cudaTanhGradDropout

#---------no grad kernels-----------
#kernel 1 = fill_cols
#kernel 2 = finish_delt
#kernel 3 = elMul
#kernel 4 = cudaTanh

#2D thread block size for GPU
K = 32
threads = CuDim((K, K))

function getTypes(x)
    if isbits(x)
        typeof(x)
    else
        Ptr{eltype(x)}
    end
end

function run_kernel(kernel::CuFunction, N::Int64, M::Int64, inputs...)
	blocks = CuDim((ceil(Int, N/K), ceil(Int, M/K)))
    cudacall(kernel, blocks, threads, (Int64, Int64, getTypes.(inputs)...), N, M, inputs...)
end



function predict(d_Thetas, d_biases, d_X, m, input_layer_size, output_layer_size, hidden_layers, D = 0.0f0)
#PREDICT Predict the value of an input given a trained neural network
#m = number of examples in X, input_layer_size = number of input values, output_layer_size = number of output values
#hidden_layers = hidden layer vector
	l = length(d_Thetas)
	num_hidden = l - 1

	#dropout scale factor
	F = 1.0f0 - D

	d_a = Array{CuArray{Float32, 2}}(l)
	if num_hidden > 0
		for i = 1:l-1
			d_a[i] = CuArray{Float32}(m, hidden_layers[i])
		end
	end
	d_a[l] = CuArray{Float32}(m, output_layer_size)

	if num_hidden > 0
		for i = 1:num_hidden
			run_kernel(kernelsNOGRAD[1], m, hidden_layers[i], d_a[i], d_biases[i])
			#CUDArt.launch(kernelsNOGRAD[1], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i], d_biases[i]))
		end
	end

	#println(string("biases 3", round(float64(to_host(d_biases[3])), 6)))
	run_kernel(kernelsNOGRAD[1], m, output_layer_size, d_a[end], d_biases[end])
	#CUDArt.launch(kernelsNOGRAD[1], blocks(m, output_layer_size), threads, (m, output_layer_size, d_a[end], d_biases[end]))

	CUBLAS.gemm!('N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])

	if num_hidden > 0
		run_kernel(kernelsNOGRAD[4], m, hidden_layers[1], d_a[1])
		#CUDArt.launch(kernelsNOGRAD[4], blocks(m, hidden_layers[1]), threads, (m, hidden_layers[1], d_a[1])) 


		if num_hidden > 1
			for i = 2:num_hidden
				CUBLAS.gemm!('N', 'T', F, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				run_kernel(kernelsNOGRAD[4], m, hidden_layers[i], d_a[i])
				#CUDArt.launch(kernelsNOGRAD[4], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i])) 
			end
		end

		CUBLAS.gemm!('N', 'T', F, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
	end
	synchronize()
	return d_a[end]
end

function nnCostFunction(d_Thetas::Array{CuArray{Float32, 2}, 1}, d_biases::Array{CuArray{Float32, 1}, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Vector, m::Int64, d_ones::CuArray{Float32, 1}, d_a::Array{CuArray{Float32, 2}, 1}, d_tanh_grad_z::Array{CuArray{Float32, 2}, 1}, d_deltas::Array{CuArray{Float32, 2}, 1}, d_Theta_grads::Array{CuArray{Float32, 2}, 1}, d_bias_grads::Array{CuArray{Float32, 1}, 1}, d_X::CuArray{Float32, 2}, d_y::CuArray{Float32, 2},lambda::Float32, D = 0.0f0)

	num_hidden = length(hidden_layers)

	if num_hidden > 0
		if lambda > 0.0f0
			CUBLAS.blascopy!(input_layer_size*hidden_layers[1], d_Thetas[1], 1, d_Theta_grads[1], 1)
			if num_hidden > 1
				for i = 2:num_hidden
					CUBLAS.blascopy!(hidden_layers[i-1]*hidden_layers[i], d_Thetas[i],1, d_Theta_grads[i],1)
				end
			end
			CUBLAS.blascopy!(hidden_layers[num_hidden], d_Thetas[end],1, d_Theta_grads[end],1)
		end


		for i = 1:num_hidden
			run_kernel(kernels[1], m, hidden_layers[i], d_a[i], d_biases[i])
			#CUDArt.launch(kernels[1], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i], d_biases[i]))
		end
	end

	#println(string("biases 3", round(float64(to_host(d_biases[3])), 6)))

	run_kernel(kernels[1], m, output_layer_size, d_a[end], d_biases[end])
	#CUDArt.launch(kernels[1], blocks(m, output_layer_size), threads, (m, output_layer_size, d_a[end], d_biases[end]))


	CUBLAS.gemm!('N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])

	if num_hidden > 0

		if D == 0.0f0
			run_kernel(kernels[4], m, hidden_layers[1], d_a[1], d_tanh_grad_z[1])
			#CUDArt.launch(kernels[4], blocks(m, hidden_layers[1]), threads, (m, hidden_layers[1], d_a[1], d_tanh_grad_z[1])) 
		else
			run_kernel(kernels[5], m, hidden_layers[1], d_a[1], d_tanh_grad_z[1], rand(UInt32), D)
			#CUDArt.launch(kernels[5], blocks(m, hidden_layers[1]), threads, (m, hidden_layers[1], d_a[1], d_tanh_grad_z[1], rand(UInt32), D)) 
		end


		if num_hidden > 1
			for i = 2:num_hidden
				CUBLAS.gemm!('N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				if D == 0.0f0
					run_kernel(kernels[4], m, hidden_layers[i], d_a[i], d_tanh_grad_z[i])
					#CUDArt.launch(kernels[4], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i], d_tanh_grad_z[i])) 
				else
					run_kernel(kernels[5], m, hidden_layers[i], d_a[i], d_tanh_grad_z[i], rand(UInt32), D)
				end
			end
		end

		CUBLAS.gemm!('N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
	end

	run_kernel(kernels[2], m, output_layer_size, d_a[end], d_y, d_deltas[end])
	#CUDArt.launch(kernels[2], blocks(m, 1), threads, (m, output_layer_size, d_a[end], d_y, d_deltas[end]))


	i = num_hidden
	while i >= 1
		#CUDArt.launch(kernels[3], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_deltas[i], CUBLAS.gemm('N', 'N', d_deltas[i+1], d_Thetas[i+1]), d_tanh_grad_z[i]))
		gemm!('N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 0.0f0, d_deltas[i]) #do part 1 of line 1 in place
		run_kernel(kernels[3], m, hidden_layers[i], d_deltas[i], d_tanh_grad_z[i])
		#CUDArt.launch(kernels[3], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_deltas[i], d_tanh_grad_z[i])) #do part 2 of line 1 in place
		i = i - 1
	end

	CUBLAS.gemm!('T', 'N', 1.0f0/m, d_deltas[1], d_X, lambda/m, d_Theta_grads[1])


	CUBLAS.gemv!('T', 1.0f0/m, d_deltas[1], d_ones, 0.0f0, d_bias_grads[1])
	#d_bias_grads[1] = CUBLAS.gemv('T', 1.0f0/m, d_deltas[1], d_ones)

	if num_hidden > 0
		for i = 2:num_hidden+1
			CUBLAS.gemm!('T', 'N', 1.0f0/m, d_deltas[i], d_a[i-1], lambda/m, d_Theta_grads[i])
			#d_bias_grads[i] = CUBLAS.gemv('T', 1.0f0/m, d_deltas[i], d_ones)
			CUBLAS.gemv!('T', 1.0f0/m, d_deltas[i], d_ones, 0.0f0, d_bias_grads[i])
		end
	end
	synchronize()
end


function nnCostFunctionNOGRAD(d_Thetas::Array{CuArray{Float32, 2}, 1}, d_biases::Array{CuArray{Float32, 1}, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Vector, m::Int64, d_a::Array{CuArray{Float32, 2}, 1}, d_X::CuArray{Float32, 2}, d_y::CuArray{Float32, 2},lambda::Float32, D = 0.0f0)

	#dropout scale factor
	F = (1.0f0 - D)

	num_hidden = length(hidden_layers)

	if num_hidden > 0
		for i = 1:num_hidden
			run_kernel(kernelsNOGRAD[1], m, hidden_layers[i], d_a[i], d_biases[i])
			#CUDArt.launch(kernelsNOGRAD[1], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i], d_biases[i]))
		end
	end

	#println(string("biases 3", round(float64(to_host(d_biases[3])), 6)))

	run_kernel(kernelsNOGRAD[1], m, output_layer_size, d_a[end], d_biases[end])
	#CUDArt.launch(kernelsNOGRAD[1], blocks(m, output_layer_size), threads, (m, output_layer_size, d_a[end], d_biases[end]))


	CUBLAS.gemm!('N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])

	if num_hidden > 0

		run_kernel(kernelsNOGRAD[4], m, hidden_layers[1], d_a[1])
		#CUDArt.launch(kernelsNOGRAD[4], blocks(m, hidden_layers[1]), threads, (m, hidden_layers[1], d_a[1])) 


		if num_hidden > 1
			for i = 2:num_hidden
				CUBLAS.gemm!('N', 'T', F, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				run_kernel(kernelsNOGRAD[4], m, hidden_layers[i], d_a[i])
				#CUDArt.launch(kernelsNOGRAD[4], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i])) 
			end
		end

		CUBLAS.gemm!('N', 'T', F, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
	end

	CUBLAS.axpy!(output_layer_size*m, -1.0f0, d_y, 1, d_a[end], 1)

	CUBLAS.asum(d_a[end])/m
end

# function nnCostFunctionAdv(d_Thetas::Array{CuArray{Float32, 2}, 1}, d_biases::Array{CuArray{Float32, 1}, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Array{Int64, 1}, m::Int64, d_ones::CuArray{Float32, 1}, d_a::Array{CuArray{Float32, 2}, 1}, d_tanh_grad_z::Array{CuArray{Float32, 2}, 1}, d_deltas::Array{CuArray{Float32, 2}, 1}, d_Theta_grads::Array{CuArray{Float32, 2}, 1}, d_bias_grads::Array{CuArray{Float32, 1}, 1}, d_advX::CuArray{Float32, 2}, d_X::CuArray{Float32, 2}, d_y::CuArray{Float32, 2},lambda::Float32, blocks::Function, threads::Tuple{Int64, Int64}, kernels)

# num_hidden = length(hidden_layers)


# if lambda > 0.0f0
# CUBLAS.blascopy!(input_layer_size*hidden_layers[1], d_Thetas[1], 1, d_Theta_grads[1], 1)
# if num_hidden > 1
# 	for i = 2:num_hidden
# 		CUBLAS.blascopy!(hidden_layers[i-1]*hidden_layers[i], d_Thetas[i],1, d_Theta_grads[i],1)
# 	end
# end
# CUBLAS.blascopy!(hidden_layers[num_hidden], d_Thetas[end],1, d_Theta_grads[end],1)
# end


# for i = 1:num_hidden
# 	CUDArt.launch(kernels[1], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i], d_biases[i]))
# end

# #println(string("biases 3", round(float64(to_host(d_biases[3])), 6)))

# CUDArt.launch(kernels[1], blocks(m, output_layer_size), threads, (m, output_layer_size, d_a[end], d_biases[end]))


# CUBLAS.gemm!('N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])


# CUDArt.launch(kernels[4], blocks(m, hidden_layers[1]), threads, (m, hidden_layers[1], d_a[1], d_tanh_grad_z[1])) 


# if num_hidden > 1
# 	for i = 2:num_hidden
# 		CUBLAS.gemm!('N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
		
# 		CUDArt.launch(kernels[4], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_a[i], d_tanh_grad_z[i])) 
# 	end
# end

# CUBLAS.gemm!('N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])

# CUDArt.launch(kernels[2], blocks(m, 1), threads, (m, output_layer_size, d_a[end], d_y, d_deltas[end]))


# i = num_hidden
# while i >= 1
# 	#CUDArt.launch(kernels[3], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_deltas[i], CUBLAS.gemm('N', 'N', d_deltas[i+1], d_Thetas[i+1]), d_tanh_grad_z[i]))
# 	gemm!('N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 0.0f0, d_deltas[i]) #do part 1 of line 1 in place
# 	CUDArt.launch(kernels[3], blocks(m, hidden_layers[i]), threads, (m, hidden_layers[i], d_deltas[i], d_tanh_grad_z[i])) #do part 2 of line 1 in place
# 	i = i - 1
# end

# CUBLAS.gemm!('N', 'N', 1.0f0/m, d_deltas[1], d_Thetas[1], 0.0f0, d_advX)
# CUDArt.launch(kernels[5], blocks(m, input_layer_size), threads, (m, input_layer_size, d_X, d_advX))

# CUBLAS.gemm!('T', 'N', 1.0f0/m, d_deltas[1], d_X, lambda/m, d_Theta_grads[1])


# CUBLAS.gemv!('T', 1.0f0/m, d_deltas[1], d_ones, 0.0f0, d_bias_grads[1])
# #d_bias_grads[1] = CUBLAS.gemv('T', 1.0f0/m, d_deltas[1], d_ones)


# for i = 2:num_hidden+1
# 	CUBLAS.gemm!('T', 'N', 1.0f0/m, d_deltas[i], d_a[i-1], lambda/m, d_Theta_grads[i])
# 	#d_bias_grads[i] = CUBLAS.gemv('T', 1.0f0/m, d_deltas[i], d_ones)
# 	CUBLAS.gemv!('T', 1.0f0/m, d_deltas[i], d_ones, 0.0f0, d_bias_grads[i])
# end

# end
