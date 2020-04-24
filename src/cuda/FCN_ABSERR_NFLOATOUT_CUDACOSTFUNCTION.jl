using NVIDIALibraries, NVIDIALibraries.DeviceArray

@using_nvidialib_settings 

costfunc_kernel_names = ("fill_cols", "swap_matrix_col", "finish_delta", "elMul", "tanhGradient", "tanhGradientDropout", "noactivationGradient", "tanhActivation")

#for cuda version 8 tensor ops are not available so default to regular GEMM algorithm
algo = try
	CUBLAS_GEMM_DEFAULT_TENSOR_OP
catch
	CUBLAS_GEMM_DFALT
end

function cu_module_load()
	#------use nvcc to compile .ptx files from .cu kernels and load module------------
    filepath = joinpath(@__DIR__, "NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.cu")
    cost_md = if isfile("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
        cuModuleLoad("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
    else
        run(`nvcc -ptx $filepath`)
        cuModuleLoad("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")
    end

    filepath = joinpath(@__DIR__, "ADAMAX_INTRINSIC_KERNELS.cu")
    adamax_md = if isfile("ADAMAX_INTRINSIC_KERNELS.ptx")
        cuModuleLoad("ADAMAX_INTRINSIC_KERNELS.ptx")
    else
        run(`nvcc -ptx $filepath`)
        cuModuleLoad("ADAMAX_INTRINSIC_KERNELS.ptx")
    end
    #------------------------------------------------------------------------
    (adamax_md, cost_md)
end

function create_kernels(md, knames)
#create module kernels in global scope
	for kname in knames
         @eval global $(Symbol(kname)) = cuModuleGetFunction($md, $kname)
    end
end

function create_errorfunction_dicts(cost_md)
	err_kernel_list = map(kname -> cuModuleGetFunction(cost_md, kname), costFuncNames)
    err_deriv_kernel_list = map(kname -> cuModuleGetFunction(cost_md, string(kname, "Deriv")), costFuncNames)

    #make error kernels available in global scope
    global costFuncKs = Dict(zip(costFuncNames, err_kernel_list))
    global costFuncDerivKs = Dict(zip(costFuncNames, err_deriv_kernel_list))
end

function switch_device(d::Int64)
	if current_device == devlist[d]
		println("Already using device $(devlist[d])")
	else
		println("Switching from $current_device to $(devlist[d])")
		#destroy existing cublas_handle
		cublasDestroy_v2(cublas_handle)

		#set cuda device for kernel launches and cublas handles to a new device d
	    cudaSetDevice(devlist[d])

	    #create cublas handle to reference for calls on the new device
	    global cublas_handle = cublasCreate_v2()
	    global current_device = devlist[d]

	    #------load ptx modules in new context------------
        (adamax_md, cost_md) = cu_module_load()
        # cost_md = cuModuleLoad("NFLOATOUT_COSTFUNCTION_INTRINSIC_KERNELS.ptx")        
        # adamax_md = cuModuleLoad("ADAMAX_INTRINSIC_KERNELS.ptx")
       
        #----------use cuda driver api to create cuFunction pointers-------------
        #create adamax train and cost function kernels in global scope
   		create_kernels(adamax_md, adamax_kernel_names)
   		create_kernels(cost_md, costfunc_kernel_names)
        
        # #create error function and derivatives kernel lists
        # err_kernel_list = map(kname -> cuModuleGetFunction(cost_md, kname), costFuncNames)
        # err_deriv_kernel_list = map(kname -> cuModuleGetFunction(cost_md, string(kname, "Deriv")), costFuncNames)

        # #make error kernels available in global scope
        # global costFuncKs = Dict(zip(costFuncNames, err_kernel_list))
        # global costFuncDerivKs = Dict(zip(costFuncNames, err_deriv_kernel_list))

        #make error kernels available in global scope
        create_errorfunction_dicts(cost_md) 
	end
	println("Current device set to $(devlist[d])")
	return current_device
end

function getTypes(x)
    if isbits(x)
        typeof(x)
    elseif typeof(x) <: NVIDIALibraries.DeviceArray.CUDAArray
       Ptr{x.element_type}
    end
end

function run_kernel(kernel::CUfunction, N::Int64, M::Int64, inputs...; stream = CUstream(C_NULL))
	K = 16
	threads = Cuint.((K, K))
	blocks = Cuint.((ceil(Int, N/K), ceil(Int, M/K)))
    cuLaunchKernel(kernel, dim3(blocks...), dim3(threads...), (Cint, Cint, getTypes.(inputs)...), Cint(N), Cint(M), inputs..., stream = stream)
    # cuCtxSynchronize()
end

function run_kernel_1D(kernel::CUfunction, N::Int64, inputs...; stream = CUstream(C_NULL))
	K = 256
	threads = Cuint.((K,))
	blocks = Cuint.((ceil(Int, N/K),))
    cuLaunchKernel(kernel, dim3(blocks...), dim3(threads...), (Cint, getTypes.(inputs)...), Cint(N), inputs..., stream = stream)
    # cuCtxSynchronize()
end

function kernelTests()
	println("Testing cudaTanh kernel launch")
	N = 10
	M = 10
	run_kernel(cudaTanh, N, M, cuda_allocate(rand(Float32, 10, 10)))
	println("Test success")

	println("Testing elSq kernel launch")
	N = 10
	M = 10
	run_kernel(elSq, N, M, cuda_allocate(rand(Float32, 10, 10)))
	println("Test success")
end

function device_allocate(host_array::Vector{Array{Float32, N}}, scale = 1.0f0) where N
	l = length(host_array)
	device_array = Vector{CUDAArray}(undef, l)
	for (i, a) in enumerate(host_array)
		device_array[i] = cuda_allocate(scale .* host_array[i])
	end
	return device_array
end

function device_copy(orig::Vector{CUDAArray})
	l = length(orig)
	new = Vector{CUDAArray}(undef, l)
	for i in 1:l
		new[i] = cuda_allocate(host_allocate(orig[i]))
	end
end

device_copy(orig::CUDAArray) = cuda_allocate(host_allocate(orig))

function host_allocate(device_array::CUDAArray)
	T = device_array.element_type
	s = device_array.size
	host_array = Array{T, length(s)}(undef, s...)
	memcpy!(host_array, device_array)
	return host_array
end

function host_allocate(device_array::Vector{CUDAArray})
	l = length(device_array)
	host_array = Vector{Array{device_array[1].element_type, length(device_array[1].size)}}(undef, l)
	for (i, d_a) in enumerate(device_array)
		host_array[i] = host_allocate(device_array[i])
	end
	return host_array
end

function clear_gpu_data(device_array::Vector{CUDAArray})
	l = length(device_array)
	for d_a in device_array
		deallocate!(d_a)
	end
end


function cublasSaxpy(handle::cublasHandle_t, alpha::Float32, x::CUDAArray, y::CUDAArray)::Nothing
    @assert ((x.element_type == Float32) &&
            (y.element_type == Float32))
   	
   	dims = length(x.size)
    local m::Cint = x.size[1]

    local n::Cint

    if dims > 1
    	n = x.size[2]
    	n2 = y.size[2]
    else
    	n = 1
    	n2 = 1
    end

    local num = Cint(m*n)

    # get increments for x and y
    local incx::Cint = 1
    local incy::Cint = 1

    # check if dimensions are wrong
    if (n != n2) || (m != y.size[1])
    	throw(DimentionMismatch("The dimensions of x, $((m, n)) does nto equal the dimensions of y $((y.size[1], n2))"))
    end
    tmp = [alpha]
    local result::cublasStatus_t = cublasSaxpy_v2(handle, num, pointer(tmp), Ptr{Float32}(x.ptr), incx, Ptr{Float32}(y.ptr), incy)
    @assert (result == cudaSuccess) ("cublasSaxpy() error: " * cublasGetErrorName(result))
end

function cublasSscal(handle::cublasHandle_t, alpha::Float32, x::CUDAArray)::Nothing
    @assert (x.element_type == Float32)
    
    dims = length(x.size)
    local m::Cint = x.size[1]
    local n::Cint
    n = (dims > 1) ? x.size[2] : 1

    local num = Cint(m*n)

    # get increments for x and y
    local incx::Cint = 1

    tmp = [alpha]
    local result::cublasStatus_t = cublasSscal_v2(handle, num, pointer(tmp), Ptr{Float32}(x.ptr), incx)
    @assert (result == cudaSuccess) ("cublasSscal() error: " * cublasGetErrorName(result))
end

function forwardNOGRAD!(d_a::Vector{CUDAArray}, d_Thetas::Vector{CUDAArray}, d_biases::Vector{CUDAArray}, hidden_layers::Vector, d_X::CUDAArray, resLayers::Int64=0; activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))
#modifies d_a with forward activations
	num_hidden = length(hidden_layers)
	m = d_X.size[1]

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== hidden_layers[1]) "hidden layers do not share a dimension" 
	end

	if num_hidden > 0
		for i = 1:num_hidden
			run_kernel(fill_cols, m, hidden_layers[i], d_a[i], d_biases[i])
		end
	end

	run_kernel(fill_cols, m, d_a[end].size[2], d_a[end], d_biases[end])

	cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])
	# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])

	if num_hidden > 0

		activation_list[1] && run_kernel(tanhActivation, m, hidden_layers[1], d_a[1])

		if num_hidden > 1
			for i = 2:num_hidden
				cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])	
				if (resLayers != 0) && (((i - 1) % resLayers) == 0)
					#calculate residual skip every resLayers layers past the first hidden layer
					cublasSaxpy(cublas_handle, 1.0f0, d_a[i-resLayers], d_a[i])
				end
				activation_list[i] && run_kernel(tanhActivation, m, hidden_layers[i], d_a[i])	
			end
		end
		# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
		cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
	end
	# cuCtxSynchronize()
end

function nnCostFunctionNOGRAD(d_Thetas::Array{CUDAArray, 1}, d_biases::Array{CUDAArray, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Vector, m::Int64, d_a::Array{CUDAArray, 1}, d_X::CUDAArray, d_y::CUDAArray,lambda::Float32, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))

	@assert d_a[end].size[1] == d_y.size[1]

	if occursin("Log", costFunc)
		@assert d_a[end].size[2] == 2*d_y.size[2]
	else
		@assert d_a[end].size[2] == d_y.size[2]
	end

	#define size of output data separately from output layer
	n = d_y.size[2]
	
	forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list = activation_list)
	#launch across output data size rather than output layer size
	run_kernel(costFuncKs[costFunc], m, n, d_a[end], d_y)
	# cuCtxSynchronize()
	#changed from absolute sum to regular sum because the actual error values are stored in d_a[end]
	tmp_out = host_allocate(d_a[end]) 
	@fastmath sum(tmp_out)/m
end

function nnCostFunctionNOGRAD(d_Thetas::Array{CUDAArray, 1}, d_biases::Array{CUDAArray, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Vector, m::Int64, d_a::Array{CUDAArray, 1}, d_X::CUDAArray,lambda::Float32, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))

	@assert d_a[end].size[1] == d_X.size[1]

	if occursin("Log", costFunc)
		@assert d_a[end].size[2] == 2*d_X.size[2]
	else
		@assert d_a[end].size[2] == d_X.size[2]
	end

	#define size of output data separately from output layer
	n = d_X.size[2]
	
	forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list=activation_list)
	#launch across output data size rather than output layer size
	run_kernel(costFuncKs[costFunc], m, n, d_a[end], d_X)
	# cuCtxSynchronize()
	#changed from absolute sum to regular sum because the actual error values are stored in d_a[end]
	tmp_out = host_allocate(d_a[end]) 
	@fastmath sum(tmp_out)/m
end

function form_activations(d_Thetas::Vector{CUDAArray}, m::Int64)
	l = length(d_Thetas)
	d_a = Vector{CUDAArray}(undef, l)

	for i = 1:l
		d_a[i] = cuda_allocate(Array{Float32}(undef, m, d_Thetas[i].size[1]))
	end

	return d_a
end

function predict(d_Thetas, d_biases, d_X, input_layer_size, output_layer_size, hidden_layers, resLayers::Int64 = 0; layerout = length(hidden_layers)+1, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))
#PREDICT Predict the value of an input given a trained neural network
#m = number of examples in X, input_layer_size = number of input values, output_layer_size = number of output values
#hidden_layers = hidden layer vector
	l = length(d_Thetas)
	num_hidden = l - 1
	m = d_X.size[1]

	d_a = form_activations(d_Thetas, m)
	
	forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list=activation_list)

	return (d_a[layerout], host_allocate(d_a[layerout]))
end

function predict!(d_Thetas, d_biases, d_X, d_a, resLayers::Int64=0; activation_list::AbstractVector{Bool} = fill(true, length(d_Thetas)-1))
	l = length(d_Thetas)
	num_hidden = l - 1
	m = d_X.size[1]
	h = [B.size[1] for B in d_biases]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end

	forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list=activation_list)
end

function predictBatches(d_Thetas, d_biases, batches, input_layer_size, output_layer_size, hidden_layers, resLayers::Int64 = 0; activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))
#PREDICT Predict the value of an input given a trained neural network
#m = number of examples in X, input_layer_size = number of input values, output_layer_size = number of output values
#hidden_layers = hidden layer vector
	l = length(d_Thetas)
	num_hidden = l - 1
	m = size(batches[1], 1)
	#dropout scale factor
	# F = 1.0f0 - D

	d_a = form_activations(d_Thetas, m)

	out = mapreduce(vcat, batches) do X
		d_X = cuda_allocate(collect(X))

		forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list=activation_list)
		return host_allocate(d_a[end])
	end

	clear_gpu_data(d_a)
	(cuda_allocate(out), out)
end

function predictMulti(multiParams, d_X, input_layer_size, output_layer_size, hidden_layers, resLayers::Int64 = 0; activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))
#PREDICT Predict the value of an input given a trained neural network
#m = number of examples in X, input_layer_size = number of input values, output_layer_size = number of output values
#hidden_layers = hidden layer vector
	l = length(multiParams[1][1])
	num_hidden = l - 1

	#dropout scale factor
	# F = 1.0f0 - D
	m = d_X.size[1]

	d_a = form_activations(multiParams[1][1], m)

	[begin
		d_Thetas = params[1]
		d_biases = params[2]

		forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list=activation_list)
		host_allocate(d_a[end])
	end
	for params in multiParams]
end

function predictMultiBatches(multiParams, batches, input_layer_size, output_layer_size, hidden_layers, resLayers::Int64=0; activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))
#PREDICT Predict the value of an input given a trained neural network
#m = number of examples in X, input_layer_size = number of input values, output_layer_size = number of output values
#hidden_layers = hidden layer vector
	l = length(multiParams[1][1])
	num_hidden = l - 1
	m = size(batches[1], 1)
	d_a = form_activations(multiParams[1][1], m)

	outputs = map(batches) do X
		d_X = cuda_allocate(collect(X))
		[begin
			d_Thetas = params[1]
			d_biases = params[2]
			forwardNOGRAD!(d_a, d_Thetas, d_biases, hidden_layers, d_X, resLayers, activation_list=activation_list)
			host_allocate(d_a[end])
		end
		for params in multiParams]
	end

	clear_gpu_data(d_a)
	multiOut = map(i -> mapreduce(out -> out[i], vcat, outputs), 1:length(multiParams))
end

function nnCostFunction(d_Thetas::Array{CUDAArray, 1}, d_biases::Array{CUDAArray, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Vector, m::Int64, d_ones::CUDAArray, d_a::Array{CUDAArray, 1}, d_tanh_grad_z::Array{CUDAArray, 1}, d_deltas::Array{CUDAArray, 1}, d_Theta_grads::Array{CUDAArray, 1}, d_bias_grads::Array{CUDAArray, 1}, d_X::CUDAArray, d_y::CUDAArray,lambda::Float32, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== hidden_layers[1]) "hidden layers do not share a dimension" 
	end

	# kernelTests()

	@assert d_a[end].size[1] == d_y.size[1]

	if occursin("Log", costFunc)
		@assert d_a[end].size[2] == 2*d_y.size[2]
	else
		@assert d_a[end].size[2] == d_y.size[2]
	end

	#define size of output data separately from output layer
	n = d_y.size[2]

	if num_hidden > 0
		if lambda > 0.0f0
			for i in 1:length(d_Thetas)
				memcpy!(d_Theta_grads[i], d_Thetas[i])
			end
		end

		for i = 1:num_hidden
			run_kernel(fill_cols, m, hidden_layers[i], d_a[i], d_biases[i])
		end
	end
	run_kernel(fill_cols, m, output_layer_size, d_a[end], d_biases[end])
	cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])
	# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])
	if num_hidden > 0
		if activation_list[1]
			if D == 0.0f0
				run_kernel(tanhGradient, m, hidden_layers[1], d_a[1], d_tanh_grad_z[1])
			else
				run_kernel(tanhGradientDropout, m, hidden_layers[1], d_a[1], d_tanh_grad_z[1], rand(UInt32), D)
			end
		else
			run_kernel(noactivationGradient, m, hidden_layers[1], d_a[1], d_tanh_grad_z[1], rand(UInt32), D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				if (resLayers != 0) && (((i - 1) % resLayers) == 0)
					#calculate residual skip every resLayers layers past the first hidden layer
					cublasSaxpy(cublas_handle, 1.0f0, d_a[i-resLayers], d_a[i])
				end
				if activation_list[i]
					if D == 0.0f0
						run_kernel(tanhGradient, m, hidden_layers[i], d_a[i], d_tanh_grad_z[i])
					else
						run_kernel(tanhGradientDropout, m, hidden_layers[i], d_a[i], d_tanh_grad_z[i], rand(UInt32), D)
					end
				else
					run_kernel(noactivationGradient, m, hidden_layers[i], d_a[i], d_tanh_grad_z[i], rand(UInt32), D)
				end
			end
		end
		cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
		# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
	end

	#launch across output data size rather than output layer size
	run_kernel(costFuncDerivKs[costFunc], m, n, d_a[end], d_y, d_deltas[end])

	i = num_hidden
	while i >= 1
		cublasGemmEx(cublas_handle, algo, 'T', 'N', 1.0f0/m, d_deltas[i+1], d_a[i], lambda/m, d_Theta_grads[i+1])
		# cublasSgemm(cublas_handle, 'T', 'N', 1.0f0/m, d_deltas[i+1], d_a[i], lambda/m, d_Theta_grads[i+1])
		cublasSgemv(cublas_handle, 'T', 1.0f0/m, d_deltas[i+1], d_ones, 0.0f0, d_bias_grads[i+1])
		if (resLayers != 0) && ((i <= (num_hidden-resLayers)) && (((i+resLayers-1)%resLayers)==0))
			#replace d_deltas[i] with d_deltas[i+resLayers]
			memcpy!(d_deltas[i], d_deltas[i+resLayers])
			cublasGemmEx(cublas_handle, algo, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 1.0f0, d_deltas[i])
			# cublasSgemm(cublas_handle, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 1.0f0, d_deltas[i])
		else
			cublasGemmEx(cublas_handle, algo, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 0.0f0, d_deltas[i])
			# cublasSgemm(cublas_handle, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 0.0f0, d_deltas[i]) 
		end

		run_kernel(elMul, m, hidden_layers[i], d_deltas[i], d_tanh_grad_z[i])
		i = i - 1
	end
	cublasGemmEx(cublas_handle, algo, 'T', 'N', 1.0f0/m, d_deltas[1], d_X, lambda/m, d_Theta_grads[1])
	# cublasSgemm(cublas_handle, 'T', 'N', 1.0f0/m, d_deltas[1], d_X, lambda/m, d_Theta_grads[1])
	cublasSgemv(cublas_handle, 'T', 1.0f0/m, d_deltas[1], d_ones, 0.0f0, d_bias_grads[1])
	# cuCtxSynchronize()
end


function nnCostFunction(d_Thetas::Array{CUDAArray, 1}, d_biases::Array{CUDAArray, 1}, input_layer_size::Int64, output_layer_size::Int64, hidden_layers::Vector, m::Int64, d_ones::CUDAArray, d_a::Array{CUDAArray, 1}, d_tanh_grad_z::Array{CUDAArray, 1}, d_deltas::Array{CUDAArray, 1}, d_Theta_grads::Array{CUDAArray, 1}, d_bias_grads::Array{CUDAArray, 1}, d_X::CUDAArray,lambda::Float32, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)))

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== hidden_layers[1]) "hidden layers do not share a dimension" 
	end

	# kernelTests()

	@assert d_a[end].size[1] == d_X.size[1]

	if occursin("Log", costFunc)
		@assert d_a[end].size[2] == 2*d_X.size[2]
	else
		@assert d_a[end].size[2] == d_X.size[2]
	end

	#define size of output data separately from output layer
	n = d_X.size[2]

	if num_hidden > 0
		if lambda > 0.0f0
			for i in 1:length(d_Thetas)
				memcpy!(d_Theta_grads[i], d_Thetas[i])
			end
		end

		for i = 1:num_hidden
			run_kernel(fill_cols, m, hidden_layers[i], d_a[i], d_biases[i])
		end
	end
	run_kernel(fill_cols, m, output_layer_size, d_a[end], d_biases[end])
	cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])
	# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_X, d_Thetas[1], 1.0f0, d_a[1])
	if num_hidden > 0

		if activation_list[1]
			if D == 0.0f0
				run_kernel(tanhGradient, m, hidden_layers[1], d_a[1], d_tanh_grad_z[1])
			else
				run_kernel(tanhGradientDropout, m, hidden_layers[1], d_a[1], d_tanh_grad_z[1], rand(UInt32), D)
			end
		else
			run_kernel(noactivationGradient, m, hidden_layers[1], d_a[1], d_tanh_grad_z[1], rand(UInt32), D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_a[i-1], d_Thetas[i], 1.0f0, d_a[i])
				if (resLayers != 0) && (((i - 1) % resLayers) == 0)
					#calculate residual skip every resLayers layers past the first hidden layer
					cublasSaxpy(cublas_handle, 1.0f0, d_a[i-resLayers], d_a[i])
				end
				if activation_list[i]
					if D == 0.0f0
						run_kernel(tanhGradient, m, hidden_layers[i], d_a[i], d_tanh_grad_z[i])
					else
						run_kernel(tanhGradientDropout, m, hidden_layers[i], d_a[i], d_tanh_grad_z[i], rand(UInt32), D)
					end
				else
					run_kernel(noactivationGradient, m, hidden_layers[i], d_a[i], d_tanh_grad_z[i], rand(UInt32), D)
				end
			end
		end
		cublasGemmEx(cublas_handle, algo, 'N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
		# cublasSgemm(cublas_handle, 'N', 'T', 1.0f0, d_a[end-1], d_Thetas[end], 1.0f0, d_a[end])
	end

	#launch across output data size rather than output layer size
	run_kernel(costFuncDerivKs[costFunc], m, n, d_a[end], d_X, d_deltas[end])

	i = num_hidden
	while i >= 1
		cublasGemmEx(cublas_handle, algo, 'T', 'N', 1.0f0/m, d_deltas[i+1], d_a[i], lambda/m, d_Theta_grads[i+1])
		# cublasSgemm(cublas_handle, 'T', 'N', 1.0f0/m, d_deltas[i+1], d_a[i], lambda/m, d_Theta_grads[i+1])
		cublasSgemv(cublas_handle, 'T', 1.0f0/m, d_deltas[i+1], d_ones, 0.0f0, d_bias_grads[i+1])
		if (resLayers != 0) && ((i <= (num_hidden-resLayers)) && (((i+resLayers-1)%resLayers)==0))
			#replace d_deltas[i] with d_deltas[i+resLayers]
			memcpy!(d_deltas[i], d_deltas[i+resLayers])
			cublasGemmEx(cublas_handle, algo, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 1.0f0, d_deltas[i])
			# cublasSgemm(cublas_handle, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 1.0f0, d_deltas[i])
		else
			cublasGemmEx(cublas_handle, algo, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 0.0f0, d_deltas[i])
			# cublasSgemm(cublas_handle, 'N', 'N', 1.0f0, d_deltas[i+1], d_Thetas[i+1], 0.0f0, d_deltas[i]) 
		end

		run_kernel(elMul, m, hidden_layers[i], d_deltas[i], d_tanh_grad_z[i])
		i = i - 1
	end
	cublasGemmEx(cublas_handle, algo, 'T', 'N', 1.0f0/m, d_deltas[1], d_X, lambda/m, d_Theta_grads[1])
	# cublasSgemm(cublas_handle, 'T', 'N', 1.0f0/m, d_deltas[1], d_X, lambda/m, d_Theta_grads[1])
	cublasSgemv(cublas_handle, 'T', 1.0f0/m, d_deltas[1], d_ones, 0.0f0, d_bias_grads[1])
	# cuCtxSynchronize()
end