# using Base.LinAlg.BLAS
using LinearAlgebra.BLAS
# import Base.BLAS: gemv!

#-----------------Types of Cost Functions---------------------------------
function absErr(a, y)
	abs(a-y)
end

function absErrDeriv(a, y)
	ifelse(a > y, 1.0f0, ifelse(a < y, -1.0f0, 0.0f0))
end

function sqErr(a, y)
	(a-y)^2
end

function sqErrDeriv(a, y)
	2*(a - y)
end

#the exponential of a2 is the sigma parameter, this ensures it is always positive
function normLogErr(a1, a2, y)
	0.5f0*exp(2*a2)*a1^2 - exp(2*a2)*a1*y + 0.5f0*exp(2*a2)*y*y - a2 + 0.9189385332
	#0.5f0*(a2^2)*(a1-y)^2 - log(abs.(a2)) + 0.9189385332f0
end

function normLogErrDeriv(a1, a2, y)
	((a1 - y)*exp(2*a2), exp(2*a2)*(y-a1)^2 - 1)
end 

#the exponential of a2 is the inverse of the scale parameter, this ensures it is always positive
#exp(a2)=-1/b where b is the scale parameter => b = -1/exp(a2) and => log(exp(a2)/2) = log(-1/2b)
#-abs(x-u)/b - log(2b) = -abs(x-u)*exp(a2)+log(0.5*exp(a2))
function cauchyLogErr(a1, a2, y)
	exp(a2)*abs(a1-y) - log(0.5f0*exp(a2))
end

function cauchyLogErrDeriv(a1, a2, y)
	(exp(a2)*ifelse(a1 > y, 1.0f0, ifelse(a1 < y, -1.0f0, 0.0f0)), exp(a2)*abs(a1-y) - 1)
end


#names, functions, and function derivatives must all be in order here
costFuncNames = ("absErr", "sqErr", "normLogErr", "cauchyLogErr")
costFuncList = eval.(Symbol.(costFuncNames))
costFuncDerivsList = eval.(Symbol.(map(a -> "$(a)Deriv", costFuncNames)))
#--------------------------------------------------------------------------

function calcJ(m, n, delta, lambda, Thetas)
	accum1 = 0.0f0
	@simd for i = 1:m*n
		@inbounds accum1 += delta[i]
	end
	accum1 = accum1 / m

	accum2 = 0.0f0
	if lambda > 0.0f0
		for i = 1:length(Thetas)
			@simd for j = 1:length(Thetas[i])
				@inbounds accum2 += Thetas[i][j]*Thetas[i][j]
			end
		end
		accum2 = accum2 * lambda/(2.0*m)
	end
	#println(string("cost is ", accum1+accum2))
	return accum1 + accum2
end

costFuncs = Dict(zip(costFuncNames, costFuncList))
costFuncDerivs = Dict(zip(costFuncNames, costFuncDerivsList))

#in case output layer predicts a value and a range, check relative size
#between the two to dispatch to the proper error function
function calcDeltaOut!(costFuncDeriv, deltas, a, y, m, n)
#calculates derivative of the cost function
	if length(a) == 2*length(y)
		@simd for i = 1:m*n
			@inbounds (d1, d2) = costFuncDeriv(a[i], a[i+(m*n)], y[i])
			@inbounds deltas[i] = d1
			@inbounds deltas[i+(m*n)] = d2
			#@inbounds deltas[i] = absErrDeriv(a[i], y[i])
		end
	elseif length(a) == length(y)
		@simd for i = 1:m*n
			@inbounds deltas[i] = costFuncDeriv(a[i], y[i])
			#@inbounds deltas[i] = absErrDeriv(a[i], y[i])
		end
	else
		error("output layer does not match data")
	end
end

function calcFinalOut!(costFunc, a, y, m, n)
	if length(a) == 2*length(y)
		@simd for i = 1:m*n
			@inbounds a[i] = costFunc(a[i], a[i+(m*n)], y[i])
			#@inbounds a[i] = absErr(a[i], y[i])
		end
	elseif length(a) == length(y)
		@simd for i = 1:m*n
			@inbounds a[i] = costFunc(a[i], y[i])
			#@inbounds a[i] = absErr(a[i], y[i])
		end
	else
		error("output layer does not match data")
	end
end

function fillAs!(a::Array{Matrix{Float32}, 1}, biases::Array{Vector{Float32}, 1}, m)
	for i = 1:length(a)
		for j = 1:length(biases[i])
			start = m*(j-1) + 1
			b = biases[i][j]
			@simd for k = start:start+m-1
				@inbounds a[i][k] = b
			end
		end
	end
end

function fillThetaGrads!(Theta_grads, Thetas)
	for i = 1:length(Thetas)
		@simd for j = 1:length(Thetas[i])
			@inbounds Theta_grads[i][j] = Thetas[i][j]
		end
	end
end

function finishDelta!(delta, tanh_grad_z)
	@simd for i = 1:length(delta)
		@inbounds delta[i] = delta[i] * tanh_grad_z[i]
	end
end

function relu(x::T) where T <: AbstractFloat
	ifelse(x < zero(T), zero(T), x)
end

function relu_grad(x::T) where T <: AbstractFloat
	ifelse(x < zero(T), zero(T), one(T))
end

function relu_gradient!(z::Matrix{Float32}, grad_z::Matrix{Float32})
	l = length(z)
	@inbounds @simd for i in 1:l
		grad_z[i] = relu_grad(z[i])
		z[i] = relu(z[i])
	end
end

function fast_tanh(x::Float32)
	x2 = x*x
	a = x * (135135.0f0 + x2 * (17325.0f0 + x2 * (378.0f0 + x2)))
	b = 135135.0f0 + x2 * (62370.0f0 + x2 * (3150.0f0 + x2 * 28.0f0))	
	ifelse(x > 4.97f0, 1.0f0, ifelse(x<-4.97f0, -1.0f0, a/b))
end

function tanhGradient!(z::Matrix{Float32}, tanh_grad_z::Matrix{Float32})
	l = length(z)
	@inbounds @simd for i = 1:l	
		z[i] = 1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end
	#z.=1.7159f0*fast_tanh.(2.0f0*z/3.0f0)

	@inbounds @simd for i = 1:l 
		tanh_grad_z[i] = 1.7159f0 * (1.0f0 - z[i]*z[i]/(1.7159f0*1.7159f0)) * 2.0f0 / 3.0f0
	end
	#tanh_grad_z.=1.7159f0 * (1.0f0 - z.*z/(1.7159f0*1.7159f0)) * 2.0f0 / 3.0f0
end

function noactivationGradient!(z::Matrix{Float32}, tanh_grad_z::Matrix{Float32}, D::Float32)
	l = length(z)
	F = 1.0f0/(1.0f0 - D) #added scaling factor so dropout trained network can be treated normally during inference
	if D != 0
		@inbounds for i = 1:l
			tanh_grad_z[i] = Float32(rand() > D)*F
			z[i] = tanh_grad_z[i]*z[i]
		end
	else
		@inbounds for i in 1:l
			tanh_grad_z[i] = 1.0f0
		end
	end
end

function tanhGradient!(z::Matrix{Float32}, tanh_grad_z::Matrix{Float32}, D::Float32)
	l = length(z)
	F = 1.0f0/(1.0f0 - D) #added scaling factor so dropout trained network can be treated normally during inference
	@inbounds for i = 1:l
		tanh_grad_z[i] = Float32(rand() > D)*F
	end

	@inbounds @simd for i = 1:l	
		z[i] = tanh_grad_z[i]*1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end

	@inbounds @simd for i = 1:l 
		tanh_grad_z[i] = tanh_grad_z[i]*1.7159f0 * (1.0f0 - z[i]*z[i]/(1.7159f0*1.7159f0)) * 2.0f0 / 3.0f0
	end	
end

function relu_gradient!(z::Matrix{T}, grad_z::Matrix{T}, D::T) where T <: AbstractFloat
	l = length(z)
	F = one(T)/(one(T) - D)
	@inbounds @simd for i in 1:l
		z[i] = relu(z[i])
		grad_z[i] = ifelse(rand(T) > D, F*relu_grad(z[i]), zero(T))
	end
end

function tanhActivation!(z::Matrix{Float32})
	@simd for i = 1:length(z)
		@inbounds z[i] = 1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end
end

function relu_activation!(z::Matrix{T}) where T <: AbstractFloat
	@simd for i = 1:length(z)
		@inbounds z[i] = relu(z[i])
	end
end

function form_activations(Thetas::Vector{Matrix{Float32}}, m::Int64)
	l = length(Thetas)
	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(Thetas[i], 1))
	end

	return a
end

function form_tanh_grads(hidden_layers::AbstractVector{Int64}, m::Int64)
	num_hidden = length(hidden_layers)
	tanh_grad_z = Array{Matrix{Float32}, 1}(undef, num_hidden)
	if num_hidden > 0
		for i in 1:num_hidden
			tanh_grad_z[i] = Matrix{Float32}(undef, m, hidden_layers[i])
		end
	end
	return tanh_grad_z
end

function forwardNOGRAD!(a::Vector{Matrix{Float32}}, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, hidden_layers::Vector, X::Matrix{Float32}, resLayers::Int64=0; activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), userelu = false)
	m = size(X, 1)
	l = length(thetas)
	n = size(thetas[end], 1)
	num_hidden = length(hidden_layers)
	@assert num_hidden == l-1

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== hidden_layers[1]) "hidden layers do not share a dimension" 
	end

	fillAs!(a, biases, m)
	gemm!('N', 'T', 1.0f0, X, thetas[1], 1.0f0, a[1])

	if l > 1
		activation_list[1] && (userelu ? relu_activation!(a[1]) : tanhActivation!(a[1]))
		if (l-1) > 1
			for i = 2:l-1
				gemm!('N', 'T', 1.0f0, a[i-1], thetas[i], 1.0f0, a[i])
				if (resLayers != 0) && ((i - 1) % resLayers) == 0
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				activation_list[i] && (userelu ? relu_activation!(a[i]) : tanhActivation!(a[i]))
			end
		end
		gemm!('N', 'T', 1.0f0, a[end-1], thetas[end], 1.0f0, a[end])
	end
end

function nnCostFunctionNOGRAD(Thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, input_layer_size::Int64, hidden_layers::Vector, X::Matrix{Float32}, y::Matrix{Float32}, lambda::Float32, a::Vector{Matrix{Float32}}, D::Float32 = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), userelu = false)

	num_hidden = length(hidden_layers)

	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	# F = 1.0f0 - D
	
	if occursin("Log", costFunc)
		@assert 2*n == size(a[end], 2)
	else
		@assert n == size(a[end], 2)
	end

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list = activation_list, userelu = userelu)

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], y, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)
end

function nnCostFunctionNOGRAD(Thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, input_layer_size::Int64, hidden_layers::Vector, X::Matrix{Float32}, lambda::Float32, a::Vector{Matrix{Float32}}, D::Float32 = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), userelu = false)

	num_hidden = length(hidden_layers)

	#Setup some useful variables
	(m, n) = size(X)
	# F = 1.0f0 - D
	
	if occursin("Log", costFunc)
		@assert 2*n == size(a[end], 2)
	else
		@assert n == size(a[end], 2)
	end

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list = activation_list, userelu = userelu)

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], X, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)
end

function predict(Thetas, biases, X::Matrix{Float32}, resLayers::Int64 = 0; layerout=length(Thetas), activation_list::AbstractVector{Bool} = fill(true, length(Thetas) - 1), userelu = false)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(Thetas)
	num_hidden = l-1
	
	h = [length(B) for B in biases]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end
	#dropout scale factor
	# F = (1.0f0 - D)
	a = form_activations(Thetas, m)

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list = activation_list, userelu = userelu)
	return a[layerout]
end

function predict!(Thetas, biases, X::Matrix{Float32}, a::Vector{Matrix{Float32}}, resLayers::Int64 = 0; activation_list = fill(true, length(Thetas)-1), userelu = false)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(Thetas)
	num_hidden = l-1
	
	h = [length(B) for B in biases]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end
	#dropout scale factor
	# F = (1.0f0 - D)

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list = activation_list, userelu = userelu)
end

function predictBatches(Thetas, biases, batches::Vector{Matrix{Float32}}, resLayers::Int64 = 0; layerout=length(Thetas), activation_list = fill(true, length(Thetas)-1), userelu = false)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(batches[1], 1)
	l = length(Thetas)
	num_hidden = l-1

	h = [length(B) for B in biases]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end
	a = form_activations(Thetas, m)

	outputlength = mapreduce(a -> size(a, 1), +, batches)
	(batchlength, outputwidth) = size(a[layerout])
	output = Matrix{Float32}(undef, outputlength, outputwidth)
	#dropout scale factor
	# F = (1.0f0 - D)
	row = 1
	# mapreduce(vcat, batches) do X
	for X in batches
		forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list, userelu = userelu)
		output[row:row+batchlength-1, :] .= a[layerout]
		row = row+batchlength
		# output[copy(a[layerout])
	end
end

function predictMulti(multiParams, X::Matrix{Float32}, resLayers::Int64 = 0; layerout=length(multiParams[1][1]), activation_list=fill(true, length(multiParams[1][1])), userelu = false)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(multiParams[1][1])
	num_hidden = l-1

	h = [length(B) for B in multiParams[1][2]]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end

	a = form_activations(multiParams[1][1], m)
	[begin
		Thetas = params[1]
		biases = params[2]
		forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list, userelu = userelu)
		copy(a[layerout])
	end
	for params in multiParams]
end

function predictMulti!(multiParams, X::Matrix{Float32}, a, outputs, resLayers::Int64 = 0; activation_list=fill(true, length(multiParams[1][1])), userelu = false)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(multiParams[1][1])
	num_hidden = l-1

	h = [length(B) for B in multiParams[1][2]]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end

	for (i, params) in enumerate(multiParams)
		Thetas = params[1]
		biases = params[2]
		forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list, userelu = userelu)
		outputs[i] .= a[end]
	end
end

function predictMultiBatches(multiParams, batches::Vector{Matrix{Float32}}, resLayers::Int64 = 0; layerout=length(multiParams[1][1]), activation_list=fill(true, length(multiParams[1][1])), userelu = false)
#PREDICT Predict the value of an input in batches given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(batches[1], 1)
	l = length(multiParams[1][1])
	num_hidden = l-1

	h = [length(B) for B in multiParams[1][2]]
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		h[1:l-1]
	end
	#dropout scale factor
	# F = (1.0f0 - D)
	a = form_activations(multiParams[1][1], m)

	[begin
		mapreduce(vcat, batches) do X
			Thetas = params[1]
			biases = params[2]
			forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list, userelu = userelu)
			return copy(a[layerout])
		end
	end
	for params in multiParams]
end

function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::Vector, X::Matrix{Float32}, y::Matrix{Float32},lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), userelu = false)

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== hidden_layers[1]) "hidden layers do not share a dimension" 
	end


	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	gradfunc! = userelu ? relu_gradient! : tanhGradient!

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				gradfunc!(a[1], tanh_grad_z[1])
			else
				gradfunc!(a[1], tanh_grad_z[1], D)
			end
		else
			noactivationGradient!(a[1], tanh_grad_z[1], D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
				if (resLayers != 0) && (((i - 1) % resLayers) == 0)
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				if activation_list[i]
					if D == 0.0f0
						gradfunc!(a[i], tanh_grad_z[i])
					else
						gradfunc!(a[i], tanh_grad_z[i], D)
					end
				else
					noactivationGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	#mean abs error cost function
	calcDeltaOut!(costFuncDerivs[costFunc], deltas[end], a[end], y, m, n)

	# println("deltas[end] CPU is $(deltas[end])")


	i = num_hidden
	
	while i >= 1
		gemm!('T', 'N', 1.0f0/m, deltas[i+1], a[i], lambda/m, Theta_grads[i+1])
		gemv!('T', 1.0f0/m, deltas[i+1], onesVec, 0.0f0, Bias_grads[i+1])
		#deltas[i] = (deltas[i+1]*Thetas[i+1]) .* tanh_grad_z[i]
		if (resLayers != 0) && ((i <= (num_hidden - resLayers)) && (((i + resLayers - 1) % resLayers) == 0))
			#replace deltas[i] with deltas[i+resLayers]
			# scal!(length(deltas[i]), 0.0f0, deltas[i], 1)
			# axpy!(1.0f0, deltas[i+resLayers], deltas[i]) 
			blascopy!(length(deltas[i]), deltas[i+resLayers], 1, deltas[i], 1)
			#propagate derivative back to deltas from the original input to the residual layers
			gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 1.0f0, deltas[i])
			# gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i])
		else
			gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i]) #do part 1 of line 1 in place
		end
		finishDelta!(deltas[i], tanh_grad_z[i]) #do part 2 of line 1 in place
		i = i - 1
	end


	gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]
end

function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::Vector, X::Matrix{Float32}, lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), userelu = false)

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== hidden_layers[1]) "hidden layers do not share a dimension" 
	end


	#Setup some useful variables
	(m, n) = size(X)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end

	gradfunc! = userelu ? relu_gradient! : tanhGradient!


	fillAs!(a, biases, m)

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				gradfunc!(a[1], tanh_grad_z[1])
			else
				gradfunc!(a[1], tanh_grad_z[1], D)
			end
		else
			noactivationGradient!(a[1], tanh_grad_z[1], D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
				if (resLayers != 0) && (((i - 1) % resLayers) == 0)
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				
				if activation_list[i]
					if D == 0.0f0
						gradfunc!(a[i], tanh_grad_z[i])
					else
						gradfunc!(a[i], tanh_grad_z[i], D)
					end
				else
					noactivationGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	#mean abs error cost function
	calcDeltaOut!(costFuncDerivs[costFunc], deltas[end], a[end], X, m, n)

	# println("deltas[end] CPU is $(deltas[end])")


	i = num_hidden
	
	while i >= 1
		gemm!('T', 'N', 1.0f0/m, deltas[i+1], a[i], lambda/m, Theta_grads[i+1])
		gemv!('T', 1.0f0/m, deltas[i+1], onesVec, 0.0f0, Bias_grads[i+1])
		#deltas[i] = (deltas[i+1]*Thetas[i+1]) .* tanh_grad_z[i]
		if (resLayers != 0) && ((i <= (num_hidden - resLayers)) && (((i + resLayers - 1) % resLayers) == 0))
			#replace deltas[i] with deltas[i+resLayers]
			# scal!(length(deltas[i]), 0.0f0, deltas[i], 1)
			# axpy!(1.0f0, deltas[i+resLayers], deltas[i]) 
			blascopy!(length(deltas[i]), deltas[i+resLayers], 1, deltas[i], 1)
			#propagate derivative back to deltas from the original input to the residual layers
			gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 1.0f0, deltas[i])
			# gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i])
		else
			gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i]) #do part 1 of line 1 in place
		end
		finishDelta!(deltas[i], tanh_grad_z[i]) #do part 2 of line 1 in place
		i = i - 1
	end


	gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]
end


function nnCostFunctionAdv(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::Vector{Int}, advX::Matrix{Float32}, X::Matrix{Float32}, y::Matrix{Float32},lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32})

	num_hidden = length(hidden_layers)


	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	tanhGradient!(a[1], tanh_grad_z[1])

	if num_hidden > 1
		for i = 2:num_hidden
			gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
			tanhGradient!(a[i], tanh_grad_z[i])
		end
	end

	gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])

	#mean abs error cost function
	@simd for i = 1:m*n
		@inbounds deltas[end][i] = ifelse(a[end][i] > y[i], 1.0f0, ifelse(a[end][i] < y[i], -1.0f0, 0.0f0))
	end


	i = num_hidden
	while i >= 1
		#deltas[i] = (deltas[i+1]*Thetas[i+1]) .* tanh_grad_z[i]
		gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i]) #do part 1 of line 1 in place
		finishDelta!(deltas[i], tanh_grad_z[i]) #do part 2 of line 1 in place
		i = i - 1
	end

	gemm!('N', 'N', 1.0f0/m, deltas[1], Thetas[1], 0.0f0, advX)

	@simd for i = 1:m*input_layer_size
		@inbounds advX[i] = (X[i] + (1.0f-8)*sign(advX[i]))
	end


	gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])

	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]

	for i = 2:num_hidden+1
		gemm!('T', 'N', 1.0f0/m, deltas[i], a[i-1], lambda/m, Theta_grads[i])
		gemv!('T', 1.0f0/m, deltas[i], onesVec, 0.0f0, Bias_grads[i]) #calculate below line in place
		#Bias_grads[i] = (ones(Float32, 1, m)*deltas[i]/m)[:]
	end

end