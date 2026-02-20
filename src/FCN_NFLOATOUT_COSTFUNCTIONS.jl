# using Base.LinAlg.BLAS
using LinearAlgebra.BLAS
# import Base.BLAS: gemv!

#-----------------Types of Cost Functions---------------------------------
"""
    absErr(a, y)

Compute the absolute error between predicted value `a` and true value `y`.

# Arguments
- `a::Float32`: Predicted/model output value
- `y::Float32`: True/target value

# Returns
- `Float32`: Absolute difference |a - y|

# Description
Primary cost function for measuring prediction error in the neural network.
Used for both single values and element-wise in matrices for batch processing.
Part of the mean absolute error (MAE) cost metric when averaged across all outputs.

# See also
[`sqErr`](@ref): Squared error cost function
[`normLogErr`](@ref): Log-likelihood based error for probabilistic outputs
[`absErrDeriv`](@ref): Derivative of absolute error for gradient computations
"""
function absErr(a, y)
    abs(a-y)
end

"""
    absErrDeriv(a, y)

Compute the derivative of the absolute error with respect to the predicted value.

# Arguments
- `a::Float32`: Predicted/model output value
- `y::Float32`: True/target value

# Returns
- `Float32`: Derivative value: +1 if a > y, -1 if a < y, 0 if a == y

# Description
Calculates the gradient of the absolute error cost function for backpropagation.
Returns the sign of the difference between predicted and true values, making it 
suitable for gradient-based optimization where the exact magnitude of the error
is less important than its direction.

# Mathematical Details
The derivative is discontinuous at a = y, but this generally doesn't cause issues
in practice for neural network training. The derivative is defined as:
- +1 when a > y (prediction too high)
- -1 when a < y (prediction too low)
- 0 when a = y (perfect prediction)

# See also
[`absErr`](@ref): The corresponding cost function
[`sqErrDeriv`](@ref): Derivative of squared error alternative
"""
function absErrDeriv(a, y)
    ifelse(a > y, 1.0f0, ifelse(a < y, -1.0f0, 0.0f0))
end

function sqErr(a, y)
	(a-y)^2
end

function sqErrDeriv(a, y)
	2*(a - y)
end

outputIndex = Returns(nothing)
outputIndexDeriv = Returns(nothing)
crossEntropy = Returns(nothing)
crossEntropyDeriv = Returns(nothing)

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
costFuncNames = ("absErr", "sqErr", "normLogErr", "cauchyLogErr", "outputIndex", "crossEntropy")
costFuncList = eval.(Symbol.(costFuncNames))
costFuncDerivsList = eval.(Symbol.(map(a -> "$(a)Deriv", costFuncNames)))
#--------------------------------------------------------------------------

function calculate_l2(Thetas::Vector{M}) where {T<:Real, M<:Matrix{T}}
	accum = zero(T) 
	for i in eachindex(Thetas)
		@simd for j = eachindex(Thetas[i])
			@inbounds accum += Thetas[i][j]*Thetas[i][j]
		end
	end
	return accum
end

function calcJ(m::Integer, n::Integer, delta::M, lambda::T, Thetas::Vector{M}) where {T<:Real, M <: Matrix{T}}
	accum1 = zero(T)
	@inbounds @simd for i in 1:m*n
		accum1 += delta[i]
	end
	accum1 /= m
	iszero(lambda) && return accum1
	accum2 = lambda*calculate_l2(Thetas) / (T(2) * m)
	#println(string("cost is ", accum1+accum2))
	return accum1 + accum2
end

#if the number of output columns is not included, the values are all stored in the first column and correspond to the loss with just a single element from each example
function calcJ(m::Integer, delta::M, lambda::T, Thetas::Vector{M}) where {T<:Real, M <: Matrix{T}}
	accum1 = zero(T)
	@inbounds @simd for i in 1:m
		accum1 += delta[i, 1]
	end
	accum1 /= m
	iszero(lambda) && return accum1
	accum2 = lambda*calculate_l2(Thetas) / (T(2) * m)
	#println(string("cost is ", accum1+accum2))
	return accum1 + accum2
end

#calculate output for output which is just selecting one of the indices from the available outputs
function calcJ(delta::M, output_index::Integer, lambda::T, Thetas::Vector{M}) where {T<:Real, M<:Matrix{T}}
	m = size(delta, 1)
	accum1 = zero(T) 
	@inbounds @simd for i = 1:m
		accum1 += delta[i, output_index]
	end

	accum1 /= m
	iszero(lambda) && return accum1
	accum2 = lambda*calculate_l2(Thetas) / (T(2) * m)
	#println(string("cost is ", accum1+accum2))
	return accum1 + accum2
end

function calcJ(delta::M, output_indices::Vector{I}, lambda::T, Thetas::Vector{M}) where {I<:Integer, T<:Real, M<:Matrix{T}}
	m = size(delta, 1)
	accum1 = zero(T) 
	@inbounds @simd for i = 1:m
		accum1 += delta[i, output_indices[i]]
	end
	accum1 = accum1 / m
	iszero(lambda) && return accum1
	accum2 = lambda*calculate_l2(Thetas) / (T(2) * m)
	#println(string("cost is ", accum1+accum2))
	return accum1 + accum2
end

#calculate output for output which is just selecting one of the indices from the available outputs
function calcJ(delta::V, output_index::Integer, lambda::T, Thetas::Vector{M}) where {T<:Real, V<:Vector{T}, M<:Matrix{T}}
	accum1 = delta[output_index]
	iszero(lambda) && return accum1 
	accum2 = lambda*calculate_l2(Thetas) / T(2)
	#println(string("cost is ", accum1+accum2))
	return accum1 + accum2
end

costFuncs = Dict(zip(costFuncNames, costFuncList))
costFuncDerivs = Dict(zip(costFuncNames, costFuncDerivsList))

#in case output layer predicts a value and a range, check relative size
#between the two to dispatch to the proper error function
function calcDeltaOut!(costFuncDeriv::Function, deltas::Matrix{Float32}, a::Matrix{Float32}, y::Matrix{Float32}, m, n)
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

function calcDeltaOut!(costFuncDeriv::Function, deltas::Matrix{Float32}, a::Matrix{Float32}, y::Vector{Float32}, indices::Vector{I}, m::Integer) where I<:Integer
	deltas .= zero(Float32)
	#calculates derivative of the cost function
	@inbounds @simd for i in 1:m
		j = indices[i]
		deltas[i, j] = costFuncDeriv(a[i, j], y[i])
	end
end

#when the output derivative function is just one of the output indices and there is just one example
function calcDeltaOut!(deltas::Vector{T}, index::Integer) where T<:Real
	deltas .= zero(T)
	deltas[index] = one(T)
end

#when the output derivative function is just one of the output indices
function calcDeltaOut!(deltas::Matrix{T}, index::Integer) where T<:Real
	deltas .= zero(T)
	@inbounds @simd for i in 1:size(deltas, 1)
		deltas[i, index] = one(T) # sign(a[i, index])
	end
end


#when the output derivative function is just one of the output indices which can vary for each example
function calcDeltaOut!(deltas::Matrix{T}, indices::Vector{I}) where {T<:Real, I<:Integer}
	deltas .= zero(T)
	@inbounds @simd for i in eachindex(indices) 
		deltas[i, indices[i]] = one(T) # sign(a[i, index])
	end
end

abstract type LossType end
struct OutputIndex <: LossType end
struct CrossEntropyLoss <: LossType end

calcDeltaOut!(::OutputIndex, deltas::Array{T, N}, a::Array{T, N}, index) where {T<:Real, N} = calcDeltaOut!(deltas, index)

#perform the calculation of a softmax put derivative where the maximum value of each example is subtracted before computing the softmax to ensure numerical stability.  this case is for a single example so the deltas and activations are vectors rather than matrices. there is also only a single index for the output since there is only one example
function calcDeltaOut!(::CrossEntropyLoss, deltas::Vector{T}, a::Vector{T}, index::Integer) where {T<:Real}
	n = length(a)
	
	#compute maximum activation value
	max_value = maximum(a)

	#initialize variables to track the denominator of the softmax
	denominator = zero(T)

	#go back through all of the activations and calculate the numerator of the softmax output and accumulate the denominator into the second column of deltas
	@inbounds @simd for i in 1:n
		h = exp(a[i] - max_value)
		denominator += h
		deltas[i] = h
	end

	deltas ./= denominator
	deltas[index] -= one(T)
	return deltas
end

#perform the first part of the calculation of a softmax put derivative where the maximum value of each example is subtracted before computing the softmax to ensure numerical stability
function crossEntropyDeltaOut!(deltas::Matrix{T}, a::Matrix{T}) where {T<:Real}
	(m, n) = size(a)

	#initialize the first column of deltas with the first column of activations, this index will store the max value of all activations
	@inbounds @simd for i in 1:m
		deltas[i, 1] = a[i, 1]
	end

	#initialize the final column of deltas with 0 to keep track of the sum required for the denominator of the softmax
	@inbounds @simd for i in 1:m
		deltas[i, n] = zero(T)
	end

	#go through the remaining columns of activations and calculate the maximum value for each row
	@inbounds for j in 2:n
		@simd for i in 1:m
			deltas[i, 1] = max(deltas[i, 1], a[i, j])
		end
	end

	#go back through all of the activations and calculate the numerator of the softmax output and accumulate the denominator into the second column of deltas
	@inbounds for j in 1:n
		@simd for i in 1:m
			h = exp(a[i, j] - deltas[i, 1])
			deltas[i, n] += h
			a[i, j] = h
		end
	end

	#note that the final column of deltas will be overwritten last after it has been used for the other outputs
	@inbounds for j in 1:n
		#update the deltas to be the softmax output
		@simd for i in 1:m
			deltas[i, j] = a[i, j] / deltas[i, n]
		end
	end
end

#finish the calculation started above when there is a separate index per example
function crossEntropyDeltaOut!(deltas::Matrix{T}, indices::Vector{I}) where {T<:Real, I<:Integer}
	#subtract 1 for the desired index with p=1 for that example
	@inbounds @simd for i in eachindex(indices)
		deltas[i, indices[i]] -= one(T)
	end
end

#finish the calculation started above when there is a single index per example
function crossEntropyDeltaOut!(deltas::Matrix{T}, index::Integer) where {T<:Real}
	#subtract 1 for the desired index with p=1 for that example
	@inbounds @simd for i in 1:size(deltas, 1)
		deltas[i, index] -= one(T)
	end
end

function calcDeltaOut!(::CrossEntropyLoss, deltas::Matrix{T}, a::Matrix{T}, index) where {T<:Real} 
	crossEntropyDeltaOut!(deltas, a)
	crossEntropyDeltaOut!(deltas, index)
end

function calcFinalOut!(costFunc::Function, a::Matrix{T}, y::Matrix{T}, m::Integer, n::Integer) where T<:AbstractFloat
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

function calcFinalOut!(costFunc::Function, a::Matrix{T}, y::Vector{T}, output_indices::Vector{I}, m::Integer) where {T<:AbstractFloat, I<:Integer}
	@simd for i = 1:m
		@inbounds a[i, 1] = costFunc(a[i, output_indices[i]], y[i])
	end
end

function calcFinalOut!(::OutputIndex, a, output)
end

#compute the cross entropy loss of the softmax of a for a single example where the desired output is given by index
function calcFinalOut!(::CrossEntropyLoss, a::Vector{T}, index::Integer) where T<:AbstractFloat
	m = maximum(a)
	s = zero(T)
	@inbounds @simd for i in eachindex(a)
		h = a[i] - m
		x = exp(h)
		s += x
		a[i] = h * (i == index)
	end
	a[index] = -a[index] + log(s) 
end

#compute the cross entropy loss of the softmax of a for multiple examples contained in the rows of a and the elements of indices
function calcFinalOut!(::CrossEntropyLoss, a::Matrix{T}, index::Integer) where {T<:AbstractFloat}
	(m, n) = size(a)
	@inbounds @simd for i in 1:m
		max_value = zero(T)
		for j in 1:n
			max_value = max(max_value, a[i, j])
		end

		s = zero(T)
		for j in 1:n
			h = a[i, j] - max_value
			x = exp(h)
			a[i, j] = h*(j == index)
			s += x
		end
		a[i, index] = -a[i, index] + log(s)
	end
end
#compute the cross entropy loss of the softmax of a for multiple examples contained in the rows of a and the elements of indices
function calcFinalOut!(::CrossEntropyLoss, a::Matrix{T}, indices::Vector{I}) where {T<:AbstractFloat, I<:Integer}
	(m, n) = size(a)
	@inbounds @simd for i in eachindex(indices)
		max_value = zero(T)
		for j in 1:n
			max_value = max(max_value, a[i, j])
		end

		s = zero(T)
		for j in 1:n
			h = a[i, j] - max_value
			x = exp(h)
			a[i, j] = h*(j == indices[i])
			s += x
		end
		a[i, indices[i]] = -a[i, indices[i]] + log(s)
	end
end

#fill activations for a single example
function fillAs!(a::Array{Vector{Float32}, 1}, biases::Array{Vector{Float32}, 1})
	@inbounds for i = 1:length(a)
		a[i] .= biases[i]
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
	for i = eachindex(Thetas)
		@simd for j = eachindex(Thetas[i])
			@inbounds Theta_grads[i][j] = Thetas[i][j]
		end
	end
end

function finishDelta!(delta, tanh_grad_z)
	@simd for i = eachindex(delta)
		@inbounds delta[i] = delta[i] * tanh_grad_z[i]
	end
end


function fast_tanh(x::Float32)
	x2 = x*x
	a = x * (135135.0f0 + x2 * (17325.0f0 + x2 * (378.0f0 + x2)))
	b = 135135.0f0 + x2 * (62370.0f0 + x2 * (3150.0f0 + x2 * 28.0f0))	
	ifelse(x > 4.97f0, 1.0f0, ifelse(x<-4.97f0, -1.0f0, a/b))
end

function tanhGradient!(z::Array{T, N}, tanh_grad_z::Array{T, N}) where {T<:Float32, N}
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

function noactivationGradient!(z::Array{Float32, N}, tanh_grad_z::Array{Float32, N}, D::Float32) where N
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

function tanhGradient!(z::Array{Float32, N}, tanh_grad_z::Array{Float32, N}, D::Float32) where N
	iszero(D) && tanhGradient!(z, tanh_grad_z)
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

function tanhActivation!(z::Array{Float32, N}) where N
	@simd for i = 1:length(z)
		@inbounds z[i] = 1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end
end

#form activation matrices for m examples
function form_activations(Thetas::Vector{Matrix{Float32}}, m::Int64)
	l = length(Thetas)
	a = Array{Matrix{Float32}, 1}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(Thetas[i], 1))
	end

	return a
end

#form activation vectors for 1 example
function form_activations(Thetas::Vector{Matrix{Float32}})
	l = length(Thetas)
	a = Array{Vector{Float32}, 1}(undef, l)
	for i in eachindex(Thetas) 
		a[i] = Vector{Float32}(undef, size(Thetas[i], 1))
	end
	return a
end


#form tanh gradient vectors for 1 example
function form_tanh_grads(hidden_layers::AbstractVector{Int64})
	num_hidden = length(hidden_layers)
	tanh_grad_z = Array{Vector{Float32}, 1}(undef, num_hidden)
	if num_hidden > 0
		for i in 1:num_hidden
			tanh_grad_z[i] = Vector{Float32}(undef, hidden_layers[i])
		end
	end
	return tanh_grad_z
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

"""
    forwardNOGRAD_base!(a, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, x, resLayers::Int64)

Compute a forward pass through a neural network without gradient computations, using either tanh activation
functions or residual connections between layers.

# Methods
    forwardNOGRAD_base!(a::Vector{Vector{Float32}}, thetas, biases, x::Vector{Float32}, resLayers)

Process a single input example through the network.

    forwardNOGRAD_base!(a::Vector{Matrix{Float32}}, thetas, biases, X::Matrix{Float32}, resLayers)

Process multiple input examples simultaneously through the network.

# Arguments
- `a`: Pre-allocated vector of activation arrays (vectors for single example, matrices for batch)
- `thetas::Vector{Matrix{Float32}}`: Weight matrices for each layer
- `biases::Vector{Vector{Float32}}`: Bias vectors for each layer
- `x`: Input data (vector for single example, matrix with examples as rows for batch)
- `resLayers::Int64`: Number of layers between residual connections (0 for no residual connections)

# Effects
- Updates the activation arrays in `a` in-place
- Applies tanh activation to hidden layers
- If `resLayers > 0`, adds residual connections every `resLayers` layers

# Notes
- Residual connections require at least two hidden layers
- When using residual connections, hidden layers must have matching dimensions

# See also
[`forwardNOGRAD!`](@ref): Higher-level version with activation function options and checks
[`tanhActivation!`](@ref): The activation function used in hidden layers
[`form_activations`](@ref): Creates the activation array structure
[`nnCostFunctionNOGRAD`](@ref): Cost computation without gradients using this forward pass
"""
function forwardNOGRAD_base!(a::Vector{Vector{Float32}}, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, x, resLayers::Int64)
	l = length(thetas)
	fillAs!(a, biases)
	gemv!('N', 1.0f0, thetas[1], x, 1.0f0, a[1])

	if l > 1
		tanhActivation!(a[1])
		if (l-1) > 1
			@inbounds for i = 2:l-1
				gemv!('N', 1.0f0, thetas[i], a[i-1], 1.0f0, a[i])
				if (resLayers != 0) && ((i - 1) % resLayers) == 0
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				tanhActivation!(a[i])
			end
		end
		gemv!('N', 1.0f0, thetas[end], a[end-1], 1.0f0, a[end])
	end
end


function forwardNOGRAD_base!(a::Vector{Matrix{Float32}}, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, X, resLayers::Int64; input_orientation::Char = 'N')
	l = length(thetas)
	m = size(X, ifelse(input_orientation == 'N', 1, 2))
	fillAs!(a, biases, m)
	gemm!(input_orientation, 'T', 1.0f0, X, thetas[1], 1.0f0, a[1])

	if l > 1
		tanhActivation!(a[1])
		if (l-1) > 1
			@inbounds for i = 2:l-1
				gemm!('N', 'T', 1.0f0, a[i-1], thetas[i], 1.0f0, a[i])
				if (resLayers != 0) && ((i - 1) % resLayers) == 0
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				tanhActivation!(a[i])
			end
		end
		gemm!('N', 'T', 1.0f0, a[end-1], thetas[end], 1.0f0, a[end])
	end
end

#computes the forward pass of the network without any checks, allowing optionality to provide a list of Bools showing which if any layers are not active in the case of one example
function forwardNOGRAD!(a::Vector{Vector{Float32}}, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, x, resLayers::Int64, activation_list::AbstractVector{B}) where B <: Bool
	isempty(activation_list) && return forwardNOGRAD_base!(a, thetas, biases, x, resLayers)
	
	l = length(thetas)
	fillAs!(a, biases)
	gemv!('N', 1.0f0, thetas[1], x, 1.0f0, a[1])

	if l > 1
		activation_list[1] && tanhActivation!(a[1])
		if (l-1) > 1
			@inbounds for i = 2:l-1
				gemv!('N', 1.0f0, thetas[i], a[i-1], 1.0f0, a[i])
				if (resLayers != 0) && ((i - 1) % resLayers) == 0
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				activation_list[i] && tanhActivation!(a[i])
			end
		end
		gemv!('N', 1.0f0, thetas[end], a[end-1], 1.0f0, a[end])
	end
end

#computes the forward pass of the network without any checks, allowing optionality to provide a list of Bools showing which if any layers are not active
function forwardNOGRAD!(a::Vector{Matrix{Float32}}, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, X, resLayers::Int64, activation_list::AbstractVector{B}; input_orientation::Char = 'N') where B <: Bool
	isempty(activation_list) && return forwardNOGRAD_base!(a, thetas, biases, X, resLayers; input_orientation = input_orientation)
	
	l = length(thetas)
	m = size(X, ifelse(input_orientation == 'N', 1, 2))
	fillAs!(a, biases, m)
	gemm!(input_orientation, 'T', 1.0f0, X, thetas[1], 1.0f0, a[1])

	if l > 1
		activation_list[1] && tanhActivation!(a[1])
		if (l-1) > 1
			@inbounds for i = 2:l-1
				gemm!('N', 'T', 1.0f0, a[i-1], thetas[i], 1.0f0, a[i])
				if (resLayers != 0) && ((i - 1) % resLayers) == 0
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				activation_list[i] && tanhActivation!(a[i])
			end
		end
		gemm!('N', 'T', 1.0f0, a[end-1], thetas[end], 1.0f0, a[end])
	end
end

#checks that hidden layers are compatible
function forwardNOGRAD!(a::Vector{Array{Float32, N}}, thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, hidden_layers, X, resLayers::Int64=0; activation_list = Vector{Bool}(), kwargs...) where N
	l = length(thetas)
	num_hidden = length(hidden_layers)
	@assert num_hidden == l-1

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert prod(hidden_layers .== first(hidden_layers)) "hidden layers do not share a dimension" 
	end

	forwardNOGRAD!(a, thetas, biases, X, resLayers, activation_list; kwargs...)
end

#calculate forward pass for dataset with an input and output matrix and a cost function that updates a matrix with the some function of the input and output element
function nnCostFunctionNOGRAD(Thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, input_layer_size::Int64, hidden_layers, X, y::Matrix{Float32}, lambda::Float32, a::Vector{Matrix{Float32}}, D::Float32 = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), input_orientation::Char = 'N')
	#Setup some useful variables
	m = input_orientation == 'N' ? size(X, 1) : size(X, 2)
	n = size(y, 2)
	# F = 1.0f0 - D
	
	if occursin("Log", costFunc)
		@assert 2*n == size(a[end], 2)
	else
		@assert n == size(a[end], 2)
	end

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list = activation_list; input_orientation = input_orientation)

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], y, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)
end

#calculate forward pass for dataset with an input matrix, output vector, and output index indicator.  The function output is trying to match the values in the output vector per example at the output index given by the indicator
function nnCostFunctionNOGRAD(Thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, input_layer_size::Int64, hidden_layers, X, y::Vector{Float32}, output_indices::Vector{I}, lambda::Float32, a::Vector{Matrix{Float32}}, D::Float32 = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), input_orientation::Char = 'N', kwargs...) where I <: Integer
	#Setup some useful variables
	m = input_orientation == 'N' ? size(X, 1) : size(X, 2)
	# F = 1.0f0 - D
	
	occursin("Log", costFunc) && error("log cost function is not compatible with output index format")

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers; activation_list = activation_list, input_orientation = input_orientation, kwargs...)

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], y, output_indices, m)

	J = calcJ(m, a[end], lambda, Thetas)
end

function nnCostFunctionNOGRAD(Thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, input_layer_size::Int64, hidden_layers, X, lambda::Float32, a::Vector{Matrix{Float32}}, D::Float32 = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), kwargs...)
	#Setup some useful variables
	(m, n) = size(X)
	# F = 1.0f0 - D
	
	if occursin("Log", costFunc)
		@assert 2*n == size(a[end], 2)
	else
		@assert n == size(a[end], 2)
	end

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers; activation_list = activation_list, kwargs...)

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], X, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)
end

#forward pass where the output gradient is either the output at a particular index or the cross entropy loss of the softmax of the output activations with a desired output index
function nnCostFunctionNOGRAD(Thetas::Vector{Matrix{Float32}}, biases::Vector{Vector{Float32}}, hidden_layers, X, output::Union{Integer, Vector{I}}, lambda::Float32, a::Vector{Array{Float32, N}}, D::Float32 = 0.0f0; resLayers::Int64 = 0, activation_list = fill(true, length(hidden_layers)), loss_type::LossType = OutputIndex(), kwargs...) where {I <: Integer, N}

	num_hidden = length(hidden_layers)

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers; activation_list = activation_list, kwargs...)

	calcFinalOut!(loss_type, a[end], output)

	J = calcJ(a[end], output, lambda, Thetas)
end

function predict!(Thetas, biases, X, a::Vector{Array{Float32, N}}, resLayers::Int64 = 0; kwargs...) where N
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	l = length(Thetas)
	num_hidden = l-1
	hidden_layers = if num_hidden==0
		Vector{Int64}()
	else
		(length(biases[i]) for i in 1:num_hidden)
	end
	#dropout scale factor
	# F = (1.0f0 - D)

	forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers; kwargs...)
end

function predict(Thetas, biases, X, resLayers::Int64 = 0; layerout=length(Thetas), kwargs...) 
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	if N == 1
		a = form_activations(Thetas)
	else
		m = size(X, 1)
		a = form_activations(Thetas, m)
	end
	predict!(Thetas, biases, X, a, resLayers; kwargs...)
	return a[layerout]
end

function predictBatches(Thetas, biases, batches::Vector{Matrix{Float32}}, resLayers::Int64 = 0; layerout=length(Thetas), activation_list = fill(true, length(Thetas)-1))
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
		forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list)
		output[row:row+batchlength-1, :] .= a[layerout]
		row = row+batchlength
		# output[copy(a[layerout])
	end
	return output
end

function predictMulti(multiParams, X::Matrix{Float32}, resLayers::Int64 = 0; layerout=length(multiParams[1][1]), activation_list=fill(true, length(multiParams[1][1])))
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
		forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list)
		copy(a[layerout])
	end
	for params in multiParams]
end

function predictMulti!(multiParams, X::Matrix{Float32}, a, outputs, resLayers::Int64 = 0; activation_list=fill(true, length(multiParams[1][1])))
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
		forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list)
		outputs[i] .= a[end]
	end
	return outputs
end

function predictMultiBatches(multiParams, batches::Vector{Matrix{Float32}}, resLayers::Int64 = 0; layerout=length(multiParams[1][1]), activation_list=fill(true, length(multiParams[1][1])))
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
			forwardNOGRAD!(a, Thetas, biases, hidden_layers, X, resLayers, activation_list=activation_list)
			return copy(a[layerout])
		end
	end
	for params in multiParams]
end

function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::AbstractVector{I}, X::Matrix{Float32}, y::Matrix{Float32},lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), input_orientation::Char = 'N') where I <: Integer

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert all(h == hidden_layers[1] for h in hidden_layers) "hidden layers do not share a dimension" 
	end


	#Setup some useful variables
	mdim = input_orientation == 'N' ? 1 : 2
	m = size(X, mdim)
	n = size(y, 2)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!(input_orientation, 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				tanhGradient!(a[1], tanh_grad_z[1])
			else
				tanhGradient!(a[1], tanh_grad_z[1], D)
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
						tanhGradient!(a[i], tanh_grad_z[i])
					else
						tanhGradient!(a[i], tanh_grad_z[i], D)
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


	gemm!('T', input_orientation, 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]
end

function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, hidden_layers::AbstractVector{I}, X::Matrix{Float32}, y::Vector{Float32}, indices::Vector{I}, lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), input_orientation::Char = 'N') where I <: Integer

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert all(h == hidden_layers[1] for h in hidden_layers) "hidden layers do not share a dimension" 
	end

	mdim = input_orientation == 'N' ? 1 : 2
	#Setup some useful variables
	m = size(X, mdim)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!(input_orientation, 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				tanhGradient!(a[1], tanh_grad_z[1])
			else
				tanhGradient!(a[1], tanh_grad_z[1], D)
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
						tanhGradient!(a[i], tanh_grad_z[i])
					else
						tanhGradient!(a[i], tanh_grad_z[i], D)
					end
				else
					noactivationGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	#mean abs error cost function
	calcDeltaOut!(costFuncDerivs[costFunc], deltas[end], a[end], y, indices, m)

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


	gemm!('T', input_orientation, 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]
end

function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::AbstractVector{I}, X::Matrix{Float32}, lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr", resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), input_orientation::Char = 'N') where I <: Integer

	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert all(h == hidden_layers[1] for h in hidden_layers) "hidden layers do not share a dimension" 
	end


	#Setup some useful variables
	if input_orientation == 'N'
		(m, n) = size(X)
	else
		(n, m) = size(X)
	end
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!(input_orientation, 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				tanhGradient!(a[1], tanh_grad_z[1])
			else
				tanhGradient!(a[1], tanh_grad_z[1], D)
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
						tanhGradient!(a[i], tanh_grad_z[i])
					else
						tanhGradient!(a[i], tanh_grad_z[i], D)
					end
				else
					noactivationGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end

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


	gemm!('T', input_orientation, 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]
end

#output is either an index or list of indices.  Cost function is either the output at the index or the cross entropy loss of the softmax of the output vector with the desired output index
function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, hidden_layers::AbstractVector{I}, X::Matrix{Float32}, output::Union{Integer, Vector{Int64}}, lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), loss_type::LossType = OutputIndex(), input_orientation::Char = 'N') where I <: Integer
	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert all(h == hidden_layers[1] for h in hidden_layers) "hidden layers do not share a dimension" 
	end


	#Setup some useful variables
	if input_orientation == 'N'
		(m, input_size) = size(X)
	else
		(input_size, m) = size(X)
	end
	(m2, output_size) = size(a[end])
	
	@assert m == m2

	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!(input_orientation, 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				tanhGradient!(a[1], tanh_grad_z[1])
			else
				tanhGradient!(a[1], tanh_grad_z[1], D)
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
						tanhGradient!(a[i], tanh_grad_z[i])
					else
						tanhGradient!(a[i], tanh_grad_z[i], D)
					end
				else
					noactivationGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end
	
	#this fixes whatever is going wrong with calcDeltaOut! below.  Somehow not all the values of deltas[end] are getting set to 0
	# deltas[end] .= 0f0
	# if isinteger(output)
	# 	deltas[end][:, output] .= 1f0
	# else
	# 	deltas[end][:, output] .= 1f0 
	# end

	calcDeltaOut!(loss_type, deltas[end], a[end], output)	

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


	gemm!('T', input_orientation, 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]
end

#Single example cost functoin with output as an index.  Cost function is either the output at the index or the cross entropy loss of the softmax of the output vector with the desired output index
function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, hidden_layers::AbstractVector{I}, x, output::Integer, lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Vector{Float32}, 1}, a::Array{Vector{Float32}, 1}, deltas::Array{Vector{Float32}, 1}, D = 0.0f0; resLayers::Int64 = 0, activation_list::AbstractVector{Bool} = fill(true, length(hidden_layers)), loss_type::LossType = OutputIndex()) where I <: Integer
	num_hidden = length(hidden_layers)

	if resLayers != 0
		@assert num_hidden > 1 "Must have at least two hidden layers"
		@assert ((num_hidden - 1) % resLayers) == 0 "The length of hidden_layers - 1 ($(num_hidden-1)) is not a multiple of the number of residual layers ($resLayers)"
		@assert all(h == hidden_layers[1] for h in hidden_layers) "hidden layers do not share a dimension" 
	end

	#Setup some useful variables
	input_size = length(x)
	output_size = length(a[end])
	
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end

	fillAs!(a, biases)

	gemv!('N', 1.0f0, Thetas[1], x, 1.0f0, a[1])

	if length(Thetas) > 1
		if activation_list[1]
			if D == 0.0f0
				tanhGradient!(a[1], tanh_grad_z[1])
			else
				tanhGradient!(a[1], tanh_grad_z[1], D)
			end
		else
			noactivationGradient!(a[1], tanh_grad_z[1], D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				gemv!('N', 1.0f0, Thetas[i], a[i-1], 1.0f0, a[i])
				if (resLayers != 0) && (((i - 1) % resLayers) == 0)
					#calculate residual skip every resLayers layers past the first hidden layer
					axpy!(1.0f0, a[i-resLayers], a[i])
				end
				
				if activation_list[i]
					if D == 0.0f0
						tanhGradient!(a[i], tanh_grad_z[i])
					else
						tanhGradient!(a[i], tanh_grad_z[i], D)
					end
				else
					noactivationGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemv!('N', 1.0f0, Thetas[end], a[end-1], 1.0f0, a[end])
	end
	
	calcDeltaOut!(loss_type, deltas[end], a[end], output)	

	# println("deltas[end] CPU is $(deltas[end])")

	i = num_hidden
	
	while i >= 1
		gemm!('N', 'T', 1.0f0, deltas[i+1], a[i], lambda, Theta_grads[i+1])
		copy!(Bias_grads[i+1], deltas[i+1])
		# gemv!('T', 1.0f0/m, deltas[i+1], onesVec, 0.0f0, Bias_grads[i+1])
		#deltas[i] = (deltas[i+1]*Thetas[i+1]) .* tanh_grad_z[i]
		if (resLayers != 0) && ((i <= (num_hidden - resLayers)) && (((i + resLayers - 1) % resLayers) == 0))
			#replace deltas[i] with deltas[i+resLayers]
			# scal!(length(deltas[i]), 0.0f0, deltas[i], 1)
			# axpy!(1.0f0, deltas[i+resLayers], deltas[i]) 
			blascopy!(length(deltas[i]), deltas[i+resLayers], 1, deltas[i], 1)
			#propagate derivative back to deltas from the original input to the residual layers
			gemv!('T', 1.0f0, Thetas[i+1], deltas[i+1], 1.0f0, deltas[i])
			# gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i])
		else
			gemv!('T', 1.0f0, Thetas[i+1], deltas[i+1], 0.0f0, deltas[i]) #do part 1 of line 1 in place
		end
		finishDelta!(deltas[i], tanh_grad_z[i]) #do part 2 of line 1 in place
		i = i - 1
	end

	gemm!('N', 'T', 1.0f0, deltas[1], x, lambda, Theta_grads[1])
	copy!(Bias_grads[1], deltas[1])
	# gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
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