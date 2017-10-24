using Base.LinAlg.BLAS
import Base.BLAS: gemv!

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
function normLogLikelihoodErr(a1, a2, y)
	0.5f0*exp(2*a2)*a1^2 - exp(2*a2)*a1*y + 0.5f0*exp(2*a2)*y*y - a2 + 0.9189385332
	#0.5f0*(a2^2)*(a1-y)^2 - log(abs.(a2)) + 0.9189385332f0
end

function normLogLikelihoodDeriv(a1, a2, y)
	((a1 - y)*exp(2*a2), exp(2*a2)*(y-a1)^2 - 1)
end 

#the exponential of a2 is the scale parameter, this ensures it is always positive
function cauchyLogLikelihoodErr(a1, a2, y)
	exp(a2)*abs(a1-y) - log(0.5f0*exp(a2))
end

function cauchyLogLikelihoodDeriv(a1, a2, y)
	(exp(a2)*ifelse(a1 > y, 1.0f0, ifelse(a1 < y, -1.0f0, 0.0f0)), exp(a2)*abs(a1-y) - 1)
end


#names, functions, and function derivatives must all be in order here
costFuncList = [absErr, sqErr, normLogLikelihoodErr, cauchyLogLikelihoodErr]
costFuncDerivsList = [absErrDeriv, sqErrDeriv, normLogLikelihoodDeriv, cauchyLogLikelihoodDeriv]
costFuncNames = ["absErr", "sqErr", "normLog", "cauchyLog"]
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

function tanhGradient!(z::Matrix{Float32}, tanh_grad_z::Matrix{Float32}, D::Float32)
	l = length(z)
	@inbounds for i = 1:l
		tanh_grad_z[i] = Float32(rand() > D)
	end

	@inbounds @simd for i = 1:l	
		z[i] = tanh_grad_z[i]*1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end

	@inbounds @simd for i = 1:l 
		tanh_grad_z[i] = tanh_grad_z[i]*1.7159f0 * (1.0f0 - z[i]*z[i]/(1.7159f0*1.7159f0)) * 2.0f0 / 3.0f0
	end	
end

function tanhActivation!(z::Matrix{Float32})
	@simd for i = 1:length(z)
		@inbounds z[i] = 1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end
end
	
function predict(Thetas, biases, X, D = 0.0f0)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D)

	# Useful values
	m = size(X, 1)
	l = length(Thetas)
	n = size(Thetas[end], 1)

	#dropout scale factor
	F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(l)

	for i = 1:l
		a[i] = Array{Float32}(m, size(Thetas[i], 1))
	end

	#a[1] = X * Thetas[1]' .+ biases[1]'
	#applyBias!(a[1], X*Thetas[1]', biases[1], m, length(biases[1]))

	for ii = 1:l
		for i = 1:length(biases[ii])
			a[ii][:, i] = fill(biases[ii][i], m)
		end
	end

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if l > 1
		tanhActivation!(a[1])

		if (l-1) > 1
			for i = 2:l-1
				gemm!('N', 'T', F, a[i-1], Thetas[i], 1.0f0, a[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', F, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	return a[end]

end



function nnCostFunction(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::Vector, X::Matrix{Float32}, y::Matrix{Float32},lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr")

	num_hidden = length(hidden_layers)


	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	fillAs!(a, biases, m)

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
		if D == 0.0f0
			tanhGradient!(a[1], tanh_grad_z[1])
		else
			tanhGradient!(a[1], tanh_grad_z[1], D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
				if D == 0.0f0
					tanhGradient!(a[i], tanh_grad_z[i])
				else
					tanhGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	#mean abs error cost function
	calcDeltaOut!(costFuncDerivs[costFunc], deltas[end], a[end], y, m, n)


	i = num_hidden
	if num_hidden > 0
		while i >= 1
			#deltas[i] = (deltas[i+1]*Thetas[i+1]) .* tanh_grad_z[i]
			gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i]) #do part 1 of line 1 in place
			finishDelta!(deltas[i], tanh_grad_z[i]) #do part 2 of line 1 in place
			i = i - 1
		end
	end

	gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])

	gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	#Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]

	if num_hidden > 0
		for i = 2:num_hidden+1
			gemm!('T', 'N', 1.0f0/m, deltas[i], a[i-1], lambda/m, Theta_grads[i])
			gemv!('T', 1.0f0/m, deltas[i], onesVec, 0.0f0, Bias_grads[i]) #calculate below line in place
			#Bias_grads[i] = (ones(Float32, 1, m)*deltas[i]/m)[:]
		end
	end

end

function nnCostFunctionNOGRAD(Thetas, biases, input_layer_size, hidden_layers, X, y, lambda, a, D = 0.0f0; costFunc = "absErr")

	num_hidden = length(hidden_layers)


	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	F = 1.0f0 - D

	fillAs!(a, biases, m)

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
	#case for a network with at least 1 hidden layer
		tanhActivation!(a[1])

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', F, a[i-1], Thetas[i], 1.0f0, a[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', F, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], y, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)

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