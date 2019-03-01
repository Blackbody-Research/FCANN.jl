# using Base.LinAlg.BLAS
using LinearAlgebra.BLAS
# import Base.BLAS: gemv!

e = 1f-4

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

#the exponential of a2 is the inverse of the scale parameter, this ensures it is always positive
#exp(a2)=-1/b where b is the scale parameter => b = -1/exp(a2) and => log(exp(a2)/2) = log(-1/2b)
#-abs(x-u)/b - log(2b) = -abs(x-u)*exp(a2)+log(0.5*exp(a2))
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

function layerNorm!(a, g, b)
	#simplest version of code
	(N, M) = size(a)
	if M > 1
		u = mean(a, dims = 2)[:]
		# s = if M > 2
		# 	std(a, dims = 2, corrected = false)[:]
		# else
		# 	fill(1.0f0, N)
		# end
		s = sqrt.(mean(a .^2, dims = 2)[:])
		# println("a = $a")
		println("s = $s")
		println("prenorm a = $a")
		# a .= (g' .* (a .- u) ./ (s .+ e)) .+ b'
		a .= ((g' .*a) ./s) .+ b'
		println("layer norm a = $a")
	else
		u = fill(0.0f0, N)
		s = fill(NaN, N)
		a .= g[1] .*a .+ b[1]
	end


	# 	# s = std(a, dims = 2)[:]
		
	# 	# println(u, s)

	# 	# g' .* (a .- u) ./ s .+ b'

	# 	#calculate u and s through a single pass of a

	# 	#once I have u and s want to iterate through a just once
	# 	# println(N, M)
	# 	for j in 1:M
	# 		@simd for i in 1:N
	# 			# z = (a[i, j] - u[i]) / s[i]
	# 			# a[i, j] =  (g[j] * z) + b[j]
	# 			@inbounds a[i, j] = (g[j] * (a[i, j] - u[i]) / (s[i] + e)) + b[j]
	# 		end
	# 	end
	# else
	# 	u = fill(0.0f0, N)
	# 	s = fill(NaN, N)
	# 	@simd for i in 1:N
	# 		@inbounds a[i] = g[1]*(a[i] - u[i]) + b[1]
	# 	end
	# end

	return (u, s)
end

function layerNorm!(a, g, b, u, s)
	#used for treating u and s as a constant for gradient checking
	# u = mean(a, dims = 2)
	# s = std(a, dims = 2)
	
	# println(u, s)

	# g' .* (a .- u) ./ s .+ b'

	#calculate u and s through a single pass of a

	#once I have u and s want to iterate through a just once
	(N, M) = size(a)
	if M > 1
		# println(N, M)
		for j in 1:M
			@simd for i in 1:N
				# z = (a[i, j] - u[i]) / s[i]
				# a[i, j] =  (g[j] * z) + b[j]
				@inbounds a[i, j] = (g[j] * (a[i, j] - u[i]) / (s[i] + e)) + b[j]
			end
		end
	else
		@simd for i in 1:N
			@inbounds a[i] = g[1]*(a[i] - u[i]) + b[1]
		end
	end
end

function layerNormGrad!(a, agrad, z, g, b)
	#simplest version of code
	

	# out = g' .* (a .- u) ./ s .+ b'
	# z = (a .- u) ./ s
	# agrad = F * g' ./ s

	#calculate u and s through a single pass of a

	#once I have u and s want to iterate through a just once
	(N, M) = size(a)
	
	if M > 1
		# u = mean(a, dims = 2)
		# s = if M > 2
		# 	std(a, dims = 2, corrected = false)[:]
		# else
		# 	fill(1.0f0, N)
		# end
		u = mean(a .^2, dims = 2)[:]
		s = sqrt.(u)
		println("u = $u")
		println("s = $s")
		println("g = $g")
		println("b = $b")

		eye = Matrix{Float32}(M, M)
		# z .= (a .- u) ./ (s .+ e)
		z .= a ./ s

		# if M == 2
		# 	agrad .= (M .- 1) .* g' ./ (M .* (s .+ e))
		# else
		# 	agrad .= (g' ./ (M .* (s .+ e))) .* (M .- 1 .- z.^2)
		# end
		agrad .= (g' ./ s) .* (1 .- ((z .^2) ./ M))
		# agrad .= (a'*a) ./ (M .* u)
		println("agrad = $agrad")
		println("prenorm a = $a")
		a .= (g' .* z) .+ b'
		println("layer norm a = $a")
	else
		z .= a
		agrad .= g[1]
		a .= g[1].*z .+ b[1]
	end

	# if M > 1
	# 	#store the row means in the last column of z
	# 	for j in 1:M
	# 		@simd for i in 1:N
	# 			@inbounds z[i, M] += a[i, j]/M
	# 		end
	# 	end

	# 	#store the variance in the last column of agrad only if M > 2
	# 	if M > 2
	# 		for j in 1:M
	# 			@simd for i in 1:N
	# 				@inbounds agrad[i, j] = ((a[i,j] - z[i, M])^2)/M
	# 			end
	# 		end
	# 	end

	# 	#update last column of agrad to store the inverse std or 1 if M = 2
	# 	if M == 2
	# 		@simd for i in 1:N
	# 			# @inbounds tmp = 1 / (sqrt(agrad[i, M]) + e)
	# 			# @inbounds agrad[i, M] = tmp
	# 			@inbounds agrad[i, M] = 1.0f0
	# 		end
	# 	else	
	# 		@simd for i in 1:N
	# 			# @inbounds tmp = 1 / (sqrt(agrad[i, M]) + e)
	# 			# @inbounds agrad[i, M] = tmp
	# 			@inbounds agrad[i, M] = 1.0f0 / (sqrt(agrad[i, M]) + e)
	# 		end
	# 	end

	# 	# u = mean(a, dims = 2)[:]
	# 	# s = 1 ./(std(a, dims = 2)[:] .+ e)
	# 	for j in 1:M - 1
	# 		# @inbounds z[j] = 0.0f0
	# 		@simd for i in 1:N
	# 			@inbounds agrad[i, j] = agrad[i, M] #s[i] #1/(s[i] + e)
	# 			@inbounds z[i, j] = (a[i, j] - z[i, M])*agrad[i, j] #(a[i, j] - u[i])*agrad[i, j]
	# 			@inbounds agrad[i, j] = g[j]*F*agrad[i, j]
	# 			@inbounds a[i, j] = g[j]*z[i, j] + b[j]
	# 			# @inbounds z[j] += z
	# 		end
	# 		# @inbounds z[j] = z[j] / N
	# 	end

	# 	@simd for i in 1:N
	# 		@inbounds z[i, M] = (a[i, M] - z[i, M])*agrad[i, M] #(a[i, M] - u[i])*agrad[i, M]
	# 		@inbounds agrad[i, M] = g[M]*agrad[i, M]
	# 		@inbounds a[i, M] = g[M]*z[i, M] + b[M]
	# 		# @inbounds z[j] += z
	# 	end
	# else
	# 	@simd for i in 1:N
	# 		@inbounds agrad[i] = 1.0f0
	# 		@inbounds z[i] = a[i]*agrad[i]
	# 		@inbounds agrad[i] = g[1]*agrad[i]
	# 		@inbounds a[i] = g[1]*z[i] + b[1]
	# 		# @inbounds z[j] += z
	# 	end
	# end	
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

function finishLayerNormDelta!(delta, gain)
	(N, M) = size(delta)
	for j in 1:M
		@simd for i = 1:M
			@inbounds delta[i, j] = delta[i] * gain[j]
		end
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

function tanhActivation!(z::Matrix{Float32})
	@simd for i = 1:length(z)
		@inbounds z[i] = 1.7159f0*fast_tanh(2.0f0*z[i]/3.0f0)
	end
end
	
function predict(Thetas, biases, X)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(Thetas)
	n = size(Thetas[end], 1)

	#dropout scale factor
	# F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(Thetas[i], 1))
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
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end
	return a[end]
end

function predict(Thetas, biases, gains, X)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(Thetas)
	n = size(Thetas[end], 1)

	#dropout scale factor
	# F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(Thetas[i], 1))
	end

	#a[1] = X * Thetas[1]' .+ biases[1]'
	#applyBias!(a[1], X*Thetas[1]', biases[1], m, length(biases[1]))

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
	layerNorm!(a[1], gains[1], biases[1])

	if l > 1
		tanhActivation!(a[1])

		if (l-1) > 1
			for i = 2:l-1
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 0.0f0, a[i])
				layerNorm!(a[i], gains[i], biases[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 0.0f0, a[end])
		layerNorm!(a[end], gains[end], biases[end])
	end
	return a[end]
end

function predictBatches(Thetas, biases, batches)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(batches[1], 1)
	l = length(Thetas)
	n = size(Thetas[end], 1)

	#dropout scale factor
	# F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(Thetas[i], 1))
	end

	#a[1] = X * Thetas[1]' .+ biases[1]'
	#applyBias!(a[1], X*Thetas[1]', biases[1], m, length(biases[1]))

	mapreduce(vcat, batches) do X
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
					gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
					tanhActivation!(a[i])
				end
			end

			gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
		end
		return copy(a[end])
	end
end

function predictBatches(Thetas, biases, gains, batches)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(batches[1], 1)
	l = length(Thetas)
	n = size(Thetas[end], 1)

	#dropout scale factor
	# F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(Thetas[i], 1))
	end

	#a[1] = X * Thetas[1]' .+ biases[1]'
	#applyBias!(a[1], X*Thetas[1]', biases[1], m, length(biases[1]))

	mapreduce(vcat, batches) do X
		gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
		layerNorm!(a[1], gains[1], biases[1])
		if l > 1
			tanhActivation!(a[1])

			if (l-1) > 1
				for i = 2:l-1
					gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
					layerNorm!(a[i], gains[i], biases[i])
					tanhActivation!(a[i])
				end
			end

			gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
			layerNorm!(a[end], gains[end], biases[end])
		end
		return copy(a[end])
	end
end

function predictMulti(multiParams, X)
#PREDICT Predict the value of an input given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(X, 1)
	l = length(multiParams[1][1])
	n = size(multiParams[1][1][end], 1)

	#dropout scale factor
	# F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(multiParams[1][1][i], 1))
	end

	#a[1] = X * Thetas[1]' .+ biases[1]'
	#applyBias!(a[1], X*Thetas[1]', biases[1], m, length(biases[1]))
	if length(multiParams[1]) == 2
		[begin
			Thetas = params[1]
			biases = params[2]
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
						gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
						tanhActivation!(a[i])
					end
				end

				gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
			end
			copy(a[end])
		end
		for params in multiParams]
	else
		[begin
			Thetas = params[1]
			biases = params[2]
			gains = params[3]

			gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
			layerNorm!(a[1], gains[1], biases[1])

			if l > 1
				tanhActivation!(a[1])

				if (l-1) > 1
					for i = 2:l-1
						gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 0.0f0, a[i])
						layerNorm!(a[i], gains[i], biases[i])
						tanhActivation!(a[i])
					end
				end

				gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 0.0f0, a[end])
				layerNorm!(a[end], gains[end], biases[end])
			end
			copy(a[end])
		end
		for params in multiParams]
	end
end

function predictMultiBatches(multiParams, batches)
#PREDICT Predict the value of an input in batches given a trained neural network trained with dropout
#factor D.  D is assumed to be 0 by default meaning no dropout.  The incoming weights to neurons
#that had dropout applied to them are scaled by (1-D).  No longer necessary with new dropout cost function
#that applies scaling during training so the network can be used with the same functions
	# Useful values
	m = size(batches[1], 1)
	l = length(multiParams[1][1])
	n = size(multiParams[1][1][end], 1)

	#dropout scale factor
	# F = (1.0f0 - D)

	a = Array{Matrix{Float32}}(undef, l)

	for i = 1:l
		a[i] = Array{Float32}(undef, m, size(multiParams[1][1][i], 1))
	end

	#a[1] = X * Thetas[1]' .+ biases[1]'
	#applyBias!(a[1], X*Thetas[1]', biases[1], m, length(biases[1]))
	[begin
		if length(params) == 2
			mapreduce(vcat, batches) do X
				Thetas = params[1]
				biases = params[2]
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
							gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
							tanhActivation!(a[i])
						end
					end

					gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
				end
				return copy(a[end])
			end
		else
			mapreduce(vcat, batches) do X
				Thetas = params[1]
				biases = params[2]
				gains = params[3]

				gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
				layerNorm!(a[1], gains[1], biases[1])

				if l > 1
					tanhActivation!(a[1])

					if (l-1) > 1
						for i = 2:l-1
							gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 0.0f0, a[i])
							layerNorm!(a[i], gains[i], biases[i])
							tanhActivation!(a[i])
						end
					end

					gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 0.0f0, a[end])
					layerNorm!(a[end], gains[end], biases[end])
				end
				return copy(a[end])
			end
		end
	end
	for params in multiParams]
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
	# F = 1.0f0 - D

	fillAs!(a, biases, m)

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
	#case for a network with at least 1 hidden layer
		tanhActivation!(a[1])

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 1.0f0, a[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 1.0f0, a[end])
	end

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], y, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)

end

function nnCostFunctionLayerNorm(Thetas::Array{Matrix{Float32},1}, biases::Array{Vector{Float32}, 1}, gains::Array{Vector{Float32}, 1}, input_layer_size::Int, hidden_layers::Vector, X::Matrix{Float32}, y::Matrix{Float32},lambda::Float32, Theta_grads::Array{Matrix{Float32}, 1}, Bias_grads::Array{Vector{Float32}, 1},  gain_grads::Array{Vector{Float32}, 1}, tanh_grad_z::Array{Matrix{Float32}, 1}, a::Array{Matrix{Float32}, 1}, agrad::Array{Matrix{Float32}, 1}, z::Array{Matrix{Float32}, 1}, deltas::Array{Matrix{Float32}, 1}, onesVec::Vector{Float32}, D = 0.0f0; costFunc = "absErr")

	num_hidden = length(hidden_layers)


	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	         
	if lambda > 0.0f0
		fillThetaGrads!(Theta_grads, Thetas)
	end


	gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
	layerNormGrad!(a[1], agrad[1], z[1], gains[1], biases[1])
	# println(agrad[1])

	if length(Thetas) > 1
		if D == 0.0f0
			tanhGradient!(a[1], tanh_grad_z[1])
		else
			tanhGradient!(a[1], tanh_grad_z[1], D)
		end

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 0.0f0, a[i])
				layerNormGrad!(a[i], agrad[i], z[i], gains[i], biases[i])
				if D == 0.0f0
					tanhGradient!(a[i], tanh_grad_z[i])
				else
					tanhGradient!(a[i], tanh_grad_z[i], D)
				end
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 0.0f0, a[end])
		layerNormGrad!(a[end], agrad[end], z[end], gains[end], biases[end])
	end

	#mean abs error cost function
	calcDeltaOut!(costFuncDerivs[costFunc], deltas[end], a[end], y, m, n)
	gemv!('T', 1.0f0/m, deltas[end], onesVec, 0.0f0, Bias_grads[end])
	gemv!('T', 1.0f0/m, deltas[end] .* z[end], onesVec, 0.0f0, gain_grads[end])
	gemm!('N', 'N', 1.0f0/m, deltas[end], agrad[end], 1.0f0, eye[end])
	finishLayerNormDelta!(deltas[end], gains[end])
	

	i = num_hidden
	if num_hidden > 0
		gemm!('T', 'N', 1.0f0/m, deltas[end], a[end-1], lambda/m, Theta_grads[end])
		while i >= 1
			#deltas[i] = (deltas[i+1]*Thetas[i+1]) .* tanh_grad_z[i]
			gemm!('N', 'N', 1.0f0, deltas[i+1], Thetas[i+1], 0.0f0, deltas[i]) #do part 1 of line 1 in place
			finishDelta!(deltas[i], tanh_grad_z[i]) #do part 2 of line 1 in place
			gemv!('T', 1.0f0/m, deltas[i] .* z[i], onesVec, 0.0f0, gain_grads[i])
			gemv!('T', 1.0f0/m, deltas[i], onesVec, 0.0f0, Bias_grads[i])
			finishDelta!(deltas[i], agrad[i])
			if i > 1
				gemm!('T', 'N', 1.0f0/m, deltas[i], a[i-1], lambda/m, Theta_grads[i])
			else
				gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
			end
			i = i - 1
		end
	else
		gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	end

	# gemm!('T', 'N', 1.0f0/m, deltas[1], X, lambda/m, Theta_grads[1])
	# gemv!('T', 1.0f0/m, deltas[1] .*z[1], onesVec, 0.0f0, gain_grads[1])

	# gemv!('T', 1.0f0/m, deltas[1], onesVec, 0.0f0, Bias_grads[1]) #calculate below line in place
	# #Bias_grads[1] = (ones(Float32, 1, m)*deltas[1]/m)[:]

	# if num_hidden > 0
	# 	for i = 2:num_hidden+1
	# 		gemm!('T', 'N', 1.0f0/m, deltas[i], a[i-1], lambda/m, Theta_grads[i])
	# 		gemv!('T', 1.0f0/m, deltas[i] .* z[i], onesVec, 0.0f0, gain_grads[i])
	# 		gemv!('T', 1.0f0/m, deltas[i], onesVec, 0.0f0, Bias_grads[i]) #calculate below line in place
	# 		#Bias_grads[i] = (ones(Float32, 1, m)*deltas[i]/m)[:]
	# 	end
	# end

end

function nnCostFunctionLayerNormNOGRAD(Thetas, biases, gains, input_layer_size, hidden_layers, X, y, lambda, a, D = 0.0f0; costFunc = "absErr")

	num_hidden = length(hidden_layers)

	
	Us = Array{Vector{Float32}, 1}(undef, length(a))
	Sigs = Array{Vector{Float32}, 1}(undef, length(a))
	

	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	# F = 1.0f0 - D

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
	# println(a[1][:])
	(Us[1], Sigs[1]) = layerNorm!(a[1], gains[1], biases[1])

	# println(a[1][:]) #after layerNorm! the a's do not change with Tplus vs Tminus, because we must treat u and sig as constants, if you derive them with respect to Theta then the Theta derivatives are always 0 because you have x1 - x1 basically
	# fillAs!(a, biases, m)

	# gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
	#case for a network with at least 1 hidden layer
		tanhActivation!(a[1])

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 0.0f0, a[i])
				(Us[i], Sigs[i]) = layerNorm!(a[i], gains[i], biases[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 0.0f0, a[end])
		(Us[end], Sigs[end]) = layerNorm!(a[end], gains[end], biases[end])
	end

	#mean abs error cost function
	calcFinalOut!(costFuncs[costFunc], a[end], y, m, n)

	J = calcJ(m, n, a[end], lambda, Thetas)
	return (J, Us, Sigs)

end


function nnCostFunctionLayerNormNOGRAD(Thetas, biases, gains, input_layer_size, hidden_layers, X, y, lambda, a, Us, Sigs, D = 0.0f0; costFunc = "absErr")

	num_hidden = length(hidden_layers)

	
	# Us = Array{Vector{Float32}, 1}(undef, length(a))
	# Sigs = Array{Vector{Float32}, 1}(undef, length(a))
	

	#Setup some useful variables
	m = size(X, 1)
	n = size(y, 2)
	# F = 1.0f0 - D

	gemm!('N', 'T', 1.0f0, X, Thetas[1], 0.0f0, a[1])
	# println(a[1][:])
	layerNorm!(a[1], gains[1], biases[1], Us[1], Sigs[1])
	# println(a[1][:]) #after layerNorm! the a's do not change with Tplus vs Tminus, because we must treat u and sig as constants, if you derive them with respect to Theta then the Theta derivatives are always 0 because you have x1 - x1 basically
	# fillAs!(a, biases, m)

	# gemm!('N', 'T', 1.0f0, X, Thetas[1], 1.0f0, a[1])

	if length(Thetas) > 1
	#case for a network with at least 1 hidden layer
		tanhActivation!(a[1])

		if num_hidden > 1
			for i = 2:num_hidden
				gemm!('N', 'T', 1.0f0, a[i-1], Thetas[i], 0.0f0, a[i])
				layerNorm!(a[i], gains[i], biases[i], Us[i], Sigs[i])
				tanhActivation!(a[i])
			end
		end

		gemm!('N', 'T', 1.0f0, a[end-1], Thetas[end], 0.0f0, a[end])
		layerNorm!(a[end], gains[end], biases[end], Us[end], Sigs[end])
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