function params2Theta(input_layer_size, hidden_layer_size, output_layer_size, nn_params)
#convert a vector of nn parameters into the matrices used for the predict
#function based on the input and hidden layer sizes.

num_hidden = length(hidden_layer_size)
Thetas = Array{Matrix{Float32}}(undef, num_hidden+1)
Biases = Array{Vector{Float32}}(undef, num_hidden+1)
Theta_elements = zeros(num_hidden+1)

if num_hidden > 0
	Theta_elements = input_layer_size*hidden_layer_size[1]
	Biases[1] = nn_params[1:hidden_layer_size[1]]
	Thetas[1] = reshape(nn_params[hidden_layer_size[1] + 1:hidden_layer_size[1] + Theta_elements], hidden_layer_size[1], input_layer_size)
	currentIndex = hidden_layer_size[1] + Theta_elements

	if num_hidden > 1
		for i = 2:num_hidden
			Theta_elements = hidden_layer_size[i-1] * hidden_layer_size[i]
			Biases[i] = nn_params[currentIndex+1:currentIndex+hidden_layer_size[i]]
			Thetas[i] = reshape(nn_params[currentIndex+hidden_layer_size[i]+1:currentIndex+hidden_layer_size[i]+Theta_elements], hidden_layer_size[i], hidden_layer_size[i-1])
			currentIndex = currentIndex+hidden_layer_size[i]+Theta_elements
		end
	end
	Theta_elements = hidden_layer_size[num_hidden]*output_layer_size
	Biases[num_hidden+1] = nn_params[currentIndex+1:currentIndex+output_layer_size]
	Thetas[num_hidden+1] = reshape(nn_params[currentIndex+output_layer_size+1:currentIndex+output_layer_size+Theta_elements], output_layer_size, hidden_layer_size[num_hidden])
else
	Biases[1] = nn_params[1:output_layer_size]
	Thetas[1] = reshape(nn_params[output_layer_size+1:end], output_layer_size, input_layer_size)
end

return Thetas, Biases
end

function theta2Params(Biases, Thetas)
#convert Theta and Bias arrays to vector of parameters
l = length(Thetas)
nn_params = []
for i = 1:l
	nn_params = [nn_params; Biases[i]; Thetas[i][:]]
end

return map(Float32, nn_params)
end

function makeorthonormalrand(n::Integer, m::Integer)
	if min(n, m) == 1
		randn(n, m)
	else
		Q = rand(Haar(1), max(n, m))
		Q = Q[1:n, 1:m]
		Q .* sqrt(max(n, m)) #rescale to 1 variance because var(Q)=1/max(n, m)
	end
end

function initializeParams(input_layer_size, hidden_layers, output_layer_size, reslayers=0, use_μP = false)
	num_hidden = length(hidden_layers)
	scale = (reslayers == 0) ? 1 : num_hidden/(reslayers+1)
	Thetas = Array{Matrix{Float32}}(undef, num_hidden+1)
	Biases = Array{Vector{Float32}}(undef, num_hidden+1)
	if num_hidden > 0
		Thetas[1] = map(Float32, randn(hidden_layers[1], input_layer_size) .* (scale*input_layer_size)^(-.5f0))
		Biases[1] = map(Float32, randn(hidden_layers[1]) .* (scale*input_layer_size)^(-.5f0))
		
		if num_hidden > 1
			for i in 2:num_hidden
				Thetas[i] = map(Float32, randn(hidden_layers[i], hidden_layers[i-1]) .* (scale*hidden_layers[i-1])^(-.5f0))
				Biases[i] = map(Float32, randn(hidden_layers[i]) .* (scale*hidden_layers[i-1])^(-.5f0))
			end
		end
		c = scale*hidden_layers[num_hidden]
		f = use_μP ? 1.0f0 / c : c^(-0.5f0)
		Thetas[num_hidden+1] = map(Float32, randn(output_layer_size, hidden_layers[num_hidden]) .* f)
		Biases[num_hidden+1] = map(Float32, randn(output_layer_size) .* (scale*hidden_layers[num_hidden])^(-.5f0))
	else
		Thetas[1] = randn(Float32, output_layer_size, input_layer_size)*input_layer_size^(-.5f0)
		Biases[1] = randn(Float32, output_layer_size)*input_layer_size^(-0.5f0)
	end
	return Thetas, Biases
end

function initializeparams_saxe(input_layer_size, hidden_layers, output_layer_size, reslayers=0; use_μP = false)
	num_hidden = length(hidden_layers)
	scale = (reslayers == 0) ? 1 : num_hidden/(reslayers+1) + 1
	Thetas = Array{Matrix{Float32}}(undef, num_hidden+1)
	Biases = Array{Vector{Float32}}(undef, num_hidden+1)
	if num_hidden > 0
		Thetas[1] = map(Float32, makeorthonormalrand(hidden_layers[1], input_layer_size) .* (scale*input_layer_size)^(-.5f0))
		Biases[1] = zeros(Float32, hidden_layers[1])
			if num_hidden > 1
				for i in 2:num_hidden
					Thetas[i] = map(Float32, makeorthonormalrand(hidden_layers[i], hidden_layers[i-1]) .* (scale*hidden_layers[i-1])^(-.5f0))
					Biases[i] = zeros(Float32, hidden_layers[i])
				end
			end
		c = scale*hidden_layers[num_hidden]
		f = use_μP ? 1.0f0 / c : c^(-0.5f0)
		Thetas[num_hidden+1] = map(Float32, makeorthonormalrand(output_layer_size, hidden_layers[num_hidden]) .* f)
		Biases[num_hidden+1] = zeros(Float32, output_layer_size)
	else
		Thetas[1] = Float32.(makeorthonormalrand(output_layer_size, input_layer_size)*input_layer_size^(-.5f0))
		Biases[1] = zeros(Float32, output_layer_size)
	end
	return Thetas, Biases
end

#initialize parameters overwriting existing parameters
function initializeparams_saxe!(Thetas, Biases, reslayers=0; use_μP = false)
	num_hidden = length(Thetas) - 1
	scale = (reslayers == 0) ? 1 : num_hidden/(reslayers+1) + 1
	if num_hidden > 0
		Thetas[1] .= map(Float32, makeorthonormalrand(size(Thetas[1])...) .* (scale*size(Thetas[1], 2))^(-.5f0))
		Biases[1] .= zero(Float32)
			if num_hidden > 1
				for i in 2:num_hidden
					Thetas[i] .= map(Float32, makeorthonormalrand(size(Thetas[i])...) .* (scale*size(Thetas[i], 2))^(-.5f0))
					Biases[i] .= zero(Float32)
				end
			end
		c = scale*size(Thetas[end], 2)
		f = use_μP ? 1.0f0 / c : c^(-0.5f0)
		Thetas[num_hidden+1] .= map(Float32, makeorthonormalrand(size(Thetas[num_hidden+1])...) .* f)
		Biases[num_hidden+1] .= zero(Float32)
	else
		Thetas[1] .= Float32.(makeorthonormalrand(size(Thetas[1])...)*size(Thetas[1], 2)^(-.5f0))
		Biases[1] .= zeros(Float32)
	end
	return Thetas, Biases
end

function getNumParams(M, H, O)
	num = 0
	if length(H) > 0
		num += M*H[1] + H[1]
		if length(H) > 1
			for i = 2:length(H)
				num += H[i-1]*H[i] + H[i]
			end
		end
		num += H[end]*O + O
	else
		num += M*O+O
	end
	return num
end

#estimate the memory usage for inference given parameters T, B, and the number
#of examples in each batch m
function estPredictMemUsage(T, B, m, numType = Float32)
	lengthA = mapreduce(t -> size(t, 1)*m, +, T)
	sizeof(numType)*lengthA
end

#get the maximum batch size that will fit in remaining free memory for an inference task
#with parameters T, B.  free is an integer representing the number of available bytes
function getMaxBatchSize(T, B, free, numType=Float32)
	neuronCount = mapreduce(t -> size(t, 1), +, T)
	floor(Int64, max(0, free)/(sizeof(numType)*neuronCount))
end

#get the maximum batch size that will fit in remaining GPU free memory for an inference task
#with parameters T, B.  free is an integer representing the number of available bytes
function getMaxGPUBatchSize(T, B, free, numType=Float32)
	neuronCount = mapreduce(t -> size(t, 1), +, T)
	#also add allocation for the new input and output batches which need to be moved to the GPU
	#account for 2 sets of output batches as well for error calculations
	floor(Int64, max(0, free)/(sizeof(numType)*(neuronCount + size(T[1], 2) + 2*length(B[end]))))
end

#given a fixed M and O.  Try to generate networks with different numbers of hidden layers
#that contain a set number of total parameters.  For simplicity assume each hidden layer 
#has the same number of neurons.

#Let's say we want P total parameters with L total layers.  What is H?  where H is the size
#of each hidden layer.  P = MxH + H + (L-1)*(HxH + H) + HxO + O
#H^2(L-1) + H(M+1+L-1+O) + (O-P) = 0
#H^2(L-1) + H(M+L+O) + (O-P) = 0
#H = (-(M+L+O) +- sqrt((M+L+O)^2 - 4x(L-1)(O-P)))/(2x(L-1))
#If L = 1, P = MxH + H + HxO + O -> H = (P - O)/(M+1+O)
function getHiddenSize(M, O, L, P)
	if L == 0
		0
	elseif L == 1
		(P-O)/(M+O+1)
	else
		a = (L - 1)
		b = (M + L + O)
		c = (O - P)
		proot = (-b + sqrt(b^2 - 4*a*c))/(2*a)
	end
end

function getNetworkDims(T::Array{Array{Float32,2},1}, B::Array{Array{Float32,1},1})
#return M (input layer size), H (hidden layer sizes), and O (output layer size)
#from a tuple of network parameters
	M = size(T[1], 2)
	H = if length(B) > 1
		map(b -> length(b), B[1:end-1])
	else
		[]
	end
	O = length(B[end])
	return (M, H, O)
end

function writeparams!(f::IO, params)
	for P in params
		(T, B) = (P[1], P[2])
		nnParams = theta2Params(B, T)
		(M, H, O) = getNetworkDims(T, B)
		n = length(H)
		write(f, n)
		#println(string("Writing hidden layer length: ", n))
		write(f, M)
		if n != 0
			write(f, H)
		end
		write(f, O)
		write(f, nnParams)
	end
end

function writeParams(params::Array{Tuple{Array{Array{Float32,2},1},Array{Array{Float32,1},1}},1}, filename::String)
#write a list of NN parameters in binary representation to file.  There is a leading set of integers which will help
#the read function determine how to reconstruct the T and B arrays.  The encoding is as follows: first value N indicates
#how many hidden layers are in the network.  If n = 1 then 3 integers will preceed the parameters namely, M, H, and O.  
#Larger values of n will mean more preceeding ints need to be read.
	if isfile(filename)
		rm(filename)
	end
	f = open(filename, "a")
	writeparams!(f, params)
	close(f)
end

function writeArray(input::Array{Float32,2}, filename::String)
#write a Float32 array in binary representation to file.  There is a leading set of integers which will help
#the read function determine how to reconstruct the array.  The encoding is as follows: first value N indicates
#how many rows are in the array.  The second value M indicates how many columns are in the array.  All of the
#following lines are the data.
	if isfile(filename)
		rm(filename)
	end
	f = open(filename, "a")
	(N, M) = size(input)
	write(f, N)
	write(f, M)
	if N*M != 0
		write(f, input)
	end
	close(f)
end

function readBinParams(filename::String)
	#read parameters in binary form from a file.  File may contain more than one
	#set of parameters in general. 
	f = open(filename)
	out = readBinParams(f)
	close(f)
	return out
end

function readBinParams(f::IO)
	#get length of params array
	n = read(f, Int64)
	M = read(f, Int64)
	H = if n != 0
		read!(f, Array{Int64}(undef, n)) # read(f, Int64, n)
	else
		[]
	end
	O = read(f, Int64)
	println(string("Got the following network dimensions: ", M, H, O))
	l = getNumParams(M, H, O)
	println(string("Reading ", l, " parameters"))
	nnParams = read!(f, Array{Float32}(undef, l)) # read(f, Float32, l)
	(T, B) = params2Theta(M, H, O, nnParams)
	out = [(T, B)]
	while !eof(f)
		n = read(f, Int64)
		M = read(f, Int64)
		H = if n != 0
			read!(f, Array{Int64}(undef, n)) #read(f, Int64, n)
		else
			[]
		end
		O = read(f, Int64)
		l = getNumParams(M, H, O)
		nnParams = read!(f, Array{Float32}(undef, l)) #read(f, Float32, l)
		(T, B) = params2Theta(M, H, O, nnParams)
		out = [out; (T, B)]
	end
	return out
end

function readBinArray(filename::String)
#read array in binary form from a file.  
	f = open(filename)
	#get length of params array
	N = read(f, Int64)
	M = read(f, Int64)
	# println(string("Got the following array dimensions: ", N, " rows ", M, " columns"))
	out =read!(f, Array{Float32}(undef, N, M)) #read(f, Float32, N, M)
end

function readInput(name)
	X = Float32.(readdlm(string("Xtrain_", name, ".csv")))
	Xtest = Float32.(readdlm(string("Xtest_", name, ".csv")))
	Y = Float32.(readdlm(string("ytrain_", name, ".csv")))
	Ytest = Float32.(readdlm(string("ytest_", name, ".csv")))
	(X, Xtest, Y, Ytest)
end

function readBinInput(name)
	X = readBinArray(string("Xtrain_", name, ".bin"))
	Xtest = readBinArray(string("Xtest_", name, ".bin"))
	Y = readBinArray(string("ytrain_", name, ".bin"))
	Ytest = readBinArray(string("ytest_", name, ".bin"))
	(X, Xtest, Y, Ytest)
end

