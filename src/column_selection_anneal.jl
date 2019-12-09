RecordType{T} = Dict{Set{Int64}, T}

function maketimestr(t::AbstractFloat)
	hours = floor(Integer, t / 60 / 60)
	minutes = floor(Integer, t / 60) - hours*60
	seconds = t - (minutes*60) - (hours*60*60)
	secstr = string(round(seconds, digits = 2))
	"$hours:$minutes:$(secstr[1:min(4, length(secstr))])"
end

roundstr(num) = round(num, sigdigits=5)

function printcolsvec(origcolsvec, colsvec, switchind, acc)
	nmax = length(digits(length(colsvec))) #maximum number of digits for indicator column
	emax = 30
	c = 0
	l = 1
	print(repeat(" ", nmax+1)) #add nmax+1 spaces of padding for the row labels
	for i in 1:emax print(string(lpad(i, 2), " ")) end
	println()
	print(repeat(" ", nmax+1))
	for i in eachindex(colsvec)
    	#highlight cells with attempted changes
    	bracketcolor = if i == switchind
    		#if an attempted change is accepted make the brackets blue else red
    		acc ? :blue : :red
    	else
    		#if no change leave green
    		:green
    	end

    	fillstate = ((i == switchind) && acc) ? !colsvec[i] : colsvec[i]
    	#fill cell with X if being used and nothing if not
    	fillchar = fillstate ? 'X' : ' '
    	#if current state differs from original highlight in yellow
    	fillcolor = fillstate == origcolsvec[i] ? :default : :reverse

    	printstyled(IOContext(stdout, :color => true), "[", color = bracketcolor)
    	printstyled(IOContext(stdout, :color => true), fillchar, color = fillcolor)
    	printstyled(IOContext(stdout, :color => true), "]", color = bracketcolor)
        c += 1
        if i != length(colsvec)
	        if c == emax
	        	println()
	        	print(string(lpad(emax*l, nmax), " "))
	        	c = 0
	        	l += 1
	        end
	    else
	    	newcolchange = if acc
	    		colsvec[switchind] ?  -1 : 1
	    	else
	    		0
	    	end
	    	print(string(" ", sum(colsvec) + newcolchange, "/", length(colsvec)))
	    end
    end
    return l
end

function calc_accept_rate(C::Vector{T}) where T <: AbstractFloat
	l = length(C)
	numacc = 0
	for i in 1:(l-1)
		if C[i] != C[i+1]
			numacc += 1
		end
	end
	numacc/(l-1)
end

function extract_record(err_record)
	errs = [a[3] for a in err_record]
	sortind = sortperm(errs)
	errdelt = try 
		quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25) 
	catch 
		0.0 
	end
	bestcols = err_record[sortind[1]][2]
	besterr = err_record[sortind[1]][3]
	(besterr, errdelt, bestcols) 
end

function run_temp_calibrate(model, input_data::Matrix{Float32}, output_data::Matrix{Float32}, input_data_copy::Matrix{Float32}, a, v; seed = 1, colsrecord::RecordType{T} = RecordType{T}(), printiter = true, updateinterval = 2.0, chi = 0.9, multiout=Vector{Matrix{Float32}}()) where T <: AbstractFloat
	Random.seed!(seed)
	
	n = size(input_data, 2)

	initialcols = Set{Int64}()

	if length(model) == 4
		err = calcOutputCPU!(input_data, output_data, model[1], model[2], a, costFunc=model[4], resLayers = model[3])
	elseif length(model) == 3
		_,err,_ = calcMultiOutCPU!(input_data, output_data, model[1], a, multiout, costFunc=model[3], resLayers = model[2])
	else
		error("model input is not a valid format")
	end


	println("Beginning temperature calibration with a target select rate of $chi")
	println("Got error of: $(roundstr(err)) with no shuffled columns")
	println("--------------------------------------------------------------------")
	println("Setting up $n columns for step updates")
	
	println()

	#add line padding for print update if print is turned on
	if printiter
		lines = ceil(Int64, n/30) + 3
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		numsteps = 2*round(Int64, n*log(n))+1
		println("Starting $numsteps steps of sampling process without printing updates")
	end

	push!(colsrecord, initialcols => err)
	calibrate_gibbs_temp(model, input_data, output_data, input_data_copy, a, v, initialcols, colsrecord, err; printstep = printiter, updateinterval = updateinterval, chi = chi, multiout=multiout)
end

function calibrate_gibbs_temp(model, input_data::Matrix{Float32}, output_data::Matrix{Float32}, input_data_copy::Matrix{Float32}, a, v, currentcols, colsrecord::RecordType{T}, currenterr; printstep = true, updateinterval = 2.0, chi = 0.9, multiout=Vector{Matrix{Float32}}()) where T <: AbstractFloat where N
	
	##########Set up initial values and print first iteration##############################
	t_stepstart = time()
	t_iter = 0.0
	accept_rate = 0.0
	dicthitrate = 0.0
	
	n = size(input_data, 2)
	
	l = length(output_data)
	m = 2*round(Int64, n*log(n))
	tsteps = zeros(m+1)
	accs = zeros(Int64, m)
	err_record = Vector{Tuple{String, Set{Int64}, Float64}}(undef, m)
	deltCbar = 0.0
	m1 = 0
	m2 = 0
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	T0 = tsteps[1]

	#########################################Perform first step iteration######################################
	changeind = rand(1:n)
	threshold = rand()
	# threshold = 1
	rmcol = in(changeind, currentcols)
	newcols = if rmcol
		setdiff(currentcols, changeind)
	else
		union(currentcols, changeind)
	end

	if length(model) == 4
		(newerr, acc) = evalnewcol(model[1], model[2], input_data, output_data, input_data_copy, a, v, currentcols, changeind, currenterr, threshold, T0; rng=1, reslayers=model[3], costFunc = model[4])
	elseif length(model) == 3
		(newerr, acc) = evalnewcol(model[1], input_data, output_data, input_data_copy, a, multiout, v, currentcols, changeind, currenterr, threshold, T0; rng=1, reslayers=model[2], costFunc = model[3])
	else
		error("model is not a valid format")
	end

	origcolsvec = [in(c, currentcols) for c in 1:n]
	colsvec = [in(c, newcols) for c in 1:n]

	if printstep
		lines = ceil(Int64, n/30) + 3
		for i in 1:lines
			print("\33[2K\033[A\r")
		end
	end
	printcolsvec(origcolsvec, colsvec, changeind, acc)
	println()
	println("Current error of $(roundstr(currenterr)) after first step of calibration")

	###############################Purge record if insufficient memory#######
	keycheck = haskey(colsrecord, newcols)
	push!(colsrecord, newcols => newerr)
	accept_rate = Float64(acc)
	dicthitrate = Float64(keycheck)
	deltC = newerr - currenterr
	if deltC > 0
		deltCbar += deltC
		m2 += 1
	else
		m1 += 1
	end

	tsteps[2] = deltCbar / log(m2 / (m2*chi - (1-chi)*m1)) / m2
	
	if acc
		accs[1] = 1
		currentcols = newcols
		currenterr = newerr
		if rmcol
			colstr = "-$changeind"
		else
			colstr = "+$changeind"
		end
	else
		colstr = ""
	end

	err_record[1] = (colstr, currentcols, currenterr)
	t_iter = (time() - t_stepstart)
	t_lastprint = time()

	for i in 2:length(tsteps)-1
		#print second iteration and subsequent once every 1 second by default
		printiter = if printstep
			if i == 2
				true
			elseif (time() - t_lastprint) > updateinterval
				true
			else
				false
			end
		else
			false
		end

		t_stepstart = time()
		changeind = rand(1:n)
		threshold = rand()
		# threshold = 1
		
		rmcol = in(changeind, currentcols)
		newcols = if rmcol
			setdiff(currentcols, changeind)
		else
			union(currentcols, changeind)
		end
		keycheck = haskey(colsrecord, newcols)

		if keycheck
			newerr = colsrecord[newcols]
			p = if newerr < currenterr
				1.0
			else
				exp(-(newerr-currenterr)/tsteps[i])
			end

			acc = (p >= threshold)

			if acc
				if rmcol
					#if change is accepted then restore column in data copy to unshuffled version
					view(input_data_copy, :, changeind) .= view(input_data, :, changeind)
				else
					v .= view(input_data, :, changeind)
					#fill shuffle column if it is being added
					shuffle!(MersenneTwister(1234), v)
					view(input_data_copy, :, changeind) .= v
				end
			end
		else
			if length(model) == 4
				(newerr, acc) = evalnewcol(model[1], model[2], input_data, output_data, input_data_copy, a, v, currentcols, changeind, currenterr, threshold, tsteps[i]; rng=1, reslayers=model[3], costFunc = model[4])
			elseif length(model) == 3
				(newerr, acc) = evalnewcol(model[1], input_data, output_data, input_data_copy, a, multiout, v, currentcols, changeind, currenterr, threshold, tsteps[i]; rng=1, reslayers=model[2], costFunc = model[3])
			else
				error("model is not a valid format")
			end
			push!(colsrecord, newcols => newerr)
		end

		dicthitrate = dicthitrate*f + (1-f)*keycheck
		
		accept_rate = f*accept_rate + (1-f)*acc

		deltC = newerr - currenterr
		if deltC > 0
			deltCbar += deltC
			m2 += 1
		else
			m1 += 1
		end

		c = m2 / (m2*chi - (1-chi)*m1)
		tsteps[i+1] = if c <= 0
			0.0
		elseif c <= 1
			Inf 
		else
			deltCbar / log(c) / m2
		end

		if printiter
			lines = ceil(Int64, n/30) + 3
			for i in 1:lines
				print("\33[2K\033[A\r")
			end

			origcolsvec = [in(c, currentcols) for c in 1:n]
			colsvec = [in(c, newcols) for c in 1:n]
			printcolsvec(origcolsvec, colsvec, changeind, acc)
			println()
			println("Current error: $(roundstr(currenterr)), accept rate:  $(roundstr(accept_rate)), dictionary hit rate: $(roundstr(dicthitrate))")
			println("Current temperature = $(roundstr(tsteps[i+1])) after $i steps of calibration out of $m")
		end

		#in second half try resetting values to get more accurate calibration
		if i == round(Int64, length(tsteps)/2) #((i < round(Int64, length(Tsteps)*0.75)) && ((i % N) == 0))
			m1 = 0
			m2 = 0
			deltCbar = 0.0
		end
		
		if acc
			accs[i] = 1
			currentcols = newcols
			currenterr = newerr
			if rmcol
				colstr = "-$changeind"
			else
				colstr = "+$changeind"
			end
		else
			colstr = ""
		end

		err_record[i] = (colstr, currentcols, currenterr)

		t_iter = f*t_iter + (1-f)*(time() - t_stepstart)

		t_lastprint = printiter ? time() : t_lastprint
	end
	(err_record, colsrecord, tsteps, accs)
end



function evalnewcol(T::Vector{Matrix{Float32}}, B::Vector{Vector{Float32}}, input_data::Matrix{Float32}, output_data::Matrix{Float32}, input_data_copy::Matrix{Float32}, a::Vector{Matrix{Float32}}, v, shuffleinds::Set{Int64}, changeind::Int64, currenterr, threshold, temp; rng=1234, reslayers=0, costFunc = "sqErr")
	
	newcol = !in(changeind, shuffleinds)
	if newcol
		v .= view(input_data, :, changeind)
		#fill shuffle column if it is being added
		shuffle!(MersenneTwister(rng), v)
		view(input_data_copy, :, changeind) .= v
	else
		v .= view(input_data_copy, :, changeind)
		view(input_data_copy, :, changeind) .= view(input_data, :, changeind)
	end

	errs = calcOutputCPU!(input_data_copy, output_data, T, B, a, costFunc=costFunc, resLayers = reslayers)

	acc = if errs < currenterr
		1.0
	else
		exp(-(errs-currenterr)/temp)
	end

	
	#restore input_data_copy to initial state if step is rejected
	if acc < threshold
		if newcol
			view(input_data_copy, :, changeind) .= view(input_data, :, changeind)
		else
			view(input_data_copy, :, changeind) .= v
		end
	end

	return errs, (acc >= threshold)
end

function evalnewcol(multiparams, input_data::Matrix{Float32}, output_data::Matrix{Float32}, input_data_copy::Matrix{Float32}, a::Vector{Matrix{Float32}}, multiout::Vector{Matrix{Float32}}, v, shuffleinds::Set{Int64}, changeind::Int64, currenterr, threshold, temp; rng=1234, reslayers=0, costFunc = "sqErr")
	
	newcol = !in(changeind, shuffleinds)
	if newcol
		v .= view(input_data, :, changeind)
		#fill shuffle column if it is being added
		shuffle!(MersenneTwister(rng), v)
		view(input_data_copy, :, changeind) .= v
	else
		v .= view(input_data_copy, :, changeind)
		view(input_data_copy, :, changeind) .= view(input_data, :, changeind)
	end

	_,errs,_ = calcMultiOutCPU!(input_data_copy, output_data, multiparams, a, multiout, costFunc=costFunc, resLayers = reslayers)

	acc = if errs < currenterr
		1.0
	else
		exp(-(errs-currenterr)/temp)
	end

	
	#restore input_data_copy to initial state if step is rejected
	if acc < threshold
		if newcol
			view(input_data_copy, :, changeind) .= view(input_data, :, changeind)
		else
			view(input_data_copy, :, changeind) .= v
		end
	end

	return errs, (acc >= threshold)
end

function run_gibbs_step(model, input_data, output_data, input_data_copy, a, v, currentcols, err_record, colsrecord::RecordType, tsteps; printstep = true, updateinterval = 2.0, accept_rate = 0.0, dicthitrate = 0.0, itertime = 0.0, calibrate = true, multiout=Vector{Matrix{Float32}}())
	
	##############################Initialize Values###############################################
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	t_stepstart = time()
	deltCT = 0.0
	deltST = 0.0
	deltCasum = 0.0
	temprecord = Vector{Tuple{Float64, Float64}}()
	costsequence = Vector{Float64}()
	T0 = calibrate ? Inf : tsteps[1]

	#############################Get dimensions of reg data#####################
	l = size(input_data, 1)
	n = size(input_data, 2)
	m = length(tsteps)

	###############################Get initial values from first iteration#######
	err = colsrecord[currentcols]
	push!(costsequence, err)
	changeind = rand(1:n)
	threshold = rand()
	rmcol = in(changeind, currentcols)
	if rmcol
		newcols = setdiff(currentcols, changeind)
		changestring = "-$changeind"
	else
		newcols = union(currentcols, changeind)
		changestring = "+$changeind"
	end

	if length(model) == 4
		(newerr, acc) = evalnewcol(model[1], model[2], input_data, output_data, input_data_copy, a, v, currentcols, changeind, err, threshold, T0, reslayers=model[3], costFunc = model[4])
	elseif length(model) == 3
		(newerr, acc) = evalnewcol(model[1], input_data, output_data, input_data_copy, a, multiout, v, currentcols, changeind, err, threshold, T0, reslayers=model[2], costFunc = model[3])
	else
		error("model is not a valid format")
	end

	if printstep
		lines = ceil(Int64, n/30) + 3
		for i in 1:lines
			print("\33[2K\033[A\r")
		end
		origcolsvec = [in(c, currentcols) for c in 1:n]
		colsvec = [in(c, newcols) for c in 1:n]
		printcolsvec(origcolsvec, colsvec, changeind, acc)
		println()
		println("Current error of $(roundstr(err)) after no steps")
	end

	
	###############################Purge record if insufficient memory###########
	keycheck = haskey(colsrecord, newcols)
	dicthitrate = Float64(keycheck)
	

	push!(colsrecord, newcols => newerr)
	accept_rate = f*accept_rate + (1-f)*acc
	deltCk = newerr- err
	deltCasum += abs(deltCk)
	
	if acc
		push!(err_record, (changestring, newcols, newerr))
		currentcols = newcols
		err = newerr
		deltCT += deltCk
	end
	push!(costsequence, err)

	t_iter = time() - t_stepstart
	itertime = if itertime == 0
		t_iter
	else
		f*itertime + (1-f)*t_iter
	end
	t_lastprint = time()

	currenttemp = T0 #for initial calibration accept all changes 
	calsteps = calibrate ? 5*length(colsvec) : 0 #number of calibration steps to determine initial temperature

	for i in 2:length(tsteps)
		#print second iteration and subsequent once every 1 second by default
		printiter = if printstep
			if i == 2
				true
			elseif (time() - t_lastprint) > updateinterval
				true
			else
				false
			end
		else
			false
		end
		
		################################Perform gibbs step iteration#############
		t_stepstart = time()

		changeind = rand(1:n)
		threshold = rand()
		
		rmcol = in(changeind, currentcols)
		if rmcol
			newcols = setdiff(currentcols, changeind)
			changestring = "-$changeind"
		else
			newcols = union(currentcols, changeind)
			changestring = "+$changeind"
		end
		keycheck = haskey(colsrecord, newcols)

		if keycheck
			newerr = colsrecord[newcols]
			p = if newerr < err
				1.0
			else
				exp(-(newerr-err)/tsteps[i])
			end

			acc = (p >= threshold)

			if acc
				if rmcol
					#if change is accepted then restore column in data copy to unshuffled version
					view(input_data_copy, :, changeind) .= view(input_data, :, changeind)
				else
					v .= view(input_data, :, changeind)
					#fill shuffle column if it is being added
					shuffle!(MersenneTwister(1234), v)
					view(input_data_copy, :, changeind) .= v
				end
			end
		else
			if length(model) == 4
				(newerr, acc) = evalnewcol(model[1], model[2], input_data, output_data, input_data_copy, a, v, currentcols, changeind, err, threshold, tsteps[i], reslayers=model[3], costFunc = model[4])
			elseif length(model) == 3
				(newerr, acc) = evalnewcol(model[1], input_data, output_data, input_data_copy, a, multiout, v, currentcols, changeind, err, threshold, tsteps[i], reslayers=model[2], costFunc = model[3])
			else
				error("model is not a valid format")
			end
			push!(colsrecord, newcols => newerr)
		end
		
		###############################Purge record if insufficient memory#######
		dicthitrate = dicthitrate*f + (1-f)*keycheck

		accept_rate = f*accept_rate + (1-f)*acc
		deltCk = newerr - err
		deltCasum += abs(deltCk)

		#after calibration steps are done calculate T0 based on average cost variation of steps
		if i == calsteps
			T0 = -deltCasum/(log(0.8)*calsteps)
			deltCT = 0.0 #reset deltCT
			currenttemp = T0
			# println("T0 = $T0")
		end
		
		if acc
			push!(err_record, (changestring, newcols, newerr))
			currentcols = newcols
			err = newerr
		end

		if printiter
			lines = ceil(Int64, n/30) + 3
			for i in 1:lines
				print("\33[2K\033[A\r")
			end

			origcolsvec = [in(c, currentcols) for c in 1:n]
			colsvec = [in(c, newcols) for c in 1:n]
			printcolsvec(origcolsvec, colsvec, changeind, acc)
			println()
			println("Current error: $(roundstr(err)), accept rate:  $(roundstr(accept_rate)), dictionary hit rate: $(roundstr(dicthitrate))")
			if i > calsteps
				println("On step $i of $(length(tsteps)) temperature = $(roundstr(currenttemp)), T0 = $(roundstr(T0)) after $calsteps steps of calibration")
			else
				println("On step $i of $(length(tsteps)) temperature = $(roundstr(currenttemp)), waiting $calsteps steps for T0 calibration")
			end
		end


		if i > calsteps
			#update entropy variation
			if deltCk > 0
				deltST -= deltCk/currenttemp
			end

			if acc 
				push!(temprecord, (currenttemp, newerr))
			else
				push!(temprecord, (currenttemp, err))
			end

			if calibrate
				currenttemp = currenttemp * 0.95^(1000/(length(tsteps)-calsteps))
			else
				currenttemp = tsteps[i]
			end
		end

		push!(costsequence, err)

		itertime = f*itertime + (1-f)*(time() - t_stepstart)

		t_lastprint = printiter ? time() : t_lastprint
	end

	return (err_record, colsrecord, temprecord, costsequence, currentcols)
end

function run_stationary_steps(model, input_data, output_data, input_data_copy, a, v, startingcols, startingerr, initialcolsrecord, initialtemp, C0; seed = 1, delt = delt=0.001, updateinterval=2.0, printiter=true, n = size(input_data, 2), m = m = round(Int64, n*log(n)), multiout=Vector{Matrix{Float32}}())
	
	Random.seed!(seed)
	
	println()
	println("-------------------------------------------------------------------------------------------------------------------")
	println("Beginning quasistatic steps at temperature of $initialtemp with a delta of $delt and equilibrium plateau of $m steps")
	println("-------------------------------------------------------------------------------------------------------------------")
	println()

		
	println("Starting error of: $startingerr shuffling $(length(startingcols))/$n columns")
	println("--------------------------------------------------------------------")
	println("Setting up $n columns for step updates")
	println()
	println()

	tsteps = fill(initialtemp, m)
	
	#add line padding for print update if print is turned on
	lines = ceil(Int64, n/30) + 3
	if printiter
		println("On initial step using starting temperature of $(roundstr(initialtemp))")
		for j in 1:lines-1
			println()
		end
	else
		println("Starting $m steps of sampling process without printing updates")
	end

	(err_record, colsrecord, temprecord, costsequence, currentcols) = run_gibbs_step(model, input_data, output_data, input_data_copy, a, v, startingcols, [("", startingcols, startingerr)], push!(initialcolsrecord, startingcols => startingerr), tsteps, printstep = printiter, updateinterval = updateinterval, calibrate = false, multiout=multiout)
	Cs = mean(costsequence)
	ar = calc_accept_rate(costsequence)

	fulltemprecord = [(initialtemp, Cs, ar)]

	sig = std(costsequence)
	newtemp = initialtemp/(1+(log(1+delt)*initialtemp/(3*sig)))
	dT = newtemp - initialtemp
	dC = Cs-C0
	thresh = abs(dC/dT * newtemp/C0)
	i = 1

	t_report = time()
	while thresh > eps(Cs)
		if time() - t_report > updateinterval
			printcheck = true
			t_report = time()
		else
			printcheck = false
		end
		if printiter && printcheck
			print("\u001b[$(lines+4)F") #move cursor to beginning of lines lines+1 lines up
			print("\u001b[2K") #clear entire line
			println("Reducing temperature from #$(i-1):$(round(initialtemp, sigdigits = 3)) to #$i:$(round(newtemp, sigdigits = 3)) with thresh:$(round(thresh, digits = 3))")
			print("\u001b[$(lines+4)E") #move cursor to beginning of lines lines+1 lines down
		end
		tsteps = fill(newtemp, m)
		initialtemp = newtemp
		(err_record, colsrecord, temprecord, costsequence, currentcols) = run_gibbs_step(model, input_data, output_data, input_data_copy, a, v, currentcols, err_record, colsrecord, tsteps, printstep = printiter, updateinterval = updateinterval, calibrate = false, multiout=multiout)

		ar = calc_accept_rate(costsequence)
		push!(fulltemprecord, (newtemp, mean(costsequence), ar))
		sig = std(costsequence)
		newtemp = initialtemp/(1+(log(1+delt)*initialtemp/(3*sig)))
		dC = mean(costsequence) - Cs
		Cs = mean(costsequence)
		dT = newtemp - initialtemp
		thresh = abs(dC/dT * newtemp/C0)
		i += 1
	end

	colscheck = true
	while colscheck
		if time() - t_report > updateinterval
			printcheck = true
			t_report = time()
		else
			printcheck = false
		end
		if printiter && printcheck
			print("\u001b[$(lines+4)F") #move cursor to beginning of lines lines+1 lines up
			print("\u001b[2K") #clear entire line
			println("Confirming local minimum at a temperature of 0.0")
			print("\u001b[$(lines+4)E") #move cursor to beginning of lines lines+1 lines down
		end
		tsteps = fill(0.0, round(Int64, n*log(n)))
		(err_record, colsrecord, temprecord, costsequence, currentcols) = run_gibbs_step(model, input_data, output_data, input_data_copy, a, v, currentcols, err_record, colsrecord, tsteps, printstep = printiter, updateinterval = updateinterval, calibrate = false, multiout=multiout)
		ar = calc_accept_rate(costsequence)
		push!(fulltemprecord, (0.0, mean(costsequence), ar))
		colscheck = length(unique(costsequence)) > 1 #verify that sequence is still fluctuating
	end
	(err_record, colsrecord, fulltemprecord)
end


function run_quasistatic_anneal_process(model, input_data, output_data; seed = 1, colnames = map(a -> string("Col ", a), 1:size(input_data, 2)), printiter=true, updateinterval = 2.0, chi = 0.9, delt = 0.001, M = round(Int64, size(input_data, 2)*log(size(input_data, 2))), initialcolsrecord::RecordType{Float64} = RecordType{Float64}())

	(m, n) = size(input_data)
	input_data_copy = copy(input_data)
	a = if length(model) == 4
		form_activations(model[1], m)
	else
		form_activations(model[1][1][1], m)
	end

	v = Vector{Float32}(undef, m) 
	if length(model) == 3
		num = length(model[1])
		multiout = [copy(a[end]) for _ in 1:num]
	else
		multiout = Vector{Matrix{Float32}}()
	end
	

	@assert m == length(output_data) "X and y must have the same number of rows"


	namePrefix = "QuasistaticAnnealReheatShuffleNNErrs"
	
	(err_record, colsrecord, tsteps, accs) = run_temp_calibrate(model, input_data, output_data, input_data_copy, a, v, seed = seed, colsrecord=initialcolsrecord, printiter = printiter, updateinterval = updateinterval, chi = chi, multiout=multiout)
	
	l = length(accs)
	accrate = mean(accs[round(Int64, l/2):end])
	newtemp = tsteps[end]
	println("Achieved an acceptance rate of $accrate compared to a target of $chi with a final temperature of $newtemp")


	# (newBestTestErr, bestInd) = findmin([a[3][2] for a in errsRecord1])
	
	bestrecord = err_record[end]
	newbesterr = bestrecord[3]
	bestcols = bestrecord[2]
	startingcols= bestcols

	errs = [a[3] for a in err_record]
	C0 = mean(errs[round(Int64, l/2):end])
	besterr = Inf
	fulltemprecord = []
	origseed = seed
	while newbesterr < besterr
		besterr = newbesterr
		t_start = time()
		(newerr_record, colsrecord, temprecord) = run_stationary_steps(model, input_data, output_data, input_data_copy, a, v, startingcols, besterr, colsrecord, newtemp, C0, seed = seed, delt = delt, multiout=multiout)

		# run_stationary_steps(model, input_data, output_data, input_data_copy, a, v, startingcols, startingerr, initialcolsrecord, initialtemp, C0; seed = 1, delt = delt=0.001, updateinterval=2.0, printiter=true, n = size(input_data, 2), m = m = round(Int64, n*log(n)))

		seed = rand(UInt32)

		startingcols = newerr_record[end][2]
		temps = [a[1] for a in temprecord]
		ARs = [a[3] for a in temprecord]
		avgerrs = [a[2] for a in temprecord]
		ind = findfirst(a -> a < 0.25, ARs)

		if ind > 1
			newtemp = (temps[ind] + temps[ind-1])/2
			C0 = (avgerrs[ind] + avgerrs[ind-1])/2
		else
			newtemp = temps[ind]
			C0 = avgerrs[ind]
		end

		iterseconds = time() - t_start
		itertimestr = maketimestr(iterseconds)

		for r in newerr_record[2:end]
			push!(err_record, r)
		end

		for r in temprecord
			push!(fulltemprecord, r)
		end

		
		(newbesterr, errdelt, bestcols) = extract_record(newerr_record) 

		println("Quasitatic annealing complete after $itertimestr with $(sum(bestcols)) columns\nwith error of $newbesterr")
		if newbesterr < besterr
			println(string("Resetting temperature to ", newtemp, " to try to find a better configuration"))
		else
			println("No improvement found after reheating so terminating process")
		end
	end
	usedcolscheck = [in(i, bestcols) ? "x\t$n" : " \t$n" for (i, n) in enumerate(colnames)]
	(bestcols, usedcolscheck, newbesterr, colsrecord, fulltemprecord, err_record)
end