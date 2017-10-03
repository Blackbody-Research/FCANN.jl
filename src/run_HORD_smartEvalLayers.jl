include("HORDEvalLayersFunction.jl")

#-------User Defined Variables------------------------
#list of potential hyperparameters to tune given a fixed target number of parameters
#   1. num epochs from 100 to 1000
#   2. target num params from 100 to 1000 (will change with problem)
#   3. num layers from 1 to 10
#   4. max norm constant inverse from 0 (c = Inf) to 2 (c = 0.5...)
#   5. number of ensemble networks to use from 1 to 100

#name of training and test sets 
name = "predictCMcapScaledNewIndic_4Q_Offsets_fullLabels"
#for each of the above 7 variables either define a default value to remain fixed or define a range for optimization as a tuple
OPT_PARAMS = [  100,     #num epochs selector
                (100, 3200),    #target params selector
                (1, 10),       #num layers selector
                0,         #max norm selector
                1         #ensemble size selector
            ]
#optionally define an initial starting point of default values believed to perform well (note the values not being tuned will simply be igonred)
ISP = [ 100,    #num epochs
        628,    #target params
        9,      #num layers
        0,    #max norm
        1       #ensemble size
        ]
#if no ISP is desired uncomment the following line
#ISP = []

trialID = 4

#maximum number of function evaluations
Nmax = 100 
##----------------------------------

HORDEvalLayers(name, OPT_PARAMS, trialID, Nmax, ISP)