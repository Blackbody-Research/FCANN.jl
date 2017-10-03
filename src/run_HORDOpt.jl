include("HORDOptFunction.jl")

#-------User Defined Variables------------------------
#list of potential hyperparameters to tune given a fixed target number of parameters
#   1. learning rate alpha from 0 to 0.1
#   2. decay rate R from 0 to 0.2
#   3. num epochs from 100 to 1000
#   4. target num params from 100 to 1000 (will change with problem)
#   5. num layers from 1 to 10
#   6. max norm constant inverse from 0 (c = Inf) to 2 (c = 0.5...)
#   7. number of ensemble networks to use from 1 to 100

#name of training and test sets 
name = "predictCMcapScaledNewIndic_4Q_Offsets_fullLabels"
#for each of the above 7 variables either define a default value to remain fixed or define a range for optimization as a tuple
OPT_PARAMS = [  (0.001, 0.01),     #alpha selector
                (0.0, 0.2),     #R selector
                (10, 1000),     #num epochs selector
                691,    #target params selector
                9,       #num layers selector
                (0, 2),         #max norm selector
                (1, 10)         #ensemble size selector
            ]
#optionally define an initial starting point of default values believed to perform well (note the values not being tuned will simply be igonred)
ISP = [ 0.005236, #alpha
        0.0,    #R
        94,    #num epochs
        691,    #target params
        9,      #num layers
        0,    #max norm
        2       #ensemble size
        ]
#if no ISP is desired uncomment the following line
#ISP = []

trialID = 2

#maximum number of function evaluations
Nmax = 100 
##----------------------------------

HORDOpt(name, OPT_PARAMS, trialID, Nmax, ISP)