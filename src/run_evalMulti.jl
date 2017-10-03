@everywhere include("MASTER_FCN_ABSERR_NFLOATOUT.jl")


name = "predictCMcapScaledNewIndic_4Q_Offsets_fullLabels"
hidden = ones(Int64, 9)*2
alpha = 0.005188f0
R = 0.0f0
lambda = 0.0f0
c = Inf
IDList = [1]

evalMulti(name, hidden, lambda, c, alpha, R, IDList = IDList)
