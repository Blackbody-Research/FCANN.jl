@everywhere include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

name = "sharesScaledNewIndic_4Q_Offsets"

Plist = [2560000]
layers = [4, 6, 8, 10]
N = 100
batchSize = 1024

smartEvalLayers(name, N, batchSize, Plist, tau = 0.05f0, layers = layers)