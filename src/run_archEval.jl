@everywhere include("MASTER_FCN_ABSERR_NFLOATOUT.jl")

name = "sharesScaledNewIndic_4Q_Offsets"

hiddenList = [[1], [2], [4], [8], [2, 2], [4, 4], [8, 8], [4, 4, 4], [8, 8, 8], [16], [32]]
N = 100
batchSize = 1024


archEval(name, N, batchSize, hiddenList)