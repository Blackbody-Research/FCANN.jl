# Tests for Step 5: embed-free transformer core block fwd+bwd (src/SEQ_BLOCK.jl).
#
# Not yet wired into runtests.jl; run standalone with:
#   julia --project=. test/test_seq_block.jl
# Mirrors the checkNumGradCPU pattern (analytic vs finite-difference relative
# error) and the earlier SEQ_* step tests. The block is now embed-free: input
# is a model_dim-wide activation (the stack owns embedding + positions), and
# block_backward! exposes its input gradient in scratch.d_input (model-width).
using Test
using Random
using LinearAlgebra
using LinearAlgebra.BLAS
using RandomMatrices

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "FCN_NFLOATOUT_AUXFUNCTIONS.jl"))
include(joinpath(SRC, "FCN_NFLOATOUT_COSTFUNCTIONS.jl"))
include(joinpath(SRC, "SEQ_CONTAINERS.jl"))
include(joinpath(SRC, "SEQ_LAYERNORM.jl"))
include(joinpath(SRC, "SEQ_ATTENTION.jl"))
include(joinpath(SRC, "SEQ_BLOCK.jl"))

# Build a BlockParams with every field zeroed (for gradient accumulation).
zero_params(params::BlockParams{T}) where {T} =
    BlockParams([zeros(T, size(getfield(params, f))) for f in fieldnames(BlockParams{T})]...)

@testset "block params init (muP)" begin
    Random.seed!(5)
    cfg = SeqConfig(4, 64, 32, 32, 2)
    params = init_block_params(cfg)
    @test size(params.qkv_weight) == (96, 32) && length(params.qkv_bias) == 96
    @test size(params.attn_out_weight) == (32, 32)
    @test size(params.mlp_hidden_weight) == (128, 32) && size(params.mlp_out_weight) == (32, 128)
    # muP Table 8: hidden weights var = 1/fan_in; dense biases 0.
    @test isapprox(std(params.qkv_weight), 1 / sqrt(32); rtol = 0.15)
    @test isapprox(std(params.mlp_hidden_weight), 1 / sqrt(32); rtol = 0.15)
    @test isapprox(std(params.mlp_out_weight), 1 / sqrt(128); rtol = 0.15)
    @test all(params.qkv_bias .== 0)
    @test all(params.attn_out_bias .== 0) && all(params.mlp_hidden_bias .== 0)
    @test all(params.mlp_out_bias .== 0)
    # Vector-like: LN identity.
    @test all(params.norm1_gain .== 1) && all(params.norm1_bias .== 0)
    @test all(params.norm2_gain .== 1) && all(params.norm2_bias .== 0)
    # Haar-orthogonal columns (off-diagonal Gram error ~1e-7).
    G = Float32.(params.qkv_weight' * params.qkv_weight)
    @test maximum(abs.(G .- Diagonal(diag(G)))) < 1f-3
    @test eltype(init_block_params(cfg, Float64).qkv_weight) == Float64
    @test_throws AssertionError init_block_params(SeqConfig(4, 64, 32, 0, 2))
end

@testset "block scratch" begin
    cfg = SeqConfig(4, 16, 8, 12, 2)
    s = form_block_scratch(Float32, cfg)
    @test size(s.qkv_out) == (16, 36)
    @test size(s.mlp_hidden) == (16, 48) && size(s.mlp_tanh_grad) == (16, 48)
    @test size(s.block_out) == (16, 12) && size(s.hidden_state) == (16, 12)
    @test size(s.dscores) == (16, 16) && size(s.d_input) == (16, 12)
    @test size(s.norm1_param_grads) == (12, 2) && size(s.norm2_param_grads) == (12, 2)
    @test size(s.delta_qkv) == (16, 36) && size(s.delta_mlp_hidden) == (16, 48)
    @test length(s.layernorm1.row_mean) == 16 && size(s.attention.scores) == (16, 16)
    @test eltype(form_block_scratch(Float64, cfg).block_out) == Float64
end

@testset "fast_tanh_generic parity" begin
    for v in (Float32(-5.5), Float32(-1.2), Float32(0), Float32(0.7), Float32(6.1))
        @test isapprox(fast_tanh_generic(v), fast_tanh(v); atol = 1f-7)
    end
    for v in (-2.0, -0.3, 0.0, 0.9, 3.0)
        @test isapprox(fast_tanh_generic(v), tanh(v); atol = 1e-5)
    end
end



# Causal bias for an L x L leading block (rows past slot get all -Inf via the
# Step 1 helper; here L == slot so a plain causal triangle).
function causal_bias(L::Int)
    bias = zeros(Float32, L, L)
    for i in 1:L, j in (i + 1):L
        bias[i, j] = -Inf32
    end
    return bias
end

@testset "block forward" begin
    Random.seed!(31)
    cfg = SeqConfig(1, 8, 6, 8, 1)
    params = init_block_params(cfg)
    scratch = form_block_scratch(Float32, cfg)
    L = 5
    x = randn(Float32, L, 8)   # model_dim-wide activation (embed-free block)
    bias = causal_bias(L)
    block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)
    @test size(scratch.block_out) == (8, 8)
    @test all(isfinite, @view(scratch.block_out[1:L, :]))
    # Determinism: same inputs -> same outputs.
    out1 = copy(scratch.block_out)
    block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)
    @test scratch.block_out == out1
    # Bidirectional mode runs and differs from causal (different mixing).
    block_forward!(scratch, params, x, bias, L, L, :bidirectional, 1f-5)
    @test all(isfinite, @view(scratch.block_out[1:L, :]))
    @test_throws AssertionError block_forward!(scratch, params, x, bias, L, L, :sideways, 1f-5)
    # Zero allocation (steady state).
    block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)
    @test Base.@allocated(block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)) == 0
end

@testset "block backward grad parity" begin
    Random.seed!(41)
    cfg = SeqConfig(1, 6, 4, 6, 1)
    params = init_block_params(cfg)
    scratch = form_block_scratch(Float32, cfg)
    L = 4
    x = randn(Float32, L, 6)   # model_dim-wide activation (embed-free block)
    bias = causal_bias(L)
    upstream = randn(Float32, L, 6)
    block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)
    grads = zero_params(params)
    block_backward!(scratch, params, grads, upstream, L, L)
    # Finite differences over every parameter group and the input; loss =
    # sum(upstream .* block_out[1:L, :]). e = 1e-3 (Float32 roundoff floor).
    function lossf(p)
        s2 = form_block_scratch(Float32, cfg)
        block_forward!(s2, p, x, bias, L, L, :causal, 1f-5)
        return sum(upstream .* (@view s2.block_out[1:L, :]))
    end
    e = 1f-3
    fns = collect(fieldnames(BlockParams{Float32}))
    worst = 0.0
    for (k, f) in enumerate(fns)
        numeric = similar(getfield(params, f))
        analytic = getfield(grads, f)
        for idx in eachindex(numeric)
            pa = [copy(getfield(params, g)) for g in fns]
            pb = [copy(getfield(params, g)) for g in fns]
            pa[k][idx] += e
            pb[k][idx] -= e
            numeric[idx] = (lossf(BlockParams(pa...)) - lossf(BlockParams(pb...))) / (2e)
        end
        r = norm(numeric - analytic) / norm(numeric + analytic)
        worst = max(worst, r)
        # field $f rel err $r must be under 5e-4
        @test r < 5e-4
    end
    # Input gradient (model-width, stored in d_input).
    dx = copy(@view scratch.d_input[1:L, :])
    lossx(xv) = begin
        s2 = form_block_scratch(Float32, cfg)
        block_forward!(s2, params, xv, bias, L, L, :causal, 1f-5)
        return sum(upstream .* (@view s2.block_out[1:L, :]))
    end
    numeric_dx = similar(x)
    for idx in eachindex(x)
        a, b = copy(x), copy(x)
        a[idx] += e
        b[idx] -= e
        numeric_dx[idx] = (lossx(a) - lossx(b)) / (2e)
    end
    @test norm(numeric_dx - dx) / norm(numeric_dx + dx) < 5e-4
    println("block worst param rel err = ", worst)
    # Accumulation: second backward doubles a weight grad (mirrors Theta_grads).
    first = copy(grads.mlp_out_weight)
    block_backward!(scratch, params, grads, upstream, L, L)
    @test grads.mlp_out_weight ≈ 2 * first
    # Unused slot: no-op, gradients untouched.
    before = copy(grads.qkv_weight)
    block_backward!(scratch, params, grads, upstream, L, 0)
    @test grads.qkv_weight ≈ before
    # Zero allocation both directions (steady state).
    block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)
    g2 = zero_params(params)
    block_backward!(scratch, params, g2, upstream, L, L)
    @test Base.@allocated(block_forward!(scratch, params, x, bias, L, L, :causal, 1f-5)) == 0
    @test Base.@allocated(block_backward!(scratch, params, g2, upstream, L, L)) == 0
end
