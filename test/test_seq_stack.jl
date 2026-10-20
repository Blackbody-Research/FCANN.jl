# Tests for Step 5: stacked sequence transformer forward + backward
# (src/SEQ_STACK.jl).
#
# Not yet wired into runtests.jl; run standalone with:
#   julia --project=. test/test_seq_stack.jl
#
# Gradient parity is checked in Float64 (authoritative, ~1e-7 threshold):
# Float32 finite differences hit a roundoff floor (~6e-4) once gradients pass
# through 2-3 stacked blocks, exactly as documented for the LayerNorm step.
# Float32 is used for determinism / 0-alloc / N=1-vs-manual regression, where
# no division-by-epsilon amplification occurs.
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
include(joinpath(SRC, "SEQ_STACK.jl"))

# Causal bias for a standalone L×L leading block (Step 1 idiom; L == slot).
causal_bias(L::Int) = begin
    b = zeros(Float32, L, L)
    for i in 1:L, j in (i + 1):L
        b[i, j] = -Inf32
    end
    b
end

@testset "stack params + scratch init" begin
    Random.seed!(7)
    cfg = SeqConfig(2, 16, 5, 8, 1)
    NB = 3
    p = init_stack_params(cfg, NB)
    @test size(p.embed_weight) == (8, 5) && length(p.embed_bias) == 8
    @test size(p.pos_table) == (16, 8) && length(p.blocks) == NB
    @test size(p.blocks[1].qkv_weight) == (24, 8)   # embed-free core block
    # μP: embed var 1, embed_bias/pos zero; block biases zero; LN identity.
    @test isapprox(std(p.embed_weight), 1.0; rtol = 0.2)
    @test all(p.embed_bias .== 0) && all(p.pos_table .== 0)
    @test all(p.blocks[1].norm1_gain .== 1) && all(p.blocks[1].qkv_bias .== 0)
    @test eltype(init_stack_params(cfg, NB, Float64).embed_weight) == Float64
    @test_throws AssertionError init_stack_params(cfg, 0)
    # zero_stack_params: all-zero, same depth, rebuilds the blocks vector.
    z = zero_stack_params(p)
    @test all(z.embed_weight .== 0) && length(z.blocks) == NB
    @test all(z.blocks[2].qkv_weight .== 0) && all(z.pos_table .== 0)
    # scratch
    s = form_stack_scratch(Float32, cfg, NB)
    @test size(s.embedded) == (16, 8) && size(s.dx_input) == (16, 5)
    @test size(s.bias) == (16, 16) && length(s.blocks) == NB
    @test size(s.blocks[1].qkv_out) == (16, 24)
    @test eltype(form_stack_scratch(Float64, cfg, NB).embedded) == Float64
    @test_throws AssertionError form_stack_scratch(Float32, cfg, 0)
end

@testset "fill_stack_attention_bias! patterns" begin
    # causal, L=4, slot=2: rows 1,2 triangular; rows 3,4 all blocked.
    c = zeros(Float32, 4, 4)
    fill_stack_attention_bias!(c, 4, 2, :causal)
    @test c[1, 1] == 0 && c[1, 2] == typemin(Float32)
    @test c[2, 1] == 0 && c[2, 2] == 0 && c[2, 3] == typemin(Float32)
    @test all(c[3, :] .== typemin(Float32)) && all(c[4, :] .== typemin(Float32))
    # bidirectional, L=4, slot=2: valid rows see keys 1:2; dummy rows blocked.
    b = zeros(Float32, 4, 4)
    fill_stack_attention_bias!(b, 4, 2, :bidirectional)
    @test b[1, 1] == 0 && b[1, 2] == 0 && b[1, 3] == typemin(Float32)
    @test b[2, 1] == 0 && b[2, 2] == 0 && b[2, 3] == typemin(Float32)
    @test all(b[3, :] .== typemin(Float32)) && all(b[4, :] .== typemin(Float32))
    @test_throws AssertionError fill_stack_attention_bias!(zeros(Float32, 4, 4), 4, 2, :sideways)
end

@testset "stack forward (Float32): determinism, modes, N=1 regression, 0-alloc" begin
    Random.seed!(11)
    cfg = SeqConfig(1, 16, 5, 8, 1)
    NB = 3
    p = init_stack_params(cfg, NB)
    s = form_stack_scratch(Float32, cfg, NB)
    L = 6
    x = randn(Float32, L, 5)
    stack_forward!(s, p, x, L, L, :causal, 1f-5)
    out = @view s.blocks[NB].block_out[1:L, :]
    @test size(out) == (L, 8) && all(isfinite, out)
    # Determinism.
    out1 = copy(out)
    stack_forward!(s, p, x, L, L, :causal, 1f-5)
    @test s.blocks[NB].block_out[1:L, :] == out1
    # Bidirectional runs and is finite.
    stack_forward!(s, p, x, L, L, :bidirectional, 1f-5)
    @test all(isfinite, @view(s.blocks[NB].block_out[1:L, :]))
    # N=1 stack == manual embed + single block (embed/pos applied exactly once).
    p1 = init_stack_params(cfg, 1)
    s1 = form_stack_scratch(Float32, cfg, 1)
    stack_forward!(s1, p1, x, L, L, :causal, 1f-5)
    manual = form_block_scratch(Float32, cfg)
    E = x * p1.embed_weight' .+ p1.embed_bias' .+ p1.pos_table[1:L, :]
    block_forward!(manual, p1.blocks[1], E, causal_bias(L), L, L, :causal, 1f-5)
    @test maximum(abs.(s1.blocks[1].block_out[1:L, :] - manual.block_out[1:L, :])) < 1f-4
    # 0-alloc steady state + no-op slot + bad mode.
    stack_forward!(s, p, x, L, L, :causal, 1f-5)
    @test Base.@allocated(stack_forward!(s, p, x, L, L, :causal, 1f-5)) == 0
    stack_forward!(s, p, x, L, 0, :causal, 1f-5)   # no-op, no error
    @test_throws AssertionError stack_forward!(s, p, x, L, L, :sideways, 1f-5)
end

@testset "stack backward grad parity (Float64, N=1,2,3)" begin
    # Authoritative finite-difference parity. Loss = sum(upstream .* stack_out).
    function check_stack64(NB::Int)
        Random.seed!(300 + NB)
        cfg = SeqConfig(1, 12, 4, 6, 1)
        params = init_stack_params(cfg, NB, Float64)
        scratch = form_stack_scratch(Float64, cfg, NB)
        L = 5
        x = randn(Float64, L, 4)
        upstream = randn(Float64, L, 6)
        stack_forward!(scratch, params, x, L, L, :causal, 1e-7)
        grads = zero_stack_params(params)
        stack_backward!(scratch, params, grads, upstream, x, L, L)
        lossf(p) = (s2 = form_stack_scratch(Float64, cfg, NB);
            stack_forward!(s2, p, x, L, L, :causal, 1e-7);
            sum(upstream .* (@view s2.blocks[NB].block_out[1:L, :])))
        e = 1e-6
        worst = 0.0
        # top-level embed/pos
        for f in (:embed_weight, :embed_bias, :pos_table)
            numeric = similar(getfield(params, f))
            for idx in eachindex(numeric)
                pa = deepcopy(params); pb = deepcopy(params)
                getfield(pa, f)[idx] += e; getfield(pb, f)[idx] -= e
                numeric[idx] = (lossf(pa) - lossf(pb)) / (2e)
            end
            ana = getfield(grads, f)
            worst = max(worst, norm(numeric - ana) / norm(numeric + ana))
        end
        # every block's params
        fns = collect(fieldnames(BlockParams{Float64}))
        for bi in 1:NB, f in fns
            numeric = similar(getfield(params.blocks[bi], f))
            for idx in eachindex(numeric)
                pa = deepcopy(params); pb = deepcopy(params)
                getfield(pa.blocks[bi], f)[idx] += e; getfield(pb.blocks[bi], f)[idx] -= e
                numeric[idx] = (lossf(pa) - lossf(pb)) / (2e)
            end
            ana = getfield(grads.blocks[bi], f)
            worst = max(worst, norm(numeric - ana) / norm(numeric + ana))
        end
        # input gradient (feature-width)
        dx = copy(@view scratch.dx_input[1:L, :])
        lossx(xv) = (s2 = form_stack_scratch(Float64, cfg, NB);
            stack_forward!(s2, params, xv, L, L, :causal, 1e-7);
            sum(upstream .* (@view s2.blocks[NB].block_out[1:L, :])))
        numeric_dx = similar(x)
        for idx in eachindex(x)
            a, b = copy(x), copy(x); a[idx] += e; b[idx] -= e
            numeric_dx[idx] = (lossx(a) - lossx(b)) / (2e)
        end
        worst = max(worst, norm(numeric_dx - dx) / norm(numeric_dx + dx))
        @test worst < 1e-7
        println("stack NB=$NB worst rel err = ", worst)
    end
    check_stack64(1)
    check_stack64(2)
    check_stack64(3)
end

@testset "stack backward (Float32): accumulation, no-op, 0-alloc" begin
    Random.seed!(200)
    cfg = SeqConfig(1, 8, 3, 5, 1)
    NB = 2
    p = init_stack_params(cfg, NB)
    s = form_stack_scratch(Float32, cfg, NB)
    x = randn(Float32, 4, 3)
    up = randn(Float32, 4, 5)
    stack_forward!(s, p, x, 4, 4, :causal, 1f-5)
    g = zero_stack_params(p)
    stack_backward!(s, p, g, up, x, 4, 4)
    ew1 = copy(g.embed_weight); bw1 = copy(g.blocks[2].qkv_weight)
    # Accumulation: second backward doubles weight grads (mirrors Theta_grads).
    stack_backward!(s, p, g, up, x, 4, 4)
    @test g.embed_weight ≈ 2 * ew1
    @test g.blocks[2].qkv_weight ≈ 2 * bw1
    # No-op slot: gradients untouched.
    stack_backward!(s, p, g, up, x, 4, 0)
    @test g.embed_weight ≈ 2 * ew1
    # 0-alloc both directions (steady state).
    stack_forward!(s, p, x, 4, 4, :causal, 1f-5)
    g2 = zero_stack_params(p)
    stack_backward!(s, p, g2, up, x, 4, 4)
    @test Base.@allocated(stack_forward!(s, p, x, 4, 4, :causal, 1f-5)) == 0
    @test Base.@allocated(stack_backward!(s, p, g2, up, x, 4, 4)) == 0
end

