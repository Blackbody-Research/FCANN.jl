# Tests for Step 3: single-head attention forward + backward (src/SEQ_ATTENTION.jl).
#
# Mirrors the checkNumGradCPU pattern in ADAMAXTRAIN_FCN_NFLOATOUT.jl: an
# analytic pass compared against finite differences via relative error
# norm(num - analytic)/norm(num + analytic). Not yet wired into runtests.jl;
# run standalone with: julia test/test_seq_attention.jl
using Test
using Random
using LinearAlgebra
using LinearAlgebra.BLAS

include(joinpath(@__DIR__, "..", "src", "SEQ_ATTENTION.jl"))
include(joinpath(@__DIR__, "..", "src", "SEQ_CONTAINERS.jl"))

@testset "attention scratch" begin
    scratch = form_attention_scratch(Float32, 64)
    @test size(scratch.scores) == (64, 64)
    @test size(scratch.probs) == (64, 64)
    @test length(scratch.row_dot) == 64
    @test eltype(scratch.scores) == Float32
    scratch64 = form_attention_scratch(Float64, 8)
    @test eltype(scratch64.row_dot) == Float64
    @test_throws AssertionError form_attention_scratch(Float32, 0)
end

@testset "attention forward causal values" begin
    queries = Float32[1 0; 0 1; 1 1]
    keys = Float32[1 0; 0 1; 1 1]
    values = Float32[1 2; 3 4; 5 6]
    bufs = SeqBuffers(SeqConfig(1, 4, 2, 4, 1))
    fill_causal_padding_bias!(bufs, 3, 3)
    bias = @view bufs.attention_bias[1:3, 1:3]
    scratch = form_attention_scratch(Float32, 4)
    output = zeros(Float32, 3, 2)
    attention_forward!(output, scratch, queries, keys, values, bias, 3, 3, :causal)
    # Row 1 sees key 1 only: output = V[1].
    @test output[1, :] ≈ Float32[1, 2]
    # Row 2 mixes V[1], V[2] with softmax([0, 1/sqrt2]).
    s = 1 / sqrt(2f0)
    w1 = exp(-s) / (exp(-s) + 1)
    w2 = 1 / (exp(-s) + 1)
    @test output[2, :] ≈ Float32[w1 * 1 + w2 * 3, w1 * 2 + w2 * 4]
    # Row 3 mixes all three with softmax([s, s, 2s]).
    S3 = Float32[s, s, 2s]
    m = maximum(S3)
    e3 = exp.(S3 .- m)
    P3 = e3 ./ sum(e3)
    @test output[3, :] ≈ [P3[1] * 1 + P3[2] * 3 + P3[3] * 5, P3[1] * 2 + P3[2] * 4 + P3[3] * 6]
end

@testset "attention forward bidirectional + edge cases" begin
    queries = Float32[1 0; 0 1; 1 1]
    keys = Float32[1 0; 0 1; 1 1]
    values = Float32[1 2; 3 4; 5 6]
    scratch = form_attention_scratch(Float32, 4)
    output = zeros(Float32, 3, 2)
    # Padding-only mask: rows 1-2 see keys 1..2.
    bias_mat = fill(-Inf32, 3, 3)
    bias_mat[1:2, 1:2] .= 0.0f0
    attention_forward!(output, scratch, queries, keys, values, bias_mat, 3, 2, :bidirectional)
    s = 1 / sqrt(2f0)
    S2 = Float32[1 * s 0; 0 1 * s]
    for i in 1:2
        m = maximum(S2[i, :])
        e = exp.(S2[i, :] .- m)
        P = e ./ sum(e)
        @test output[i, :] ≈ [P[1] * 1 + P[2] * 3, P[1] * 2 + P[2] * 4]
    end
    # Unused slot: output untouched.
    filled = fill(Float32(9), 3, 2)
    attention_forward!(filled, scratch, queries, keys, values, bias_mat, 3, 0, :causal)
    @test all(filled .== 9.0f0)
    # Bad mode rejected.
    @test_throws AssertionError attention_forward!(output, scratch, queries, keys, values, bias_mat, 3, 2, :sideways)
end

@testset "attention backward grad parity" begin
    Random.seed!(21)
    active_length, head_dim = 4, 3
    queries = randn(Float32, active_length, head_dim)
    keys = randn(Float32, active_length, head_dim)
    values = randn(Float32, active_length, head_dim)
    upstream = randn(Float32, active_length, head_dim)
    bufs = SeqBuffers(SeqConfig(1, 6, head_dim, head_dim, 1))
    fill_causal_padding_bias!(bufs, active_length, active_length)
    bias = @view bufs.attention_bias[1:active_length, 1:active_length]
    scratch = form_attention_scratch(Float32, 6)
    output = zeros(Float32, active_length, head_dim)
    attention_forward!(output, scratch, queries, keys, values, bias, active_length, active_length, :causal)
    dqueries = zeros(Float32, active_length, head_dim)
    dkeys = zeros(Float32, active_length, head_dim)
    dvalues = zeros(Float32, active_length, head_dim)
    dscores = zeros(Float32, 6, 6)
    attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, upstream,
        queries, keys, values, active_length, active_length)
    # Finite differences over (Q, K, V); loss = sum(upstream .* attention).
    loss_fn(Qv, Kv, Vv) = begin
        out = similar(Qv)
        sc = form_attention_scratch(Float32, 6)
        attention_forward!(out, sc, Qv, Kv, Vv, bias, active_length, active_length, :causal)
        sum(upstream .* out)
    end
    e = 1f-3
    function numgrad(base, slot)
        grads = similar(base)
        for idx in eachindex(base)
            a, b = copy(base), copy(base)
            a[idx] += e
            b[idx] -= e
            fa = slot == 1 ? loss_fn(a, keys, values) : slot == 2 ? loss_fn(queries, a, values) : loss_fn(queries, keys, a)
            fb = slot == 1 ? loss_fn(b, keys, values) : slot == 2 ? loss_fn(queries, b, values) : loss_fn(queries, keys, b)
            grads[idx] = (fa - fb) / (2e)
        end
        return grads
    end
    num_dq, num_dk, num_dv = numgrad(queries, 1), numgrad(keys, 2), numgrad(values, 3)
    @test norm(num_dq - dqueries) / norm(num_dq + dqueries) < 2f-3
    @test norm(num_dk - dkeys) / norm(num_dk + dkeys) < 2f-3
    @test norm(num_dv - dvalues) / norm(num_dv + dvalues) < 2f-3
    # Masked positions carry no score gradient (causal row 1: keys 2+ blocked).
    @test all(dscores[1, 2:active_length] .== 0.0f0)
end

@testset "attention backward accumulation + no-op" begin
    Random.seed!(23)
    active_length, head_dim = 3, 2
    queries = randn(Float32, active_length, head_dim)
    keys = randn(Float32, active_length, head_dim)
    values = randn(Float32, active_length, head_dim)
    upstream = randn(Float32, active_length, head_dim)
    bufs = SeqBuffers(SeqConfig(1, 4, head_dim, head_dim, 1))
    fill_causal_padding_bias!(bufs, active_length, active_length)
    bias = @view bufs.attention_bias[1:active_length, 1:active_length]
    scratch = form_attention_scratch(Float32, 4)
    output = zeros(Float32, active_length, head_dim)
    attention_forward!(output, scratch, queries, keys, values, bias, active_length, active_length, :causal)
    dqueries = zeros(Float32, active_length, head_dim)
    dkeys = zeros(Float32, active_length, head_dim)
    dvalues = zeros(Float32, active_length, head_dim)
    dscores = zeros(Float32, 4, 4)
    attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, upstream,
        queries, keys, values, active_length, active_length)
    # Second call accumulates (+=), mirroring Theta_grads across batches.
    first_dq = copy(dqueries)
    attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, upstream,
        queries, keys, values, active_length, active_length)
    @test dqueries ≈ 2 * first_dq
    # Unused slot: no-op, gradients untouched.
    before = copy(dqueries)
    attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, upstream,
        queries, keys, values, active_length, 0)
    @test dqueries ≈ before
end

@testset "attention zero allocation" begin
    active_length, head_dim = 32, 16
    scratch = form_attention_scratch(Float32, active_length)
    queries = randn(Float32, active_length, head_dim)
    keys = randn(Float32, active_length, head_dim)
    values = randn(Float32, active_length, head_dim)
    upstream = randn(Float32, active_length, head_dim)
    bias_mat = zeros(Float32, active_length, active_length)
    output = similar(queries)
    dqueries, dkeys, dvalues = similar(queries), similar(keys), similar(values)
    dscores = zeros(Float32, active_length, active_length)
    attention_forward!(output, scratch, queries, keys, values, bias_mat, active_length, active_length, :bidirectional)
    attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, upstream,
        queries, keys, values, active_length, active_length)
    @test Base.@allocated(attention_forward!(output, scratch, queries, keys, values, bias_mat, active_length, active_length, :bidirectional)) == 0
    @test Base.@allocated(attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, upstream, queries, keys, values, active_length, active_length)) == 0
end

