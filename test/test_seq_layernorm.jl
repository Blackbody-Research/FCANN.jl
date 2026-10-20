# Tests for Step 2: LayerNorm forward + backward (src/SEQ_LAYERNORM.jl).
#
# Mirrors the checkNumGradCPU pattern in ADAMAXTRAIN_FCN_NFLOATOUT.jl: an
# analytic pass compared against finite differences via relative error
# norm(num - analytic)/norm(num + analytic). Not yet wired into runtests.jl;
# run standalone with: julia test/test_seq_layernorm.jl
using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "SEQ_LAYERNORM.jl"))

@testset "layernorm init" begin
    gain, bias = init_layernorm_params(8)
    @test eltype(gain) == Float32
    @test all(gain .== 1.0f0) && all(bias .== 0.0f0)
    gain64, _ = init_layernorm_params(4, Float64)
    @test eltype(gain64) == Float64
    @test_throws AssertionError init_layernorm_params(0)
end

@testset "layernorm scratch" begin
    scratch = form_layernorm_scratch(Float32, 128, 64)
    @test length(scratch.row_mean) == 128
    @test length(scratch.row_inv_std) == 128
    @test length(scratch.row_dot) == 128
    @test size(scratch.row_normed) == (128, 64)
    @test_throws AssertionError form_layernorm_scratch(Float32, 0, 4)
    @test_throws AssertionError form_layernorm_scratch(Float32, 8, -1)
end

@testset "layernorm forward known values" begin
    gain, bias = init_layernorm_params(4)
    scratch = form_layernorm_scratch(Float32, 8, 4)
    input = Float32[1 2 3 4; 5 5 5 5]
    output = zeros(Float32, 2, 4)
    layernorm_forward!(output, scratch, input, gain, bias, 2, 1f-5)
    # Row 1: mu = 2.5, var = 1.25. Row 2 (constant): x_hat = 0.
    r = 1 / sqrt(1.25 + 1e-5)
    @test output[1, :] ≈ Float32[r*(1-2.5), r*(2-2.5), r*(3-2.5), r*(4-2.5)]
    @test output[2, :] ≈ zeros(Float32, 4)
    @test scratch.row_mean[1] ≈ 2.5f0
    @test scratch.row_inv_std[1] ≈ Float32(r)
    # Non-identity gain/bias shift the normalized row.
    layernorm_forward!(output, scratch, input, Float32[2, 2, 2, 2],
        Float32[1, 0, -1, 0.5], 1, 1f-5)
    @test output[1, :] ≈ Float32[2*r*(1-2.5)+1, 2*r*(2-2.5), 2*r*(3-2.5)-1, 2*r*(4-2.5)+0.5]
end

@testset "layernorm forward padding + no-op" begin
    gain, bias = init_layernorm_params(4)
    scratch = form_layernorm_scratch(Float32, 8, 4)
    big = fill(Float32(9), 8, 4)
    big_out = fill(Float32(7), 8, 4)
    # Zero tokens: nothing written.
    layernorm_forward!(big_out, scratch, big, gain, bias, 0, 1f-5)
    @test all(big_out .== 7.0f0)
    # Only rows 1:3 processed; padding rows untouched.
    layernorm_forward!(big_out, scratch, big, gain, bias, 3, 1f-5)
    @test all(big_out[4:8, :] .== 7.0f0)
    # Epsilon floor rejects values below eps.
    small = zeros(Float32, 1, 4)
    small_out = zeros(Float32, 1, 4)
    @test_throws AssertionError layernorm_forward!(small_out, scratch, small, gain, bias, 1, 1f-9)
end


@testset "layernorm backward grad parity" begin
    Random.seed!(7)
    model_dim, num_tokens, epsilon = 6, 5, 1f-5
    input = randn(Float32, num_tokens, model_dim)
    gain = randn(Float32, model_dim)
    bias = randn(Float32, model_dim)
    upstream = randn(Float32, num_tokens, model_dim)
    scratch = form_layernorm_scratch(Float32, num_tokens, model_dim)
    output = similar(input)
    layernorm_forward!(output, scratch, input, gain, bias, num_tokens, epsilon)
    dx = similar(input)
    param_grads = zeros(Float32, model_dim, 2)
    layernorm_backward!(dx, param_grads, upstream, scratch, gain, num_tokens)
    # Finite differences over (input, gain, bias); loss = sum(upstream .* LN).
    # Float32 roundoff dominates below e = 1e-3 (verified by epsilon sweep),
    # so e = 1e-3 with a 2e-4 threshold; Float64 below proves exactness.
    loss_fn(xv, gv, bv) = begin
        out = similar(xv)
        sc = form_layernorm_scratch(Float32, num_tokens, model_dim)
        layernorm_forward!(out, sc, xv, gv, bv, num_tokens, epsilon)
        sum(upstream .* out)
    end
    e = 1f-3
    num_dx = similar(input)
    for t in 1:num_tokens, j in 1:model_dim
        xp, xm = copy(input), copy(input)
        xp[t, j] += e
        xm[t, j] -= e
        num_dx[t, j] = (loss_fn(xp, gain, bias) - loss_fn(xm, gain, bias)) / (2e)
    end
    num_dg, num_db = similar(gain), similar(bias)
    for j in 1:model_dim
        gp, gm = copy(gain), copy(gain)
        gp[j] += e
        gm[j] -= e
        num_dg[j] = (loss_fn(input, gp, bias) - loss_fn(input, gm, bias)) / (2e)
        bp, bm = copy(bias), copy(bias)
        bp[j] += e
        bm[j] -= e
        num_db[j] = (loss_fn(input, gain, bp) - loss_fn(input, gain, bm)) / (2e)
    end
    @test norm(num_dx - dx) / norm(num_dx + dx) < 2f-4
    @test norm(num_dg - param_grads[:, 1]) / norm(num_dg + param_grads[:, 1]) < 1f-4
    @test norm(num_db - param_grads[:, 2]) / norm(num_db + param_grads[:, 2]) < 1f-4
    # Float64 reference: formula-exact to ~1e-10.
    xd, gd, bd, dyd = Float64.(input), Float64.(gain), Float64.(bias), Float64.(upstream)
    eps64 = Float64(epsilon)
    scd = form_layernorm_scratch(Float64, num_tokens, model_dim)
    oud = similar(xd)
    layernorm_forward!(oud, scd, xd, gd, bd, num_tokens, eps64)
    dxd = similar(xd)
    pgd = zeros(Float64, model_dim, 2)
    layernorm_backward!(dxd, pgd, dyd, scd, gd, num_tokens)
    loss64(xv) = begin
        o = similar(xv)
        sc = form_layernorm_scratch(Float64, num_tokens, model_dim)
        layernorm_forward!(o, sc, xv, gd, bd, num_tokens, eps64)
        sum(dyd .* o)
    end
    e64, num_dxd = 1e-6, similar(xd)
    for t in 1:num_tokens, j in 1:model_dim
        xp, xm = copy(xd), copy(xd)
        xp[t, j] += e64
        xm[t, j] -= e64
        num_dxd[t, j] = (loss64(xp) - loss64(xm)) / (2 * e64)
    end
    @test norm(num_dxd - dxd) / norm(num_dxd + dxd) < 1e-8
end

@testset "layernorm backward accumulation + no-op" begin
    Random.seed!(11)
    model_dim, num_tokens = 4, 3
    scratch = form_layernorm_scratch(Float32, num_tokens, model_dim)
    gain, bias = init_layernorm_params(model_dim)
    input = randn(Float32, num_tokens, model_dim)
    upstream = randn(Float32, num_tokens, model_dim)
    output = similar(input)
    layernorm_forward!(output, scratch, input, gain, bias, num_tokens, 1f-5)
    dx = similar(input)
    param_grads = zeros(Float32, model_dim, 2)
    layernorm_backward!(dx, param_grads, upstream, scratch, gain, num_tokens)
    # Second call accumulates (+=), mirroring Theta_grads across batches.
    first = copy(param_grads)
    layernorm_backward!(dx, param_grads, upstream, scratch, gain, num_tokens)
    @test param_grads ≈ 2 * first
    # Zero tokens: no-op, gradients untouched.
    before = copy(param_grads)
    layernorm_backward!(dx, param_grads, upstream, scratch, gain, 0)
    @test param_grads ≈ before
end

@testset "layernorm zero allocation" begin
    model_dim, num_tokens = 32, 64
    scratch = form_layernorm_scratch(Float32, num_tokens, model_dim)
    gain, bias = init_layernorm_params(model_dim)
    input = randn(Float32, num_tokens, model_dim)
    upstream = randn(Float32, num_tokens, model_dim)
    output, dx = similar(input), similar(input)
    param_grads = zeros(Float32, model_dim, 2)
    layernorm_forward!(output, scratch, input, gain, bias, num_tokens, 1f-5)
    layernorm_backward!(dx, param_grads, upstream, scratch, gain, num_tokens)
    @test Base.@allocated(layernorm_forward!(output, scratch, input, gain, bias, num_tokens, 1f-5)) == 0
    @test Base.@allocated(layernorm_backward!(dx, param_grads, upstream, scratch, gain, num_tokens)) == 0
end
