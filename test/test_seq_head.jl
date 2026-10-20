# Tests for Step 6: sequence output heads forward + backward (src/SEQ_HEAD.jl).
#
# Not yet wired into runtests.jl; run standalone with:
#   julia --project=. test/test_seq_head.jl
#
# Mirrors the checkNumGradCPU pattern (analytic vs finite-difference relative
# error) and the earlier SEQ_* step tests. Both branches are exercised: the
# set carries one single-output head (O==1, dot/axpy! fast path) and one wide
# head (O==3, gemv!/ger! general path).
#
# Note: `head_forward!` asserts slot_length >= 1 (it cannot no-op — there is
# no valid row to read), while `head_backward!` treats slot_length == 0 as a
# no-op. The tests encode exactly that split.
using Test
using Random
using LinearAlgebra
using LinearAlgebra.BLAS
using RandomMatrices

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "FCN_NFLOATOUT_AUXFUNCTIONS.jl"))
include(joinpath(SRC, "SEQ_HEAD.jl"))

@testset "head params init (muP)" begin
    Random.seed!(5)
    set = init_seq_output_set(8, [1, 3])
    @test length(set.heads) == 2 && set.read_mode === :last
    @test size(set.heads[1].weight) == (1, 8) && length(set.heads[1].bias) == 1
    @test size(set.heads[2].weight) == (3, 8) && length(set.heads[2].bias) == 3
    # μP hidden-weight rule: variance 1/fan_in; biases zero.
    @test isapprox(std(set.heads[2].weight), 1 / sqrt(8); rtol = 0.2)
    @test all(set.heads[1].bias .== 0) && all(set.heads[2].bias .== 0)
    # Widths off the weight matrices, never cached.
    @test size(set.heads[2].weight, 1) == 3 && size(set.heads[2].weight, 2) == 8
    @test eltype(init_seq_output_head(8, 3, Float64).weight) == Float64
    @test_throws AssertionError init_seq_output_head(0, 3)
    @test_throws AssertionError init_seq_output_head(8, 0)
    @test_throws AssertionError init_seq_output_set(8, Int[])
    @test_throws AssertionError init_seq_output_set(8, [2], :sideways)
    # zero_output_set: all-zero, same head count/widths/read_mode.
    z = zero_output_set(set)
    @test length(z.heads) == 2 && z.read_mode === :last
    @test all(z.heads[1].weight .== 0) && all(z.heads[2].bias .== 0)
    @test size(z.heads[2].weight) == (3, 8)
end

@testset "head scratch" begin
    set = init_seq_output_set(8, [1, 3])
    s = form_output_scratch(Float32, set, 16)
    @test length(s.read_pos) == 8 && length(s.d_read_pos) == 8
    # sumO = 1 + 3 = 4 contiguous; ranges tile it exactly (heterogeneous OK).
    @test length(s.outputs) == 4 && length(s.dout) == 4 && length(s.out_staged) == 4
    @test s.head_ranges == [1:1, 2:4]
    @test size(s.trunk_grad) == (16, 8)
    @test eltype(form_output_scratch(Float64, set, 16).read_pos) == Float64
    # Fan-in mismatch across hand-built heads fails fast.
    mixed = SeqOutputSet([init_seq_output_head(8, 1), init_seq_output_head(9, 2)], :last)
    @test_throws AssertionError form_output_scratch(Float32, mixed, 16)
    @test_throws AssertionError form_output_scratch(Float32, set, 0)
end

@testset "head forward hand values" begin
    Random.seed!(11)
    set = init_seq_output_set(8, [1, 3])
    scratch = form_output_scratch(Float32, set, 16)
    trunk = randn(Float32, 16, 8)
    slot = 5
    head_forward!(scratch, set, trunk, slot)
    # read_pos echoes the trunk row; outputs equal W*h .+ b computed naively.
    @test scratch.read_pos == trunk[slot, :]
    let h = trunk[slot, :]
        for k in 1:2
            @test (@view scratch.outputs[scratch.head_ranges[k]]) ≈ set.heads[k].weight * h .+ set.heads[k].bias
        end
    end
    # Reserved modes construct but forward rejects them.
    meanset = init_seq_output_set(8, [2], :mean)
    @test_throws AssertionError head_forward!(scratch, meanset, trunk, slot)
    # Forward cannot no-op: slot 0 has no row to read.
    @test_throws AssertionError head_forward!(scratch, set, trunk, 0)
    # Zero allocation (steady state).
    head_forward!(scratch, set, trunk, slot)
    @test Base.@allocated(head_forward!(scratch, set, trunk, slot)) == 0
end


@testset "head backward grad parity (Float64, authoritative)" begin
    # Finite differences run in Float64 (threshold ~1e-7): Float32 hits the
    # same documented roundoff floor (~7e-4) once the linear head divides by
    # the finite-difference step, exactly as in the LayerNorm/block steps.
    Random.seed!(41)
    D, O2, Tmax, slot = 6, 4, 12, 4
    set = init_seq_output_set(D, [1, O2], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    trunk = randn(Float64, Tmax, D)
    head_forward!(scratch, set, trunk, slot)
    # dout now lives in scratch (contiguous sumO); the test stages a fixed
    # upstream there, exactly as seq_cost! will do.
    dout_up = [randn(Float64, 1), randn(Float64, O2)]
    copyto!(view(scratch.dout, scratch.head_ranges[1]), dout_up[1])
    copyto!(view(scratch.dout, scratch.head_ranges[2]), dout_up[2])
    grads = zero_output_set(set)
    head_backward!(scratch, set, grads, slot, slot)
    # loss = sum_k dout_k' * out_k (raw logits/value, no activation).
    function lossf(s)
        s2 = form_output_scratch(Float64, s, Tmax)
        head_forward!(s2, s, trunk, slot)
        return sum(dout_up[1] .* (@view s2.outputs[s2.head_ranges[1]])) +
               sum(dout_up[2] .* (@view s2.outputs[s2.head_ranges[2]]))
    end
    e = 1e-6
    worst = 0.0
    for k in 1:2
        for (fname, P) in ((:weight, set.heads[k].weight), (:bias, set.heads[k].bias))
            numeric = similar(P)
            for idx in eachindex(P)
                pa = deepcopy(set)
                pb = deepcopy(set)
                getfield(pa.heads[k], fname)[idx] += e
                getfield(pb.heads[k], fname)[idx] -= e
                numeric[idx] = (lossf(pa) - lossf(pb)) / (2e)
            end
            analytic = getfield(grads.heads[k], fname)
            r = norm(numeric - analytic) / norm(numeric + analytic)
            worst = max(worst, r)
            @test r < 1e-7
        end
    end
    # Input gradient lives only in trunk_grad[slot, :].
    dh = copy(@view scratch.trunk_grad[slot, :])
    lossh(hv) = begin
        s2 = form_output_scratch(Float64, set, Tmax)
        t2 = copy(trunk)
        t2[slot, :] = hv
        head_forward!(s2, set, t2, slot)
        return sum(dout_up[1] .* (@view s2.outputs[s2.head_ranges[1]])) +
               sum(dout_up[2] .* (@view s2.outputs[s2.head_ranges[2]]))
    end
    h0 = copy(trunk[slot, :])
    numeric_dh = similar(h0)
    for j in eachindex(h0)
        a, b = copy(h0), copy(h0)
        a[j] += e
        b[j] -= e
        numeric_dh[j] = (lossh(a) - lossh(b)) / (2e)
    end
    @test norm(numeric_dh - dh) / norm(numeric_dh + dh) < 1e-7
    println("head worst param rel err = ", worst)
    # Rows other than the read row must be zero (composition contract).
    @test maximum(abs.(scratch.trunk_grad[1:(slot - 1), :])) == 0
end

@testset "head backward (Float32): accumulation, no-op, 0-alloc" begin
    Random.seed!(41)
    D, O2, Tmax, slot = 6, 4, 12, 4
    set = init_seq_output_set(D, [1, O2])
    scratch = form_output_scratch(Float32, set, Tmax)
    trunk = randn(Float32, Tmax, D)
    head_forward!(scratch, set, trunk, slot)
    # Stage the upstream once in scratch; repeated backward calls reuse it.
    dout_up = [randn(Float32, 1), randn(Float32, O2)]
    copyto!(view(scratch.dout, scratch.head_ranges[1]), dout_up[1])
    copyto!(view(scratch.dout, scratch.head_ranges[2]), dout_up[2])
    grads = zero_output_set(set)
    head_backward!(scratch, set, grads, slot, slot)
    # Accumulation: second backward doubles a weight grad (mirrors Theta_grads).
    first = copy(grads.heads[2].weight)
    head_backward!(scratch, set, grads, slot, slot)
    @test grads.heads[2].weight ≈ 2 * first
    # Unused slot: no-op, gradients untouched.
    before = copy(grads.heads[1].weight)
    head_backward!(scratch, set, grads, slot, 0)
    @test grads.heads[1].weight ≈ before
    # Zero allocation (steady state).
    g2 = zero_output_set(set)
    @test Base.@allocated(head_backward!(scratch, set, g2, slot, slot)) == 0
end

