# Tests for Step 7 item 2: cost normalization layer (src/SEQ_COST.jl).
#
# Not yet wired into runtests.jl; run standalone with:
#   julia --project=. test/test_seq_cost.jl
#
# Covers only the SeqCost carriers/constructors for now; per-head application,
# the multi-head combiner, and gradient parity arrive with later items of
# Step 7 and will be appended here.
using Test
using LinearAlgebra
using Random
using RandomMatrices

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "FCN_NFLOATOUT_AUXFUNCTIONS.jl"))
include(joinpath(SRC, "FCN_NFLOATOUT_COSTFUNCTIONS.jl"))
# SEQ_HEAD before SEQ_COST: the seq_head_* methods in SEQ_COST reference
# SeqOutputScratch, and the item-3 tests exercise head_forward!.
include(joinpath(SRC, "SEQ_HEAD.jl"))
include(joinpath(SRC, "SEQ_COST.jl"))

# Elementwise names (carry a real (cost, deriv) pair) and the names whose
# concepts dispatch through a LossType - mirrors the SEQ_COST_BY_NAME table.
const ELEMENTWISE_NAMES = ("absErr", "sqErr", "normLogErr", "cauchyLogErr",
                           "sqErrIndex", "absErrIndex")
const LOSSTYPE_NAMES = ("outputIndex", "outputIndexBatch",
                        "crossEntropy", "crossEntropyBatch",
                        "crossEntropyDist", "crossEntropyDistBatch")

@testset "SeqCost normalization: name style" begin
    # Every FCN cost name resolves (the load-time table assert proves coverage
    # - loading SEQ_COST.jl above would have thrown otherwise).
    @test length(costFuncNames) == 12
    for n in ELEMENTWISE_NAMES
        c = SeqCost(n)
        @test c isa SeqDerivCost
        # *Index names normalize to the base error's pair; index selection is
        # carried by the target shape at the call site.
        if startswith(n, "sqErr")
            @test c.cost === sqErr && c.deriv === sqErrDeriv
        elseif startswith(n, "absErr")
            @test c.cost === absErr && c.deriv === absErrDeriv
        end
    end
    @test SeqCost("normLogErr").cost === normLogErr
    @test SeqCost("normLogErr").deriv === normLogErrDeriv
    @test SeqCost("cauchyLogErr").cost === cauchyLogErr
    @test SeqCost("cauchyLogErr").deriv === cauchyLogErrDeriv
    for n in LOSSTYPE_NAMES
        @test SeqCost(n) isa SeqLossCost
    end
    for n in ("outputIndex", "outputIndexBatch")
        @test SeqCost(n).loss isa OutputIndex
    end
    for n in ("crossEntropy", "crossEntropyBatch",
              "crossEntropyDist", "crossEntropyDistBatch")
        @test SeqCost(n).loss isa CrossEntropyLoss
    end
    # Unknown name throws and names the valid ones.
    @test_throws ArgumentError SeqCost("noSuchCost")
    err = try SeqCost("noSuchCost") catch e; e end
    @test err isa ArgumentError
    @test occursin("absErr", err.msg)   # error message lists valid names
end

@testset "SeqCost normalization: function and pair styles" begin
    # A function resolves to the same carrier whether passed as the cost or
    # the deriv half, and that carrier === the name-based carrier (singletons).
    @test SeqCost(sqErr) === SeqCost("sqErr")
    @test SeqCost(sqErrDeriv) === SeqCost("sqErr")
    @test SeqCost(sqErrDeriv) === SeqCost("sqErrIndex")  # concept mapping
    @test SeqCost(absErrDeriv) === SeqCost("absErr")
    @test SeqCost(normLogErrDeriv) === SeqCost("normLogErr")
    # Non-FCN functions are rejected.
    @test_throws ArgumentError SeqCost(sum)
    # Returns(nothing) placeholders are rejected on purpose.
    @test_throws ArgumentError SeqCost(Returns(nothing))
    # Explicit pair bypasses the lists and keeps its halves as given.
    p = SeqCost(absErr, sqErrDeriv)
    @test p isa SeqDerivCost
    @test p.cost === absErr && p.deriv === sqErrDeriv
end

@testset "SeqCost normalization: LossType style and metadata" begin
    @test SeqCost(OutputIndex()) isa SeqLossCost
    @test SeqCost(OutputIndex()).loss isa OutputIndex
    # beta dtype/value preserved exactly through normalization.
    @test SeqCost(CrossEntropyLoss()).loss.beta isa Float32  # FCN default
    c64 = SeqCost(CrossEntropyLoss(0.5))
    @test c64.loss.beta == 0.5
    @test c64.loss isa CrossEntropyLoss{Float64}
    # Carrier construction is singleton-stable: same name -> same object.
    @test SeqCost("absErr") === SeqCost("absErr")
    @test SeqCost("crossEntropy") === SeqCost("crossEntropy")
end

println("SEQ_COST item2 tests done")

# ---------------------------------------------------------------------------
# Step 7 item 3: per-head deriv-cost application (seq_head_value!/seq_head_grad!)
# ---------------------------------------------------------------------------

@testset "seq_head_value! matches direct FCN kernel (Float64)" begin
    Random.seed!(71)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost1 = SeqCost("absErr")           # head 1: O == 1
    cost2 = SeqCost("sqErr")            # head 2: O == 3
    t1 = randn(Float64, 1)
    t2 = randn(Float64, 3)
    outs_before = copy(scratch.outputs)
    # Direct FCN reference: its kernel clobbers `a`, so pass a copy. Same
    # operations in the same order -> results must agree tightly.
    r1 = scratch.head_ranges[1]
    a1 = reshape(collect(@view scratch.outputs[r1]), 1, 1)
    calcFinalOut!(cost1.cost, a1, reshape(copy(t1), 1, 1), 1, 1)
    @test seq_head_value!(cost1, scratch, 1, t1) ≈ sum(a1)
    r2 = scratch.head_ranges[2]
    a2 = reshape(collect(@view scratch.outputs[r2]), 1, 3)
    calcFinalOut!(cost2.cost, a2, reshape(copy(t2), 1, 3), 1, 3)
    @test seq_head_value!(cost2, scratch, 2, t2) ≈ sum(a2)
    # Staging invariant: costs never touch the live outputs.
    @test scratch.outputs == outs_before
end

@testset "seq_head_grad! matches direct FCN kernel (Float64)" begin
    Random.seed!(72)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost1 = SeqCost("absErr")
    cost2 = SeqCost("sqErr")
    t1 = randn(Float64, 1)
    t2 = randn(Float64, 3)
    r1 = scratch.head_ranges[1]
    r2 = scratch.head_ranges[2]
    # Direct FCN references (copies of the predictions; this kernel only
    # reads them, but copying keeps the two paths obviously independent).
    deltas_ref1 = zeros(Float64, 1, 1)
    calcDeltaOut!(cost1.deriv, deltas_ref1,
                  reshape(collect(@view scratch.outputs[r1]), 1, 1),
                  reshape(copy(t1), 1, 1), 1, 1)
    deltas_ref2 = zeros(Float64, 1, 3)
    calcDeltaOut!(cost2.deriv, deltas_ref2,
                  reshape(collect(@view scratch.outputs[r2]), 1, 3),
                  reshape(copy(t2), 1, 3), 1, 3)
    seq_head_grad!(cost1, scratch, 1, t1)
    seq_head_grad!(cost2, scratch, 2, t2)
    @test vec(deltas_ref1) ≈ collect(@view scratch.dout[r1])
    @test vec(deltas_ref2) ≈ collect(@view scratch.dout[r2])
end

println("SEQ_COST item3 part1 tests done")

@testset "two-output cost (normLogErr): value + grad (Float64)" begin
    Random.seed!(73)
    D, Tmax, slot = 5, 6, 4
    set = init_seq_output_set(D, [2], :last, Float64)   # head width 2 = (a1, a2)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("normLogErr")
    target = randn(Float64, 1)                          # length = width ÷ 2
    r = scratch.head_ranges[1]
    # Direct value reference: length(a)==2*length(y) triggers the dual branch;
    # the loss lands in the first half only.
    a = reshape(collect(@view scratch.outputs[r]), 1, 2)
    calcFinalOut!(cost.cost, a, reshape(copy(target), 1, 1), 1, 1)
    @test seq_head_value!(cost, scratch, 1, target) ≈ a[1]
    # Direct grad reference: deltas receive (d1, d2) across both halves.
    deltas_ref = zeros(Float64, 1, 2)
    calcDeltaOut!(cost.deriv, deltas_ref,
                  reshape(collect(@view scratch.outputs[r]), 1, 2),
                  reshape(copy(target), 1, 1), 1, 1)
    seq_head_grad!(cost, scratch, 1, target)
    dout_slice = collect(@view scratch.dout[r])
    @test vec(deltas_ref) ≈ dout_slice
    @test all(isfinite, dout_slice)
end

@testset "numeric dJ/doutputs == seq_head_grad! dout (Float64)" begin
    Random.seed!(74)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    e = 1.0e-6
    # Head 2 (O == 3): two-sided difference of seq_head_value! against the
    # gradient seq_head_grad! wrote into dout.
    cost2 = SeqCost("sqErr")
    t2 = randn(Float64, 3)
    r2 = scratch.head_ranges[2]
    seq_head_grad!(cost2, scratch, 2, t2)
    analytic2 = collect(@view scratch.dout[r2])
    numeric2 = similar(analytic2)
    for (j, i) in enumerate(r2)
        orig = scratch.outputs[i]
        scratch.outputs[i] = orig + e
        jp = seq_head_value!(cost2, scratch, 2, t2)
        scratch.outputs[i] = orig - e
        jm = seq_head_value!(cost2, scratch, 2, t2)
        scratch.outputs[i] = orig
        numeric2[j] = (jp - jm) / (2e)
    end
    @test norm(numeric2 - analytic2) / norm(numeric2 + analytic2) < 1.0e-7
    # Head 1 (O == 1), scalar form of the same check.
    cost1 = SeqCost("absErr")
    t1 = randn(Float64, 1)
    seq_head_grad!(cost1, scratch, 1, t1)
    i1 = first(scratch.head_ranges[1])
    orig = scratch.outputs[i1]
    scratch.outputs[i1] = orig + e
    jp = seq_head_value!(cost1, scratch, 1, t1)
    scratch.outputs[i1] = orig - e
    jm = seq_head_value!(cost1, scratch, 1, t1)
    scratch.outputs[i1] = orig
    numeric1 = (jp - jm) / (2e)
    analytic1 = scratch.dout[i1]
    @test abs(numeric1 - analytic1) / max(abs(numeric1), abs(analytic1)) < 1.0e-5
end

@testset "seq_head_*: mismatch error, eltype assert, 0-alloc" begin
    Random.seed!(75)
    D, Tmax, slot = 5, 6, 2
    set = init_seq_output_set(D, [3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("sqErr")
    # Target length matches neither O nor O÷2 -> FCN kernel length check fires.
    @test_throws ErrorException seq_head_value!(cost, scratch, 1, randn(Float64, 2))
    @test_throws ErrorException seq_head_grad!(cost, scratch, 1, randn(Float64, 2))
    # Element type must match the scratch.
    @test_throws AssertionError seq_head_value!(cost, scratch, 1, randn(Float32, 3))
    @test_throws AssertionError seq_head_grad!(cost, scratch, 1, randn(Float32, 3))
    # Head index bounds.
    @test_throws AssertionError seq_head_value!(cost, scratch, 0, randn(Float64, 3))
    @test_throws AssertionError seq_head_grad!(cost, scratch, 2, randn(Float64, 3))
    # Steady-state 0-alloc (views were pre-built in form_output_scratch).
    t = randn(Float64, 3)
    seq_head_value!(cost, scratch, 1, t)   # warmup
    seq_head_grad!(cost, scratch, 1, t)
    @test Base.@allocated(seq_head_value!(cost, scratch, 1, t)) == 0
    @test Base.@allocated(seq_head_grad!(cost, scratch, 1, t)) == 0
end

println("SEQ_COST item3 tests done")

# ---------------------------------------------------------------------------
# Step 7 item 4: LossType cost application (seq_head_value!/seq_head_grad!)
# ---------------------------------------------------------------------------

@testset "CE index value matches FCN kernel (Float64)" begin
    Random.seed!(81)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("crossEntropy")
    @test cost isa SeqLossCost
    index = 2
    outs_before = copy(scratch.outputs)
    # Direct FCN reference (kernel clobbers `a`, so work on a copy).
    r = scratch.head_ranges[1]
    a = copy(reshape(collect(@view scratch.outputs[r]), 1, 4))
    calcFinalOut!(cost.loss, a, index)
    j_ref = a[index]
    j = seq_head_value!(cost, scratch, 1, index)
    @test j ≈ j_ref
    @test scratch.outputs == outs_before   # staging invariant
end

@testset "CE distribution value matches FCN kernel (Float64)" begin
    Random.seed!(82)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("crossEntropy")
    targets = [0.0, 1.0, 0.0, 0.0]
    outs_before = copy(scratch.outputs)
    r = scratch.head_ranges[1]
    a = copy(reshape(collect(@view scratch.outputs[r]), 1, 4))
    calcFinalOut!(cost.loss, a, targets)
    j_ref = a[1]   # kernel writes loss into entry 1
    j = seq_head_value!(cost, scratch, 1, targets)
    @test j ≈ j_ref
    @test scratch.outputs == outs_before   # staging invariant
end

@testset "OutputIndex value is raw activation (Float64)" begin
    Random.seed!(83)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("outputIndex")
    index = 3
    r = scratch.head_ranges[1]
    @test seq_head_value!(cost, scratch, 1, index) ≈ scratch.outputs[r][index]
end

println("SEQ_COST item4 value tests done")

@testset "CE index grad matches FCN kernel (Float64)" begin
    Random.seed!(84)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("crossEntropy")
    index = 2
    r = scratch.head_ranges[1]
    # Direct FCN reference (kernel reads `a`, writes deltas; copies for independence).
    deltas_ref = zeros(Float64, 1, 4)
    calcDeltaOut!(cost.loss, deltas_ref,
                  reshape(collect(@view scratch.outputs[r]), 1, 4),
                  index)
    seq_head_grad!(cost, scratch, 1, index)
    @test vec(deltas_ref) ≈ collect(@view scratch.dout[r])
end

@testset "CE distribution grad matches FCN kernel (Float64)" begin
    Random.seed!(85)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("crossEntropy")
    targets = [0.0, 1.0, 0.0, 0.0]
    r = scratch.head_ranges[1]
    deltas_ref = zeros(Float64, 4)
    calcDeltaOut!(cost.loss, deltas_ref,
                  collect(@view scratch.outputs[r]),
                  targets)
    seq_head_grad!(cost, scratch, 1, targets)
    @test vec(deltas_ref) ≈ collect(@view scratch.dout[r])
end

@testset "OutputIndex grad is one-hot at index (Float64)" begin
    Random.seed!(86)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("outputIndex")
    index = 3
    r = scratch.head_ranges[1]
    deltas_ref = zeros(Float64, 1, 4)
    calcDeltaOut!(cost.loss, deltas_ref,
                  reshape(collect(@view scratch.outputs[r]), 1, 4),
                  index)
    seq_head_grad!(cost, scratch, 1, index)
    @test vec(deltas_ref) ≈ collect(@view scratch.dout[r])
    @test scratch.dout[r][index] == 1.0
    @test all(i -> i == index ? scratch.dout[r][i] == 1.0 : scratch.dout[r][i] == 0.0, eachindex(r))
end

@testset "numeric dJ/doutputs == seq_head_grad! dout (CE index, Float64)" begin
    Random.seed!(87)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    cost = SeqCost("crossEntropy")
    index = 2
    r = scratch.head_ranges[1]
    seq_head_grad!(cost, scratch, 1, index)
    analytic = collect(@view scratch.dout[r])
    e = 1.0e-6
    numeric = similar(analytic)
    for (j, i) in enumerate(r)
        orig = scratch.outputs[i]
        scratch.outputs[i] = orig + e
        jp = seq_head_value!(cost, scratch, 1, index)
        scratch.outputs[i] = orig - e
        jm = seq_head_value!(cost, scratch, 1, index)
        scratch.outputs[i] = orig
        numeric[j] = (jp - jm) / (2e)
    end
    @test norm(numeric - analytic) / norm(numeric + analytic) < 1.0e-7
end

@testset "seq_head_*: index/target asserts, eltype assert, 0-alloc" begin
    Random.seed!(88)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [4], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    ce = SeqCost("crossEntropy")
    oi = SeqCost("outputIndex")
    # Index out of range.
    @test_throws AssertionError seq_head_value!(ce, scratch, 1, 0)
    @test_throws AssertionError seq_head_value!(ce, scratch, 1, 5)
    @test_throws AssertionError seq_head_grad!(ce, scratch, 1, 0)
    # Target length mismatch for distribution CE.
    @test_throws AssertionError seq_head_value!(ce, scratch, 1, [0.0, 1.0])
    @test_throws AssertionError seq_head_grad!(ce, scratch, 1, [0.0, 1.0])
    # Element type mismatch.
    @test_throws MethodError seq_head_value!(ce, scratch, 1, Float32[0.0, 1.0, 0.0, 0.0])
    @test_throws MethodError seq_head_grad!(ce, scratch, 1, Float32[0.0, 1.0, 0.0, 0.0])
    # Head index bounds.
    @test_throws AssertionError seq_head_value!(ce, scratch, 2, 1)
    @test_throws AssertionError seq_head_grad!(oi, scratch, 0, 1)
    # Steady-state 0-alloc (views pre-built).
    t4 = [0.0, 1.0, 0.0, 0.0]
    seq_head_value!(ce, scratch, 1, 2)   # warmup
    seq_head_grad!(ce, scratch, 1, 2)
    @test Base.@allocated(seq_head_value!(ce, scratch, 1, 2)) == 0
    @test Base.@allocated(seq_head_grad!(ce, scratch, 1, 2)) == 0
    @test Base.@allocated(seq_head_value!(oi, scratch, 1, 3)) == 0
    @test Base.@allocated(seq_head_grad!(oi, scratch, 1, 3)) == 0
end

println("SEQ_COST item4 grad tests done")

# ---------------------------------------------------------------------------
# Step 7 item 5: SeqOutputCosts collection + constructors
# ---------------------------------------------------------------------------

@testset "SeqHeadCost construction" begin
    cost = SeqCost("sqErr")
    target = [1.0, 2.0, 3.0]
    hc = SeqHeadCost(cost, target)
    @test hc.cost === cost
    @test hc.target === target
end

@testset "SeqOutputCosts construction: default weights" begin
    costs = [SeqCost("sqErr"), SeqCost("absErr")]
    targets = [[1.0, 2.0], [3.0]]
    spec = SeqOutputCosts(costs, targets)
    @test length(spec.heads) == 2
    @test spec.heads[1].cost === costs[1]
    @test spec.heads[1].target === targets[1]
    @test spec.heads[2].cost === costs[2]
    @test spec.heads[2].target === targets[2]
    @test spec.weights == ones(Float64, 2)
end

@testset "SeqOutputCosts construction: explicit weights" begin
    costs = [SeqCost("sqErr"), SeqCost("absErr")]
    targets = [[1.0, 2.0], [3.0]]
    weights = [2.0, 0.5]
    spec = SeqOutputCosts(costs, targets, weights)
    @test spec.weights == weights
    @test spec.heads[1].cost === costs[1]
    @test spec.heads[2].cost === costs[2]
end

@testset "SeqOutputCosts(set, ...) convenience constructor" begin
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    costs = [SeqCost("sqErr"), SeqCost("crossEntropy")]
    targets = [[0.5], 2]
    spec = SeqOutputCosts(set, costs, targets)
    @test length(spec.heads) == length(set.heads)
    @test spec.weights == ones(Float64, 2)
    @test spec.heads[1].target === targets[1]
    @test spec.heads[2].target === targets[2]
end

@testset "SeqOutputCosts: length mismatch errors" begin
    costs = [SeqCost("sqErr"), SeqCost("absErr")]
    targets = [[1.0]]   # only 1 target for 2 costs
    @test_throws AssertionError SeqOutputCosts(costs, targets)

    weights = [1.0, 2.0, 3.0]   # 3 weights for 2 costs
    @test_throws AssertionError SeqOutputCosts(costs, targets, weights)

    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    costs2 = [SeqCost("sqErr")]   # only 1 cost for 2 heads
    targets2 = [[0.5]]
    @test_throws AssertionError SeqOutputCosts(set, costs2, targets2)
end

println("SEQ_COST item5 constructor tests done")

# ---------------------------------------------------------------------------
# Step 7 item 5: seq_cost_grads! / seq_cost_values!
# ---------------------------------------------------------------------------

@testset "seq_cost_grads! writes weighted dout (Float64)" begin
    Random.seed!(91)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    # Head 1: sqErr, target [0.5], weight 2.0 → dout = 2.0*2*(out-0.5)
    #   (sqErrDeriv(a,y) = 2*(a-y); weight multiplies the kernel gradient)
    # Head 2: sqErr, target [0.1, 0.2, 0.3], weight 0.5 → dout = 0.5*2*(out-target)
    costs = [SeqCost("sqErr"), SeqCost("sqErr")]
    targets = [[0.5], [0.1, 0.2, 0.3]]
    weights = [2.0, 0.5]
    spec = SeqOutputCosts(costs, targets, weights)
    seq_cost_grads!(spec, scratch)
    # Verify head 1 dout
    r1 = scratch.head_ranges[1]
    dout1 = collect(scratch.dout_view[1])
    out1 = collect(scratch.outputs_view[1])
    @test dout1[1] ≈ 4.0 * (out1[1] - 0.5)
    # Verify head 2 dout
    r2 = scratch.head_ranges[2]
    dout2 = collect(scratch.dout_view[2])
    out2 = collect(scratch.outputs_view[2])
    @test all(dout2 .≈ 1.0 .* (out2 .- targets[2]))
    # Verify composition: dout feeds stack_backward!
    grads = zero_output_set(set)
    stack_scratch = scratch  # same scratch used by head_backward!
    head_backward!(scratch, set, grads, slot, slot)
    # dout should be unchanged after backward (head_backward! reads it)
    @test collect(scratch.dout_view[1]) ≈ dout1
    @test collect(scratch.dout_view[2]) ≈ dout2
end

@testset "seq_cost_values! returns weighted sum (Float64)" begin
    Random.seed!(92)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [1, 3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    costs = [SeqCost("sqErr"), SeqCost("absErr")]
    targets = [[0.5], [0.7, 0.6, 0.8]]
    weights = [3.0, 0.25]
    spec = SeqOutputCosts(costs, targets, weights)
    J = seq_cost_values!(spec, scratch)
    # Manual computation (each head's target length matches its width)
    out1 = collect(scratch.outputs_view[1])[1]
    out2 = collect(scratch.outputs_view[2])
    J_manual = 3.0 * (out1 - 0.5)^2 + 0.25 * sum(abs.(out2 .- targets[2]))
    @test J ≈ J_manual
end

@testset "seq_cost_grads! + head_backward! composition (Float64)" begin
    Random.seed!(93)
    D, Tmax, slot = 6, 8, 3
    set = init_seq_output_set(D, [2, 1], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    # Head 1: absErr, target [1.0, 0.5], weight 1.0 → dout = sign(out-target)
    # Head 2: sqErr, target [0.25], weight 0.0 → dout = 0
    costs = [SeqCost("absErr"), SeqCost("sqErr")]
    targets = [[1.0, 0.5], [0.25]]
    weights = [1.0, 0.0]
    spec = SeqOutputCosts(costs, targets, weights)
    seq_cost_grads!(spec, scratch)
    # Head 2 dout should be exactly zero (weight 0)
    r2 = scratch.head_ranges[2]
    @test all(iszero, scratch.dout_view[2])
    # Verify grads flow through stack_backward!
    grads = zero_output_set(set)
    head_backward!(scratch, set, grads, slot, slot)
    # trunk_grad should have nonzero gradient at slot row (from head 1)
    @test any(!iszero, scratch.trunk_grad[slot, :])
end

@testset "seq_cost_grads! 0-alloc (Float64)" begin
    Random.seed!(94)
    D, Tmax, slot = 5, 6, 2
    set = init_seq_output_set(D, [3], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    costs = [SeqCost("sqErr")]
    targets = [[0.0, 0.0, 0.0]]
    spec = SeqOutputCosts(costs, targets)
    # Warmup
    seq_cost_grads!(spec, scratch)
    @test Base.@allocated(seq_cost_grads!(spec, scratch)) == 0
end

@testset "seq_cost_values! weight scaling (Float64)" begin
    Random.seed!(95)
    D, Tmax, slot = 5, 6, 2
    set = init_seq_output_set(D, [2], :last, Float64)
    scratch = form_output_scratch(Float64, set, Tmax)
    head_forward!(scratch, set, randn(Float64, Tmax, D), slot)
    costs = [SeqCost("sqErr")]
    targets = [[0.0, 0.0]]
    # weight=0 → J=0
    spec0 = SeqOutputCosts(costs, targets, [0.0])
    @test seq_cost_values!(spec0, scratch) == 0.0
    # weight=1 → J = loss
    spec1 = SeqOutputCosts(costs, targets, [1.0])
    J1 = seq_cost_values!(spec1, scratch)
    out1 = collect(scratch.outputs_view[1])
    @test J1 ≈ sum(out1 .^ 2)
    # weight=2 → J = 2*loss
    spec2 = SeqOutputCosts(costs, targets, [2.0])
    @test seq_cost_values!(spec2, scratch) ≈ 2.0 * J1
end

println("SEQ_COST item5 grad/value tests done")