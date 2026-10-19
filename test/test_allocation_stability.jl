# Allocation, runtime-dispatch and type-stability checks for the low-level forward
# and backward pass kernels (calcFinalOut!, calcDeltaOut!, crossEntropyDeltaOut!,
# calcJ) plus the end-to-end nnCostFunction / nnCostFunctionNOGRAD wrappers.
#
# These kernels sit on the hot path of every training step, so they must be
# allocation free (no `fill`/`copy`/broadcast temporaries), statically dispatched
# (concrete element types, no Union splitting) and type stable.  Each check warms
# up (compiles) the method first, then measures `@allocated` of a second call.

using FCANN
using Test
using Random

# internal (unexported) low-level API
const calcDeltaOut!    = FCANN.calcDeltaOut!
const calcFinalOut!    = FCANN.calcFinalOut!
const crossEntropyDeltaOut! = FCANN.crossEntropyDeltaOut!
const calcJ            = FCANN.calcJ
const form_activations = FCANN.form_activations
const nnCostFunction   = FCANN.nnCostFunction
const nnCostFunctionNOGRAD = FCANN.nnCostFunctionNOGRAD

# warm up (compile) once, then measure a second call
function alloc_after_warmup(f, args...)
    f(args...)                      # warmup / compile
    return @allocated f(args...)
end

@testset "low-level kernel allocations" begin
    Random.seed!(1234)
    T = Float32
    m, n = 7, 5
    beta = T(0.5)
    loss = CrossEntropyLoss(beta)
    loss0 = CrossEntropyLoss()

    a_vec      = randn(T, n)
    d_vec      = similar(a_vec)
    targets_v  = (t = rand(T, n); t ./ sum(t))

    a_mat      = randn(T, m, n)
    d_mat      = similar(a_mat)
    idx_scalar = 2
    idx_vec    = [rand(1:n) for _ in 1:m]
    vals_vec   = randn(T, m)
    targets_m  = (t = rand(T, m, n); t ./ sum(t, dims = 2))

    Thetas = [randn(T, 4, 3), randn(T, n, 4)]

    # ---- single-example backward (vector activations) ----
    @test alloc_after_warmup((d, a, i) -> calcDeltaOut!(loss, d, a, i),
                             d_vec, copy(a_vec), idx_scalar) == 0
    @test alloc_after_warmup((d, a, i) -> calcDeltaOut!(loss0, d, a, i),
                             d_vec, copy(a_vec), idx_scalar) == 0
    @test alloc_after_warmup((d, a, i, v) -> calcDeltaOut!(loss, d, a, i, v),
                             d_vec, copy(a_vec), idx_scalar, T(2.5)) == 0
    @test alloc_after_warmup((d, a, i, v) -> calcDeltaOut!(loss0, d, a, i, v),
                             d_vec, copy(a_vec), idx_scalar, T(2.5)) == 0
    @test alloc_after_warmup((d, a, i) -> calcDeltaOut!(OutputIndex(), d, a, i),
                             d_vec, copy(a_vec), idx_scalar) == 0
    @test alloc_after_warmup((d, a, i, v) -> calcDeltaOut!(OutputIndex(), d, a, i, v),
                             d_vec, copy(a_vec), idx_scalar, T(2.5)) == 0
    @test alloc_after_warmup((d, a, t) -> calcDeltaOut!(loss, d, a, t),
                             d_vec, copy(a_vec), copy(targets_v)) == 0

    # ---- single-example forward (vector activations) ----
    @test alloc_after_warmup((a, i) -> calcFinalOut!(loss, a, i),
                             copy(a_vec), idx_scalar) == 0
    @test alloc_after_warmup((a, i) -> calcFinalOut!(loss0, a, i),
                             copy(a_vec), idx_scalar) == 0
    @test alloc_after_warmup((a, i, v) -> calcFinalOut!(loss, a, i, v),
                             copy(a_vec), idx_scalar, T(2.5)) == 0
    @test alloc_after_warmup((a, t) -> calcFinalOut!(loss, a, t),
                             copy(a_vec), copy(targets_v)) == 0
    @test alloc_after_warmup((a, i) -> calcFinalOut!(OutputIndex(), a, i),
                             copy(a_vec), idx_scalar) == 0

    # ---- batch backward (matrix activations) ----
    @test alloc_after_warmup((d, a, i) -> calcDeltaOut!(loss, d, a, i),
                             d_mat, copy(a_mat), idx_scalar) == 0
    @test alloc_after_warmup((d, a, i) -> calcDeltaOut!(loss0, d, a, i),
                             d_mat, copy(a_mat), idx_scalar) == 0
    @test alloc_after_warmup((d, a, ix) -> calcDeltaOut!(loss, d, a, ix),
                             d_mat, copy(a_mat), idx_vec) == 0
    @test alloc_after_warmup((d, a, ix, vs) -> calcDeltaOut!(loss, d, a, ix, vs),
                             d_mat, copy(a_mat), idx_vec, vals_vec) == 0
    @test alloc_after_warmup((d, a, ix, v) -> calcDeltaOut!(loss, d, a, ix, v),
                             d_mat, copy(a_mat), idx_vec, T(2.5)) == 0
    @test alloc_after_warmup((d, a, i, vs) -> calcDeltaOut!(loss, d, a, i, vs),
                             d_mat, copy(a_mat), idx_scalar, vals_vec) == 0
    @test alloc_after_warmup((d, a, i, v) -> calcDeltaOut!(loss, d, a, i, v),
                             d_mat, copy(a_mat), idx_scalar, T(2.5)) == 0
    @test alloc_after_warmup((d, a, t) -> calcDeltaOut!(loss, d, a, t),
                             d_mat, copy(a_mat), copy(targets_m)) == 0
    @test alloc_after_warmup((d, a, i) -> calcDeltaOut!(OutputIndex(), d, a, i),
                             d_mat, copy(a_mat), idx_scalar) == 0
    @test alloc_after_warmup((d, a, ix, vs) -> calcDeltaOut!(OutputIndex(), d, a, ix, vs),
                             d_mat, copy(a_mat), idx_vec, vals_vec) == 0

    # ---- raw softmax helper ----
    @test alloc_after_warmup((d, a) -> crossEntropyDeltaOut!(d, a),
                             d_mat, copy(a_mat)) == 0
    @test alloc_after_warmup((d, ix) -> crossEntropyDeltaOut!(d, ix),
                             d_mat, idx_vec) == 0
    @test alloc_after_warmup((d, ix, vs) -> crossEntropyDeltaOut!(d, ix, vs),
                             d_mat, idx_vec, vals_vec) == 0
    @test alloc_after_warmup((d, ix, v) -> crossEntropyDeltaOut!(d, ix, v),
                             d_mat, idx_vec, T(2.5)) == 0
    @test alloc_after_warmup((d) -> crossEntropyDeltaOut!(d, idx_scalar),
                             d_mat) == 0
    @test alloc_after_warmup((d, v) -> crossEntropyDeltaOut!(d, idx_scalar, v),
                             d_mat, T(2.5)) == 0

    # ---- batch forward (matrix activations) ----
    @test alloc_after_warmup((a, i) -> calcFinalOut!(loss, a, i),
                             copy(a_mat), idx_scalar) == 0
    @test alloc_after_warmup((a, i) -> calcFinalOut!(loss0, a, i),
                             copy(a_mat), idx_scalar) == 0
    @test alloc_after_warmup((a, ix) -> calcFinalOut!(loss, a, ix),
                             copy(a_mat), idx_vec) == 0
    @test alloc_after_warmup((a, ix, vs) -> calcFinalOut!(loss, a, ix, vs),
                             copy(a_mat), idx_vec, vals_vec) == 0
    @test alloc_after_warmup((a, ix, v) -> calcFinalOut!(loss, a, ix, v),
                             copy(a_mat), idx_vec, T(2.5)) == 0
    @test alloc_after_warmup((a, i, vs) -> calcFinalOut!(loss, a, i, vs),
                             copy(a_mat), idx_scalar, vals_vec) == 0
    @test alloc_after_warmup((a, i, v) -> calcFinalOut!(loss, a, i, v),
                             copy(a_mat), idx_scalar, T(2.5)) == 0
    @test alloc_after_warmup((a, t) -> calcFinalOut!(loss, a, t),
                             copy(a_mat), copy(targets_m)) == 0
    @test alloc_after_warmup((a, i) -> calcFinalOut!(OutputIndex(), a, i),
                             copy(a_mat), idx_scalar) == 0

    # ---- cost aggregation ----
    @test alloc_after_warmup((d, i, th) -> calcJ(d, i, T(0.1), th),
                             copy(a_mat), idx_scalar, Thetas) == 0
    @test alloc_after_warmup((d, ix, th) -> calcJ(d, ix, T(0.1), th),
                             copy(a_mat), idx_vec, Thetas) == 0
    @test alloc_after_warmup((d, ix, vs, th) -> calcJ(d, ix, vs, T(0.1), th),
                             copy(a_mat), idx_vec, vals_vec, Thetas) == 0
    @test alloc_after_warmup((d, i, th) -> calcJ(d, i, T(0.1), th),
                             copy(a_vec), idx_scalar, Thetas) == 0
    @test alloc_after_warmup((d, i, v, th) -> calcJ(d, i, v, T(0.1), th),
                             copy(a_vec), idx_scalar, T(2.5), Thetas) == 0
end

@testset "low-level kernel type stability" begin
    Random.seed!(1234)
    T = Float32
    m, n = 7, 5
    loss = CrossEntropyLoss(T(0.5))
    a_vec     = randn(T, n)
    d_vec     = similar(a_vec)
    a_mat     = randn(T, m, n)
    d_mat     = similar(a_mat)
    idx_vec   = [rand(1:n) for _ in 1:m]
    vals_vec  = randn(T, m)
    targets_v = (t = rand(T, n); t ./ sum(t))
    Thetas    = [randn(T, 4, 3), randn(T, n, 4)]

    # concrete return types (Nothing / Float32) prove inference sees through
    # the loss structs and value containers with no Union splitting
    @inferred calcDeltaOut!(loss, d_vec, copy(a_vec), 2)
    @inferred calcDeltaOut!(loss, d_vec, copy(a_vec), 2, 1.5f0)
    @inferred calcDeltaOut!(loss, d_mat, copy(a_mat), idx_vec, vals_vec)
    @inferred calcDeltaOut!(loss, d_mat, copy(a_mat), idx_vec, 1.5f0)
    @inferred calcDeltaOut!(loss, d_mat, copy(a_mat), 2, vals_vec)
    @inferred calcFinalOut!(loss, copy(a_vec), 2)
    @inferred calcFinalOut!(loss, copy(a_mat), idx_vec, vals_vec)
    @inferred calcFinalOut!(loss, copy(a_mat), idx_vec, 1.5f0)
    @inferred calcFinalOut!(loss, copy(a_mat), 2, vals_vec)
    @inferred calcFinalOut!(loss, copy(a_vec), targets_v)
    @test @inferred(calcJ(copy(a_mat), idx_vec, vals_vec, 0.1f0, Thetas)) isa Float32
    @test @inferred(calcJ(copy(a_vec), 2, 0.1f0, Thetas)) isa Float32
    @test @inferred(calcJ(copy(a_vec), 2, 1.5f0, 0.1f0, Thetas)) isa Float32
end

@testset "end-to-end cost function stability" begin
    Random.seed!(1234)
    T = Float32
    input_layer_size, hidden_layers, output_layer_size = 3, [4, 4], 2
    m = 6
    T0, B0 = initializeParams(input_layer_size, hidden_layers, output_layer_size)
    X = randn(T, m, input_layer_size)
    x = randn(T, input_layer_size)
    idx_vec = [rand(1:output_layer_size) for _ in 1:m]
    vals_vec = randn(T, m)
    one_idx = rand(1:output_layer_size)
    one_val = vals_vec[1]
    onesVec = ones(T, m)
    al = fill(true, length(hidden_layers))

    for loss_type in (OutputIndex(), CrossEntropyLoss(T(0.0)), CrossEntropyLoss(T(0.5)))
        Theta_grads = deepcopy(T0); Bias_grads = deepcopy(B0)
        a  = form_activations(T0, m); tgz = deepcopy(a); del = deepcopy(a)
        kwargs = (activation_list = al, loss_type = loss_type)

        # batch, shared index
        nnCostFunction(T0, B0, hidden_layers, X, one_idx, 0.0f0,
                       Theta_grads, Bias_grads, tgz, a, del, onesVec; kwargs...)
        allocs = alloc_after_warmup(
            (A, TG, BG, TZ, D) -> nnCostFunction(T0, B0, hidden_layers, X, one_idx, 0.0f0,
                                                 TG, BG, TZ, A, D, onesVec; kwargs...),
            a, Theta_grads, Bias_grads, tgz, del)
        @test allocs == 0

        # batch, per-example indices
        nnCostFunction(T0, B0, hidden_layers, X, idx_vec, 0.0f0,
                       Theta_grads, Bias_grads, tgz, a, del, onesVec; kwargs...)
        allocs = alloc_after_warmup(
            (A, TG, BG, TZ, D) -> nnCostFunction(T0, B0, hidden_layers, X, idx_vec, 0.0f0,
                                                 TG, BG, TZ, A, D, onesVec; kwargs...),
            a, Theta_grads, Bias_grads, tgz, del)
        @test allocs == 0

        # batch, per-example indices + values
        nnCostFunction(T0, B0, hidden_layers, X, idx_vec, vals_vec, 0.0f0,
                       Theta_grads, Bias_grads, tgz, a, del, onesVec; kwargs...)
        allocs = alloc_after_warmup(
            (A, TG, BG, TZ, D) -> nnCostFunction(T0, B0, hidden_layers, X, idx_vec, vals_vec, 0.0f0,
                                                 TG, BG, TZ, A, D, onesVec; kwargs...),
            a, Theta_grads, Bias_grads, tgz, del)
        @test allocs == 0

        # single example
        a1  = form_activations(T0); tgz1 = deepcopy(a1); del1 = deepcopy(a1)
        nnCostFunction(T0, B0, hidden_layers, x, one_idx, 0.0f0,
                       Theta_grads, Bias_grads, tgz1, a1, del1; kwargs...)
        allocs = alloc_after_warmup(
            (A, TG, BG, TZ, D) -> nnCostFunction(T0, B0, hidden_layers, x, one_idx, 0.0f0,
                                                 TG, BG, TZ, A, D; kwargs...),
            a1, Theta_grads, Bias_grads, tgz1, del1)
        @test allocs == 0

        # single example + scalar value
        nnCostFunction(T0, B0, hidden_layers, x, one_idx, one_val, 0.0f0,
                       Theta_grads, Bias_grads, tgz1, a1, del1; kwargs...)
        allocs = alloc_after_warmup(
            (A, TG, BG, TZ, D) -> nnCostFunction(T0, B0, hidden_layers, x, one_idx, one_val, 0.0f0,
                                                 TG, BG, TZ, A, D; kwargs...),
            a1, Theta_grads, Bias_grads, tgz1, del1)
        @test allocs == 0

        # forward NOGRAD costs (return Float32)
        an = form_activations(T0, m)
        J = nnCostFunctionNOGRAD(T0, B0, hidden_layers, X, idx_vec, 0.0f0, an; kwargs...)
        @test J isa Float32
        allocs = alloc_after_warmup(
            (A) -> nnCostFunctionNOGRAD(T0, B0, hidden_layers, X, idx_vec, 0.0f0, A; kwargs...),
            an)
        @test allocs == 0
        J = nnCostFunctionNOGRAD(T0, B0, hidden_layers, X, idx_vec, vals_vec, 0.0f0, an; kwargs...)
        @test J isa Float32
        allocs = alloc_after_warmup(
            (A) -> nnCostFunctionNOGRAD(T0, B0, hidden_layers, X, idx_vec, vals_vec, 0.0f0, A; kwargs...),
            an)
        @test allocs == 0
        J = nnCostFunctionNOGRAD(T0, B0, hidden_layers, X, one_idx, one_val, 0.0f0, an; kwargs...)
        @test J isa Float32
        allocs = alloc_after_warmup(
            (A) -> nnCostFunctionNOGRAD(T0, B0, hidden_layers, X, one_idx, one_val, 0.0f0, A; kwargs...),
            an)
        @test allocs == 0
    end
end