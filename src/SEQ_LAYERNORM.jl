# Sequence LayerNorm for FCANN (Step 2, work in progress).
#
# Per-token LayerNorm with explicit forward + backward, in the FCN philosophy
# (`tanhGradient!` / `finishDelta!` idiom: in-place `!`, `@inbounds @simd`,
# pre-allocated scratch, generic AbstractFloat). This file is built one
# function at a time for review. Nothing here is included by `FCANN.jl` yet,
# so existing behavior is unchanged.

"""
    init_layernorm_params(model_dim::Int, ::Type{FloatType}=Float32) -> (gain, bias)

Create LayerNorm scale + shift parameters for a `model_dim`-wide token row.
`gain` starts at ones and `bias` at zeros, so the transform is the identity
until training moves it — the same known-good-start philosophy as
`initializeParams` in `FCN_NFLOATOUT_AUXFUNCTIONS.jl`, minus fan-in scaling,
which does not apply to a normalize-then-scale parameter.

# Arguments
- `model_dim`: token width; must be positive.
- `FloatType`: element type (`Float32` default, `Float64`/`Float16` allowed).

# Returns
- `gain::Vector{FloatType}`: per-column scale, all ones.
- `bias::Vector{FloatType}`: per-column shift, all zeros.
"""
function init_layernorm_params(model_dim::Int, ::Type{FloatType}=Float32) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert model_dim > 0 "model_dim must be positive, got $model_dim"
    # Identity start: y = 1 .* x_hat + 0 = x_hat (see layernorm_forward!).
    gain = ones(FloatType, model_dim)
    bias = zeros(FloatType, model_dim)
    return gain, bias
end

"""
    LayerNormScratch(row_mean, row_inv_std, row_normed)

Pre-allocated workspace for one LayerNorm forward + backward pass over a
batch of tokens. Created by `form_layernorm_scratch`; users should not
construct it field-by-field.

# Fields
- `row_mean::Vector{FloatType}`: per-token mean `μ`, length `max_tokens`.
- `row_inv_std::Vector{FloatType}`: per-token `r = 1/sqrt(σ² + ϵ)`, cached for
  backward (avoids recomputing the square root per element).
- `row_normed::Matrix{FloatType}`: per-token normalized rows `x̂`,
  `(max_tokens, model_dim)`, cached for backward — the same role
  `tanh_grad_z` plays in `nnCostFunction`.
- `row_dot::Vector{FloatType}`: per-token `s2 = Σ_j dy_j * x̂_j` staging for
  backward (length `max_tokens`). The fused backward needs two row scalars
  (`s1 = Σ dy` reuses `row_mean`; `r` must stay live in `row_inv_std`), so a
  dedicated vector keeps every pass contiguous with zero allocation.
"""
struct LayerNormScratch{T}
    row_mean::Vector{T}
    row_inv_std::Vector{T}
    row_normed::Matrix{T}
    row_dot::Vector{T}
end

"""
    form_layernorm_scratch(::Type{FloatType}, max_tokens::Int, model_dim::Int) -> LayerNormScratch

Allocate zero-filled LayerNorm scratch for up to `max_tokens` tokens of width
`model_dim`. Mirrors `form_tanh_grads(hidden_layers, m)` in
`FCN_NFLOATOUT_COSTFUNCTIONS.jl`: one allocation up front, zero allocation
afterwards.

# Sizing
`max_tokens` is the caller's upper bound on valid tokens per forward call —
typically `max_batch_size * max_seq_length`, or smaller when the caller loops
slot-by-slot and reuses one slot-sized scratch. Only rows `1:num_tokens` are
written per call; the rest is never read.

# Arguments
- `FloatType`: element type matching the gain/bias/input dtype.
- `max_tokens`: maximum tokens per call; must be positive.
- `model_dim`: token width; must be positive.
"""
function form_layernorm_scratch(::Type{FloatType}, max_tokens::Int, model_dim::Int) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert max_tokens > 0 "max_tokens must be positive, got $max_tokens"
    @assert model_dim > 0 "model_dim must be positive, got $model_dim"
    # Zero-fill: unwritten rows read as 0, so a caller bug that reads past
    # num_tokens degrades to zeros rather than NaNs leaking silently.
    row_mean = zeros(FloatType, max_tokens)
    row_inv_std = zeros(FloatType, max_tokens)
    row_normed = zeros(FloatType, max_tokens, model_dim)
    row_dot = zeros(FloatType, max_tokens)
    return LayerNormScratch(row_mean, row_inv_std, row_normed, row_dot)
end

"""
    layernorm_forward!(output, scratch, input, gain, bias, num_tokens, epsilon) -> Nothing

Normalize each of the first `num_tokens` rows of `input` to zero mean / unit
variance, then scale by `gain` and shift by `bias`, writing rows into
`output`. Caches per-token `μ`, `r = 1/sqrt(σ² + ϵ)`, and `x̂` in `scratch`
for `layernorm_backward!`. All writes in place; allocates nothing.

Math per token row `x` (independent across rows):

    μ   = (1/d) Σ_j x_j
    σ²  = (1/d) Σ_j (x_j - μ)²
    r   = 1/sqrt(σ² + ϵ)
    x̂_j = (x_j - μ) * r
    y_j = gain_j * x̂_j + bias_j

# Arguments
- `output`: `(>= num_tokens, model_dim)` write target, any strided matrix
  (e.g. a `@view` into a larger activation buffer).
- `scratch`: `LayerNormScratch` sized for `>= num_tokens` rows.
- `input`: `(>= num_tokens, model_dim)` read source; only rows `1:num_tokens`
  are read, so padding rows are never touched — no mask argument needed.
- `gain`, `bias`: `(model_dim,)` vectors from `init_layernorm_params`.
- `num_tokens`: rows to process; `0` is a no-op (all-empty batch).
- `epsilon`: small floor inside the square root; must satisfy
  `epsilon >= eps(FloatType)` so `r` cannot overflow. Default `1e-5` in the
  working dtype (see Step 2 spec §4 for the per-type rationale).
"""
function layernorm_forward!(output::AbstractMatrix{T}, scratch::LayerNormScratch{T},
        input::AbstractMatrix{T}, gain::AbstractVector{T}, bias::AbstractVector{T},
        num_tokens::Int, epsilon::T=T(1e-5)) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert num_tokens >= 0 "num_tokens must be non-negative, got $num_tokens"
    @assert epsilon >= eps(T) "epsilon $epsilon below eps($T) = $(eps(T)); inv_std could overflow"
    model_dim = length(gain)
    @assert length(bias) == model_dim "gain/bias length mismatch: $(length(gain)) vs $(length(bias))"
    @assert size(input, 2) == model_dim && size(output, 2) == model_dim "input/output width must equal model_dim $model_dim"
    @assert size(input, 1) >= num_tokens && size(output, 1) >= num_tokens "input/output need >= num_tokens $num_tokens rows"
    @assert length(scratch.row_mean) >= num_tokens "scratch sized for $(length(scratch.row_mean)) tokens, need $num_tokens"
    inv_dim = one(T) / T(model_dim)
    # Memory order note: Julia arrays are column-major, so element (t, j)
    # sits at t + (j-1)*stride — varying t streams contiguously, varying j
    # strides by the leading dimension. The loops below are therefore
    # feature-outer / token-inner: the inner @simd loop always walks t
    # contiguously. A token-outer / feature-inner nesting would stride
    # through memory on every inner step and trash the cache.
    # Pass 1: per-token mean into scratch.row_mean.
    scratch.row_mean .= zero(T)
    for j in 1:model_dim
        @inbounds @simd for t in 1:num_tokens
            scratch.row_mean[t] += input[t, j]
        end
    end
    scratch.row_mean .*= inv_dim
    
    # Pass 2: per-token variance -> inv std. Accumulate squared deviations
    # with the same contiguous nesting.
    for t in 1:num_tokens
        @inbounds scratch.row_inv_std[t] = zero(T)
    end
    for j in 1:model_dim
        @inbounds @simd for t in 1:num_tokens
            centered = input[t, j] - scratch.row_mean[t]
            scratch.row_inv_std[t] += centered * centered
        end
    end
    # Epsilon floors the radicand so a constant row (σ² == 0) yields
    # r = 1/sqrt(ϵ) instead of Inf.
    @inbounds @simd for t in 1:num_tokens
        scratch.row_inv_std[t] = one(T) / sqrt(scratch.row_inv_std[t] * inv_dim + epsilon)
    end
    # Pass 3: normalize, cache x̂, scale + shift into output. Gain/bias are
    # per-column, so hoisting the j loop outside keeps them loop-invariant
    # while t streams.
    for j in 1:model_dim
        gain_j = gain[j]
        bias_j = bias[j]
        @inbounds @simd for t in 1:num_tokens
            x_hat = (input[t, j] - scratch.row_mean[t]) * scratch.row_inv_std[t]
            scratch.row_normed[t, j] = x_hat
            output[t, j] = gain_j * x_hat + bias_j
        end
    end
    return nothing
end

"""
    layernorm_backward!(dx, param_grads, dy, scratch, gain, num_tokens) -> Nothing

Backpropagate upstream `dy` through the `layernorm_forward!` result, writing
`dx` and accumulating into `param_grads` (gain grads in column 1, bias grads
in column 2). Reads cached `x̂` and `r` from `scratch`. All writes in place;
allocates nothing.

Math per token row (row subscript suppressed; `d = model_dim`):

    db_j += dy_j                          # bias is additive
    dg_j += dy_j * x̂_j                    # gain scales x̂
    s1    = Σ_k gain_k * dy_k             # row scalar, pass 1
    s2    = Σ_k gain_k * dy_k * x̂_k       # row scalar, pass 1
    dx_j  = (r / d) * (d*gain_j*dy_j - s1 - x̂_j * s2)

Derivation sketch: `y_j = g_j * x̂_j + b_j` with `x̂_j = (x_j - μ) r`. The
Jacobian `∂x̂_j/∂x_i = r (δ_ji - 1/d - x̂_j x̂_i / d)` (mean-subtraction gives
`-1/d`, variance path gives `-x̂_j x̂_i/d`) contracts with the gain-weighted
upstream `g_j * dy_j` to the fused form above — no `d×d` matrix is ever
formed. Note the gain sits *inside* the row sums: `s1 = Σ g·dy`, not `Σ dy`.
The grad-check in `test/test_seq_layernorm.jl` verifies this against finite
differences.

Loop nesting is feature-outer / token-inner throughout (see the memory-order
note in `layernorm_forward!`): inner `@simd` loops always walk contiguous `t`.
Staging: `s1` reuses `scratch.row_mean` (forward-only cache, already
consumed); `s2` goes in `scratch.row_dot`; `r` stays live in
`scratch.row_inv_std` throughout.

# Arguments
- `dx`: `(>= num_tokens, model_dim)` write target for input gradients.
- `param_grads`: `(model_dim, 2)` accumulation target; column 1 += gain
  grads, column 2 += bias grads (`+=`, not `=`, so repeated calls sum —
  mirrors how `Theta_grads` accumulate across batches in `nnCostFunction`).
- `dy`: `(>= num_tokens, model_dim)` upstream gradients; only rows
  `1:num_tokens` are read.
- `scratch`: `LayerNormScratch` filled by the matching forward call.
- `gain`: `(model_dim,)` forward gain (needed in the `dx` scale factor).
- `num_tokens`: rows to process; `0` is a no-op.

The token width `model_dim` is derived from `length(gain)` rather than passed,
mirroring `layernorm_forward!` — the gain vector is structure already in hand,
so re-deriving avoids trusting a separate scalar.
"""
function layernorm_backward!(dx::AbstractMatrix{T}, param_grads::AbstractMatrix{T},
        dy::AbstractMatrix{T}, scratch::LayerNormScratch{T}, gain::AbstractVector{T},
        num_tokens::Int) where {T <: AbstractFloat}
    # Derive the width from the gain vector (structure in hand), as the forward does.
    model_dim = length(gain)
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert num_tokens >= 0 "num_tokens must be non-negative, got $num_tokens"
    @assert size(dx, 2) == model_dim && size(dy, 2) == model_dim "dx/dy width must equal model_dim $model_dim"
    @assert size(dx, 1) >= num_tokens && size(dy, 1) >= num_tokens "dx/dy need >= num_tokens $num_tokens rows"
    @assert size(param_grads) == (model_dim, 2) "param_grads must be (model_dim, 2), got $(size(param_grads))"
    @assert length(scratch.row_mean) >= num_tokens "scratch sized for $(length(scratch.row_mean)) tokens, need $num_tokens"
    inv_dim = one(T) / T(model_dim)
    dim_T = T(model_dim)
    x_hat = scratch.row_normed
    r_vec = scratch.row_inv_std
    s1_vec = scratch.row_mean
    s2_vec = scratch.row_dot
    # Pass 1: row scalars s1 = Σ g·dy into s1_vec, s2 = Σ g·dy*x̂ into s2_vec.
    # Both reductions are j-outer / t-inner so every inner loop streams.
    # Gain is folded in here (not just outside): the Jacobian contracts with
    # g_j * dy_j, so the row sums must be gain-weighted.
    @inbounds @simd for t in 1:num_tokens
        s1_vec[t] = zero(T)
        s2_vec[t] = zero(T)
    end
    for j in 1:model_dim
        gain_j = gain[j]
        @inbounds @simd for t in 1:num_tokens
            gdy = gain_j * dy[t, j]
            s1_vec[t] += gdy
            s2_vec[t] += gdy * x_hat[t, j]
        end
    end
    # Pass 2: param grads (column streaming) + dx (fused form). r/s1/s2 are
    # per-token cached scalars; gain_j scales only its own column's dy term.
    for j in 1:model_dim
        gain_j = gain[j]
        @inbounds @simd for t in 1:num_tokens
            dy_tj = dy[t, j]
            x_hat_tj = x_hat[t, j]
            param_grads[j, 1] += dy_tj * x_hat_tj
            param_grads[j, 2] += dy_tj
            scale = r_vec[t] * inv_dim
            dx[t, j] = scale * (dim_T * gain_j * dy_tj - s1_vec[t] - x_hat_tj * s2_vec[t])
        end
    end
    return nothing
end
