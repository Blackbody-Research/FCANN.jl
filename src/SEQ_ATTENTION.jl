# Sequence single-head attention for FCANN (Step 3, work in progress).
#
# Scaled dot-product attention with explicit forward + backward, in the FCN
# philosophy (in-place `!`, `@inbounds @simd`, pre-allocated scratch, generic
# AbstractFloat). Q/K/V are direct inputs; projections arrive in Step 4.
# This file is built one function at a time for review. Nothing here is
# included by `FCANN.jl` yet, so existing behavior is unchanged.

"""
    AttentionScratch(scores, probs)

Pre-allocated workspace for one single-head attention forward + backward pass
over a sequence of up to `max_seq_length` tokens. Created by
`form_attention_scratch`; users should not construct it field-by-field.

# Fields
- `scores::Matrix{FloatType}`: raw affinities `S = Q*K'/sqrt(dk) + bias`,
  `(max_seq_length, max_seq_length)`. Only the leading
  `active_length × active_length` block is written per call.
- `probs::Matrix{FloatType}`: row-softmax weights `P`, same shape. Cached for
  backward — the attention analogue of `tanh_grad_z` in `nnCostFunction` and
  `row_normed` in `LayerNormScratch`.
- `row_dot::Vector{FloatType}`: per-query `Σ_k dP[i,k]*P[i,k]` staging for the
  softmax-Jacobian backward (length `max_seq_length`). Lets both sweeps run
  column-streaming with zero allocation — mirrors `LayerNormScratch.row_dot`.
"""
struct AttentionScratch{T}
    scores::Matrix{T}
    probs::Matrix{T}
    row_dot::Vector{T}
end

"""
    form_attention_scratch(::Type{FloatType}, max_seq_length::Int) -> AttentionScratch

Allocate zero-filled attention scratch for sequences up to `max_seq_length`.
Mirrors `form_layernorm_scratch` in `SEQ_LAYERNORM.jl`: one allocation up
front, zero allocation afterwards.

# Sizing
Both matrices are `(max_seq_length, max_seq_length)` — 32KB at
`max_seq_length = 64`, 128KB at 128 (Float32, both matrices). Only the
leading `active_length × active_length` block is written per call; the rest
is never read.

# Arguments
- `FloatType`: element type matching the Q/K/V/output dtype.
- `max_seq_length`: maximum tokens per call; must be positive.
"""
function form_attention_scratch(::Type{FloatType}, max_seq_length::Int) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert max_seq_length > 0 "max_seq_length must be positive, got $max_seq_length"
    # Zero-fill: unwritten blocks read as 0, so a caller bug that reads past
    # active_length degrades to zeros rather than NaNs leaking silently.
    scores = zeros(FloatType, max_seq_length, max_seq_length)
    probs = zeros(FloatType, max_seq_length, max_seq_length)
    row_dot = zeros(FloatType, max_seq_length)
    return AttentionScratch(scores, probs, row_dot)
end

"""
    attention_forward!(output, scratch, queries, keys, values, bias, active_length, slot_length, mode) -> Nothing

Single-head scaled dot-product attention over the first `active_length`
positions, writing `output` rows `1:active_length`. All writes in place;
allocates nothing.

Math (row `i` = query position, column `j` = key position):

    S[i,j] = dot(Q[i,:], K[j,:]) / sqrt(dk) + bias[i,j]
    P[i,:] = softmax(S[i,:])          # max-subtracted, numerically stable
    O[i,:] = sum_j P[i,j] * V[j,:]

The `bias` matrix carries the mask: allowed keys hold `0`, blocked keys hold
`-Inf` (so `exp(blocked - max) = 0` and they get exactly zero weight). Two
mask patterns, selected by `mode`:

- `:causal`: query `i` sees keys `j <= min(i, slot_length)` — prediction at
  `i` never peeks at the future. One forward pass yields `T` predictions at
  `T` masking levels (autoregressive training).
- `:bidirectional`: every query sees keys `1..slot_length` — the full valid
  history mixes into every output row. Read `O[slot_length,:]` for a
  sequence-level output head.

Rows `i > slot_length` (dummy queries) read an all-`-Inf` bias row; their
outputs are meaningless and must be discarded by the caller (loss/gradient
entries zeroed downstream). Only the leading `active_length` rows/columns of
`scratch` are written.

Column-major note: `Q*K'` and `P*V` go through `BLAS.gemm!` on strided views
(the Step 1-tested pattern); the row-softmax follows the `max`-subtraction
idiom of `crossEntropyDeltaOut!` (`FCN_NFLOATOUT_COSTFUNCTIONS.jl:336-372`),
with query-outer / key-inner nesting so each row's max/exp/normalize pass
streams contiguously.

# Arguments
- `output`: `(>= active_length, dk)` write target.
- `scratch`: `AttentionScratch` sized for `>= active_length`.
- `queries`, `keys`, `values`: `(>= active_length, dk)` inputs; only rows
  `1:active_length` are read.
- `bias`: `(>= active_length, >= active_length)` mask bias (`0`/`-Inf`),
  e.g. the `attention_bias` block written by `fill_causal_padding_bias!`.
  `:bidirectional` callers build the padding-only pattern (all valid rows
  identical) with a later helper.
- `active_length`: positions to compute (`maximum(batch_lengths)`).
- `slot_length`: valid tokens in this slot (`0` = unused; no-op).
- `mode`: `:causal` or `:bidirectional` (asserted, not branched around —
  both share scores/softmax/mixing; only the bias pattern differs).
"""
function attention_forward!(output::AbstractMatrix{T}, scratch::AttentionScratch{T},
        queries::AbstractMatrix{T}, keys::AbstractMatrix{T}, values::AbstractMatrix{T},
        bias::AbstractMatrix{T}, active_length::Int, slot_length::Int, mode::Symbol) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert mode === :causal || mode === :bidirectional "mode must be :causal or :bidirectional, got $mode"
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    head_dim = size(queries, 2)
    @assert head_dim > 0 "head_dim must be positive"
    @assert size(keys, 2) == head_dim && size(values, 2) == head_dim "keys/values width must equal queries width $head_dim"
    @assert size(output, 2) == head_dim "output width must equal head_dim $head_dim"
    @assert size(queries, 1) >= active_length && size(keys, 1) >= active_length "queries/keys need >= active_length $active_length rows"
    @assert size(values, 1) >= active_length && size(output, 1) >= active_length "values/output need >= active_length $active_length rows"
    @assert size(bias, 1) >= active_length && size(bias, 2) >= active_length "bias needs >= active_length $active_length rows/cols"
    @assert size(scratch.scores, 1) >= active_length && size(scratch.scores, 2) >= active_length "scratch sized for $(size(scratch.scores, 1)), need $active_length"
    # Unused slot: nothing valid to compute; leave scratch/output untouched so
    # a caller bug reading them degrades visibly rather than silently.
    slot_length == 0 && return nothing
    scores = scratch.scores
    probs = scratch.probs
    # Pass 1: raw affinities S = Q*K'/sqrt(dk) into the leading block. Views
    # are strided (Step 1-tested pattern), so gemm! takes them directly with
    # zero allocation and zero copy.
    query_view = @view queries[1:active_length, :]
    key_view = @view keys[1:active_length, :]
    score_view = @view scores[1:active_length, 1:active_length]
    inv_scale = one(T) / sqrt(T(head_dim))
    BLAS.gemm!('N', 'T', inv_scale, query_view, key_view, zero(T), score_view)
    # Pass 2: add the mask bias in place. Both modes share this: the bias
    # pattern already encodes causal-vs-bidirectional (written by the mask
    # helper), so no branch on mode here.
    # Deliberately NOT BLAS.axpy!: score_view is a leading block of a larger
    # matrix, i.e. a strided SubArray, and axpy!(n, a, X, 1, Y, 1) treats its
    # inputs as flat contiguous vectors — on a partial block it silently adds
    # the wrong elements and scribbles outside the block (verified
    # /tmp/check_axpy_valid.jl: block correct = false, outside clobbered).
    # The explicit column-streaming loop is correct at all sizes and 0-alloc.
    @inbounds for j in 1:active_length
        @simd for i in 1:active_length
            score_view[i, j] += bias[i, j]
        end
    end
    # Pass 3: max-subtracted row-softmax S -> P, the crossEntropyDeltaOut!
    # idiom (FCN_NFLOATOUT_COSTFUNCTIONS.jl:336-372): per-row max, exp/sum,
    # normalize. Query-outer / key-inner nesting streams each row
    # contiguously. -Inf entries yield exp(-Inf - max) = 0 exactly; dummy
    # query rows (all -Inf) force row_sum == 0 and are set to an exact zero
    # row (inv_sum = 0) rather than NaN, so NaN never enters the PV gemm!
    # or the backward d(V) = P' * d(O).
    prob_view = @view probs[1:active_length, 1:active_length]
    for i in 1:active_length
        # ifelse (not ternary) guarantees branchless max codegen regardless of
        # compiler heuristic; benchmarked identical-or-faster at all N.
        row_max = score_view[i, 1]
        @inbounds @simd for j in 2:active_length
            row_max = ifelse(score_view[i, j] > row_max, score_view[i, j], row_max)
        end
        row_sum = zero(T)
        @inbounds @simd for j in 1:active_length
            activated = exp(score_view[i, j] - row_max)
            prob_view[i, j] = activated
            row_sum += activated
        end
        inv_sum = one(T) / row_sum
        # All-masked dummy query row (every key -Inf): row_sum == 0, so
        # inv_sum would be Inf and every prob NaN.  Force inv_sum = 0 so the
        # row is exactly zero instead — prevents NaN from leaking into the
        # PV gemm! (0 * NaN) and the backward d(V) = P' * d(O) (NaN * 0).
        # Masked keys already yield exp(-Inf - max) = 0 individually; this
        # only handles the all-masked row case.
        if iszero(row_sum)
            inv_sum = zero(T)
        end
        @inbounds @simd for j in 1:active_length
            prob_view[i, j] *= inv_sum
        end
    end
    # Pass 4: mix O = P*V into output rows. Same strided-view gemm! pattern.
    output_view = @view output[1:active_length, :]
    value_view = @view values[1:active_length, :]
    BLAS.gemm!('N', 'N', one(T), prob_view, value_view, zero(T), output_view)
    return nothing
end

"""
    attention_backward!(dqueries, dkeys, dvalues, dscores, scratch, output_grad, queries, keys, values, active_length, slot_length) -> Nothing

Backpropagate upstream `output_grad` (dL/dO) through the `attention_forward!`
result, writing `dqueries`/`dkeys`/`dvalues` and staging the score gradients
in `dscores`. Reads cached `probs` from `scratch`. All writes in place;
allocates nothing.

Math per query row `i` (subscript suppressed; `P` = cached probs row):

    dV[j,:] += P[i,j] * dO[i,:]            # mixing is linear in V
    dP[i,j]  = dot(dO[i,:], V[j,:])        # mixing is linear in P
    s        = Σ_k dP[i,k] * P[i,k]         # row scalar
    dS[i,j]  = P[i,j] * (dP[i,j] - s)      # softmax Jacobian (full, unfused:
                                           # no loss is fused here, unlike the
                                           # p - y shortcut in crossEntropyDeltaOut!)
    dQ[i,:] += Σ_j dS[i,j] * K[j,:] / sqrt(dk)
    dK[j,:] += dS[i,j] * Q[i,:] / sqrt(dk)

Blocked keys (`P = 0` from `-Inf` bias) yield `dS = 0` exactly, so no
gradient flows through masked positions — including dummy query rows, whose
upstream must be zeroed by the caller (loss/gradient entries for `i >
slot_length` are never valid). Pass structure mirrors forward: `gemm!` for
the two linear maps (`dP = dO*V'`, `dQ = dS*K`, `dK = dS'*Q`), explicit
`ifelse`-free loops for the softmax Jacobian (same branchless rationale as
the forward max). The grad-check in `test/test_seq_attention.jl` verifies all
three gradient maps against finite differences.

# Arguments
- `dqueries`, `dkeys`, `dvalues`: `(>= active_length, head_dim)` write
  targets (`+=` accumulation, mirroring `Theta_grads` across batches).
- `dscores`: `(>= active_length, >= active_length)` staging for `dS`
  (overwritten, not accumulated — internal to one backward call).
- `scratch`: `AttentionScratch` filled by the matching forward call.
- `output_grad`: `(>= active_length, head_dim)` upstream dL/dO; rows
  `1:active_length` read, rows past `slot_length` must already be zero.
- `queries`, `keys`, `values`: forward inputs (needed for the Q/K maps).
- `active_length`, `slot_length`: same values as the forward call.

The key/query width `head_dim` is derived from `size(queries, 2)` rather than
passed, mirroring `attention_forward!` — the query matrix is structure already
in hand, so re-deriving avoids trusting a separate scalar.
"""
function attention_backward!(dqueries::AbstractMatrix{T}, dkeys::AbstractMatrix{T},
        dvalues::AbstractMatrix{T}, dscores::AbstractMatrix{T}, scratch::AttentionScratch{T},
        output_grad::AbstractMatrix{T}, queries::AbstractMatrix{T}, keys::AbstractMatrix{T},
        values::AbstractMatrix{T}, active_length::Int, slot_length::Int) where {T <: AbstractFloat}
    # Derive the width from the query matrix (structure in hand), as the forward does.
    head_dim = size(queries, 2)
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    # Straight-line asserts (not a loop over a mixed-type tuple): iterating a
    # (String, Matrix/SubArray...) tuple boxes through abstract iteration and
    # allocates ~1KB per call (verified --track-allocation). Each check below
    # is concrete-typed and 0-alloc.
    @assert 0 <= slot_length <= active_length "slot_length out of range"
    @assert head_dim > 0 "head_dim must be positive"
    @assert size(dqueries, 2) == head_dim "dqueries width mismatch"
    @assert size(dqueries, 1) >= active_length "dqueries needs >= active_length rows"
    @assert size(dkeys, 2) == head_dim "dkeys width mismatch"
    @assert size(dkeys, 1) >= active_length "dkeys needs >= active_length rows"
    @assert size(dvalues, 2) == head_dim "dvalues width mismatch"
    @assert size(dvalues, 1) >= active_length "dvalues needs >= active_length rows"
    @assert size(output_grad, 2) == head_dim "output_grad width mismatch"
    @assert size(output_grad, 1) >= active_length "output_grad needs >= active_length rows"
    @assert size(queries, 2) == head_dim "queries width mismatch"
    @assert size(queries, 1) >= active_length "queries needs >= active_length rows"
    @assert size(keys, 2) == head_dim "keys width mismatch"
    @assert size(keys, 1) >= active_length "keys needs >= active_length rows"
    @assert size(values, 2) == head_dim "values width mismatch"
    @assert size(values, 1) >= active_length "values needs >= active_length rows"
    @assert size(dscores, 1) >= active_length && size(dscores, 2) >= active_length "dscores needs >= active_length $active_length rows/cols"
    @assert size(scratch.probs, 1) >= active_length "scratch sized for $(size(scratch.probs, 1)), need $active_length"
    # Unused slot: nothing valid flowed forward, so nothing flows back.
    slot_length == 0 && return nothing
    probs = scratch.probs
    prob_view = @view probs[1:active_length, 1:active_length]
    dscore_view = @view dscores[1:active_length, 1:active_length]
    # Pass 1: dV += P' * dO and dP = dO * V' via gemm! on strided views.
    doutput_view = @view output_grad[1:active_length, :]
    dvalue_view = @view dvalues[1:active_length, :]
    value_view = @view values[1:active_length, :]
    BLAS.gemm!('T', 'N', one(T), prob_view, doutput_view, one(T), dvalue_view)
    BLAS.gemm!('N', 'T', one(T), doutput_view, value_view, zero(T), dscore_view)
    # Pass 2: convert raw dP = dot(dO, V) in place to dS via the softmax
    # Jacobian dS[i,j] = P[i,j] * (dP[i,j] - Σ_k dP[i,k]*P[i,k]).
    # The row scalar is staged in scratch.row_dot so both sweeps run
    # column-streaming (j-outer / i-inner, contiguous inner walk) — no
    # row-striding inner loop. Blocked keys have P = 0 so dS = 0 exactly —
    # no gradient through the mask.
    row_dot = scratch.row_dot
    @inbounds @simd for i in 1:active_length
        row_dot[i] = zero(T)
    end
    for j in 1:active_length
        @inbounds @simd for i in 1:active_length
            row_dot[i] += dscore_view[i, j] * prob_view[i, j]
        end
    end
    for j in 1:active_length
        @inbounds @simd for i in 1:active_length
            dscore_view[i, j] = prob_view[i, j] * (dscore_view[i, j] - row_dot[i])
        end
    end
    # Pass 3: dQ += dS * K / sqrt(dk) and dK += dS' * Q / sqrt(dk) via gemm!.
    inv_scale = one(T) / sqrt(T(head_dim))
    dquery_view = @view dqueries[1:active_length, :]
    dkey_view = @view dkeys[1:active_length, :]
    query_view = @view queries[1:active_length, :]
    key_view = @view keys[1:active_length, :]
    BLAS.gemm!('N', 'N', inv_scale, dscore_view, key_view, one(T), dquery_view)
    BLAS.gemm!('T', 'N', inv_scale, dscore_view, query_view, one(T), dkey_view)
    return nothing
end
