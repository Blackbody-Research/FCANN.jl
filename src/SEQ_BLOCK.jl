# Sequence transformer core block for FCANN (Step 5, work in progress).
#
# One pre-norm single-head *embed-free* block: LN1, fused QKV, attention
# (Step 3), out-projection, residual adds, LN2, tanh MLP. The input
# embedding + learned absolute positions live at the *stack* level (Step 5),
# not here: a stacked model embeds once and feeds every block the same
# model-width activation, so re-embedding per block would be wrong (positions
# injected N times) and shape-incompatible (model→model). Built one function
# at a time for review. Nothing here is included by `FCANN.jl` yet, so
# existing behavior is unchanged.

"""
    BlockParams(norm1_gain, norm1_bias, norm2_gain, norm2_bias,
        qkv_weight, qkv_bias, attn_out_weight, attn_out_bias,
        mlp_hidden_weight, mlp_hidden_bias, mlp_out_weight, mlp_out_bias)

Trainable parameters for one pre-norm single-head transformer *core* block.
Created by `init_block_params(cfg)`; users should not construct it
field-by-field. All projection widths equal `model_dim` so the two residual
adds match widths, mirroring the equal-width requirement of `resLayers` in
`nnCostFunction`. The input embedding (feature→model) and positional table
are *not* here — they belong to the stack (see `SEQ_STACK`), which embeds
once and passes every block a `model_dim`-wide activation.

# Fields (shapes for SeqConfig with model_dim D, length Tmax)
- `norm1_gain/bias (D,)`, `norm2_gain/bias (D,)`: LayerNorm scale + shift.
- `qkv_weight (3D,D)`, `qkv_bias (3D,)`: fused Q/K/V projection; one `gemm!`
  yields `Q‖K‖V`, split by column ranges into three views (zero-copy).
- `attn_out_weight (D,D)`, `attn_out_bias (D,)`: post-attention projection.
- `mlp_hidden_weight (4D,D)`, `mlp_hidden_bias (4D,)`: feedforward expansion.
- `mlp_out_weight (D,4D)`, `mlp_out_bias (D,)`: projection back to width D.
"""
struct BlockParams{T}
    norm1_gain::Vector{T}
    norm1_bias::Vector{T}
    norm2_gain::Vector{T}
    norm2_bias::Vector{T}
    qkv_weight::Matrix{T}
    qkv_bias::Vector{T}
    attn_out_weight::Matrix{T}
    attn_out_bias::Vector{T}
    mlp_hidden_weight::Matrix{T}
    mlp_hidden_bias::Vector{T}
    mlp_out_weight::Matrix{T}
    mlp_out_bias::Vector{T}
end


"""
    init_block_params(cfg::SeqConfig, ::Type{FloatType}=Float32) -> BlockParams

Initialize one core block's parameters with Haar-orthogonal columns scaled
per the μP Table 8 rule (Tensor Programs V): hidden weights get init
variance `1/fan_in`, and all biases get variance 0 (constant zero init).
LayerNorm gains at ones / biases at zeros follow the paper's vector-like
guidance (init mean may be nonzero; variance 0). The orthogonal base comes
from `makeorthonormalrand` in `FCN_NFLOATOUT_AUXFUNCTIONS.jl` (a Haar-random
block rescaled to unit element variance), mirroring `initializeparams_saxe!`;
columns stay exactly orthogonal (off-diagonal Gram error ~1e-7), which is the
conditioning benefit you use the orthonormal option for. Note the rescale
uses `max(rows, cols)`, so for non-square weights the column norm is
`sqrt(max/fan_in)` — variance is exactly per-spec, orthogonality structure
exact, unit-norm only when square.

# Arguments
- `cfg`: supplies `model_dim`. `feature_dim` / `max_seq_length` /
  `max_batch_size` / `num_heads` are ignored here (embed-free core block;
  single-head; batch is a buffer concern, not a parameter concern).
- `FloatType`: element type (`Float32` default, `Float64`/`Float16` allowed).

# Init table (all N(0, σ²) unless noted)
- `qkv_weight`, `attn_out_weight`, `mlp_hidden_weight`: σ² = 1/fan_in
  (hidden weights; fan_in = model_dim, model_dim, model_dim).
- `mlp_out_weight`: σ² = 1/fan_in with fan_in = 4*model_dim (hidden weight).
- `qkv_bias`, `attn_out_bias`, `mlp_hidden_bias`, `mlp_out_bias`: 0.
- `norm1/2_gain`: 1; `norm1/2_bias`: 0.
"""
function init_block_params(cfg::SeqConfig, ::Type{FloatType}=Float32) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert cfg.model_dim > 0 "model_dim must be positive, got $(cfg.model_dim)"
    model_dim = cfg.model_dim
    hidden_dim = 4 * model_dim
    # Orthonormal base with unit row variance (Haar random rescaled), then μP
    # variance scaling. Scale applied in Float64 then converted so Float16
    # does not under/overflow the sqrt. Same helper shape as
    # initializeparams_saxe!: makeorthonormalrand(rows, cols) .* factor.
    ortho_scaled(rows::Int, cols::Int, variance::Float64) =
        FloatType.(makeorthonormalrand(rows, cols) .* sqrt(variance))
    # Identity start: y = 1 .* x_hat + 0 (see init_layernorm_params).
    norm1_gain = ones(FloatType, model_dim)
    norm1_bias = zeros(FloatType, model_dim)
    norm2_gain = ones(FloatType, model_dim)
    norm2_bias = zeros(FloatType, model_dim)
    # Hidden weights: variance 1/fan_in (fan_in = columns).
    qkv_weight = ortho_scaled(3 * model_dim, model_dim, 1.0 / Float64(model_dim))
    qkv_bias = zeros(FloatType, 3 * model_dim)
    attn_out_weight = ortho_scaled(model_dim, model_dim, 1.0 / Float64(model_dim))
    attn_out_bias = zeros(FloatType, model_dim)
    mlp_hidden_weight = ortho_scaled(hidden_dim, model_dim, 1.0 / Float64(model_dim))
    mlp_hidden_bias = zeros(FloatType, hidden_dim)
    mlp_out_weight = ortho_scaled(model_dim, hidden_dim, 1.0 / Float64(hidden_dim))
    mlp_out_bias = zeros(FloatType, model_dim)
    return BlockParams(norm1_gain, norm1_bias, norm2_gain, norm2_bias,
        qkv_weight, qkv_bias, attn_out_weight, attn_out_bias,
        mlp_hidden_weight, mlp_hidden_bias, mlp_out_weight, mlp_out_bias)
end

"""
    BlockScratch(normed1, qkv_out, attn_out, hidden_state,
        normed2, mlp_hidden, mlp_tanh_grad, block_out, layernorm1, attention,
        layernorm2)

Pre-allocated workspace for one block forward + backward pass over a single
slot of up to `max_seq_length` tokens. Created by `form_block_scratch`;
users should not construct it field-by-field. Sized per slot (not per
batch): the caller loops slots and reuses one scratch, mirroring the
slot-by-slot option in `form_layernorm_scratch`.

# Fields (shapes for model_dim D, hidden 4D, length Tmax)
- `normed1 (Tmax,D)`: LN1 output feeding QKV.
- `qkv_out (Tmax,3D)`: fused Q‖K‖V rows; Q/K/V are column-range views.
- `attn_out (Tmax,D)`: attention output before the out-projection.
- `hidden_state (Tmax,D)`: post-attention residual `H = X + proj(Attn)`.
- `normed2 (Tmax,D)`: LN2 output feeding the MLP.
- `mlp_hidden (Tmax,4D)`: tanh MLP activations (backward cache).
- `mlp_tanh_grad (Tmax,4D)`: tanh derivative cache (`tanh_grad_z` role).
- `block_out (Tmax,D)`: final `Y = H + MLP(LN2(H))`.
- `layernorm1`, `layernorm2`: owned `LayerNormScratch` (max_tokens = Tmax).
- `attention`: owned `AttentionScratch` (max_seq_length = Tmax).
- `delta_*` mirrors: gradient staging with identical shapes
  (`d_input`, `delta_normed1`, `delta_qkv`, `delta_attn`,
  `delta_hidden`, `delta_normed2`, `delta_mlp_hidden`, `delta_block`).
- `norm1_param_grads`, `norm2_param_grads`: `(model_dim, 2)` staging for
  `layernorm_backward!` (column 1 = gain grads, column 2 = bias grads),
  folded into `grads.norm*_gain/bias` after each LN backward. Dedicated
  matrices (not aliased corners): correct at all size combinations.
- `dscores (Tmax,Tmax)`: `dS` staging for `attention_backward!`. Dedicated
  (not the `scores` cache): the forward `S` stays intact for re-verification.
- `d_input (Tmax,D)`: block input gradient w.r.t. the model-width activation
  this block received (the previous block's `hidden_state`, or the stack's
  embedded+positioned tokens for block 1). Written fresh by backward (not
  accumulated): each backward call owns its input gradient, while *parameter*
  grads accumulate. This is the value the stack routes to the previous block's
  upstream, or (for block 1) to the embed backward.
"""
struct BlockScratch{T}
    normed1::Matrix{T}
    qkv_out::Matrix{T}
    attn_out::Matrix{T}
    hidden_state::Matrix{T}
    normed2::Matrix{T}
    mlp_hidden::Matrix{T}
    mlp_tanh_grad::Matrix{T}
    block_out::Matrix{T}
    layernorm1::LayerNormScratch{T}
    attention::AttentionScratch{T}
    layernorm2::LayerNormScratch{T}
    d_input::Matrix{T}
    delta_normed1::Matrix{T}
    delta_qkv::Matrix{T}
    delta_attn::Matrix{T}
    delta_hidden::Matrix{T}
    delta_normed2::Matrix{T}
    delta_mlp_hidden::Matrix{T}
    delta_block::Matrix{T}
    norm1_param_grads::Matrix{T}
    norm2_param_grads::Matrix{T}
    dscores::Matrix{T}
end

"""
    form_block_scratch(::Type{FloatType}, cfg::SeqConfig) -> BlockScratch

Allocate zero-filled block scratch for one slot of up to
`cfg.max_seq_length` tokens at `cfg.model_dim` width. Mirrors
`form_layernorm_scratch`: one allocation up front, zero allocation
afterwards. Only rows `1:active_length` are written per call; the rest is
never read.

# Arguments
- `FloatType`: element type matching the block parameters.
- `cfg`: supplies `max_seq_length` and `model_dim`.
"""
function form_block_scratch(::Type{FloatType}, cfg::SeqConfig) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert cfg.max_seq_length > 0 "max_seq_length must be positive, got $(cfg.max_seq_length)"
    @assert cfg.model_dim > 0 "model_dim must be positive, got $(cfg.model_dim)"
    model_dim = cfg.model_dim
    seq_length = cfg.max_seq_length
    hidden_dim = 4 * model_dim
    # Zero-fill: unwritten rows read as 0, so a caller bug that reads past
    # active_length degrades to zeros rather than NaNs leaking silently.
    normed1 = zeros(FloatType, seq_length, model_dim)
    qkv_out = zeros(FloatType, seq_length, 3 * model_dim)
    attn_out = zeros(FloatType, seq_length, model_dim)
    hidden_state = zeros(FloatType, seq_length, model_dim)
    normed2 = zeros(FloatType, seq_length, model_dim)
    mlp_hidden = zeros(FloatType, seq_length, hidden_dim)
    mlp_tanh_grad = zeros(FloatType, seq_length, hidden_dim)
    block_out = zeros(FloatType, seq_length, model_dim)
    layernorm1 = form_layernorm_scratch(FloatType, seq_length, model_dim)
    attention = form_attention_scratch(FloatType, seq_length)
    layernorm2 = form_layernorm_scratch(FloatType, seq_length, model_dim)
    d_input = zeros(FloatType, seq_length, model_dim)
    delta_normed1 = zeros(FloatType, seq_length, model_dim)
    delta_qkv = zeros(FloatType, seq_length, 3 * model_dim)
    delta_attn = zeros(FloatType, seq_length, model_dim)
    delta_hidden = zeros(FloatType, seq_length, model_dim)
    delta_normed2 = zeros(FloatType, seq_length, model_dim)
    delta_mlp_hidden = zeros(FloatType, seq_length, hidden_dim)
    delta_block = zeros(FloatType, seq_length, model_dim)
    norm1_param_grads = zeros(FloatType, model_dim, 2)
    norm2_param_grads = zeros(FloatType, model_dim, 2)
    dscores = zeros(FloatType, seq_length, seq_length)
    return BlockScratch(normed1, qkv_out, attn_out, hidden_state,
        normed2, mlp_hidden, mlp_tanh_grad, block_out, layernorm1, attention,
        layernorm2, d_input, delta_normed1, delta_qkv, delta_attn,
        delta_hidden, delta_normed2, delta_mlp_hidden, delta_block,
        norm1_param_grads, norm2_param_grads, dscores)
end

"""
    fast_tanh_generic(x::T) -> T

Generic-`AbstractFloat` copy of `fast_tanh` (`FCN_NFLOATOUT_COSTFUNCTIONS.jl`,
`Float32`-only): the same (3,4)-rational tanh approximation with identical
clamps at ±4.97. Lives here so `block_forward!` stays dtype-generic without
touching the existing `Float32` kernel. Verified equal to `fast_tanh` to
`1e-7` on `Float32` in `test/test_seq_block.jl`.
"""
function fast_tanh_generic(x::T) where {T <: AbstractFloat}
    x2 = x * x
    a = x * (T(135135) + x2 * (T(17325) + x2 * (T(378) + x2)))
    b = T(135135) + x2 * (T(62370) + x2 * (T(3150) + x2 * T(28)))
    return ifelse(x > T(4.97), one(T), ifelse(x < -T(4.97), -one(T), a / b))
end

"""
    block_forward!(scratch, params, input, bias, active_length, slot_length, mode, epsilon) -> Nothing

One pre-norm single-head *embed-free* transformer core block over rows
`1:active_length` of `input`, writing rows into `scratch.block_out`. `input`
is a `model_dim`-wide activation (the previous block's output, or the stack's
embedded+positioned tokens for block 1) — embedding/positions are the stack's
job, not this block's. All writes in place; allocates nothing.

Pipeline (each arrow writes a named scratch field):

    normed1    = LN1(input)                            # Step 2
    qkv_out    = normed1*qkv_W' + qkv_b                 # one gemm!, 3 views
    attn_out   = Attn(Q‖K‖V, bias, mode)               # Step 3
    hidden     = input + attn_out*attn_out_W' + attn_out_b   # residual 1
    normed2    = LN2(hidden)                           # Step 2
    mlp_hidden = tanh(normed2*mlp_hidden_W' + mlp_hidden_b)      # scaled tanh
    block_out  = hidden + mlp_hidden*mlp_out_W' + mlp_out_b      # residual 2

Residual adds are in-place `axpy!`-style accumulation, mirroring `resLayers`
in `nnCostFunction`. The MLP uses your scaled `tanh`
(`1.7159*fast_tanh(2z/3)` via `tanhActivation!`-equivalent loops) with the
`tanh_grad_z` role played by `mlp_tanh_grad` (cached derivative for
backward). Q/K/V are column-range views into `qkv_out` (zero-copy split).
Only rows `1:active_length` are read/written; padding untouched, no mask
argument beyond the attention `bias`. `slot_length == 0` is a no-op.

# Arguments
- `scratch`: `BlockScratch` sized for `>= active_length`.
- `params`: `BlockParams` (same `FloatType`).
- `input`: `(>= active_length, model_dim)` activation rows.
- `bias`: `(>= active_length, >= active_length)` attention mask bias.
- `active_length`: positions to compute; `slot_length`: valid tokens in slot.
- `mode`: `:causal` or `:bidirectional` (passed to attention).
- `epsilon`: LayerNorm floor (default `1e-5` in working dtype).
"""
function block_forward!(scratch::BlockScratch{T}, params::BlockParams{T},
        input::AbstractMatrix{T}, bias::AbstractMatrix{T}, active_length::Int,
        slot_length::Int, mode::Symbol, epsilon::T=T(1e-5)) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert mode === :causal || mode === :bidirectional "mode must be :causal or :bidirectional, got $mode"
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    @assert epsilon >= eps(T) "epsilon $epsilon below eps($T)"
    model_dim = size(params.qkv_weight, 2)
    @assert size(input, 2) == model_dim && size(input, 1) >= active_length "input must be (>= $active_length, $model_dim)"
    @assert size(bias, 1) >= active_length && size(bias, 2) >= active_length "bias needs >= $active_length rows/cols"
    @assert size(scratch.normed1, 1) >= active_length "scratch sized for $(size(scratch.normed1, 1)), need $active_length"
    # Unused slot: leave scratch untouched (visible degradation on misuse).
    slot_length == 0 && return nothing
    # Stage 1: LN1 over input rows (input is already model-wide; the stack
    # owns embedding + positions, which happen exactly once before block 1).
    input_view = @view input[1:active_length, :]
    normed1_view = @view scratch.normed1[1:active_length, :]
    layernorm_forward!(normed1_view, scratch.layernorm1, input_view,
        params.norm1_gain, params.norm1_bias, active_length, epsilon)
    # Stage 2: fused QKV — one gemm! into qkv_out, then zero-copy column
    # views Q = cols 1:D, K = D+1:2D, V = 2D+1:3D.
    qkv_view = @view scratch.qkv_out[1:active_length, :]
    BLAS.gemm!('N', 'T', one(T), normed1_view, params.qkv_weight, zero(T), qkv_view)
    qkv_bias = params.qkv_bias
    @inbounds for j in 1:(3 * model_dim)
        bias_j = qkv_bias[j]
        @simd for i in 1:active_length
            qkv_view[i, j] += bias_j
        end
    end
    query_view = @view scratch.qkv_out[1:active_length, 1:model_dim]
    key_view = @view scratch.qkv_out[1:active_length, (model_dim + 1):(2 * model_dim)]
    value_view = @view scratch.qkv_out[1:active_length, (2 * model_dim + 1):(3 * model_dim)]
    # Stage 3: attention into attn_out.
    attn_view = @view scratch.attn_out[1:active_length, :]
    bias_view = @view bias[1:active_length, 1:active_length]
    attention_forward!(attn_view, scratch.attention, query_view, key_view,
        value_view, bias_view, active_length, slot_length, mode)
    # Stage 4: out-projection + residual 1 — H = input + proj(Attn). Copy
    # input into hidden first (gemm! beta can't add a *different* matrix),
    # then accumulate the projection with beta = 1.
    hidden_view = @view scratch.hidden_state[1:active_length, :]
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            hidden_view[i, j] = input_view[i, j]
        end
    end
    BLAS.gemm!('N', 'T', one(T), attn_view, params.attn_out_weight, one(T), hidden_view)
    attn_bias = params.attn_out_bias
    @inbounds for j in 1:model_dim
        bias_j = attn_bias[j]
        @simd for i in 1:active_length
            hidden_view[i, j] += bias_j
        end
    end
    # Stage 5: LN2 over hidden rows.
    normed2_view = @view scratch.normed2[1:active_length, :]
    layernorm_forward!(normed2_view, scratch.layernorm2, hidden_view,
        params.norm2_gain, params.norm2_bias, active_length, epsilon)
    # Stage 6: MLP hidden — tanh(normed2*W1' + b1), scaled tanh matching
    # tanhActivation! (1.7159*fast_tanh(2z/3)) with derivative cached in
    # mlp_tanh_grad for backward (tanh_grad_z role).
    mlp_hidden_view = @view scratch.mlp_hidden[1:active_length, :]
    BLAS.gemm!('N', 'T', one(T), normed2_view, params.mlp_hidden_weight, zero(T), mlp_hidden_view)
    mlp_bias1 = params.mlp_hidden_bias
    hidden_dim = length(mlp_bias1)
    tanh_grad_view = @view scratch.mlp_tanh_grad[1:active_length, :]
    scale_tanh = T(1.7159)
    @inbounds for j in 1:hidden_dim
        bias_j = mlp_bias1[j]
        @simd for i in 1:active_length
            pre = mlp_hidden_view[i, j] + bias_j
            # Scaled tanh forward + analytic derivative, same constants as
            # tanhGradient! (FCN_NFLOATOUT_COSTFUNCTIONS.jl).
            post = scale_tanh * fast_tanh_generic(T(2) / T(3) * pre)
            mlp_hidden_view[i, j] = post
            tanh_grad_view[i, j] = scale_tanh * (one(T) - post * post / (scale_tanh * scale_tanh)) * T(2) / T(3)
        end
    end
    # Stage 7: out-projection + residual 2 — Y = H + proj(MLP). Same
    # copy-then-accumulate pattern as stage 5.
    out_view = @view scratch.block_out[1:active_length, :]
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            out_view[i, j] = hidden_view[i, j]
        end
    end
    BLAS.gemm!('N', 'T', one(T), mlp_hidden_view, params.mlp_out_weight, one(T), out_view)
    mlp_bias2 = params.mlp_out_bias
    @inbounds for j in 1:model_dim
        bias_j = mlp_bias2[j]
        @simd for i in 1:active_length
            out_view[i, j] += bias_j
        end
    end
    return nothing
end

"""
    block_backward!(scratch, params, grads, output_grad, active_length, slot_length) -> Nothing

Backpropagate upstream `output_grad` (dL/dY) through the `block_forward!`
result, writing the block-input gradient into `scratch.d_input` and
accumulating parameter gradients into `grads` (a `BlockParams`-shaped struct
with matching fields, `+=` semantics mirroring `Theta_grads` across batches).
Reads all forward caches (`normed1/2`, `qkv_out`, attention `probs`,
`mlp_hidden`, `mlp_tanh_grad`). All writes in place; allocates nothing.

Reverse pipeline (residuals are gradient forks — upstream adds into both the
skip branch and the sub-block branch):

    dH_from_Y      = dY                                     # residual 2 skip
    dMLP_out       = dY → d(mlp_out_W/bias) via gemm!('T','N')
    dMLP_hidden    = dMLP_out * mlp_out_W .* tanh'(cached)  # finishDelta! idiom
    dMLP_in        = dMLP_hidden → d(mlp_hidden_W/bias)
    dH             = dY + dMLP_in                            # fork join
    dLN2_in        = LN2 backward(dH) → d(norm2_gain/bias)
    dAttn_proj     = dH → d(attn_out_W/bias)
    dAttn          = attention_backward!(dAttn_proj) → dQKV views
    dQKV_in        = concat(dQ‖dK‖dV) → d(qkv_W/bias)
    dNormed1       = LN1 backward(dQKV_in) → d(norm1_gain/bias)
    dInput         = dH_skip1 + dNormed1-path               # model-width, exposed

The final `dInput` is the gradient w.r.t. this block's model-width input
activation; the stack routes it to the previous block's upstream (or, for
block 1, to the embed backward). No `embed_W` multiply happens here — that
map belongs to the stack's embedding, applied once.

Q/K/V gradient views alias the three column ranges of `delta_qkv`
(zero-copy, mirroring the forward split). Tanh backward reuses the cached
`mlp_tanh_grad` exactly like `finishDelta!(delta, tanh_grad_z)`. Weight
grads use `gemm!('T','N', 1/m ...)` with `m = active_length`... no — unlike
`nnCostFunction` there is no batch-mean convention here: gradients are raw
sums over valid positions, and the outer training code owns any scaling
(your RL loop applies per-example weights via the head gradients). This is
documented explicitly because it differs from the `1.0f0/m` in
`nnCostFunction` (`COSTFUNCTIONS.jl:1398`).

# Arguments
- `scratch`: `BlockScratch` filled by the matching forward call.
- `params`: `BlockParams` forward parameters (needed for Q/K/V maps).
- `grads`: `BlockParams`-shaped accumulation target (same field names/shapes).
- `output_grad`: `(>= active_length, model_dim)` upstream dL/dY; rows past
  `slot_length` must already be zero.
- `active_length`, `slot_length`: same values as forward.

The width `model_dim` is derived from `size(params.qkv_weight, 2)` rather than
passed, mirroring `block_forward!` (single-head, so `model_dim == head_dim`).
"""
function block_backward!(scratch::BlockScratch{T}, params::BlockParams{T},
        grads::BlockParams{T}, output_grad::AbstractMatrix{T},
        active_length::Int, slot_length::Int) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    model_dim = size(params.qkv_weight, 2)
    @assert size(output_grad, 2) == model_dim && size(output_grad, 1) >= active_length "output_grad must be (>= $active_length, $model_dim)"
    # Unused slot: nothing flowed forward, nothing flows back.
    slot_length == 0 && return nothing
    hidden_dim = 4 * model_dim
    # Shorthand views over the leading block (all strided, gemm!-ready).
    dY = @view output_grad[1:active_length, :]
    d_block = @view scratch.delta_block[1:active_length, :]
    d_hidden = @view scratch.delta_hidden[1:active_length, :]
    d_mlp_hidden = @view scratch.delta_mlp_hidden[1:active_length, :]
    d_normed2 = @view scratch.delta_normed2[1:active_length, :]
    d_attn = @view scratch.delta_attn[1:active_length, :]
    d_qkv = @view scratch.delta_qkv[1:active_length, :]
    d_normed1 = @view scratch.delta_normed1[1:active_length, :]
    d_input = @view scratch.d_input[1:active_length, :]
    mlp_hidden = @view scratch.mlp_hidden[1:active_length, :]
    mlp_tanh_grad = @view scratch.mlp_tanh_grad[1:active_length, :]
    normed2 = @view scratch.normed2[1:active_length, :]
    hidden = @view scratch.hidden_state[1:active_length, :]
    attn = @view scratch.attn_out[1:active_length, :]
    normed1 = @view scratch.normed1[1:active_length, :]
    # Reverse stage 8: Y = H + proj(MLP). dH gets dY (skip fork); dMLP path
    # through mlp_out_W/bias. Copy dY into both fork targets first.
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            d_hidden[i, j] = dY[i, j]
            d_block[i, j] = dY[i, j]
        end
    end
    # d(mlp_out_W) += dY' * mlp_hidden; d(mlp_out_b) += sum_rows(dY).
    BLAS.gemm!('T', 'N', one(T), d_block, mlp_hidden, one(T), grads.mlp_out_weight)
    mlp_out_b_grad = grads.mlp_out_bias
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            mlp_out_b_grad[j] += d_block[i, j]
        end
    end
    # dMLP_hidden_input = dY * mlp_out_W .* tanh'(cached). gemm! into
    # d_mlp_hidden, then Hadamard with mlp_tanh_grad (finishDelta! idiom).
    BLAS.gemm!('N', 'N', one(T), d_block, params.mlp_out_weight, zero(T), d_mlp_hidden)
    @inbounds for j in 1:hidden_dim
        @simd for i in 1:active_length
            d_mlp_hidden[i, j] *= mlp_tanh_grad[i, j]
        end
    end
    # Reverse stage 7: d(mlp_hidden_W) += dHidden' * normed2; d(bias) += rows.
    BLAS.gemm!('T', 'N', one(T), d_mlp_hidden, normed2, one(T), grads.mlp_hidden_weight)
    mlp_b1_grad = grads.mlp_hidden_bias
    @inbounds for j in 1:hidden_dim
        @simd for i in 1:active_length
            mlp_b1_grad[j] += d_mlp_hidden[i, j]
        end
    end
    # dNormed2 = dHidden * mlp_hidden_W. Then fork-join: dH += dNormed2-path.
    BLAS.gemm!('N', 'N', one(T), d_mlp_hidden, params.mlp_hidden_weight, zero(T), d_normed2)
    # Reverse stage 6: LN2 backward. d(LN2-out) currently sits in d_normed2;
    # LN2 dx must ADD into d_hidden (residual fork join), and gain/bias grads
    # accumulate into grads.norm2_gain / grads.norm2_bias. layernorm_backward!
    # OVERWRITES dx, but d_hidden holds skip-dY needed in the join — so LN2 dx
    # goes to delta_normed1 first (free: LN1 backward runs later), then adds
    # into d_hidden. Param grads stage in norm2_param_grads, then fold.
    d_ln2_out = @view scratch.delta_normed2[1:active_length, :]
    d_ln2_hold = @view scratch.delta_normed1[1:active_length, :]
    layernorm_backward!(d_ln2_hold, scratch.norm2_param_grads, d_ln2_out,
        scratch.layernorm2, params.norm2_gain, active_length)
    grads_norm2_gain = grads.norm2_gain
    grads_norm2_bias = grads.norm2_bias
    staging2 = scratch.norm2_param_grads
    @inbounds @simd for j in 1:model_dim
        grads_norm2_gain[j] += staging2[j, 1]
        grads_norm2_bias[j] += staging2[j, 2]
    end
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            d_hidden[i, j] += d_ln2_hold[i, j]
        end
    end
    # Reverse stage 5: H = input + proj(Attn). dH (joined) flows two ways: the
    # skip fork into d_input, and the projection branch. d(attn_out_W) +=
    # dH' * attn; d(attn_out_b) += rows(dH). Then dAttn_in = dH * attn_out_W.
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            d_input[i, j] = d_hidden[i, j]
        end
    end
    BLAS.gemm!('T', 'N', one(T), d_hidden, attn, one(T), grads.attn_out_weight)
    attn_out_b_grad = grads.attn_out_bias
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            attn_out_b_grad[j] += d_hidden[i, j]
        end
    end
    BLAS.gemm!('N', 'N', one(T), d_hidden, params.attn_out_weight, zero(T), d_attn)
    # Reverse stage 4: attention backward. dQ/dK/dV views alias delta_qkv's
    # three column ranges (zero-copy mirror of the forward split); dS stages
    # in scratch.dscores (forward S cache stays intact).
    d_query = @view scratch.delta_qkv[1:active_length, 1:model_dim]
    d_key = @view scratch.delta_qkv[1:active_length, (model_dim + 1):(2 * model_dim)]
    d_value = @view scratch.delta_qkv[1:active_length, (2 * model_dim + 1):(3 * model_dim)]
    query_fwd = @view scratch.qkv_out[1:active_length, 1:model_dim]
    key_fwd = @view scratch.qkv_out[1:active_length, (model_dim + 1):(2 * model_dim)]
    value_fwd = @view scratch.qkv_out[1:active_length, (2 * model_dim + 1):(3 * model_dim)]
    # attention_backward! ACCUMULATES into dQ/dK/dV: zero the ranges first so
    # the result is exact (not polluted by a previous slot's leftovers past
    # a shorter active_length).
    @inbounds for j in 1:(3 * model_dim)
        @simd for i in 1:active_length
            d_qkv[i, j] = zero(T)
        end
    end
    attention_backward!(d_query, d_key, d_value, scratch.dscores,
        scratch.attention, d_attn, query_fwd, key_fwd, value_fwd,
        active_length, slot_length)
    # Reverse stage 3: fused QKV — d(qkv_W) += dQKV' * normed1;
    # d(qkv_b) += rows(dQKV). Then dNormed1 = dQKV * qkv_W.
    BLAS.gemm!('T', 'N', one(T), d_qkv, normed1, one(T), grads.qkv_weight)
    qkv_b_grad = grads.qkv_bias
    @inbounds for j in 1:(3 * model_dim)
        @simd for i in 1:active_length
            qkv_b_grad[j] += d_qkv[i, j]
        end
    end
    BLAS.gemm!('N', 'N', one(T), d_qkv, params.qkv_weight, zero(T), d_normed1)
    # Reverse stage 2: LN1 backward. d(Normed1) sits in d_normed1; LN1 dx must
    # ADD into d_input (residual 1 fork join: d_input already holds skip-dH).
    # LN1 dx goes to a free buffer first: delta_mlp_hidden is free (MLP
    # backward finished). Param grads stage in norm1_param_grads, then fold.
    d_ln1_hold = @view scratch.delta_mlp_hidden[1:active_length, 1:model_dim]
    layernorm_backward!(d_ln1_hold, scratch.norm1_param_grads, d_normed1,
        scratch.layernorm1, params.norm1_gain, active_length)
    grads_norm1_gain = grads.norm1_gain
    grads_norm1_bias = grads.norm1_bias
    staging1 = scratch.norm1_param_grads
    @inbounds @simd for j in 1:model_dim
        grads_norm1_gain[j] += staging1[j, 1]
        grads_norm1_bias[j] += staging1[j, 2]
    end
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            d_input[i, j] += d_ln1_hold[i, j]
        end
    end
    # Block input gradient is complete in scratch.d_input (model-width). The
    # stack routes it to the previous block's upstream, or (for block 1) to
    # the embed backward. No embed_W multiply here — that map lives in the
    # stack's embedding, applied exactly once.
    return nothing
end



