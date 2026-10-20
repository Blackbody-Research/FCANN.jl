# Stacked sequence transformer for FCANN (Step 5, work in progress).
#
# Depth wrapper over the embed-free core block (`SEQ_BLOCK.jl`): embeds the
# feature tokens + adds learned absolute positions *exactly once*, then feeds
# that model-wide activation through a stack of `num_blocks` core blocks,
# routing each block's input gradient to the previous block's upstream in
# backward. Moving embed/pos here (out of the block) is what makes stacking
# correct: positions are injected once, and every block sees a model→model
# activation rather than re-embedding. Built one function at a time for
# review. Nothing here is included by `FCANN.jl` yet, so existing behavior
# is unchanged.

"""
    StackParams(embed_weight, embed_bias, pos_table, blocks)

Trainable parameters for a stacked sequence transformer: the *shared* input
embedding + positional table (applied once, before block 1) followed by a
vector of embed-free core blocks. Created by `init_stack_params(cfg,
num_blocks)`; users should not construct it field-by-field.

# Fields (shapes for SeqConfig with feature_dim F, model_dim D, length Tmax)
- `embed_weight (D,F)`, `embed_bias (D,)`: input projection `X*W' + b`,
  mapping raw feature tokens to the model width (the finite→infinite μP input
  weight: variance 1).
- `pos_table (Tmax,D)`: learned absolute positions, row `t` added to token
  `t` (front-packed `1:len` indexing). Zero start.
- `blocks::Vector{BlockParams}`: `num_blocks` embed-free core blocks (each
  see `BlockParams`). Every block is model→model; none embeds.
"""
struct StackParams{T}
    embed_weight::Matrix{T}
    embed_bias::Vector{T}
    pos_table::Matrix{T}
    blocks::Vector{BlockParams{T}}
end


"""
    init_stack_params(cfg::SeqConfig, num_blocks::Int, ::Type{FloatType}=Float32) -> StackParams

Initialize a stacked transformer: shared embed/pos (μP input-weight rule)
followed by `num_blocks` core blocks from `init_block_params`. Mirrors the
embedding + position initialization that used to live inside a single block,
now hoisted to the stack so it happens exactly once.

# Arguments
- `cfg`: supplies `feature_dim`, `model_dim`, `max_seq_length`. Passed
  unchanged to each `init_block_params` (which reads only `model_dim`).
- `num_blocks`: depth (number of stacked core blocks); must be positive.
  No default — depth is an explicit modeling choice, mirroring how
  `length(hidden_layers)` sets FCN depth.
- `FloatType`: element type (`Float32` default, `Float64`/`Float16` allowed).

# Init table
- `embed_weight`: σ² = 1 (μP input weight, fan_in = 1); `embed_bias`: 0;
  `pos_table`: 0. Each block follows its own `init_block_params` table.
"""
function init_stack_params(cfg::SeqConfig, num_blocks::Int,
        ::Type{FloatType}=Float32) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert num_blocks > 0 "num_blocks must be positive, got $num_blocks"
    @assert cfg.feature_dim > 0 "feature_dim must be positive, got $(cfg.feature_dim)"
    @assert cfg.model_dim > 0 "model_dim must be positive, got $(cfg.model_dim)"
    @assert cfg.max_seq_length > 0 "max_seq_length must be positive, got $(cfg.max_seq_length)"
    model_dim = cfg.model_dim
    feature_dim = cfg.feature_dim
    # Orthonormal base rescaled to unit element variance, then μP scaling, in
    # Float64 before conversion (Float16-safe sqrt). Same helper shape as
    # init_block_params and initializeparams_saxe!.
    ortho_scaled(rows::Int, cols::Int, variance::Float64) =
        FloatType.(makeorthonormalrand(rows, cols) .* sqrt(variance))
    # Input weight: variance 1 (finite→infinite map, fan_in = 1 per μP).
    embed_weight = ortho_scaled(model_dim, feature_dim, 1.0)
    embed_bias = zeros(FloatType, model_dim)
    pos_table = zeros(FloatType, cfg.max_seq_length, model_dim)
    # num_blocks embed-free core blocks; each owns its own LN/QKV/attn/MLP.
    blocks = BlockParams{FloatType}[init_block_params(cfg, FloatType) for _ in 1:num_blocks]
    return StackParams(embed_weight, embed_bias, pos_table, blocks)
end


"""
    zero_stack_params(params::StackParams) -> StackParams

All-zero copy of `params` (same shapes, element type, and `num_blocks`),
for use as the gradient-accumulation target of `stack_backward!`. Mirrors
how `nnCostFunction` allocates a fresh `Theta_grads` alongside `Thetas`; the
`blocks` vector field is rebuilt element-by-element (a generic
`zeros(size(...))` cannot zero a `Vector{BlockParams}`).
"""
function zero_stack_params(params::StackParams{T}) where {T}
    zero_block(p::BlockParams{T}) where {T} =
        BlockParams{T}([zeros(T, size(getfield(p, f))) for f in fieldnames(BlockParams{T})]...)
    StackParams(zeros(T, size(params.embed_weight)),
        zeros(T, size(params.embed_bias)),
        zeros(T, size(params.pos_table)),
        BlockParams{T}[zero_block(b) for b in params.blocks])
end

"""
    StackScratch(blocks, embedded, dx_input, bias)

Pre-allocated workspace for one stacked-transformer forward + backward pass
over a single slot of up to `max_seq_length` tokens. Created by
`form_stack_scratch`; users should not construct it field-by-field. Sized per
slot (not per batch): the caller loops slots and reuses one scratch, matching
the per-slot reuse in `form_block_scratch`.

# Fields (shapes for feature_dim F, model_dim D, length Tmax)
- `blocks::Vector{BlockScratch}`: one owned `BlockScratch` per core block,
  holding that block's forward caches (`normed1/2`, `qkv_out`, `attn_out`,
  `mlp_hidden`, `mlp_tanh_grad`, `hidden_state`) and its `d_input` gradient.
  Block k reads its forward caches and writes its `d_input` in backward; the
  stack routes `blocks[k].d_input` to block k-1's upstream.
- `embedded (Tmax,D)`: `X*embed_W' + embed_b + pos` staging — the model-wide
  activation fed to block 1. Written once in forward.
- `dx_input (Tmax,F)`: stack input gradient `dX = d(embedded) * embed_W`
  (feature-width). Written fresh by backward (not accumulated), while
  *parameter* grads accumulate.
- `bias (Tmax,Tmax)`: shared attention mask bias (0 for allowed keys,
  `-Inf` for causal/padding-blocked keys). Computed *once* per forward call
  and reused by every block — the depth-efficiency win of hoisting the mask
  out of the block.

# Note on `d_embedded`
There is deliberately no separate `d_embedded` buffer: after block 1's
backward, `blocks[1].d_input` already holds the gradient w.r.t. `embedded`
(model-wide), so the embed backward reuses it directly rather than copying
into a redundant buffer.
"""
struct StackScratch{T}
    blocks::Vector{BlockScratch{T}}
    embedded::Matrix{T}
    dx_input::Matrix{T}
    bias::Matrix{T}
end


"""
    form_stack_scratch(::Type{FloatType}, cfg::SeqConfig, num_blocks::Int) -> StackScratch

Allocate zero-filled stack scratch: one `BlockScratch` per core block plus
the shared `embedded` / `dx_input` / `bias` buffers. Mirrors
`form_block_scratch`: one allocation up front, zero allocation afterwards.
Only rows `1:active_length` are written per call; the rest is never read.

# Arguments
- `FloatType`: element type matching the stack parameters.
- `cfg`: supplies `max_seq_length`, `model_dim`, `feature_dim`.
- `num_blocks`: depth; must be positive and match the parameter stack.
"""
function form_stack_scratch(::Type{FloatType}, cfg::SeqConfig,
        num_blocks::Int) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert num_blocks > 0 "num_blocks must be positive, got $num_blocks"
    @assert cfg.max_seq_length > 0 "max_seq_length must be positive, got $(cfg.max_seq_length)"
    @assert cfg.model_dim > 0 "model_dim must be positive, got $(cfg.model_dim)"
    @assert cfg.feature_dim > 0 "feature_dim must be positive, got $(cfg.feature_dim)"
    model_dim = cfg.model_dim
    feature_dim = cfg.feature_dim
    seq_length = cfg.max_seq_length
    # Zero-fill: unwritten rows read as 0, so a caller bug that reads past
    # active_length degrades to zeros rather than NaNs leaking silently.
    blocks = BlockScratch{FloatType}[form_block_scratch(FloatType, cfg) for _ in 1:num_blocks]
    embedded = zeros(FloatType, seq_length, model_dim)
    dx_input = zeros(FloatType, seq_length, feature_dim)
    bias = zeros(FloatType, seq_length, seq_length)
    return StackScratch(blocks, embedded, dx_input, bias)
end


"""
    fill_stack_attention_bias!(bias, active_length, slot_length, mode) -> Nothing

Fill the leading `active_length × active_length` block of the stack's shared
`bias` buffer for one slot holding `slot_length` valid tokens, under `mode`.
Allowed keys get `0`, blocked keys get `typemin(T)` (added to attention
scores before the rowwise softmax, so blocked keys contribute exactly zero
weight). Mirrors `fill_causal_padding_bias!` in `SEQ_CONTAINERS.jl` but writes
a raw matrix (the stack owns its own `bias`, not a `SeqBuffers`) and adds the
`:bidirectional` case the FCN-only helper does not cover:

- `:causal`: key `j` allowed from query `i` when `j <= min(i, slot_length)`
  (no future, no padding) — each position attends only to itself and earlier
  positions, so one forward pass produces a per-position output that depends
  only on the prefix up to that position.
- `:bidirectional`: key `j` allowed from query `i` when `j <= slot_length`
  (every valid query sees every valid key) — each output row mixes the whole
  valid sequence, so a single position (e.g. the last) summarizes all of it.

Query rows `i > slot_length` are all `-Inf` in both modes (dummy queries
whose outputs are discarded and whose loss/gradient entries are zeroed by the
later cost code). Writes in place; allocates nothing.
"""
function fill_stack_attention_bias!(bias::Matrix{T}, active_length::Int,
        slot_length::Int, mode::Symbol) where {T <: AbstractFloat}
    @assert mode === :causal || mode === :bidirectional "mode must be :causal or :bidirectional, got $mode"
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    @assert active_length <= size(bias, 1) "active_length $active_length exceeds bias buffer $(size(bias, 1))"
    neg = typemin(T)
    # Column-first: Julia is column-major, so filling one column at a time
    # (`i` inner over a fixed `j`) walks contiguous memory and lets each
    # `.=` vectorize to SIMD stores. Measured ~4x faster than a row-outer
    # scalar loop at L=64 and ~20x at L=128, bitwise-identical output. No
    # BLAS routine covers this (structured mask with a padding cutoff + a
    # `typemin` sentinel, not `tril!`'s zero).
    @inbounds for j in 1:active_length
        if mode === :causal
            if j <= slot_length
                # Column j: rows 1:j-1 are future (blocked), rows j:slot_length
                # are causal-past (allowed), rows slot_length+1:L are dummy.
                if j > 1
                    @views bias[1:(j - 1), j] .= neg
                end
                @views bias[j:slot_length, j] .= zero(T)
                if slot_length < active_length
                    @views bias[(slot_length + 1):active_length, j] .= neg
                end
            else
                # Key past slot_length: blocked for every query.
                @views bias[1:active_length, j] .= neg
            end
        else  # :bidirectional
            if j <= slot_length
                # Valid key: allowed for every valid query (rows 1:slot_length),
                # blocked for dummy rows.
                @views bias[1:slot_length, j] .= zero(T)
                if slot_length < active_length
                    @views bias[(slot_length + 1):active_length, j] .= neg
                end
            else
                @views bias[1:active_length, j] .= neg
            end
        end
    end
    return nothing
end


"""
    stack_forward!(scratch, params, input, active_length, slot_length, mode, epsilon) -> Nothing

Forward pass for a stacked sequence transformer over rows `1:active_length`
of `input`, writing the final activation into
`scratch.blocks[end].block_out`. Embedding + positions are applied *once*
here, then the model-wide activation is threaded through every core block;
all writes in place, allocates nothing. `slot_length == 0` is a no-op.

Pipeline (embed once, then per block):

    embedded   = input*embed_W' + embed_b + pos[1:L,:]   # once, feature→model
    block_out  = Block_k(embedded)                        # k = 1
    block_out  = Block_k(prev.block_out)                  # k = 2..N  (model→model)

The attention `bias` is built once (`fill_stack_attention_bias!`) and shared
by all blocks — the depth-efficiency payoff of hoisting the mask out of the
block. The stack output is the last block's `block_out`; read it at the last
valid position for a sequence-level output head.

# Arguments
- `scratch`: `StackScratch` sized for `>= active_length` (from
  `form_stack_scratch`).
- `params`: `StackParams` (same `FloatType`, same `num_blocks`).
- `input`: `(>= active_length, feature_dim)` raw token rows.
- `active_length`: positions to compute; `slot_length`: valid tokens in slot.
- `mode`: `:causal` or `:bidirectional` (see `fill_stack_attention_bias!`).
- `epsilon`: LayerNorm floor (default `1e-5` in working dtype).
"""
function stack_forward!(scratch::StackScratch{T}, params::StackParams{T},
        input::AbstractMatrix{T}, active_length::Int, slot_length::Int,
        mode::Symbol, epsilon::T=T(1e-5)) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert mode === :causal || mode === :bidirectional "mode must be :causal or :bidirectional, got $mode"
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    @assert epsilon >= eps(T) "epsilon $epsilon below eps($T)"
    model_dim = size(params.embed_weight, 1)
    feature_dim = size(params.embed_weight, 2)
    num_blocks = length(params.blocks)
    @assert length(scratch.blocks) == num_blocks "scratch has $(length(scratch.blocks)) blocks, params have $num_blocks"
    @assert size(input, 2) == feature_dim && size(input, 1) >= active_length "input must be (>= $active_length, $feature_dim)"
    @assert size(scratch.embedded, 1) >= active_length "scratch sized for $(size(scratch.embedded, 1)), need $active_length"
    # Unused slot: leave scratch untouched (visible degradation on misuse).
    slot_length == 0 && return nothing
    # Build the shared attention bias ONCE for all blocks.
    fill_stack_attention_bias!(scratch.bias, active_length, slot_length, mode)
    bias_view = @view scratch.bias[1:active_length, 1:active_length]
    # Embed once: E = X*embed_W' + embed_b + pos[1:L,:]. gemm! on strided
    # leading views, then column-streaming bias + position adds.
    input_view = @view input[1:active_length, :]
    embed_view = @view scratch.embedded[1:active_length, :]
    BLAS.gemm!('N', 'T', one(T), input_view, params.embed_weight, zero(T), embed_view)
    pos_view = @view params.pos_table[1:active_length, :]
    embed_bias = params.embed_bias
    @inbounds for j in 1:model_dim
        bias_j = embed_bias[j]
        @simd for i in 1:active_length
            embed_view[i, j] += bias_j + pos_view[i, j]
        end
    end
    # Thread the activation through every core block. Block 1 reads `embedded`;
    # block k reads block k-1's `block_out`. Each writes its own `block_out`.
    block_input = embed_view
    for k in 1:num_blocks
        block_forward!(scratch.blocks[k], params.blocks[k], block_input,
            bias_view, active_length, slot_length, mode, epsilon)
        block_input = @view scratch.blocks[k].block_out[1:active_length, :]
        # Dummy query rows (i > slot_length) were all-masked in attention, so
        # their softmax is NaN.  Zero them so `gemm!`'s 0*NaN contamination
        # can't reach valid queries through the next block's V, and so they
        # carry no gradient backward.  Rows 1:slot_length are untouched.
        if slot_length < active_length
            @inbounds for i in slot_length+1:active_length
                for j in 1:model_dim
                    block_input[i, j] = zero(T)
                end
            end
        end
    end
    return nothing
end


"""
    stack_backward!(scratch, params, grads, output_grad, input, active_length, slot_length) -> Nothing

Backpropagate upstream `output_grad` (dL/dY, where `Y` is the stack output
`scratch.blocks[end].block_out`) through the `stack_forward!` result,
accumulating parameter gradients into `grads` (a `StackParams`-shaped struct
with matching fields, `+=` semantics mirroring `Theta_grads` across batches)
and writing the stack input gradient into `scratch.dx_input`. Reads all
forward caches; all writes in place, allocates nothing. `slot_length == 0`
is a no-op.

Reverse pipeline (blocks in reverse, then the single embed):

    for k in N:-1:1
        upstream_k = (k == N) ? output_grad : blocks[k+1].d_input
        block_backward!(blocks[k], ..., upstream_k, ...)   # writes blocks[k].d_input
    dE          = blocks[1].d_input                        # = d(embedded), model-width
    d(embed_W) += dE' * input;  d(embed_b) += rows(dE)
    d(pos_table[1:L,:]) += dE                              # identity grad
    dx_input    = dE * embed_W                             # feature-width

The block-to-block routing is the key: block k+1's `d_input` *is* the gradient
w.r.t. block k's output (`block_out_k`), because block k+1's input is
`block_out_k`. After the loop, `blocks[1].d_input` holds `d(embedded)`, which
the embed backward consumes directly (no separate `d_embedded` buffer). The
embed is applied once in forward, so its backward runs once here.

Weight grads are raw sums over valid positions (no `1/m`); the outer training
code owns any per-example scaling (same convention as `block_backward!`).

# Arguments
- `scratch`: `StackScratch` filled by the matching forward call.
- `params`: `StackParams` forward parameters.
- `grads`: `StackParams`-shaped accumulation target (same field names/shapes).
- `output_grad`: `(>= active_length, model_dim)` upstream dL/dY; rows past
  `slot_length` must already be zero.
- `input`: `(>= active_length, feature_dim)` forward input rows (needed for
  the embed weight map `d(embed_W) += dE' * X`).
- `active_length`, `slot_length`: same values as forward.
"""
function stack_backward!(scratch::StackScratch{T}, params::StackParams{T},
        grads::StackParams{T}, output_grad::AbstractMatrix{T}, input::AbstractMatrix{T},
        active_length::Int, slot_length::Int) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    model_dim = size(params.embed_weight, 1)
    feature_dim = size(params.embed_weight, 2)
    num_blocks = length(params.blocks)
    @assert length(scratch.blocks) == num_blocks "scratch has $(length(scratch.blocks)) blocks, params have $num_blocks"
    @assert length(grads.blocks) == num_blocks "grads has $(length(grads.blocks)) blocks, params have $num_blocks"
    @assert size(output_grad, 2) == model_dim && size(output_grad, 1) >= active_length "output_grad must be (>= $active_length, $model_dim)"
    @assert size(input, 2) == feature_dim && size(input, 1) >= active_length "input must be (>= $active_length, $feature_dim)"
    # Unused slot: nothing flowed forward, nothing flows back.
    slot_length == 0 && return nothing
    # Blocks in reverse. Block N consumes the stack upstream; block k<N
    # consumes block k+1's d_input (its output gradient). Different
    # BlockScratch objects, so upstream and the write target never alias.
    upstream = @view output_grad[1:active_length, :]
    for k in num_blocks:-1:1
        block_backward!(scratch.blocks[k], params.blocks[k], grads.blocks[k],
            upstream, active_length, slot_length)
        upstream = @view scratch.blocks[k].d_input[1:active_length, :]
    end
    # blocks[1].d_input now holds d(embedded) = dE (model-width).
    d_embed = @view scratch.blocks[1].d_input[1:active_length, :]
    # Embed backward. d(embed_W) += dE' * input; d(embed_b) += rows(dE).
    input_fwd = @view input[1:active_length, :]
    BLAS.gemm!('T', 'N', one(T), d_embed, input_fwd, one(T), grads.embed_weight)
    embed_b_grad = grads.embed_bias
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            embed_b_grad[j] += d_embed[i, j]
        end
    end
    # d(pos_table[t,:]) += dE[t,:] for t <= active_length (identity gradient).
    grads_pos = grads.pos_table
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            grads_pos[i, j] += d_embed[i, j]
        end
    end
    # Stack input gradient dX = dE * embed_W (feature-width). Written fresh,
    # not accumulated: each backward call owns its input gradient.
    dx_view = @view scratch.dx_input[1:active_length, :]
    BLAS.gemm!('N', 'N', one(T), d_embed, params.embed_weight, zero(T), dx_view)
    return nothing
end

