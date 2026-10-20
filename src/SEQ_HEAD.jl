# Sequence output heads for FCANN (Step 6, work in progress).
#
# A head is a *linear output layer* — it maps the trunk's read representation
# `h` (model_dim) to `output_dim` raw values, with no activation. The meaning
# of those values is application-defined and lives entirely in the cost
# function (a later step), exactly as in the FCN where `CrossEntropyLoss` owns
# the softmax and the network just emits logits. `output_dim` is simply how
# many values the head produces per item in the output sequence. `SeqOutputSet`
# (later) combines several distinct heads into a multi-head output. Built one
# function at a time for review. Nothing here is included by `FCANN.jl` yet, so
# existing behavior is unchanged.

"""
    SeqOutputHead(weight, bias)

Trainable parameters for one linear sequence output head: a raw projection
`out = h * weight' + bias` from the trunk's read representation `h`
(`model_dim`-wide) to `output_dim` values. Created by `init_seq_output_head`;
users should not construct it field-by-field. No activation is applied here —
softmax / tanh / any output transform belongs to the cost function, mirroring
the FCN split (`calcFinalOut!` owns the loss).

# Fields (shapes for trunk model_dim D, output_dim O)
- `weight (O,D)`: projection, stored `(O, D)` so `out = h*W'` is a single
  `gemm!`/`gemv!` over rows. The output width `O = size(weight, 1)` (values per
  output item) and the trunk width `D = size(weight, 2)` are both derived from
  this matrix at use time rather than cached, so a stale field can never
  disagree with the actual shape — same convention as `BlockParams`, which
  caches no dimensions. What the `O` values *mean* is set by the cost function
  and the training data, not by this struct.
- `bias (O,)`: additive bias, zero-initialized.
"""
struct SeqOutputHead{T}
    weight::Matrix{T}
    bias::Vector{T}
end


"""
    init_seq_output_head(model_dim::Int, output_dim::Int, ::Type{FloatType}=Float32) -> SeqOutputHead

Initialize one linear output head. The weight uses Haar-orthogonal columns
rescaled to unit element variance then scaled to variance `1/fan_in` with
`fan_in = model_dim` (the input width), and the bias is zero. This `1/fan_in`
rule is the *same* one every other projection in the sequence stack uses
(`qkv`, `attn_out`, `mlp_hidden`, `mlp_out` in `SEQ_BLOCK.jl`); the head is
just another projection from the trunk, so it follows that convention rather
than the FCN's separate μP output-layer scaling. The orthogonal base comes
from `makeorthonormalrand` in `FCN_NFLOATOUT_AUXFUNCTIONS.jl`, which already
falls back to plain `randn` when `min(O, D) == 1` — so a width-1 head
(`output_dim = 1`) initializes cleanly without a special case here.

# Arguments
- `model_dim`: trunk width (the head's fan_in); must be positive.
- `output_dim`: number of values `O` generated per output item; must be
  positive. The meaning of the `O` values is set by the cost function.
- `FloatType`: element type (`Float32` default, `Float64`/`Float16` allowed).
"""
function init_seq_output_head(model_dim::Int, output_dim::Int,
        ::Type{FloatType}=Float32) where {FloatType <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert model_dim > 0 "model_dim must be positive, got $model_dim"
    @assert output_dim > 0 "output_dim must be positive, got $output_dim"
    # Orthonormal base (unit element variance) scaled to μP hidden-weight
    # variance 1/fan_in, fan_in = model_dim. Float64 intermediate keeps the
    # sqrt safe for Float16. Same helper shape as init_block_params.
    weight = FloatType.(makeorthonormalrand(output_dim, model_dim) .* sqrt(1.0 / Float64(model_dim)))
    bias = zeros(FloatType, output_dim)
    return SeqOutputHead(weight, bias)
end

"""
    SeqOutputSet(heads, read_mode)

A set of distinct linear output heads sharing one read of the trunk output,
combined into a multi-head output. Created by `init_seq_output_set`; users
should not construct it field-by-field. Each entry of `heads` is an
independent `SeqOutputHead` (its own `weight`/`bias`/`output_dim`); the set
adds only a shared `read_mode` describing how the trunk's `(Tmax, model_dim)`
output is reduced to the representation `h` that every head projects.

# Fields
- `heads::Vector{SeqOutputHead}`: the output heads, in a fixed order. Each is
  fully independent — different `output_dim`s are allowed, and the meaning of
  each head's values is set by its own cost function.
- `read_mode::Symbol`: how `h` is read from the trunk output. Only `:last`
  (`h = block_out[slot_length, :]`, the last valid position) is implemented;
  `:mean` and `:each` are reserved for future mean-pool and per-position reads
  and will error clearly if requested before they exist.

# Note on generality
There is no notion here of "value" vs "policy" vs any task. A one-element head
and a many-element head are the same object; combining several heads into one
`SeqOutputSet` is how a model emits multiple outputs from one trunk. The cost
functions (a later step) assign meaning.
"""
struct SeqOutputSet{T}
    heads::Vector{SeqOutputHead{T}}
    read_mode::Symbol
end


"""
    init_seq_output_set(model_dim::Int, output_dims::AbstractVector{Int},
        read_mode::Symbol=:last, ::Type{FloatType}=Float32) -> SeqOutputSet

Build a `SeqOutputSet` of `length(output_dims)` independent heads, each created
by `init_seq_output_head(model_dim, output_dims[k])`. The heads share the trunk
width `model_dim` (their common fan_in) but may have different `output_dim`s.

# Arguments
- `model_dim`: trunk width (each head's fan_in); must be positive.
- `output_dims`: per-head output widths, in head order; must be non-empty with
  every entry positive.
- `read_mode`: `:last` (implemented), `:mean` / `:each` (reserved). Validated
  as a known symbol here; forward errors clearly if an unimplemented mode is
  requested.
- `FloatType`: element type (`Float32` default, `Float64`/`Float16` allowed).
"""
function init_seq_output_set(model_dim::Int, output_dims::AbstractVector{Int},
        read_mode::Symbol=:last, ::Type{FloatType}=Float32) where {FloatType <: AbstractFloat}
    @assert model_dim > 0 "model_dim must be positive, got $model_dim"
    @assert !isempty(output_dims) "output_dims must name at least one head"
    @assert read_mode in (:last, :mean, :each) "read_mode must be :last, :mean, or :each, got $read_mode"
    heads = SeqOutputHead{FloatType}[init_seq_output_head(model_dim, od, FloatType) for od in output_dims]
    return SeqOutputSet(heads, read_mode)
end


"""
    zero_output_set(set::SeqOutputSet) -> SeqOutputSet

All-zero copy of `set` (same head shapes and `read_mode`), for use as the
gradient-accumulation target of the output-set backward. Mirrors
`zero_stack_params`: the `heads` vector field is rebuilt element-by-element (a
generic `zeros(size(...))` cannot zero a `Vector{SeqOutputHead}`).
"""
function zero_output_set(set::SeqOutputSet{T}) where {T}
    zero_head(h::SeqOutputHead{T}) where {T} =
        SeqOutputHead(zeros(T, size(h.weight)), zeros(T, size(h.bias)))
    SeqOutputSet(SeqOutputHead{T}[zero_head(h) for h in set.heads], set.read_mode)
end


"""
    SeqOutputScratch(read_pos, outputs, d_read_pos, trunk_grad, dout, out_staged,
                     head_ranges, outputs_view, staged_view, dout_view)

Pre-allocated workspace for one output-set forward + backward pass over a
single slot. Created by `form_output_scratch`; users should not construct it
field-by-field.

# Fields (shapes for trunk model_dim D, capacity Tmax, K heads widths O_1..O_K,
# total output sumO = O_1 + ... + O_K)
- `read_pos (D,)`: the cached read representation `h` produced by `read_mode`
  from the trunk output (for `:last`, `h = block_out[slot_length, :]`).
- `outputs (sumO,)`: all heads' raw outputs stacked contiguously
  (`out_k = h * weight_k' + bias_k`, no activation). Head `k` occupies the
  slice `head_ranges[k]`. One storage instead of `Vector{Vector}`: per-head
  slices are stride-1 views, so cost/backward kernels see the same memory
  pattern as single vectors while construction is one allocation, and the
  cost step can hand per-head slices straight to the FCN cost kernels (whose
  full-target paths accept `AbstractArray` since Step 7).
- `d_read_pos (D,)`: gradient w.r.t. `h`, summed over heads in backward.
- `trunk_grad (Tmax,D)`: full-width trunk gradient handed to `stack_backward!`
  as its `output_grad`. Backward zeroes rows `1:active_length` and writes
  `d_read_pos` into row `slot_length` only. Lives here (not caller-owned) so
  head→trunk composition stays 0-alloc.
- `dout (sumO,)`: per-head upstream gradients `∂loss_k/∂out_k` stacked the
  same way, filled by the cost step (`seq_cost!`, Step 7) and consumed by
  `head_backward!` — lives here so the cost→backward chain stays 0-alloc.
- `out_staged (sumO,)`: copy of `outputs` staged for the cost kernels. The
  FCN softmax kernel `crossEntropyDeltaOut!(deltas, a)` clobbers `a` in place
  (it stores `exp` values back into the activations as a memory trick), so
  costs that softmax must operate on this copy, never on `outputs` — which
  must stay intact for re-evaluation / grad-checks.
- `head_ranges::Vector{UnitRange{Int}}`: head `k`'s slice of
  `outputs`/`dout`/`out_staged`, derived from the head widths at
  construction (`offset_k = O_1 + ... + O_{k-1}`). Per-head widths need not
  match: any heterogeneity just shifts the offsets, and every slice is a
  stride-1 view of the same buffer, so cost/backward kernels run at single-
  vector speed regardless of the per-head spread (verified: operations on
  these buffers depend only on `sumO`, not on the number of heads or how
  `sumO` is split).
- `outputs_view / staged_view / dout_view (O_k,)`: pre-built per-head views
  over `outputs` / `out_staged` / `dout`, one per `head_ranges[k]`, for the
  cost layer (SEQ_COST.jl). A fresh `view(buffer, r)` allocates its
  `SubArray` header on every call; constructing them once here keeps cost
  calls 0-alloc. They alias the flat buffers, so a write through
  `dout_view[k]` lands in `dout[head_ranges[k]]`.
"""
# Contiguous per-head slice of one of the flat scratch buffers above. All
# heads share this concrete type, so the `Vector{SeqVecView{T}}` fields stay
# concretely typed and indexing them is type stable.
const SeqVecView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}

struct SeqOutputScratch{T}
    read_pos::Vector{T}
    outputs::Vector{T}
    d_read_pos::Vector{T}
    trunk_grad::Matrix{T}
    dout::Vector{T}
    out_staged::Vector{T}
    head_ranges::Vector{UnitRange{Int}}
    outputs_view::Vector{SeqVecView{T}}
    staged_view::Vector{SeqVecView{T}}
    dout_view::Vector{SeqVecView{T}}
end


"""
    form_output_scratch(::Type{FloatType}, set::SeqOutputSet, max_seq_length::Int) -> SeqOutputScratch

Allocate zero-filled output-set scratch: one `(model_dim,)` read buffer, one
contiguous `(sumO,)` output buffer for all heads, one `(model_dim,)`
read-gradient buffer, the `(max_seq_length, model_dim)` trunk handoff buffer,
and contiguous `(sumO,)` cost-stage buffers (`dout` for the cost gradients,
`out_staged` for the clobber-safe output copy). Head `k`'s slice of each
`sumO` buffer is `head_ranges[k]`, derived from the head widths; the
per-head `outputs_view`/`staged_view`/`dout_view` slices over those buffers
are pre-built here as well so cost calls stay allocation-free. Mirrors
`form_block_scratch`: one allocation up front, zero allocation in the hot
loop. Only the implemented `:last` read writes a single `read_pos` row per
call; the rest is never read.

The trunk width `model_dim` and each head's output width `O_k` are derived from
the heads' weights (`size(weight, 2)` / `size(weight, 1)`) rather than passed,
so there is no separate scalar that could disagree with the actual shapes.
`max_seq_length` sizes only the trunk buffer (capacity, not a head property).

# Arguments
- `FloatType`: element type matching the heads.
- `set`: the output set (its heads supply both the trunk width and the
  per-head output widths).
- `max_seq_length`: trunk capacity `Tmax` for the `trunk_grad` buffer.
"""
function form_output_scratch(::Type{FloatType},
        set::SeqOutputSet, max_seq_length::Int) where {FloatType <: AbstractFloat}
    @assert !isempty(set.heads) "set must contain at least one head"
    @assert max_seq_length > 0 "max_seq_length must be positive, got $max_seq_length"
    # Trunk width is read off the first head; a hand-built set that mixes
    # fan_ins would silently misread, so fail fast on the rest.
    model_dim = size(set.heads[1].weight, 2)
    for h in set.heads
        @assert size(h.weight, 2) == model_dim "head fan_in $(size(h.weight, 2)) != model_dim $model_dim"
    end
    # Contiguous sumO layout: head k occupies offset_k = O_1+...+O_{k-1} .+ (1:O_k).
    # Any per-head widths just shift the offsets; every slice is stride-1, so
    # heterogeneity costs nothing (see the SeqOutputScratch field docs).
    head_ranges = Vector{UnitRange{Int}}(undef, length(set.heads))
    offset = 0
    for (k, h) in enumerate(set.heads)
        O_k = size(h.weight, 1)
        head_ranges[k] = (offset + 1):(offset + O_k)
        offset += O_k
    end
    sumO = offset
    read_pos = zeros(FloatType, model_dim)
    outputs = zeros(FloatType, sumO)
    d_read_pos = zeros(FloatType, model_dim)
    trunk_grad = zeros(FloatType, max_seq_length, model_dim)
    dout = zeros(FloatType, sumO)
    out_staged = zeros(FloatType, sumO)
    # Cost-layer views are built once here (a per-call view would allocate its
    # SubArray header); they alias the flat buffers for the life of the struct.
    outputs_view = SeqVecView{FloatType}[view(outputs, r) for r in head_ranges]
    staged_view = SeqVecView{FloatType}[view(out_staged, r) for r in head_ranges]
    dout_view = SeqVecView{FloatType}[view(dout, r) for r in head_ranges]
    return SeqOutputScratch(read_pos, outputs, d_read_pos, trunk_grad, dout, out_staged,
                            head_ranges, outputs_view, staged_view, dout_view)
end


"""
    head_forward!(scratch, set, trunk_out, slot_length) -> Nothing

Forward pass for one output set over a single slot: read `h` from the trunk
output under `set.read_mode`, then compute every head's raw output
`out_k = h * weight_k' + bias_k` (no activation — any output transform belongs
to the cost function). All writes in place; allocates nothing.
`slot_length == 0` is a no-op (caller must guarantee `slot_length >= 1` for a
real call; an all-empty slot is skipped by the outer loop, mirroring
`stack_forward!`).

Only `:last` is implemented: `h = trunk_out[slot_length, :]` (the last valid
position). Per-head compute branches on `size(weight_k, 1)`: a single-output
head (`O_k == 1`) reads its row through a contiguous view and uses `dot`
(measured ~7× faster than `gemv!` on a `(1,D)` matrix); wider heads use
`gemv!` plus a bias add. Same parameters, same scratch either way — the branch
is perfectly predictable (`size(weight_k, 1)` never changes for a model), and
both paths are 0-alloc.

# Arguments
- `scratch`: `SeqOutputScratch` sized for the set (from `form_output_scratch`).
- `set`: `SeqOutputSet` (same heads the scratch was formed for).
- `trunk_out`: `(>= slot_length, model_dim)` trunk activation rows (e.g.
  `stack.blocks[end].block_out`).
- `slot_length`: the valid position to read.
"""
function head_forward!(scratch::SeqOutputScratch{T}, set::SeqOutputSet{T},
        trunk_out::AbstractMatrix{T}, slot_length::Int) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert set.read_mode === :last "read_mode $(set.read_mode) not implemented (only :last)"
    @assert !isempty(set.heads) "set must contain at least one head"
    @assert slot_length >= 1 "slot_length must be positive, got $slot_length"
    # Trunk width is read off the heads (structure in hand), never passed.
    model_dim = size(set.heads[1].weight, 2)
    for h in set.heads
        @assert size(h.weight, 2) == model_dim "head fan_in $(size(h.weight, 2)) != model_dim $model_dim"
    end
    @assert size(trunk_out, 2) == model_dim && size(trunk_out, 1) >= slot_length "trunk_out must be (>= $slot_length, $model_dim)"
    # :last read: h = trunk_out[slot_length, :] into the cached buffer. No
    # @simd here — j strides across trunk_out's columns (non-contiguous), so a
    # vectorized gather would be wrong. The write target read_pos is contiguous
    # but the read source is not; keep it a plain indexed copy.
    read_pos = scratch.read_pos
    @inbounds for j in 1:model_dim
        read_pos[j] = trunk_out[slot_length, j]
    end
    # Per-head raw outputs into their slices of the contiguous sumO buffer
    # (stride-1 views; see SeqOutputScratch field docs).
    for k in eachindex(set.heads)
        head = set.heads[k]
        out_k = @view scratch.outputs[scratch.head_ranges[k]]
        O_k = size(head.weight, 1)
        if O_k == 1
            # Single-output fast path: contiguous row view + dot.
            wrow = @view head.weight[1, :]
            out_k[1] = BLAS.dot(model_dim, wrow, 1, read_pos, 1) + head.bias[1]
        else
            BLAS.gemv!('N', one(T), head.weight, read_pos, zero(T), out_k)
            @inbounds @simd for j in 1:O_k
                out_k[j] += head.bias[j]
            end
        end
    end
    return nothing
end


"""
    head_backward!(scratch, set, grads, active_length, slot_length) -> Nothing

Backward pass for one output set over a single slot: reads per-head upstream
gradients from `scratch.dout` (the contiguous sumO buffer filled by the cost
step, `seq_cost!`; head `k`'s slice is `head_ranges[k]`, already weighted by
whatever cost produced it — the head is loss-agnostic), accumulate each head's
weight and bias gradients into `grads` (`+=` semantics, mirroring `Theta_grads`
across batches), sum every head's read-position gradient into
`scratch.d_read_pos`, zero `scratch.trunk_grad[1:active_length, :]`, and write
`d_read_pos` into its row `slot_length` only — the full-width buffer handed to
`stack_backward!` as `output_grad`. All writes in place; allocates nothing.
Mirrors the FCN convention: the cost fills scratch, backward consumes scratch;
the caller never threads deltas through its own storage.

Per-head math (subscript `k` suppressed; `d = dout`, `W` = `weight`, `h` =
cached `read_pos`):

    dW   += d * h'          # axpy!/ger!/gemv! by head width
    db   += d
    dh   += W' * d          # summed across heads into d_read_pos

Single-output heads use the `dot`/`axpy!` row-view branch (same fast path as
forward); wider heads use `gemv!`. `active_length` is passed only to zero the
correct leading block of `trunk_grad` (runtime length, not structure).

# Arguments
- `scratch`: `SeqOutputScratch` filled by the matching `head_forward!` call
  (its cached `read_pos` is the `h` used for every weight map) and by the cost
  step (its `dout` holds the per-head upstream gradients).
- `set`: `SeqOutputSet` forward parameters.
- `grads`: `SeqOutputSet`-shaped accumulation target (see `zero_output_set`).
- `active_length`: positions to zero in `trunk_grad`; `slot_length`: the valid
  position whose row carries `d_read_pos` (`0` is a no-op).
"""
function head_backward!(scratch::SeqOutputScratch{T}, set::SeqOutputSet{T},
        grads::SeqOutputSet{T}, active_length::Int, slot_length::Int) where {T <: AbstractFloat}
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert length(set.heads) == length(grads.heads) == length(scratch.head_ranges) "head count mismatch (set/grads/scratch)"
    @assert set.read_mode === grads.read_mode === :last "only :last is implemented"
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    # Widths off the heads (structure in hand), never passed.
    model_dim = size(set.heads[1].weight, 2)
    for h in set.heads
        @assert size(h.weight, 2) == model_dim "head fan_in $(size(h.weight, 2)) != model_dim $model_dim"
    end
    @assert size(scratch.trunk_grad, 1) >= active_length "trunk_grad sized for $(size(scratch.trunk_grad, 1)), need $active_length"
    # Unused slot: nothing flowed forward, nothing flows back.
    slot_length == 0 && return nothing
    h = scratch.read_pos
    dh = scratch.d_read_pos
    dout = scratch.dout
    # Zero the read-position gradient accumulator (fresh per call), then sum
    # every head's contribution into it.
    @inbounds @simd for j in 1:model_dim
        dh[j] = zero(T)
    end
    for k in eachindex(set.heads)
        head = set.heads[k]
        ghead = grads.heads[k]
        dout_k = @view dout[scratch.head_ranges[k]]
        O_k = size(head.weight, 1)
        if O_k == 1
            # Single-output fast path: the weight row is contiguous (stride 1
            # in the column-major (1,D) matrix), so it feeds BLAS directly.
            # axpy! beats a hand @simd loop from D≈32 and stays ~flat in D
            # (DCE-safe profiling: dh 94ns vs 242ns, dW 95ns vs 127ns at
            # D=128); at tiny D they tie, so axpy! is the uniform choice.
            scale = dout_k[1]
            wrow = @view head.weight[1, :]
            gwrow = @view ghead.weight[1, :]
            BLAS.axpy!(scale, h, gwrow)      # dW[1,:] += scale * h
            ghead.bias[1] += scale           # db += dout (scalar)
            BLAS.axpy!(scale, wrow, dh)      # dh += scale * W[1,:]'
        else
            # General path: dW += outer(dout, h), db += dout, dh += W'*dout.
            dW = @view ghead.weight[1:O_k, :]
            BLAS.ger!(one(T), dout_k, h, dW)
            @inbounds @simd for j in 1:O_k
                ghead.bias[j] += dout_k[j]
            end
            BLAS.gemv!('T', one(T), head.weight, dout_k, one(T), dh)
        end
    end
    # Compose with the trunk: zero the leading block, write dh at the read row.
    tg = scratch.trunk_grad
    @inbounds for j in 1:model_dim
        @simd for i in 1:active_length
            tg[i, j] = zero(T)
        end
    end
    @inbounds for j in 1:model_dim
        tg[slot_length, j] = dh[j]
    end
    return nothing
end

