# Sequence containers for FCANN (Step 1, work in progress).
#
# Long-term goal: pre-allocated, in-place sequence buffers that mirror the
# FCN philosophy (`form_activations`, `fillAs!`, no allocation in hot loop).
# This file is built one function at a time for review. Nothing here is
# included by `FCANN.jl` yet, so existing behavior is unchanged.

"""
    SeqConfig(max_batch_size, max_seq_length, feature_dim, model_dim, num_heads)

Capacity + model-size descriptor for sequence buffers. All values are
explicit (no defaults); tests may pick e.g. `max_seq_length = 32` as a
concrete size.

# Arguments
- `max_batch_size::Int`: maximum batch size (number of sequences per call).
- `max_seq_length::Int`: maximum time steps any sequence may hold (buffer capacity).
- `feature_dim::Int`:    token feature dimension (fixed per model). This is the
  raw input width (analogous to `M`, the input layer size, in `initializeParams`).
- `model_dim::Int`:      working width used inside every transformer block
  (analogous to one entry of the `hidden_layers` vector `H` in the FCN, except
  it is the *same* for every block rather than varying per layer). Stored now,
  consumed when block parameters are introduced later. It does *not* set the
  number of layers: depth (how many blocks are stacked) will live in a later
  parameter struct, mirroring how `length(hidden_layers)` sets FCN depth
  separately from the per-layer widths.
- `num_heads::Int`:      attention heads per block (stored now, used Step 3).
  Must divide `model_dim` evenly; each head works on `model_dim ÷ num_heads`
  columns.

# Depth and residuals
Depth is not a field here because `SeqConfig` sizes *data* buffers, not
parameters — the same buffers serve any number of stacked blocks. Residual
connections are central to the later block design (each block adds its input
back to its output, as `axpy!` does for `resLayers` in `nnCostFunction`),
which is exactly why `model_dim` is constant across blocks: the add requires
matching widths. Layer count and residual period will be specified alongside
the block parameters, not here.

# Example
```julia
cfg = SeqConfig(64, 32, 16, 64, 4)
```
"""
struct SeqConfig
    max_batch_size::Int
    max_seq_length::Int
    feature_dim::Int
    model_dim::Int
    num_heads::Int
end

"""
    SeqBuffers(cfg::SeqConfig, ::Type{FloatType}=Float32) -> SeqBuffers

Pre-allocated storage for a batch of variable-length sequences. Construct
directly from a `SeqConfig`; users should not construct it field-by-field.
Every call uses all `max_batch_size` slots: slots with no data get
`batch_lengths[b] == 0` and their token rows are never read. The single-slot
case (`max_batch_size == 1`) reuses this same struct with 2D token storage —
later code will specialize on views (`gemm!` for a `(len, feature_dim)` block
vs `gemv!` for one token row) rather than on a separate type.

Mirrors `form_activations(Thetas, m)` in `FCN_NFLOATOUT_COSTFUNCTIONS.jl`:
one allocation up front, zero allocation afterwards.

# Fields
- `token_storage::Array{FloatType,N}`: token data, `(max_seq_length,
  feature_dim, max_batch_size)` when `max_batch_size > 1`, `(max_seq_length,
  feature_dim)` when `max_batch_size == 1`. For slot `b`, rows
  `1:batch_lengths[b]` hold valid tokens; rows above are padding.
- `batch_lengths::Vector{Int}`: valid token count per slot;
  `batch_lengths[b] == 0` means slot `b` is unused and skipped.
  Written by `fill_token_slot!` / `clear_batch_slot!`; read by kernels as
  `bufs.batch_lengths[b]`.
- `max_active_length::Base.RefValue{Int}`: `maximum(batch_lengths)` from the
  last `finalize_batch_lengths!` call — the number of time steps later kernels must
  actually compute. A `Ref` keeps the struct immutable while letting the
  scalar update in place; read via `get_active_length(bufs)`.
- `attention_bias::Matrix{FloatType}`: `(max_seq_length, max_seq_length)`
  scratch for the causal+padding bias consumed by attention (Step 3). Lives
  here so it is allocated once.

# Storage order
`token_storage` is `(max_seq_length, feature_dim, max_batch_size)` (time,
feature, batch) when `max_batch_size > 1`. Julia arrays are column-major, so
the first index varies fastest: fixing the batch index (`storage[:, :, b]`)
selects a contiguous `(max_seq_length, feature_dim)` block whose row-prefix
`storage[1:len, :, b]` is a BLAS-compatible strided matrix view — zero
allocation, zero copy, accepted directly by `BLAS.gemm!`. When
`max_batch_size == 1`, storage is 2D `(max_seq_length, feature_dim)` and the
prefix is simply `storage[1:len, :]`. The more intuitive `(batch, time,
feature)` order fails there (`ArgumentError: Input is not strided`) because
a `(batch, time)` slice straddles non-contiguous memory; verified by probe
2026-10-06. `reshape` of such a non-strided view into `(batch*time, feature)`
is likewise rejected by `gemm!`, so the layout must make the hot slice
contiguous instead of reshaping around it.
"""
struct SeqBuffers{T, N}
    token_storage::Array{T,N}
    batch_lengths::Vector{Int}
    max_active_length::Base.RefValue{Int}
    attention_bias::Matrix{T}

    function SeqBuffers(cfg::SeqConfig, ::Type{FloatType}=Float32) where {FloatType <: AbstractFloat}
        ndims_token = cfg.max_batch_size == 1 ? 2 : 3  # single-slot case uses 2D storage
        # Zero-fill so untouched pad rows (t > batch_lengths[b]) read as 0
        # until overwritten; slicing to max_active_length means they are never
        # read in practice, but zeros keep NaNs from leaking through a
        # masking bug silently.
        token_storage = ndims_token == 2 ? zeros(FloatType, cfg.max_seq_length, cfg.feature_dim) : zeros(FloatType, cfg.max_seq_length, cfg.feature_dim, cfg.max_batch_size)
        batch_lengths = zeros(Int, cfg.max_batch_size)
        # No batch set yet: zero valid tokens everywhere, zero active length.
        max_active_length = Ref(0)
        # Scratch: attention adds it as a bias (0.0 / -Inf), so it must match
        # the activation dtype, not Bool.
        attention_bias = zeros(FloatType, cfg.max_seq_length, cfg.max_seq_length)
        new{FloatType,ndims_token}(token_storage, batch_lengths, max_active_length, attention_bias)
    end

end

"""
    clear_batch_slot!(bufs::SeqBuffers, slot::Int) -> Nothing

Mark `slot` unused by recording `bufs.batch_lengths[slot] == 0`. The token
rows are left untouched (they are never read while the count is zero) to
avoid a wasteful zero-fill pass. Call `finalize_batch_lengths!` afterwards if
`max_active_length` must reflect the removal. Writes in place; allocates
nothing.
"""
function clear_batch_slot!(bufs::SeqBuffers, slot::Int)
    @assert 1 <= slot <= length(bufs.batch_lengths) "slot $slot out of range 1..$(length(bufs.batch_lengths))"
    bufs.batch_lengths[slot] = 0
    return nothing
end

"""
    finalize_batch_lengths!(bufs::SeqBuffers) -> Int

Recompute `max_active_length` as `maximum(bufs.batch_lengths)` from the
counts recorded by `fill_token_slot!` / `clear_batch_slot!`, and return it.
Call once after filling all slots for the batch; later kernels slice views
to `1:max_active_length` so short batches skip padded time entirely instead
of computing on zeros. Single scan over `max_batch_size` entries; allocates
nothing.
"""
function finalize_batch_lengths!(bufs::SeqBuffers)
    # Manual max: avoids allocating a temporary via maximum(bufs.batch_lengths).
    active_length = 0
    for slot_length in bufs.batch_lengths
        if slot_length > active_length
            active_length = slot_length
        end
    end
    bufs.max_active_length[] = active_length
    return active_length
end

"""
    fill_causal_padding_bias!(bufs::SeqBuffers, active_length::Int, slot_length::Int)

Fill the leading `active_length × active_length` block of `bufs.attention_bias`
for one batch slot holding `slot_length` valid tokens. Allowed keys get `0`,
blocked keys get `-Inf` (added to attention scores before the rowwise softmax,
so blocked keys contribute exactly zero weight):

- key `j` allowed from query `i` when `j <= min(i, slot_length)` (causal: no
  future; padding: no keys past the true length).
- query rows `i > slot_length` are all `-Inf` (dummy queries whose outputs are
  discarded and whose loss/gradient entries are zeroed by the later cost code).

Only the leading block is written; entries outside `1:active_length` keep
whatever they had and are never read while this batch is active. Writes in
place; allocates nothing.

# Example (`active_length = 4`, `slot_length = 2`)
```julia
#     keys →  1    2    3    4
# queries ↓
#   i = 1     0  -Inf -Inf -Inf   # sees key 1 only
#   i = 2     0    0  -Inf -Inf   # sees keys 1..2
#   i = 3  -Inf -Inf -Inf -Inf   # dummy query (i > slot_length)
#   i = 4  -Inf -Inf -Inf -Inf   # dummy query
```
"""
function fill_causal_padding_bias!(bufs::SeqBuffers{T}, active_length::Int, slot_length::Int) where {T <: AbstractFloat}
    # Fail fast on bad inputs, mirroring the resLayers @asserts in
    # nnCostFunction. active_length == 0 means an all-empty batch: nothing to
    # write and no reader, so return immediately.
    @assert 0 <= slot_length <= active_length "slot_length $slot_length out of range 0..$active_length"
    @assert active_length <= size(bufs.attention_bias, 1) "active_length $active_length exceeds max_seq_length"
    bias = bufs.attention_bias
    # Row-major-ish order (i outer, j inner) matches how the later softmax
    # will read each row contiguously.
    for i in 1:active_length
        if i <= slot_length
            # Valid query: keys 1:i are causal-past, rest blocked. (Keys past
            # slot_length satisfy j > i here, so one cutoff covers both rules.)
            for j in 1:i
                bias[i, j] = zero(T)
            end
            for j in (i + 1):active_length
                bias[i, j] = typemin(T)  # -Inf for Float32/64, -Inf16 for Float16
            end
        else
            # Dummy query (i > slot_length): whole row blocked. Must write
            # every column, not just 1:active_length's tail — row i may hold a
            # previous slot's wider block.
            for j in 1:active_length
                bias[i, j] = typemin(T)
            end
        end
    end
    return nothing
end

"""
    get_active_length(bufs::SeqBuffers) -> Int

Return the number of time steps to compute for the current batch: the maximum
recorded by the last `finalize_batch_lengths!` call. O(1) read of the stored
scalar — no scan, so callers never recompute what the finalize pass already
knows.

Returns `0` when no batch has been set yet.
"""
function get_active_length(bufs::SeqBuffers)
    return bufs.max_active_length[]
end

"""
    fill_token_slot!(bufs::SeqBuffers, slot::Int, src::AbstractMatrix) -> Int

Copy one sequence into `slot` of `bufs.token_storage`, front-packed from row
1, record its length in `bufs.batch_lengths[slot]`, and return how many rows
were written. `src` holds `L` token rows (`size(src, 1) == L`,
`size(src, 2) == feature_dim`), with row 1 the oldest token and row `L` the
newest.

`L` must not exceed `max_seq_length`: oversized input is an `error`, matching
the FCN convention for mismatched data (`error("output layer does not match
data")` in `nnCostFunction`) — silently dropping the oldest rows would hide a
capacity misconfiguration. Size the buffers for the longest sequence the
caller intends to pass, or window the data explicitly before calling.

Rows above the written count are padding and are never read once
`finalize_batch_lengths!` recomputes `max_active_length` from the recorded
counts. The length lives in the buffer from the moment the data lands, so no
separate setter call can forget or contradict it. Writes in place; allocates
nothing.

# Arguments
- `bufs`: buffers created by `SeqBuffers(cfg)`.
- `slot`: batch slot `1 <= slot <= max_batch_size`.
- `src`: `(L, feature_dim)` token rows, any `AbstractMatrix`/`AbstractFloat`.
"""
function fill_token_slot!(bufs::SeqBuffers, slot::Int, src::AbstractMatrix)
    # Fail fast, mirroring the resLayers @asserts in nnCostFunction.
    @assert 1 <= slot <= length(bufs.batch_lengths) "slot $slot out of range 1..$(length(bufs.batch_lengths))"
    @assert size(src, 2) == size(bufs.token_storage, 2) "src has $(size(src, 2)) columns, expected feature_dim $(size(bufs.token_storage, 2))"
    slot_length = _fill_token_slot!(bufs.token_storage, slot, src)
    # Record the count alongside the data: single owner, no desync possible.
    bufs.batch_lengths[slot] = slot_length
    return slot_length
end

# 3D storage (max_batch_size > 1): copy into the slot's block.
function _fill_token_slot!(storage::AbstractArray{T,3}, slot::Int, src::AbstractMatrix) where {T<:AbstractFloat}
    max_seq_length = size(storage, 1)
    total_rows = size(src, 1)
    @assert total_rows > 0 "src must hold at least one token row"
    # No silent truncation: oversized input errors like any other FCN shape
    # mismatch, so a too-small max_seq_length surfaces instead of hiding.
    total_rows <= max_seq_length || error("src holds $total_rows rows, exceeding max_seq_length $max_seq_length; window the data or enlarge the buffers")
    # Column-outer order: each column of storage is contiguous, so the inner
    # loop over rows streams linearly through memory.
    for j in 1:size(src, 2)
        @inbounds @simd for i in 1:total_rows
            storage[i, j, slot] = src[i, j]
        end
    end
    return total_rows
end

# 2D storage (max_batch_size == 1): no slot index; only slot 1 is valid.
function _fill_token_slot!(storage::AbstractArray{T,2}, slot::Int, src::AbstractMatrix) where {T<:AbstractFloat}
    @assert slot == 1 "single-slot buffers only accept slot == 1"
    max_seq_length = size(storage, 1)
    total_rows = size(src, 1)
    @assert total_rows > 0 "src must hold at least one token row"
    total_rows <= max_seq_length || error("src holds $total_rows rows, exceeding max_seq_length $max_seq_length; window the data or enlarge the buffers")
    for j in 1:size(src, 2)
        @inbounds @simd for i in 1:total_rows
            storage[i, j] = src[i, j]
        end
    end
    return total_rows
end