# src/SEQ_COST.jl
#
# Normalization layer for cost functions used with sequence outputs.
#
# FCANN names an elementwise cost three different ways:
#   1. a String from costFuncNames (the costFunc::String convention of
#      nnCostFunction / nnCostFunctionNOGRAD),
#   2. a cost/deriv function pair from costFuncList / costFuncDerivsList,
#   3. a LossType instance (OutputIndex(), CrossEntropyLoss(...)) used when the
#      cost owns its output transform (softmax, index selection).
# SeqCost normalizes all three to one of two concrete, statically-dispatchable
# carriers, so downstream cost code has a single dispatch surface.
#
# Inclusion requirement: this file must come AFTER
# FCN_NFLOATOUT_COSTFUNCTIONS.jl (it reads costFuncNames and the cost pairs at
# load time).  It is not yet included by FCANN.jl; the tests include it
# directly after the FCN cost file.
#
# The name table below is asserted against costFuncNames at load time, so a
# cost added to the FCN fails fast here until it is mapped explicitly.

"""
    abstract type SeqCost end

Normalized description of one cost function for sequence outputs.

Three authoring styles are accepted by the `SeqCost` constructors:

  * a cost *name* (`String`) from `costFuncNames`,
  * a cost *function* or an explicit `(cost, deriv)` function pair from
    `costFuncList` / `costFuncDerivsList`,
  * a `LossType` instance (`OutputIndex()`, `CrossEntropyLoss(...)`).

Each style normalizes to one of two concrete subtypes carrying everything the
cost machinery needs:

  * [`SeqDerivCost`](@ref) - FCN-style elementwise costs whose loss value and
    gradient are separate functions,
  * [`SeqLossCost`](@ref) - FCN-style `LossType` costs that own their output
    transform.

Cost functions are application-neutral: which meaning is attached to each
output element is decided entirely by the cost and the target supplied with
it, exactly as in the fully connected models.
"""
abstract type SeqCost end

"""
    SeqDerivCost(cost, deriv)

FCN-style elementwise cost carried as its `(cost, deriv)` function pair, e.g.
`SeqDerivCost(sqErr, sqErrDeriv)`.

Both halves are stored because both are needed downstream: the scalar loss for
monitoring and finite-difference checks (`calcFinalOut!`), the derivative for
gradient application (`calcDeltaOut!`). The FCN itself presents costs as such
pairs (`costFuncList` / `costFuncDerivsList`).

## Fields
- ``cost`` - elementwise loss function ``cost(pred, target)``
- ``deriv`` - elementwise gradient function ``deriv(pred, target)`` with the
  same shape as ``pred``

## Notes
Which `calcDeltaOut!` / `calcFinalOut!` overload runs is selected by the
*target shape* at the call site (full target, index, index+value), mirroring
how the FCN routes a single deriv function across target formats.
"""
struct SeqDerivCost{FC, FD} <: SeqCost
    cost::FC
    deriv::FD
end

"""
    SeqLossCost(loss)

FCN-style `LossType` cost, e.g. `SeqLossCost(CrossEntropyLoss(0.05))`.

The loss instance owns the output transform (softmax, index selection, ...) and
provides both its scalar value and its gradient through the
`calcFinalOut!(loss, ...)` / `calcDeltaOut!(loss, ...)` methods.

## Fields
- ``loss`` - a `LossType` instance; its dtype-carrying fields (e.g. the
  `beta` of `CrossEntropyLoss{T}`) should match the element type the cost is
  applied to.
"""
struct SeqLossCost{L<:LossType} <: SeqCost
    loss::L
end

# Index of the first usable (non-placeholder) FCN pair containing `f` in either
# slot, or 0.  Several entries of costFuncList / costFuncDerivsList are
# `Returns(nothing)` placeholders (their concepts dispatch through LossType
# instead), and those must never be handed back as a cost pair.
function _cost_pair_index(f::Function)
    for i in eachindex(costFuncList)
        v = costFuncList[i]
        d = costFuncDerivsList[i]
        (v isa Returns || d isa Returns) && continue
        (v === f || d === f) && return i
    end
    return 0
end

"""
    SeqCost(loss::LossType) -> SeqLossCost

Normalize a `LossType` instance (passes through unchanged, keeping its
dtype-carrying fields such as `beta`).
"""
SeqCost(loss::LossType) = SeqLossCost(loss)

"""
    SeqCost(cost::Function, deriv::Function) -> SeqDerivCost

Explicit `(cost, deriv)` pair for an elementwise cost, bypassing the name
table. Use when wiring a function pair that is not in the FCN cost lists.
"""
SeqCost(cost::Function, deriv::Function) = SeqDerivCost(cost, deriv)

"""
    SeqCost(f::Function) -> SeqDerivCost

Locate `f` (as either the cost or the deriv half) inside the FCN cost pair
lists and return the pair it belongs to.

Throws an `ArgumentError` if `f` belongs to no usable pair - including the
`Returns(nothing)` placeholder entries, which are rejected on purpose.
"""
function SeqCost(f::Function)
    i = _cost_pair_index(f)
    i == 0 && throw(ArgumentError(string(
        "function is not the cost or deriv half of an FCN cost pair; ",
        "construct an explicit pair with SeqCost(cost, deriv) or use a name ",
        "from costFuncNames")))
    return SeqDerivCost(costFuncList[i], costFuncDerivsList[i])
end

# Explicit name -> SeqCost table (deliberately NOT a lookup into
# costFuncDerivsList: 8 of its 12 entries are Returns(nothing) placeholders).
#   * the four elementwise names map to their real (cost, deriv) pairs,
#   * *Index names map to the base error's pair - index selection is carried
#     by the target shape at the call site, as in
#     calcDeltaOut!(deriv, deltas, a, y, indices, m),
#   * outputIndex / crossEntropy* names map to their LossType; Vector vs
#     Matrix targets already select the kernel overload, so every spelling of
#     one concept normalizes to the same carrier.
const SEQ_COST_BY_NAME = Dict{String, SeqCost}(
    "absErr"                => SeqDerivCost(absErr, absErrDeriv),
    "sqErr"                 => SeqDerivCost(sqErr, sqErrDeriv),
    "normLogErr"            => SeqDerivCost(normLogErr, normLogErrDeriv),
    "cauchyLogErr"          => SeqDerivCost(cauchyLogErr, cauchyLogErrDeriv),
    "sqErrIndex"            => SeqDerivCost(sqErr, sqErrDeriv),
    "absErrIndex"           => SeqDerivCost(absErr, absErrDeriv),
    "outputIndex"           => SeqLossCost(OutputIndex()),
    "outputIndexBatch"      => SeqLossCost(OutputIndex()),
    "crossEntropy"          => SeqLossCost(CrossEntropyLoss()),
    "crossEntropyBatch"     => SeqLossCost(CrossEntropyLoss()),
    "crossEntropyDist"      => SeqLossCost(CrossEntropyLoss()),
    "crossEntropyDistBatch" => SeqLossCost(CrossEntropyLoss()),
)

# Fail fast at load time if FCN_NFLOATOUT_COSTFUNCTIONS.jl gains or renames a
# cost name without a corresponding entry above.
@assert Set(keys(SEQ_COST_BY_NAME)) == Set(costFuncNames) string(
    "SEQ_COST name table is out of sync with costFuncNames; ",
    "update the table in src/SEQ_COST.jl"),

"""
    seq_head_value!(cost::SeqLossCost{<:CrossEntropyLoss}, scratch, head, index::Integer) -> T

Cross-entropy value for one head at the current read position: the softmax of
the head's raw outputs, then ``-log(p[index])`` (plus ``beta * entropy`` if
the loss carries ``beta > 0``). Mirrors
`calcFinalOut!(::CrossEntropyLoss, a::Vector, index)`.

The kernel overwrites its prediction argument in place, so the head's output
is staged first; `scratch.outputs` itself is left intact. Returns the staged
loss value.

# Notes
``index`` must satisfy ``1 <= index <= O_head``. The weight applied to this
loss is set later by the multi-head combiner (`seq_cost!`); this returns the
raw unweighted value.
"""

function seq_head_value!(cost::SeqLossCost{<:CrossEntropyLoss},
        scratch::SeqOutputScratch{T}, head::Int, index::Integer) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert 1 <= index <= length(scratch.outputs_view[head])
    r = scratch.head_ranges[head]
    # Stage a copy: calcFinalOut! clobbers its prediction argument.
    copyto!(scratch.out_staged, first(r), scratch.outputs, first(r), length(r))
    calcFinalOut!(cost.loss, scratch.staged_view[head], Int(index))
    return scratch.staged_view[head][Int(index)]
end

"""
    seq_head_value!(cost::SeqLossCost{<:CrossEntropyLoss}, scratch, head, targets) -> T
"""
function seq_head_value!(cost::SeqLossCost{<:CrossEntropyLoss},
        scratch::SeqOutputScratch{T}, head::Int,
        targets::AbstractVector{T}) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert eltype(targets) == T
    @assert length(targets) == length(scratch.outputs_view[head])
    r = scratch.head_ranges[head]
    # Stage a copy: calcFinalOut! clobbers entry 1 of its prediction.
    copyto!(scratch.out_staged, first(r), scratch.outputs, first(r), length(r))
    calcFinalOut!(cost.loss, scratch.staged_view[head], targets)
    return scratch.staged_view[head][1]
end

"""
    seq_head_value!(cost::SeqLossCost{OutputIndex}, scratch, head, index::Integer) -> T
"""
function seq_head_value!(cost::SeqLossCost{OutputIndex},
        scratch::SeqOutputScratch{T}, head::Int, index::Integer) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert 1 <= index <= length(scratch.outputs_view[head])
    return scratch.outputs_view[head][Int(index)]
end

"""
    seq_head_value!(cost::SeqDerivCost, scratch, head, target) -> T
"""
function seq_head_value!(cost::SeqDerivCost, scratch::SeqOutputScratch{T}, head::Int,
        target::AbstractVector) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert eltype(target) == T
    n = length(target)
    r = scratch.head_ranges[head]
    # Stage a copy: calcFinalOut! clobbers its prediction argument.
    copyto!(scratch.out_staged, first(r), scratch.outputs, first(r), length(r))
    calcFinalOut!(cost.cost, scratch.staged_view[head], target, 1, n)
    # Sum the first n entries: for a two-output cost the second half still
    # holds the stale second-output block, not loss values.
    v = zero(T)
    @inbounds for i in 1:n
        v += scratch.staged_view[head][i]
    end
    return v
end

"""
    seq_head_grad!(cost::SeqDerivCost, scratch, head, target) -> Nothing
"""
function seq_head_grad!(cost::SeqDerivCost, scratch::SeqOutputScratch{T}, head::Int,
        target::AbstractVector) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert eltype(target) == T
    n = length(target)
    # Read predictions straight from outputs (no clobber); deltas are written
    # through dout_view, which aliases the flat dout slice head_backward! uses.
    # SubArray + AbstractArray dispatch now works without allocation because
    # calcDeltaOut! uses an explicit max loop instead of maximum(a).
    calcDeltaOut!(cost.deriv, scratch.dout_view[head], scratch.outputs_view[head],
                  target, 1, n)
    return nothing
end

"""
    SeqCost(name::AbstractString) -> SeqCost
"""
function SeqCost(name::AbstractString)
    c = get(SEQ_COST_BY_NAME, String(name), nothing)
    c === nothing && throw(ArgumentError(string(
        "unknown cost function name \"", name, "\"; valid names: ",
        join(costFuncNames, ", "))))
    return c
end

"""
    seq_head_grad!(cost::SeqLossCost{<:CrossEntropyLoss}, scratch, head, index::Integer) -> Nothing
"""
function seq_head_grad!(cost::SeqLossCost{<:CrossEntropyLoss},
        scratch::SeqOutputScratch{T}, head::Int, index::Integer) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert 1 <= index <= length(scratch.outputs_view[head])
    # Read predictions straight from outputs (no clobber); deltas written
    # through dout_view, which aliases the flat dout slice head_backward! uses.
    # SubArray + AbstractArray dispatch now works without allocation because
    # calcDeltaOut! uses an explicit max loop instead of maximum(a).
    calcDeltaOut!(cost.loss, scratch.dout_view[head], scratch.outputs_view[head],
                  Int(index))
    return nothing
end

"""
    seq_head_grad!(cost::SeqLossCost{<:CrossEntropyLoss}, scratch, head, targets) -> Nothing
"""
function seq_head_grad!(cost::SeqLossCost{<:CrossEntropyLoss},
        scratch::SeqOutputScratch{T}, head::Int,
        targets::AbstractVector{T}) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert eltype(targets) == T
    @assert length(targets) == length(scratch.outputs_view[head])
    # Read predictions straight from outputs (no clobber); deltas written
    # through dout_view, which aliases the flat dout slice head_backward! uses.
    # SubArray + AbstractArray dispatch now works without allocation because
    # calcDeltaOut! uses an explicit max loop instead of maximum(a).
    calcDeltaOut!(cost.loss, scratch.dout_view[head], scratch.outputs_view[head],
                  targets)
    return nothing
end

"""
    seq_head_grad!(cost::SeqLossCost{OutputIndex}, scratch, head, index::Integer) -> Nothing
"""
function seq_head_grad!(cost::SeqLossCost{OutputIndex},
        scratch::SeqOutputScratch{T}, head::Int, index::Integer) where {T}
    @assert 1 <= head <= length(scratch.head_ranges)
    @assert 1 <= index <= length(scratch.outputs_view[head])
    fill!(scratch.dout_view[head], zero(T))
    scratch.dout_view[head][Int(index)] = one(T)
    return nothing
end

"""
    SeqHeadCost(cost::SeqCost, target)

One head's cost function paired with its target.  The target's shape (plain
vector, index, index+values, or distribution matrix) selects which
`calcFinalOut!` / `calcDeltaOut!` overload runs at the call site.

# Fields
- ``cost`` - a normalized `SeqCost` carrier (from `SeqCost(name)`, `SeqCost(f)`,
  or `SeqCost(loss::LossType)`)
- ``target`` - per-head target data; element type must match the scratch's
  `FloatType`, and its length/shape must match the head's output width `O_k`
  (or `O_k ÷ 2` for the two-output `normLogErr`/`cauchyLogErr` convention)

# Notes
No weight is stored here: weighting lives at the `SeqOutputCosts` level so a
single head can be reused with different weights across training runs.

Both fields are parametrized (`SeqHeadCost{C,TT}`) so each instance carries
*concrete* field types.  An abstractly-typed field (`cost::SeqCost`, or an
untyped `target`) makes every call through it dispatch dynamically, which
boxes isbits costs - measured at 96 bytes per head per call.  The parametrized
form keeps the whole cost application statically dispatched and 0-alloc.
"""
struct SeqHeadCost{C<:SeqCost, TT}
    cost::C
    target::TT
end

"""
    SeqOutputCosts(heads, weights)

A collection of per-head costs paired with an output set.  `heads` is aligned
1:1 with `SeqOutputSet.heads`; `weights[k]` scales head `k`'s loss *and*
gradient (the latter flows into `head_backward!` via `dout_k`).

# Fields
- ``heads`` - a `Tuple` of `SeqHeadCost` instances, one per output head
- ``weights::Vector{T}`` - per-head scalar multiplier; defaults to `ones(T)` per
  head when `weights === nothing`

# Notes
`heads` is stored as a `Tuple`, not a `Vector`.  Heads legitimately differ in
cost and target types (e.g. a `SeqDerivCost` with a `Vector` target next to a
`SeqLossCost{CrossEntropyLoss}` with an index), so no single concrete
`Vector{...}` element type exists - any `Vector` of heads would be abstractly
element-typed and re-introduce dynamic dispatch on every access.  A `Tuple`
keeps each head's concrete type, so `enumerate`/`getindex` iteration is fully
specialized and the cost application stays 0-alloc (small head counts unroll).
Construction asserts `length(heads) == length(weights)` when both are provided.
"""
struct SeqOutputCosts{T, H<:Tuple}
    heads::H
    weights::Vector{T}
end

"""
    SeqOutputCosts(costs::Vector{<:SeqCost}, targets::Vector, weights::Vector{T}=nothing) where {T}

Build a `SeqOutputCosts` from parallel vectors of costs and targets.  When
`weights` is omitted, every head defaults to weight `one(T)`.

# Notes
`costs`, `targets`, and `weights` must all have the same length.
"""
function SeqOutputCosts(costs::Vector, targets::Vector, weights=nothing)
    @assert length(costs) == length(targets) "costs and targets must have the same length"
    if weights === nothing
        weights = ones(Float64, length(costs))
    else
        @assert length(weights) == length(costs) "weights must have the same length as costs"
    end
    # Tuple(...) materializes a heterogeneous tuple of concrete SeqHeadCosts
    # (one-time setup allocation; not on the hot path).
    heads = Tuple(SeqHeadCost(costs[k], targets[k]) for k in eachindex(costs))
    T = eltype(weights)
    return SeqOutputCosts{T, typeof(heads)}(heads, weights)
end

"""
    SeqOutputCosts(set::SeqOutputSet, costs::Vector{<:SeqCost}, targets::Vector,
        weights::Vector{T}=nothing) where {T}

Convenience constructor that also validates alignment with an output set:
`length(set.heads) == length(costs) == length(targets) == length(weights)`.

# Notes
This is the recommended entry point when building costs from an existing
`SeqOutputSet`, since it catches mismatched head counts at construction time
"""
function SeqOutputCosts(set::SeqOutputSet, costs::Vector, targets::Vector,
        weights=nothing)
    @assert length(set.heads) == length(costs) && length(costs) == length(targets)
    return SeqOutputCosts(costs, targets, weights)
end

"""
    seq_cost_grads!(spec::SeqOutputCosts, scratch::SeqOutputScratch)

Gradient-only path: computes the weighted per-head output gradient
``dout_k = weight_k * dloss_k/dout_k`` for every head and writes it into
`scratch.dout_view[k]`, which `head_backward!` consumes. No loss values are
computed - this is the path used by the backward pass.

# Arguments
- `spec`: per-head costs + targets + weights, aligned with `scratch`
- `scratch`: output-set scratch; `dout_view` is written in place

# Notes
- The gradient kernel only reads its prediction argument.
- `weight_k` is applied in-place; a weight of exactly `one(T)` skips the multiply.
- Allocation: 0 in steady state.
"""
function seq_cost_grads!(spec::SeqOutputCosts, scratch::SeqOutputScratch)
    @assert length(spec.heads) == length(scratch.head_ranges)
    T = eltype(spec.weights)
    # enumerate over the tuple: each `hc` keeps its concrete SeqHeadCost type,
    # so the seq_head_grad! call below is statically dispatched (0-alloc).
    for (k, hc) in enumerate(spec.heads)
        seq_head_grad!(hc.cost, scratch, k, hc.target)
        wk = spec.weights[k]
        if wk != one(T)
            @inbounds for i in eachindex(scratch.dout_view[k])
                scratch.dout_view[k][i] *= wk
            end
        end
    end
    return nothing
end

"""
    seq_cost_values!(spec::SeqOutputCosts, scratch::SeqOutputScratch)

Loss-value path: computes the combined loss ``J`` across all heads.
Used for gradient-checking and monitoring.

# Arguments
- `spec`: per-head costs + targets + weights, aligned with `scratch`
- `scratch`: output-set scratch

# Returns
- `J`: combined weighted loss as a scalar
"""
function seq_cost_values!(spec::SeqOutputCosts{T}, scratch::SeqOutputScratch{T}) where {T}
    @assert length(spec.heads) == length(scratch.head_ranges)
    J = zero(T)
    # enumerate over the tuple: each `hc` keeps its concrete SeqHeadCost type,
    # so the seq_head_value! call below is statically dispatched (0-alloc).
    for (k, hc) in enumerate(spec.heads)
        J += spec.weights[k] * seq_head_value!(hc.cost, scratch, k, hc.target)
    end
    return J
end
