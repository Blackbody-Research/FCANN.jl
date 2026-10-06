# Numerical gradient checking for every loss/regularization variation, on both backends.
#
# This file is the single home for all checkNumGrad testing in the package.  The other test
# files (CPU_singleCore_tests.jl, GPU_singleCore_tests.jl, CPU_multiCore_tests.jl) cover only
# higher-level functionality (training, evaluation, I/O and distributed helpers).

using FCANN
using Test

# Variation parameters exercised by every group below
const cost_funcs   = ["absErr", "sqErr", "normLogErr", "cauchyLogErr"]
const lambdas      = [0.0f0, 0.1f0, 1.0f0]
const betas        = [0.0f0, 0.1f0, 1.0f0]
const orientations = ['N', 'T']
const grad_tol     = 0.015

# Tolerance for the CPU/GPU agreement of the reported forward cost.  The two implementations
# reduce the same per-example costs in a different order, which puts an absolute float32 floor of
# a few 1e-4 on the agreement (visible as a large *relative* gap when the cost is near zero, e.g.
# the output index loss).  The tolerance below leaves ~10x margin over that noise floor while still
# failing loudly for a cost that omits a whole term: dropping beta on the m=1 entropy regularized
# check shifts the cost by ~0.3, and dropping the L2 penalty shifts it by lambda*||W||^2/(2m) ~ 0.03.
const cost_rtol = 1.0f-2
const cost_atol = 5.0f-3

# grad_errors[(backend, name)] = relative gradient error and grad_costs[(backend, name)] = the
# (gpu, cpu) forward costs reported for the same variation.  Both are used to assert that the two
# implementations agree - the gradient check alone cannot catch a cost that is computed wrongly
# (for example a GPU cost that ignores entropy regularization).
const grad_errors = Dict{Tuple{Symbol, String}, Float64}()
const grad_costs  = Dict{Tuple{Symbol, String}, Tuple{Float64, Float64}}()

# Perform a numerical gradient check for the requested backend.  On the GPU the check also reports
# the GPU and CPU forward costs (return_costs) so they can be compared directly; the CPU check is
# the reference implementation and only returns the relative gradient error.
function numgrad(backend::Symbol, args...; kwargs...)
    backend === :GPU || return checkNumGrad(args...; kwargs...)
    return checkNumGrad(args...; kwargs..., return_costs = true)
end

# Run one gradient check, assert the gradient is correct and, when available, assert that the GPU
# and CPU agree on the forward cost.  The closure is passed first so that the standard Julia
# `gradcheck(backend, name) do ... end` (which splices the closure in as the first argument) works.
function gradcheck(f, backend::Symbol, name::String)
    res = f()
    err = res isa NamedTuple ? res.grad_err : res
    grad_errors[(backend, name)] = err
    @test err < grad_tol
    if res isa NamedTuple
        grad_costs[(backend, name)] = (res.cost_gpu, res.cost_cpu)
        @test isapprox(res.cost_gpu, res.cost_cpu; rtol = cost_rtol, atol = cost_atol)
    end
    return err
end

# Available backends; GPU is skipped automatically when it could not be initialized.
const backends = :GPU in backendList ? [:CPU, :GPU] : [:CPU]
if !(:GPU in backendList)
    @info "GPU backend not available; running numerical gradient checks on CPU only"
end

@testset "Numerical Gradient Checking" begin
    for backend in backends
        tag = string(backend)
        setBackend(backend)   # once per backend; the gradient checks below dispatch on it

        # 1: every elementwise cost function over each regularization strength
        @testset "$tag default method (cost functions)" begin
            for costFunc in cost_funcs, lambda in lambdas
                gradcheck(backend, "default,Cost=$costFunc,Lambda=$lambda") do
                    numgrad(backend, lambda, costFunc = costFunc; printmsg = false)
                end
            end
        end

        # 2: residual connections between hidden layers
        @testset "$tag residual layers" begin
            for costFunc in cost_funcs, lambda in lambdas
                gradcheck(backend, "resLayers=1,Cost=$costFunc,Lambda=$lambda") do
                    numgrad(backend, lambda, costFunc = costFunc, resLayers = 1; printmsg = false)
                end
            end
        end

        # 3: mixed activation functions across hidden layers
        @testset "$tag custom activation lists" begin
            for costFunc in cost_funcs, lambda in lambdas
                gradcheck(backend, "activation_list,Cost=$costFunc,Lambda=$lambda") do
                    numgrad(backend, lambda, costFunc = costFunc, hidden_layers = [10, 10, 10],
                                 activation_list = [true, false, true]; printmsg = false)
                end
            end
        end

        # 4: a single output index selected as the loss target (log likelihood costs unsupported)
        @testset "$tag output index cost functions" begin
            for costFunc in cost_funcs
                occursin("Log", costFunc) && continue
                for lambda in lambdas
                    gradcheck(backend, "output_index=$costFunc,Lambda=$lambda") do
                        numgrad(backend, 1, lambda; printmsg = false)
                    end
                end
            end
        end

        # 5: index cost functions with either input orientation
        @testset "$tag output index orientation" begin
            for costFunc in cost_funcs
                occursin("Log", costFunc) && continue
                for lambda in lambdas, orientation in orientations
                    gradcheck(backend, "index_orientation=$costFunc,Lambda=$lambda,$orientation") do
                        numgrad(backend, lambda, costFunc, input_orientation = orientation; printmsg = false)
                    end
                end
            end
        end

        # 6: cross entropy with per-row target probability distributions
        @testset "$tag distribution targets" begin
            for lambda in lambdas, beta in betas, single_example in (false, true), orientation in orientations
                gradcheck(backend, "dist,Lambda=$lambda,beta=$beta,single=$single_example,$orientation") do
                    numgrad(backend, lambda, Val(:dist);
                                 loss_type = CrossEntropyLoss(beta),
                                 single_example = single_example,
                                 input_orientation = orientation,
                                 printmsg = false)
                end
            end
        end

        # 7: cross entropy at a per-example output index for either input orientation
        @testset "$tag batch cross entropy output index" begin
            for lambda in lambdas, beta in betas, orientation in orientations
                gradcheck(backend, "ce_index,Lambda=$lambda,beta=$beta,$orientation") do
                    numgrad(backend, lambda, orientation;
                                 loss_type = CrossEntropyLoss(beta),
                                 printmsg = false)
                end
            end
        end

        # 8: as above with an extra per-example scalar multiplying the loss
        @testset "$tag batch cross entropy output values" begin
            for lambda in lambdas, beta in betas, orientation in orientations
                gradcheck(backend, "ce_values,Lambda=$lambda,beta=$beta,$orientation") do
                    numgrad(backend, lambda, orientation;
                                 loss_type = CrossEntropyLoss(beta),
                                 use_values = true,
                                 printmsg = false)
                end
            end
        end

        # 9: no hidden layers at all
        @testset "$tag zero hidden layers" begin
            for costFunc in cost_funcs
                gradcheck(backend, "no_hidden,Cost=$costFunc") do
                    numgrad(backend, 0.0f0, costFunc = costFunc, hidden_layers = Vector{Int64}(); printmsg = false)
                end
            end
        end

        # 10: single example (m = 1)
        @testset "$tag single example" begin
            for costFunc in cost_funcs, lambda in lambdas
                gradcheck(backend, "m=1,Cost=$costFunc,Lambda=$lambda") do
                    numgrad(backend, lambda, costFunc = costFunc, m = 1; printmsg = false)
                end
            end
        end

        # 11: per-example output index vector
        @testset "$tag output vector" begin
            for lambda in lambdas
                gradcheck(backend, "output_vector,Lambda=$lambda") do
                    numgrad(backend, 1, lambda; output_vector = true, printmsg = false)
                end
            end
        end

        # 12: entropy regularized cross entropy at an output index (single example and batch)
        @testset "$tag cross entropy entropy regularization" begin
            for lambda in lambdas, beta in betas, single in (true, false)
                gradcheck(backend, "ce_entropy,Lambda=$lambda,beta=$beta,m=$(single ? 1 : 1000)") do
                    numgrad(backend, 1, lambda;
                                 loss_type = CrossEntropyLoss(beta),
                                 m = single ? 1 : 1000,
                                 printmsg = false)
                end
            end
        end

        # 13: output index loss with each supported loss type
        @testset "$tag output index loss types" begin
            for loss_type in (OutputIndex(), CrossEntropyLoss())
                gradcheck(backend, "loss_type=$(typeof(loss_type))") do
                    numgrad(backend, 1, 1.0f0; loss_type = loss_type, printmsg = false)
                end
            end
        end
    end
end

# Every variation must pass on both backends, and the CPU and GPU runs must cover exactly the
# same set of variations - this is what keeps the two implementations in step.  The gradient check
# alone is not sufficient: a forward cost that is computed incorrectly (for example one that
# ignores entropy regularization) can still produce correct gradients, so the reported GPU and CPU
# forward costs are compared here as well.
if :GPU in backends
    @testset "CPU and GPU gradient agreement" begin
        cpu = Dict(name => err for ((backend, name), err) in grad_errors if backend === :CPU)
        gpu = Dict(name => err for ((backend, name), err) in grad_errors if backend === :GPU)

        @test !isempty(cpu)
        @test sort!(collect(keys(cpu))) == sort!(collect(keys(gpu)))

        for name in sort!(collect(keys(cpu)))
            @test cpu[name] < grad_tol
            @test gpu[name] < grad_tol
        end
    end

    @testset "CPU and GPU cost agreement" begin
        # every GPU variation must have reported a cost, and the CPU and GPU costs must match
        @test !isempty(grad_costs)
        @test sort!(collect(keys(grad_costs))) ==
              sort!([(backend, name) for (backend, name) in keys(grad_errors) if backend === :GPU])

        for name in sort!(collect(keys(grad_costs)))
            (cost_gpu, cost_cpu) = grad_costs[name]
            @test isapprox(cost_gpu, cost_cpu; rtol = cost_rtol, atol = cost_atol)
        end
    end
else
    @info "Skipping CPU/GPU comparison (GPU backend unavailable)"
end
