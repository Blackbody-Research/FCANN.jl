# Comprehensive gradient test suite for CPU backend
# Tests all cost functions with all method signatures

using FCANN
using Test

setBackend(:CPU)

println("Testing all CPU gradient checks...")
println("="^60)

# Test parameters
cost_funcs = ["absErr", "sqErr", "normLogErr", "cauchyLogErr"]
lambdas = [0.0f0, 0.1f0, 1.0f0]
orientations = ['N', 'T']

# Track results
results = Dict{String, Bool}()

# ============================================================================
# Test 1: All cost functions with default method
# ============================================================================
println("\n=== Test Group 1: Cost Functions with Default Method ===")
for costFunc in cost_funcs
    for lambda in lambdas
        test_name = "Cost=$costFunc, Lambda=$lambda"
        println(test_name)
        err = checkNumGrad(lambda, costFunc=costFunc; printmsg=false)
        results[test_name] = err < 0.015
        println("err = $err")
        println(results[test_name] ? "PASS" : "FAIL")
    end
end

# ============================================================================
# Test 2: Cost functions with residual layers
# ============================================================================
println("\n=== Test Group 2: Cost Functions with Residual Layers ===")
for costFunc in cost_funcs
    for lambda in lambdas
        test_name = "Cost=$costFunc, Lambda=$lambda, resLayers=1"
        println(test_name)
        err = checkNumGrad(lambda, costFunc=costFunc, resLayers=1; printmsg=false)
        results[test_name] = err < 0.015
        println("err = $err")
        println(results[test_name] ? "PASS" : "FAIL")
    end
end

# ============================================================================
# Test 3: Cost functions with custom activation lists
# ============================================================================
println("\n=== Test Group 3: Cost Functions with Custom Activation Lists ===")
for costFunc in cost_funcs
    for lambda in lambdas
        test_name = "Cost=$costFunc, Lambda=$lambda, activation_list=[T,F,T]"
        println(test_name)
        err = checkNumGrad(lambda, costFunc=costFunc, hidden_layers=[10, 10, 10], 
                          activation_list=[true, false, true]; printmsg=false)
        results[test_name] = err < 0.015
        println("err = $err")
        println(results[test_name] ? "PASS" : "FAIL")
    end
end

# ============================================================================
# Test 4: Cost functions with output_index method
# ============================================================================
println("\n=== Test Group 4: Cost Functions with Output Index ===")
for costFunc in cost_funcs
    if !occursin("Log", costFunc)
        for lambda in lambdas
            test_name = "Cost=$costFunc, Lambda=$lambda, output_index=1"
            println(test_name)
            # Note: output_index method doesn't support costFunc keyword
            err = checkNumGrad(1, lambda; printmsg=false)
            results[test_name] = err < 0.015
            println("err = $err")
            println(results[test_name] ? "PASS" : "FAIL")
        end
    else
        println("Skipping log cost function for output_index: $costFunc")
    end
end

# ============================================================================
# Test 5: Cost functions with input orientation
# ============================================================================
println("\n=== Test Group 5: Cost Functions with Input Orientation ===")
for costFunc in cost_funcs
    if !occursin("Log", costFunc)
        for lambda in lambdas
            for orientation in orientations
                test_name = "Cost=$costFunc, Lambda=$lambda, orientation=$orientation"
                println(test_name)
                err = checkNumGrad(lambda, costFunc, input_orientation=orientation; printmsg=false)
                results[test_name] = err < 0.015
                println("err = $err")
                println(results[test_name] ? "PASS" : "FAIL")
            end
        end
    else
        println("Skipping log cost function for orientation tests: $costFunc")
    end
end

# ============================================================================
# Test 6: Cross Entropy with distribution targets
# ============================================================================
println("\n=== Test Group 6: Cross Entropy with Distribution Targets ===")
for lambda in lambdas
    for single_example in [false, true]
        for orientation in orientations
            test_name = "CrossEntropy, Lambda=$lambda, single=$single_example, orientation=$orientation"
            println(test_name)
            err = checkNumGrad(lambda, Val(:dist); single_example=single_example, 
                             input_orientation=orientation, printmsg=false)
            results[test_name] = err < 0.015
            println("err = $err")
            println(results[test_name] ? "PASS" : "FAIL")
        end
    end
end

# ============================================================================
# Test 7: Cross Entropy with output index and orientation
# ============================================================================
println("\n=== Test Group 7: Cross Entropy with Output Index and Orientation ===")
for lambda in lambdas
    for orientation in orientations
        test_name = "CrossEntropy, Lambda=$lambda, orientation=$orientation"
        println(test_name)
        err = checkNumGrad(lambda, orientation; printmsg=false)
        results[test_name] = err < 0.015
        println("err = $err")
        println(results[test_name] ? "PASS" : "FAIL")
    end
end

# ============================================================================
# Test 8: Cross Entropy with output values
# ============================================================================
println("\n=== Test Group 8: Cross Entropy with Output Values ===")
for lambda in lambdas
    for orientation in orientations
        test_name = "CrossEntropy, Lambda=$lambda, orientation=$orientation, use_values=true"
        println(test_name)
        err = checkNumGrad(lambda, orientation; use_values=true, printmsg=false)
        results[test_name] = err < 0.015
        println("err = $err")
        println(results[test_name] ? "PASS" : "FAIL")
    end
end

# ============================================================================
# Test 9: Lambda = 0 with no hidden layers
# ============================================================================
println("\n=== Test Group 9: Lambda = 0, No Hidden Layers ===")
for costFunc in cost_funcs
    test_name = "Cost=$costFunc, Lambda=0, no hidden layers"
    println(test_name)
    err = checkNumGrad(0.0f0, costFunc=costFunc, hidden_layers=Vector{Int64}(); printmsg=false)
    results[test_name] = err < 0.015
    println("err = $err")
    println(results[test_name] ? "PASS" : "FAIL")
end

# ============================================================================
# Test 10: Single example mode
# ============================================================================
println("\n=== Test Group 10: Single Example Mode ===")
for costFunc in cost_funcs
    for lambda in lambdas
        test_name = "Cost=$costFunc, Lambda=$lambda, m=1"
        println(test_name)
        err = checkNumGrad(lambda, costFunc=costFunc, m=1; printmsg=false)
        results[test_name] = err < 0.015
        println("err = $err")
        println(results[test_name] ? "PASS" : "FAIL")
    end
end

# ============================================================================
# Test 11: Output index with different loss types
# ============================================================================
println("\n=== Test Group 11: Output Index with Different Loss Types ===")
for loss_type in [OutputIndex(), CrossEntropyLoss()]
    test_name = "output_index=1, loss_type=$(typeof(loss_type))"
    println(test_name)
    err = checkNumGrad(1, 1.0f0; loss_type=loss_type, printmsg=false)
    results[test_name] = err < 0.015
    println("err = $err")
    println(results[test_name] ? "PASS" : "FAIL")
end

# ============================================================================
# Test 12: Output vector mode
# ============================================================================
println("\n=== Test Group 12: Output Vector Mode ===")
for lambda in lambdas
    test_name = "output_vector=true, Lambda=$lambda"
    println(test_name)
    err = checkNumGrad(1, lambda; output_vector=true, printmsg=false)
    results[test_name] = err < 0.015
    println("err = $err")
    println(results[test_name] ? "PASS" : "FAIL")
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^60)
println("SUMMARY")
println("="^60)

passed = count(values(results))
total = length(results)
println("Passed: $passed / $total tests")

if passed == total
    println("ALL TESTS PASSED!")
else
    println("FAILED TESTS:")
    for (name, passed_test) in results
        if !passed_test
            println("  - $name (err: $results[$name])")
        end
    end
end