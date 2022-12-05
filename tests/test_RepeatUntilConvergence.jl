import Test: @test, @test_throws, @testset
import ObjectMatching: RepeatUntilConvergence, execute!

@testset "RepeatUntilConvergence" begin
    # FIXTURES
    function update_func(frac=0.5::Float64)
        return frac*0.5
    end
    function has_converged(history::Vector{Float64})
        return last(history) < 0.0001
    end
    R = RepeatUntilConvergence{Float64}(update_func, has_converged; memory_size=20)
    # end of fixtures

    function test_right_convergence(R)
        res, _ = execute!(R, 0.001)
        return res
    end
    @test test_right_convergence(R) < 0.0001

    function test_raises_error_if_has_converged_isnot_boolean()
        function non_boolean_func()
            return 1
        end
        RepeatUntilConvergence{Float64}(update_func, non_boolean_func)
    end
    @test_throws ArgumentError test_raises_error_if_has_converged_isnot_boolean()

    function test_raises_error_if_update_func_return_wrong_type()
        function weird_update(x::Float64)
            return 1::Int64
        end
        RepeatUntilConvergence{Float64}(weird_update, has_converged)
    end
    @test_throws ArgumentError test_raises_error_if_has_converged_isnot_boolean()

    function test_raises_error_if_update_func_accept_wrong_type()
        function boolean_wrong(x::Int64)
            return true
        end
        RepeatUntilConvergence{Float64}(update_func, boolean_wrong)
    end
    @test_throws ArgumentError test_raises_error_if_has_converged_isnot_boolean()

    function test_raises_error_if_has_converged_accept_wrong_type()
        function weird_update(x::Int64)
            return 1::Float64
        end
        RepeatUntilConvergence{Float64}(weird_update, has_converged)
    end
    @test_throws ArgumentError test_raises_error_if_has_converged_isnot_boolean()

    function test_works_if_stop_function_has_default_args()
        function stop_with_defarg(x::Vector{Float64}; def=0)
            return has_converged(x)
        end
        R = RepeatUntilConvergence{Float64}(update_func, stop_with_defarg)
        res, _ = execute!(R, 0.001)
        return res
    end
    @test test_works_if_stop_has_default_args() < 0.0001
end

