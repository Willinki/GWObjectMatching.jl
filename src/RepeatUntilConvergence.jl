import DataStructures: CircularBuffer
import Base

"""This object aims at simulating a loop until convergence, we would like to get rid
of while loops in the main algorithms.

Arguments
---------

- update_func: Function. It performs calculations and returns a value.
- has_converged: Function. Takes history as input.
Returns a boolean value, if True, the loop is stopped.
- init_vals: Any. Initial values for update_func
- history: contains values of returned parameters.
# TODO: FIX DOCUMENTATION
# TODO: INSERT THAT IF MEMORY SIZE IS ONE, history is not a vector 
# TODO: maybe mutable can be removed
"""
abstract type BaseRepeatUntilConvergence{T} end

mutable struct RepeatUntilConvergence{T} <: BaseRepeatUntilConvergence{T}
    update_func::Function
    has_converged::Function
    history::CircularBuffer{T}
    init_vals::T

    function RepeatUntilConvergence{T}(
        update_func::Function,
        has_converged::Function;
        memory_size=10::Int64
        ) where T
        hasmethod(update_func, (T, )) || throw(ArgumentError(
            "The provided update_func has no method for type $(T)"
        ))
        hasmethod(has_converged, (Vector{T}, )) || throw(ArgumentError(
            "The provided has_converged has no method for type Vector{$(T)}"
        ))
        T in Base.return_types(update_func, (T,)) || throw(ArgumentError(
            "The provided update_func does not return type $(T)"
        ))
        Bool in Base.return_types(has_converged, (Vector{T},)) || throw(ArgumentError(
            "The provided has_converged does not return type Bool"
        ))
        # incomplete initialization, init_vals are set at execution
        return new{T}(update_func, has_converged, CircularBuffer{T}(memory_size))
    end
end

mutable struct LargeMemoryRepeatUntilConvergence{T} <: BaseRepeatUntilConvergence{T}
    update_func::Function
    has_converged::Function
    history::Vector{T}
    init_vals::T

    function LargeMemoryRepeatUntilConvergence{T}(
        update_func::Function,
        has_converged::Function;
        memory_size=10::Int64
        ) where T
        throw("Not implemented")
    end
end

mutable struct SingleValueRepeatUntilConvergence{T} <: BaseRepeatUntilConvergence{T}
    update_func::Function
    has_converged::Function
    history::T
    init_vals::T

    function SingleValueRepeatUntilConvergence{T}(
        update_func::Function,
        has_converged::Function;
        memory_size=10::Int64
        ) where T
        throw("Not implemented")
    end
end


"""
Initializes initial values. Performs a while loop with specified conditions.
Returns the last results and R.
# TODO: CircularBuffer is not optimal since convert takes O(n), might
require an ad hoc thing in certain cases
"""
function execute!(R::RepeatUntilConvergence, init_vals)
    R.init_vals = init_vals
    iter_results = R.update_func(R.init_vals)
    push!(R.history, iter_results)
    while !R.has_converged(convert(Vector, R.history))
       iter_results = R.update_func(iter_results)
       push!(R.history, iter_results) 
    end
    return iter_results, R
end
