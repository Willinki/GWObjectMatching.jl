import DataStructures: CircularBuffer
import Base


"""This object aims at simulating a loop until convergence. In the loop, a
a quantity is updated according to an update function. After updating, we check
convergence with the has_converged function. The history allows to compare 
values at different cylces of the iteration. Initial values can be supplied
directly at execution (execute! function).

Elements
--------
- update_func: Function. Must accept an element of type T as ONLY positional 
    arguments and return the updated element of type T. Any other argument can 
    be passed as keyword.
- has_converged: Function. Takes Vector{T} as input (the history). 
    Outputs true if the values have converged.
- init_vals: T. Initial value for the loop
- history: [specified in subclasses]. Keeps track of previous results 
of the iterations. 
"""
abstract type BaseRepeatUntilConvergence{T} end


"""
See BaseRepeatUntilConvergence.
Implements RepeatUntilConvergence where history is supposed to be a short 
(in the order of ~10) iterable of space-consuming quantities (for example
large matrices). 

In this instance history is a CircularBuffer. It keeps track of only the 
last iterations (specified at construction) efficiently.

Constructor Arguments
---------------------
- update_func: Function. Must accept an element of type T as ONLY positional 
    arguments and return the updated element of type T. Any other argument can 
    be passed as keyword.
- has_converged: Function. Takes Vector{T} as input (the history). 
    Outputs true if the values have converged.
- memory_size: Int64 [default 2]. Maximum length of the history.
"""
mutable struct RepeatUntilConvergence{T} <: BaseRepeatUntilConvergence{T}
    update_func::Function
    has_converged::Function
    history::CircularBuffer{T}
    init_vals::T

    function RepeatUntilConvergence{T}(
        update_func::Function,
        has_converged::Function;
        memory_size=2::Int64
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


"""
See BaseRepeatUntilConvergence.
Implements RepeatUntilConvergence where history is supposed to be a long 
(in the order of ~100) iterable.

In this instance history is a Vector. Keeps track of all the iterations.

Constructor Arguments
---------------------
- update_func: Function. Must accept an element of type T as ONLY positional 
    arguments and return the updated element of type T. Any other argument can 
    be passed as keyword.
- has_converged: Function. Takes Vector{T} as input (the history). 
    Outputs true if the values have converged.
"""
mutable struct LargeMemoryRepeatUntilConvergence{T} <: BaseRepeatUntilConvergence{T}
    update_func::Function
    has_converged::Function
    history::Vector{T}
    init_vals::T

    function LargeMemoryRepeatUntilConvergence{T}(
        update_func::Function,
        has_converged::Function;
        ) where T
        throw("Not implemented")
    end
end


"""
See BaseRepeatUntilConvergence.
Implements RepeatUntilConvergence where history is supposed to be only the result 
of the previous iteration .

In this instance history is of type T. Keeps track of only the previous iterations.

Constructor Arguments
---------------------
- update_func: Function. Must accept an element of type T as ONLY positional 
    arguments and return the updated element of type T. Any other argument can 
    be passed as keyword.
- has_converged: Function. Takes ::T as input (the history). 
    Outputs true if the values have converged.
"""
mutable struct SingleValueRepeatUntilConvergence{T} <: BaseRepeatUntilConvergence{T}
    update_func::Function
    has_converged::Function
    history::T
    init_vals::T

    function SingleValueRepeatUntilConvergence{T}(
        update_func::Function,
        has_converged::Function;
        ) where T
        throw("Not implemented")
    end
end


"""
Executes loop until convergence given a set of initial values.

Arguments
---------
- `R <: BaseRepeatUntilConvergence{T}`. Object that defines the loop.
- `init_vals : T`. Initial values for the iterations.

Returns
-------
`(iter_results, R) : Tuple{T, RepeatUntilConvergence{T}}`

- `iter_results` are the results of the last iteration. 
- `R` is the updated `RepeatUntilConvergence` object.
"""

function execute!(R::BaseRepeatUntilConvergence{T}, init_vals::T) where T 
    R.init_vals = deepcopy(init_vals)
    iter_results = R.update_func(R.init_vals)
    update_history!(R, iter_results)
    while has_converged_wrapper(R)
       iter_results = R.update_func(iter_results)
       update_history!(R, iter_results)
    end
    return iter_results, R
end

"""
Updates the history of a `BaseRepeatUntilConvergence{T}` object with 
the last iteration results.

## Returns nothing
"""
function update_history!(
        R::Union{RepeatUntilConvergence{T}, LargeMemoryRepeatUntilConvergence{T}},
        iter_results::T
    ) where T
    push!(R.history, deepcopy(iter_results))
end

function update_history!(
        R::SingleValueRepeatUntilConvergence{T},
        iter_results::T
    ) where T
    push!(R.history, deepcopy(iter_results))
end

"""
Checks convergence of a `BaseRepeatUntilConvergence{T}` object based 
on values in his `history`.

Returns
-------
- `true` if R.has_converged returns `true`
- `false` otherwise  
"""
function has_converged_wrapper(R::RepeatUntilConvergence{T})::Bool where T
    return !R.has_converged(convert(Vector, R.history))
end

function has_converged_wrapper(
        R::Union{LargeMemoryRepeatUntilConvergence{T}, SingleValueRepeatUntilConvergence{T}}
    )::Bool where T
    return !R.has_converged(R.history)
end