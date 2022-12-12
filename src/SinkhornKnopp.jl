import LinearAlgebra: diagm, norm
import ObjectMatching: RepeatUntilConvergence, execute!

"""
Data structure for sinkhorn-knopp (SK) iteration.
Has to be used inside RepeatUntilConvergence.
# TODO: fill documentation
# TODO: define proper constructor for this
"""
struct data_SK
    K::Matrix{Float64} 
    p::Vector{Float64} 
    q::Vector{Float64} 
    T::Matrix{Float64}       
    a::Vector{Float64}       
    b::Vector{Float64}       

    function data_SK(
            K::Matrix{Float64},
            p::Vector{Float64}, 
            q::Vector{Float64},
            T::Matrix{Float64}
        )

        (size(T,1) == size(K,1) && size(T,2) == size(K,2)) || throw(ArgumentError(
            "The size of T and K doesn't match."
        ))

        size(T,1) == length(p) || throw(ArgumentError(
            "The number of rows of T must be the same of the length of p."
        ))

        size(T,2) == length(q) || throw(ArgumentError(
            "The number of columns of T must be the same of the length of q."
        ))
        
        a = 1/(length(p))*ones(length(p))
        b = q./((K')*a)
        return new(K, p, q, T, a, b)
        
    end
end

"""
Update for single iteration
"""
function update_SK(elem::data_SK)::data_SK
    elem.a .= (elem.p)./((elem.K)*(elem.b))
    elem.b .= (elem.q)./(((elem.K)')*(elem.a))
    elem.T .= diagm(elem.a)*elem.K*diagm(elem.b)
    return elem
end

"""
First proposal for stopping criterion, stops whenever transport is stable
"""
function stop_SK_T(history::Vector{data_SK}; ϵ=1e-8::Float64)::Bool
    length(history) == 1 && return false
    a = norm(history[end].T - history[end-1].T, 1)
    return norm(history[end].T - history[end-1].T, 1)::Float64 < ϵ
end

"""
Second proposal for stopping criterion, stops whenever a and b are close
enough to p and q.
"""
function stop_SK_ab(history::Vector{data_SK}; ϵ=10^(-6)::Float64)::Bool
    return max(
        norm(history[end].a - history[end].p,1) ,
        norm(history[end].b - history[end].q,1)  
    )::Float64 < ϵ
end
