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

#TODO: write it better
function compute_marginals(elem::data_SK)
    T = elem.T
    i,j = size(T)
    p = zeros(Float64, i)
    q = zeros(Float64, j)
    for k = 1:i
        p[k] = sum(T[k,:])
    end
    for k = 1:j
        q[k] = sum(T[:,k])
    end
    return (p,q)
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
    m = max(
        norm(history[end].a - history[end].p,1) ,
        norm(history[end].b - history[end].q,1)  
    )
    a = (m<ϵ) 
    return a
end

#TODO: make possible to use it as field "has_converged" in RepeatUntilConvergence
# i.e. understand why it doesn't return a Bool type

function stop_SK(history::Vector{data_SK}; ϵ=10^(-2)::Float64)::Bool
    a = (stop_SK_T(history,ϵ) && stop_SK_ab(history,ϵ))
    return a
end
