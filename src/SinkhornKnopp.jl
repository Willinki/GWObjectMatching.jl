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
        a = ones(length(p))
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

compute_marginals(Ts::Matrix{Float64}) = (sum(Ts, dims=2)|> vec, sum(Ts, dims=1)|> vec)

"""
First proposal for stopping criterion, stops whenever transport is stable
"""
function stop_SK_T(history::Vector{data_SK}; tol::Float64)::Bool
    length(history) == 1 && return false
    return norm(history[end].T - history[end-1].T, 1)::Float64 < tol
end

"""
Old proposal
"""
function stop_SK_ab_old(history::Vector{data_SK}; tol::Float64)::Bool
    μ,ν = compute_marginals(history[end].T)
    bigger_norm = max(
        norm(μ - history[end].p,1) ,
        norm(ν - history[end].q,1)  
    )
    return bigger_norm < tol
end

"""
Second proposal for stopping criterion, stops whenever a and b are close
enough to p and q.
"""
function stop_SK_ab_new(history::Vector{data_SK}; tol::Float64)::Bool
    elem = history[end]
    first_check = (elem.a).*(elem.K*elem.b)-elem.p
    second_check = (elem.b).*(((elem.K)')*elem.a)-elem.q
    δ = max(norm(first_check,1), norm(second_check,1))
    return δ<tol
end

function stop_SK(history::Vector{data_SK}; tol::Float64)::Bool
    boolvar = (stop_SK_T(history; tol) && stop_SK_ab_new(history; tol))
    return boolvar
end
