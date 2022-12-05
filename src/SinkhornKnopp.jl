import LinearAlgebra: diagm, norm

"""
Data structure for sinkhorn-knopp (SK) iteration.
Has to be used inside RepeatUntilConvergence.
# TODO: fill documentation
"""
struct data_SK
    K::Matrix{Float64} 
    p::Vector{Float64} 
    q::Vector{Float64} 
    T::Matrix{Float64}       
    a::Vector{Float64}       
    b::Vector{Float64}       
end

"""
Update for single iteration
"""
function update_SK(elem::data_SK)::data_SK
    elem.a .= (elem.p)./(((elem.K)')*(elem.b))
    elem.b .= (elem.q)./(((elem.K)')*(elem.a))
    elem.T .= diagm(elem.a)*elem.K*diagm(elem.b)
    return elem
end

"""
First proposal for stopping criterion, stops whenever transport is stable
"""
function stop_SK_T(history::Vector{data_SK}; ϵ=10^{-8}::Float64)::Bool
    length(history) == 1 && return false
    return norm(history[end].T - history[end-1].T,1) < ϵ
end

"""
Second proposal for stopping criterion, stops whenever a and b are close
enough to p and q.
"""
function stop_SK_ab(history::Vector{data_SK}; ϵ=10^{-6}::Float64)::Bool
    return max(
        norm(history[end].a - history[end].p,1) ,
        norm(history[end].b - history[end].q,1)  
    ) < ϵ
end