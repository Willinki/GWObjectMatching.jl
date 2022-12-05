import LinearAlgebra: diagm, norm

"""
Data structure for sinkhorn iteration.
Has to be used inside RepeatUntilConvergence.
# TODO: fill ddocumentation
"""
struct iter_Sinkhorn
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
function single_iteration_Sinkhorn(elem::iter_Sinkhorn)
    elem.a .= (elem.p)./(((elem.K)')*(elem.b))
    elem.b .= (elem.q)./(((elem.K)')*(elem.a))
    elem.T .= diagm(elem.a)*elem.K*diagm(elem.b)
    return elem
end

function stop_crit_T(history::Vector{iter_Sinkhorn}; ϵ=10^{-8}::Float64)
    length(history) == 1 && return false
    return norm(history[end].T - history[end-1].T,1) < ϵ
end

function stop_crit_marg(history::Vector{iter_Sinkhorn}; ϵ=10^{-6}::Float64)
    return max(
        norm(history[end].a - history[end].p,1) ,
        norm(history[end].b - history[end].q,1)  
    ) < ϵ
end 

