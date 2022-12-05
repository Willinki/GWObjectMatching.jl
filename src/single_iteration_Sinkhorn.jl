mutable struct iter_Sinkhorn
        const K::Matrix{Float64}, #K
        T::Matrix{Float64},       #T
        const p::Vector{Float64}, #p
        const q::Vector{Float64}, #q
        a::Vector{Float64},       #a
        b::Vector{Float64}        #b
end

function single_iteration_Sinkhorn(elem::iter_Sinkhorn)
    elem.a .= (elem.p)./(((elem.K)')*(elem.b))
    elem.b .= (elem.q)./(((elem.K)')*(elem.a))
    elem.T = diagm(elem.a)*K*diagm(elem.b)
    return elem
end

function stop_crit_T(history::Vector{iter_Sinkhorn}; ϵ=10^{-8}::Float64)
    return norm(history[end].T - history[end-1].T,1) < ϵ
end

function stop_crit_marg(history::Vector{iter_Sinkhorn}; ϵ=10^{-6}::Float64)
    return max(
        norm(history[end].a - history[end-1].p,1) ,
        norm(history[end].b - history[end-1].q,1)  
    ) < ϵ
end 

