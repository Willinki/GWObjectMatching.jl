import ObjectMatching as OM
using PartialFunctions
using LinearAlgebra

function GW_barycenters(
        Cs_collection::Vector{OM.MetricMeasureSpace}, 
        λs_collection::OM.ConvexSum = OM.ConvexSum(length(Cs_collection))
        ;
        n_points::Int64    = 100,
        p::Vector{Float64} = fill(1/n_points,n_points),
        loss::OM.Loss      = OM.Loss("L2"),
        ϵ::Float64         = 0.05,
        Cp_niter::Int64    = 20,
        Ts_tol::Float64    = 1e-6,
        SK_tol::Float64    = 1e-8,
    )::OM.MetricMeasureSpace

    # maybe include args check into a function
    length(p)==n_points || throw(ArgumentError("n must be equal to the length of p.")) 
    length(Cs_collection)==length(λs_collection.v) || throw(ArgumentError(
        "λs and Cs collections have different lengths"
    ))

    Cp_initial = init_C(p)
    # we set all kwargs inside update and stopping condition
    update_barycenters_repeater = OM.RepeatUntilConvergence{OM.MetricMeasureSpace}(
        update_barycenters $ (
            Cs_collection=Cs_collection,
            λs_collection=λs_collection, 
            loss=loss,
            ϵ=ϵ, 
            Ts_tol=Ts_tol,
            SK_tol=SK_tol
        ), 
        stop_barycenter_niter $ (niter=Cp_niter,); 
        memory_size=Cp_niter
    )
    Cp_final , _ = execute!(update_barycenters_repeater, Cp_initial)
    return Cp_final
end

function init_C(p::Vector{Float64})::OM.MetricMeasureSpace
    #WLOG we can assume that p.>0, if it is not we discard its zero elements
    p = filter!(e->e!=0, p) 
    # initialize C uniform
    return OM.MetricMeasureSpace(rand(Float64, length(p), length(p)), p)
end

function stop_barycenter_niter(history::Vector{OM.MetricMeasureSpace}; niter::Int64)
    L = length(history)
    if L == 1
        if niter==1
            return true
        end
        return false
    end
    diff_mat = abs.(history[end].C - history[end-1].C)
    ratio = maximum(diff_mat./history[end].C)
    println("ITERATION: $(L)/$(niter) - $(ratio)")
    return L>=niter
end

function update_barycenters(
        Cp::OM.MetricMeasureSpace
        ;
        Cs_collection::Vector{OM.MetricMeasureSpace}, 
        λs_collection::OM.ConvexSum,
        loss::OM.Loss,      
        ϵ::Float64,         
        Ts_tol::Float64,
        SK_tol::Float64,
    )::OM.MetricMeasureSpace
    #for every Cs we compute the transport to Cp, save it to Ts_collections
    Ts_collection = [
        begin 
            update_transport_repeater = OM.RepeatUntilConvergence{Matrix{Float64}}(
                update_transport $ (Cs=Cs, Cp=Cp, loss=loss, ϵ=ϵ, SK_tol=SK_tol), 
                stop_transport $ (ratio_thresh=Ts_tol,); 
                memory_size=2
            );
            execute!(update_transport_repeater, init_Ts(Cp, Cs))[1]
        end
        for Cs in Cs_collection
    ] 
    # with the new Ts we update the barycenter
    return compute_C(
        λs_collection,
        map(x->convert(Matrix{Float64},(x)'), Ts_collection), # also tested with deepcopy 
        Cs_collection,
        Cp.μ,
        loss
    )
end

function init_Ts(Cp::OM.MetricMeasureSpace, Cs::OM.MetricMeasureSpace)::Matrix{Float64}
    # we simply start with an admissimble transport
    return (Cp.μ)*(Cs.μ)'
end

function update_transport(
        Ts::Matrix{Float64}
        ;
        Cs::OM.MetricMeasureSpace,
        Cp::OM.MetricMeasureSpace,
        loss::OM.Loss,
        ϵ::Float64,    
        SK_tol::Float64, 
    )::Matrix{Float64}
    K = OM.GW_Cost(loss, Cp, Cs, Ts, ϵ)
    SK_initial_point = OM.data_SK(K, Cp.μ, Cs.μ, Ts)
    SK_repeater = OM.RepeatUntilConvergence{OM.data_SK}(
        OM.update_SK, 
        OM.stop_SK $ (tol=SK_tol,);
        memory_size=2
    )
    final_SK , _ = execute!(SK_repeater, SK_initial_point)
    return final_SK.T
end

function stop_transport(
        history::Vector{Matrix{Float64}};
        ratio_thresh::Float64 = 1e-4
    )::Bool
    if length(history) == 1
        return false
    end
    diff_mat = abs.(history[end] - history[end-1])
    ratio = maximum(diff_mat./history[end])
    println(ratio)
    return ratio < ratio_thresh
end

function compute_C(
        λs_collection::OM.ConvexSum, 
        Ts_collection::Vector{Matrix{Float64}}, 
        Cs_collection::Vector{OM.MetricMeasureSpace},
        p::Vector{Float64},
        loss::OM.Loss
    )::OM.MetricMeasureSpace
    if loss.string == "L2"
        Ms_collection = [
            λs * Ts' * Cs.C * Ts
            for (λs, Ts, Cs) in zip(λs_collection.v, Ts_collection, Cs_collection)
        ]
        return OM.MetricMeasureSpace(sum(Ms_collection)./(p*p'), p)
    else #if loss.string == "KL"
        Ms_collection = [
            λs.v * Ts' * log.(Cs.C) * Ts
            for (λs, Ts, Cs) in zip(λs_collection.v, Ts_collection, Cs_collection)
        ]
        return OM.MetricMeasureSpace(exp.(sum(Ms_collection)./(p*p')),p)
    end
end
