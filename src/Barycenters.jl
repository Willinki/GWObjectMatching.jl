import ObjectMatching as OM
using PartialFunctions
using LinearAlgebra

function GW_barycenters(
        n_points::Int64,
        Cs_collection::Vector{OM.MetricMeasureSpace}, 
        λs_collection::OM.ConvexSum = OM.ConvexSum(length(Cs_collection)),
        p::Vector{Float64}          = fill(1/n_points,n_points),
        loss::OM.loss               = OM.loss("L2"),
        ϵ::Float64                  = 0.05,
        tol::Float64                = 1e-8,
        niter::Int64                = 15
    )::OM.MetricMeasureSpace

    length(p)==n_points || throw(ArgumentError("n must be equal to the length of p.")) 
    length(Cs_collection)==length(λs_collection.v) || throw(ArgumentError(
        "λs and Cs collections have different lengths"
    ))

    Cp_initial = init_C(p)
    update_barycenters_updated = update_barycenters $ (
        Cs_collection=Cs_collection,
        λs_collection=λs_collection, 
        loss=loss,
        ϵ=ϵ, tol=tol
    )
    update_barycenters_repeater = OM.RepeatUntilConvergence{OM.MetricMeasureSpace}(
        update_barycenters_updated, stop_barycenter_niter; memory_size=niter
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

function stop_barycenter_niter(history::Vector{OM.MetricMeasureSpace}, niter=20)
    L = length(history)
    if L == 1
        return false
    end
    diff_mat = abs.(history[end].C - history[end-1].C)
    ratio = maximum(diff_mat./history[end].C)
    print("ITERATION: $(L)/$(niter) - $(ratio)\r")
    flush(stdout)
    return L>=niter
end

function update_barycenters(
        Cp::OM.MetricMeasureSpace;
        Cs_collection::Vector{OM.MetricMeasureSpace}, 
        λs_collection::OM.ConvexSum,
        loss::OM.loss,      #loss("L2")
        ϵ::Float64,         #1e-2
        tol::Float64        #1e-8
    )::OM.MetricMeasureSpace
    #for every Cs we compute the transport to Cp, save it to Ts_collections
    Ts_collection = Vector{Matrix{Float64}}(undef, length(Cs_collection))  
    for (s, Cs) in enumerate(Cs_collection)
        Ts::Matrix{Float64} = init_Ts(Cp, Cs)
        update_transport_set = update_transport $ (
            Cs=Cs,
            Cp=Cp,
            loss=loss,
            ϵ=ϵ, tol=tol
        )
        update_transport_repeater = OM.RepeatUntilConvergence{Matrix{Float64}}(
            update_transport_set, stop_transport; memory_size=2
        )
        Ts_final, _ = execute!(update_transport_repeater, Ts)
        Ts_collection[s] = Ts_final #checked, this does not cause any bug
    end
    # given the updated barycenter
    return compute_C(
        λs_collection,
        map(x->convert(Matrix{Float64},(x)'), Ts_collection), 
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
        Ts::Matrix{Float64};
        Cs::OM.MetricMeasureSpace,
        Cp::OM.MetricMeasureSpace,
        loss::OM.loss,
        ϵ::Float64,    #stop_SK_T
        tol::Float64,  #stop_SK_ab
    )::Matrix{Float64}
    K = OM.GW_Cost(loss, Cp, Cs, Ts, ϵ)
    # define stop sk tolerance
    SK_initial_point = OM.data_SK(K, Cp.μ, Cs.μ, Ts)
    SK_repeater = OM.RepeatUntilConvergence{OM.data_SK}(OM.update_SK, OM.stop_SK)
    final_SK , _ = execute!(SK_repeater, SK_initial_point)
    return final_SK.T
end

function stop_transport(
        history::Vector{Matrix{Float64}},
        ratio_threshold::Float64 = 1e-4
    )::Bool
    if length(history) == 1
        return false
    end
    diff_mat = abs.(history[end] - history[end-1])
    ratio = maximum(diff_mat./history[end])
    println(ratio)
    return ratio < ratio_threshold
end

function compute_C(
        λs_collection::OM.ConvexSum, 
        Ts_collection::Vector{Matrix{Float64}}, 
        Cs_collection::Vector{OM.MetricMeasureSpace},
        p::Vector{Float64},
        loss::OM.loss
    )::OM.MetricMeasureSpace
    S = length(λs_collection.v)
    Ms_collection = fill(zeros(length(p),length(p)), S)
    # TODO: improve syntax
    if loss.string == "L2"
        for i = 1:S
            Ms_collection[i] = λs_collection.v[i]*(
                (Ts_collection[i]')*((Cs_collection[i].C)*(Ts_collection[i]))
                )         
        end
        return OM.MetricMeasureSpace((sum(Ms_collection)./(p*p')),p)
    else #if loss.string == "KL"
        for i = 1:S
            j = size((Cs_collection[i].C),1)
            (((Cs_collection[i].C).>=0) == ones(j,j)) || throw(ArgumentError(
                "If the loss is the KL-loss, all the Cs' must be non-negative."
            ))
            Ms_collection[i] = λs_collection.v[i]*(
                (Ts_collection[i]')*(log.(Cs_collection[i].C)*(Ts_collection[i]))
                ) 
        end
        return OM.MetricMeasureSpace(exp.(sum(Ms_collection)./(p*p')),p)
    end
end
