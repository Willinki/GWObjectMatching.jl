import ObjectMatching as OM
using PartialFunctions
using LinearAlgebra

function GW_barycenters(
        n_points::Int64,
        Cs_collection::Vector{OM.MetricMeasureSpace}, 
        λs_collection = OM.ConvexSum(length(Cs_collection))::OM.ConvexSum,
        p             = fill(1/n_points,n_points)::Vector{Float64},
        loss          = OM.loss("L2")::OM.loss,
        ϵ             = 1e-2::Float64,
        tol           = 1e-8::Float64,
        niter         = 20::Int64
    )::OM.MetricMeasureSpace

    length(p)==n_points || throw(ArgumentError("n must be equal to the length of p.")) 
    Cp_initial = initialize_C(p)
    function stop_bar_niter(history::Vector{OM.MetricMeasureSpace})
        return length(history)>=niter
    end

    update_barycenters_updated = update_barycenters $ (
        Cs_collection=Cs_collection,
        λs_collection=λs_collection, 
        loss=loss,
        ϵ=ϵ, tol=tol
    )
    update_barycenters_repeater = OM.RepeatUntilConvergence{OM.MetricMeasureSpace}(
        update_barycenters_updated, stop_bar_niter; memory_size=niter
    )
    Cp_final , _ = execute!(update_barycenters_repeater, Cp_initial)
    return Cp_final
end


#function stop_bar_update(history::Vector{OM.MetricMeasureSpace},ϵ=1e-8::Float64)
#    length(history) == 1 && return false
#    return norm(history[end].C - history[end-1].C, 1)::Float64 < ϵ
#end

function initialize_C(p::Vector{Float64})
    #WLOG we can assume that p.>0, if it is not we discard its zero elements
    p = filter!(e->e!=0, p) 
    # initialize C uniform
    C = rand(Float64, length(p), length(p))
    C[diagind(C)] .= 0 
    return OM.MetricMeasureSpace(C, p)
end

function update_barycenters(
        Cp::OM.MetricMeasureSpace;
        Cs_collection::Vector{OM.MetricMeasureSpace}, 
        λs_collection::OM.ConvexSum,
        loss::OM.loss,      #loss("L2")
        ϵ::Float64,         #1e-2
        tol::Float64        #1e-8
    )::OM.MetricMeasureSpace

    length(Cs_collection)==length(λs_collection.v) || throw(ArgumentError(
        "λs and Cs collections have different lengths"
    ))
    # obtain optimal transport
    update_transport_updated = update_transport $ (Cp=Cp, loss=loss, ϵ=ϵ, tol=tol)
    Ts_collection = map(update_transport_updated, Cs_collection)
    # compute C barycenter
    return compute_C(
        λs_collection,
        map(x->convert(Matrix{Float64},(x.T)'), Ts_collection), 
        Cs_collection,
        Cp.μ,
        loss
    )
end

function update_transport(
        Cs::OM.MetricMeasureSpace;
        Cp::OM.MetricMeasureSpace,
        loss::OM.loss,
        ϵ::Float64,    #stop_SK_T
        tol::Float64,  #stop_SK_ab
    )
    T = (Cp.μ)*(Cs.μ)' 
    K = OM.GW_Cost(loss, Cp, Cs, T, ϵ)
    # define stop sk tolerance
    SK_initial_point = OM.data_SK(K, Cp.μ, Cs.μ, T)
    SK_repeater = OM.RepeatUntilConvergence{OM.data_SK}(OM.update_SK, OM.stop_SK)
    T ,_ = execute!(SK_repeater, SK_initial_point)
    return T
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
