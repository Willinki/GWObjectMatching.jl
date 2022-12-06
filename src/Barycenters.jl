# TODO: sistemare metric measure space per avere anche le due
# TODO: specificare tipi
# componenti, quindi una discrete probability
import ObjectMatching as OM
using PartialFunctions

function update_barycenters(
        Cs_collection::Vector{OM.MetricMeasureSpace},
        λs_collection::OM.ConvexSum,
        p::OM.DiscreteProbability;
        loss=OM.loss("L2")::OM.loss,
        ϵ=1e-2::Float64,
        tol=1e-8::Float64
    )::OM.MetricMeasureSpace

    length(Cs_collection)==length(λs_collection) || throw(ArgumentError(
        "λs and Cs collections have different lengths"
    ))

    # initialize C uniform
    C = OM.MetricMeasureSpace(ones(Float64, length(p.mu), length(p.mu)), p)
    # obtain optimal transport
    update_transport_updated = update_transport $ (Cp=C, loss=loss, ϵ=ϵ, tol=tol)
    Ts_collection = map(update_transport_updated, Cs_collection)
    # compute C barycenter
    C = compute_C(Ts_collection, λs_collection, p)
end

function update_transport(
        Cs::OM.MetricMeasureSpace;
        Cp::OM.MetricMeasureSpace,
        loss::OM.loss,
        ϵ::Float64,
        tol::Float64,
    )
    Np, Ns = (length(Cp.μ), length(Cs.μ))
    T = ones(Np, Ns)./(Np*Ns)
    K = OM.GW_cost(loss, Cp, Cs, T, ϵ)
    # define stop sk tolerance
    SK_initial_point = OM.data_SK(K, Cp.μ, Cs.μ, T)
    SK_repeater = OM.RepeatUntilConvergence{OM.data_SK}(OM.update_SK, OM.stop_SK_T)
    _, T = execute!(SK_repeater, SK_initial_point)
    return T
end

function compute_C()
    throw("Not implemented")
end
