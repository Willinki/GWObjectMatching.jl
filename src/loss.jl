struct loss
    """
    Given a real valued loss function L(a,b) of the form f1(a)+f2(b)-h1(a)*h2(b), we define the inner constructor that takes as input a strings
    which describes the loss function: the possible strings input are:
        - "L2" for the squared loss L(a,b) = |a-b|^2
        - "KL" for the Kullback-Leibler loss L(a,b) = a*log(a/b)-a+b
    """
    f1::Function
    f2::Function
    h1::Function
    h2::Function

    function loss(s::String)
        if s == "L2"
            f1 = x->x^2
            f2 = x->x^2
            h1 = x->x
            h2 = x->2*x
        elseif s == "KL"
            f1 = x->x*log(x)
            f2 = x->x
            h1 = x->x
            h2 = x->log(x)
        else
            throw(ArgumentError("Not valid input: it doesn't describe any known loss function."))
        end
        new(f1,f2,h1,h2)
    end #innerconstructor
  
end #struct


function GW_Cost(L::loss, M::MetricMeasureSpace, N::MetricMeasureSpace, T::Matrix{Float64}, ϵ::Float{64})
    """
    This function evaluate the tensor product mathcal{L}(M.C,N.C) tensor T, using the decomposition of a loss function in f1,f2,h1,h2.
    About T: all the operation makes sense if T is any matrix of the correct size, but to make sense T must be a transport plan between
    M.μ and N.μ.
    """
    if (size(M.C,1) != size(T,1)) || (size(N.C,1) != size(T,2))
        throw(ArgumentError("Wrong input size, it must be possible to compute the product matrix (M.C)*T*(N.C)."))
    end

    E = zeros(size(T,1),size(T,2))

    E = (M.C)*T*(N.C)

    E = ((L.f1).(M.C))*((M.μ)*(ones(size(T,2)))') + ones(size(T,1))*((N.μ)'*((L.f2).(N.C))') - ((L.h1).(M.C))*T*((L.h2).(N.C))' 

    E .= exp.(-E./ϵ)

    return E
end #function
        
