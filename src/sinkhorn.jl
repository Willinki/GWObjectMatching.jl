function sinkhorn(K::Matrix{Float64}, p::Vector{Float64}, q::Vector{Float64}, eps::Float64)
    #check of the size of the inputs
    rows = size(K)[1];
    if (rows != size(K)[2] || rows != length(p) || rows!= length(q))
        throw(ArgumentError{"Wrong size of the inputs."})
    end

    #initialize the first vector to be the uniform distribution
    a = (1/rows)*ones(rows)
    b = q./((K')*a)
    onemat = ones(rows,rows)
    onevec = ones(rows)
    T = diagm(a)*K*diagm(b)
    x = 1/(rows)*ones(rows,rows)
    i = 0
    #sinkhorn iterations, until convergence
    while (norm(T-x)>eps)
        T = x
        #x = diagm(onevec./p)*K*(q.*(onemat./(K'*(onemat./x))))
        a = p./((K')*b)
        b = q./((K')*a)
        i+=1
        x = diagm(a)*K*diagm(b)
    end
    println(i)
    #x = onemat./x
    #x = q.*(onemat./(K'*x))
    return x
end