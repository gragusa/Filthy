@noinline function currentstate(cf::LinearStateSpace, ::Type{Val{false}}) 
    idx = cf.t[]::Int
    (cf.f.a[idx], cf.f.P[idx])
end

@noinline function currentstate(cf::LinearStateSpace, ::Type{Val{true}}) 
    idx = cf.t[]::Int
    (cf.f.a[idx], cf.f.P[idx], cf.f.Pinf[idx], cf.f.Pstar[idx])
end


# function updatelik!(ss, v, F, F⁻¹, ispd) 
#     p, m, r = size(ss)
#     s = ispd ? 0.5*(logdet(F)) : s = 0.5*(logdet(F) + v'F⁻¹*v)
#     ss.loglik[1] += -p/2*log(2π) - first(s)
# end


function updatelik!(ss, v, F, F⁻¹, Fflag) 
    p, m, r = size(ss)
    s = Fflag == 1 ? 0.5*(logdet(F)) : 0.5*(logdet(F) + v'F⁻¹*v)
    ss.loglik[1] += -p/2*log(2π) - first(s)
end

function updatelik!(ss, v, F, F⁻¹)     
    p, m, r = size(ss)
    s = 0.5*(logdet(F) + v'F⁻¹*v)
    ss.loglik[1] += -p/2*log(2π) - first(s)
end

function loglik(v, F, F⁻¹)
     - 0.5*(logdet(F) + v'F⁻¹*v)
end

function loglik(v, F, F⁻¹, flag)
    s = flag == 1 ? 0.5*(logdet(F)) : 0.5*(logdet(F) + v'F*v)
    - first(s)
end

isdiffuse(Pinf::StaticMatrix) = any(Pinf.data.!=0)
isdiffuse(Pinf::AbstractFloat) = Pinf>0
isdiffuse(Pinf::AbstractMatrix) = any(Pinf.!=0)

numstates(a::AbstractArray) = size(a,1)
numstates(a::AbstractFloat) = 1

nummeasur(a::AbstractArray) = size(a,1)
nummeasur(a::AbstractFloat) = 1

Base.size(cf::LinearStateSpace) = size(cf.p)

Base.size(p::KFParms{P}) where P<:AbstractFloat = (1,1,1)
Base.size(p::KFParms{P}) where P<:AbstractArray = (size(p.Z)..., size(p.Q, 1))

testposdef(x::AbstractArray) = isposdef(x)
testposdef(x::StaticMatrix) = all(map(u-> u[1]>0, eig(x)))

function safeinverse(x)
    flag = any(abs.(x) .> 1e-07) ? 1 : 0
    if flag > 0
        try
            (flag, inv(x))
        catch mesg
            throw(DomainError)
        end        
    else
        (flag, x)
    end        
end

toarray(b) = b
toarray(b::Real) = [b]


function checkfilterdataconsistency(ss::LinearStateSpace, y::Vector)
    T = length(y)
    p, m, r = size(ss.p)
    @assert p==1 "Inconsistent dimension. Y must be a (Tx$p) Matrix{T}."
    (T, p, m, r, ss.t[])
end

function checkfilterdataconsistency(ss::LinearStateSpace, Y::Matrix)
    q, T = size(Y)
    p, m, r = size(ss.p)
    msg = "a (Tx$p) Matrix"
    @assert p==q "Inconsistent dimension. Y must be "*msg
    (T, p, m, r, ss.t[])
end

function resizecontainer!(ss::LinearStateSpace, n::Int64)
    resize!(ss.f.a.data, n)
    resize!(ss.f.P.data, n)
    resize!(ss.c.F.data, n)
    resize!(ss.c.F⁻¹.data, n)
    resize!(ss.c.Y.data, n)
end


function Masked(a::NTuple)
    rnan = map(y->find(x -> isnan.(x), y), a)
    nnan = map(y->find(x -> !isnan.(x), y), a)
    freeitr = map(x -> enumerate(x), rnan)
    fixeditr = map(x -> enumerate(x), nnan)       
    Masked(rnan, nnan, freeitr, fixeditr, a, [1])
end

prototypical(m::Masked) = zeros(sum(map(i->length(i), m.freeitr)))

function OptimSSM(m::Masked, Z, CHH, T, R, CQQ, a1, P1, P1inf, Pstar, y)
    thetatmp = prototypical(m)
    jcfg = ForwardDiff.GradientConfig(nothing, thetatmp, ForwardDiff.Chunk(thetatmp))
    hcfg = ForwardDiff.HessianConfig(nothing, thetatmp, ForwardDiff.Chunk(thetatmp))
    OptimSSM(m, jcfg, hcfg, Z, CHH, R, T, CQQ, 
             convert(Vector,a1), 
             convert(Matrix, P1), 
             convert(Matrix, P1inf), 
             convert(Matrix, Pstar), y)
end


allfixed(x::Masked) = maximum(map(i->length(i), x.freeitr))==0 ? true : false

function remask!(m::Masked, b::NTuple, theta)
    count = m.count
    map(b, m.freeitr) do y,x        
        for (i,j) in x
            y[j] = theta[count[1]]; 
            count[1] += 1            
        end
    end    
    count[1] = 1
    map(b, m.fixeditr, m.parent) do y, x, z
        for (i,j) in x            
            @inbounds y[j] = z[j]
        end
    end
    nothing
end
    

function islowertriangular(x::AbstractMatrix{T}) where T
    r, c = size(x)
    out = true
    for j in 2:c
        for i in 1:j-1
            out = x[i,j] == zero(T)
            if !out
                break
            end
        end
    end
    out
end

@noinline function setupdiffusematrices(a1::T, P1::M, P1inf::M) where {T<:AbstractVector, M<:AbstractMatrix}
    idx = find(diag(P1inf) .> 0)
    aa = copy(convert(Vector, a1))
    PP = copy(convert(Matrix, P1))
    RR = eye(size(P1, 1))
    map(idx) do i
        PP[i,:] = 0.0
        PP[:,i] = 0.0
        RR[:,i] = 0.0
        RR[i,:] = 0.0
        aa[i] = 0.0
    end
    a1′ = convert(typeof(a1), aa)
    P1′ = convert(typeof(P1), PP)
    Pstar = convert(typeof(P1), RR*RR')
    (a1′::T, P1′::M, Pstar::M)
end



    