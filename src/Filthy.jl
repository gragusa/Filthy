module Filthy

using StaticArrays
using GrowableArrays
using Parameters

struct KalmanFilterCache{KK, FF, NN, RR}
    K::KK
    F::FF
    N::NN
    R::RR
end

struct KFParms{Zt, Ht, Tt, Rt, Qt}
    Z::Zt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
end

mutable struct KFStates{AT, PT}
    att::AT
    Ptt::PT
end

struct KFInitVal{AT, PT, PI}
    a0::AT
    P0::PT
    PI::PI
    diffuse::BitArray{1}
end

mutable struct KFCache{A, B}
    a::A
    b::B
end

struct CovarianceFilter{P<:KFParms, S<:KFStates, I<:KFInitVal, C<:KFCache, F<:Base.RefValue{Float64}, N<:Base.RefValue{Int64}}
    p::P
    s::S
    i::I
    c::C
    loglik::F
    t::N    
end

function CovarianceFilter(Z::A, H::A, T::A, R::B, Q::A, a0::AZ, P0::PZ) where {A, B, AZ, PZ}
    #checksizes(Z, H, T, R, Q, α0, P0)
    m = numstates(Z)::Int64
    att = GrowableArray(a0)
    Ptt = GrowableArray(P0)
    params = KFParms(Z, H, T, R, Q)
    states = KFStates(att, Ptt)
    inival = KFInitVal(a0, P0, P0, BitArray(m))
    caches = KFCache([],[])
    CovarianceFilter(params, states, inival, caches, Ref(0.0), Ref(1))
end


function Base.filter!(cf::CovarianceFilter, y::Vector{Float64})
    a, P = currentstate(cf)
    p, m, r = size(cf.p)
    @unpack Z, H, T, R, Q = cf.p  
    ## Time Update
    # at = T*a
    # Pt = T*P*T' + R*Q*R'
    ## Measurament update
    v = y .- Z*a
    F = Z*P*Z + H
    invF = inv(F)
    K = T*P*Z'invF
    L = T-K*Z
    att = T*a + K*v
    Ptt = T*P*L' + R*Q*R'
    push!(cf.s.att, att)
    push!(cf.s.Ptt, Ptt)
    s = 0.5*logdet(F)+v'*invF*v
    cf.loglik[] += - p/2*log(2π) - s[1]
    cf.t[] += 1
end

function Base.filter!(cf::CovarianceFilter{KFP} where KFP<:KFParms{A}, y::Float64) where A<:Float64
    a, P = currentstate(cf)
    p, m, r = size(cf.p)
    @unpack Z, H, T, R, Q = cf.p  
    ## Time Update
    # at = T*a
    # Pt = T*P*T' + R*Q*R'
    ## Measurament update
    v = y - Z*a
    F = Z*P*Z + H    
    K = T*P*Z/F
    L = T-K*Z
    att = T*a + K*v
    Ptt = T*P*L + R*Q*R
    push!(cf.s.att, att)
    push!(cf.s.Ptt, Ptt)
    cf.loglik[] += - p/2*log(2π) - 0.5*abs(F)+v*v/F
    cf.t[] += 1
end






function simulate(cf::CovarianceFilter, nout) 
    @unpack Z, H, T, R, Q = cf.p  

    p, m = size(Z) 

    Y = Array{Float64, 2}(p, nout+1)
    α = Array{Float64, 2}(m, nout+2)

    fH = chol(H)'
    fQ = chol(Q)'
    α[:,1] = T*cf.i.a0 + R*fQ*randn(m)
    for j in 1:nout+1
        Y[:, j] = Z*α[:,j] + fH*randn(p)
        α[:, j+1] = T*α[:,j] + R*fQ*randn(m)
    end
    (Y[:,1:end-1], α[:,2:end-1])
end


# function cols2tuple(A::Array{T,2}) where {T <: Float64}
#     return tuple([A[:,c] for c in 1:size(A,2)]...)
# end

include("methods.jl")

export CovarianceFilter, simulate


end # module
