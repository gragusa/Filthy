module Filthy

using StaticArrays
using GrowableArrays
using Parameters

struct KFParms{Zt, Ht, Tt, Rt, Qt}
    Z::Zt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
end

mutable struct KFFiltered{AT, PT}
    a::AT
    P::PT
end

mutable struct KFSmoothed{AT, PT}
    a::AT
    V::PT
    r::AT
    N::PT
end

struct KFInitVal{AT, PT, PI}
    a0::AT
    P0::PT
    PI::PI
    diffuse::BitArray{1}
end

struct KFCache{FF, DD}    
    F::FF  ## Z*F⁻¹ (m x m)
    Y::DD  ## Data 
end

struct CovarianceFilter{P<:KFParms, F<:KFFiltered, S<:KFSmoothed,  I<:KFInitVal, C<:KFCache, L<:Base.RefValue{Float64}, N<:Base.RefValue{Int64}}
    p::P
    f::F
    s::S
    i::I
    c::C
    loglik::L
    t::N    
end

struct SequentialFilter{P<:KFParms, F<:KFFiltered, S<:KFSmoothed,  I<:KFInitVal, C<:KFCache, L<:Base.RefValue{Float64}, N<:Base.RefValue{Int64}}
    p::P
    f::F
    s::S
    i::I
    c::C
    loglik::L
    t::N    
end

function CovarianceFilter(Z::A, H::A, T::A, R::B, Q::A, a0::AZ, P0::PZ) where {A, B, AZ, PZ}
    #checksizes(Z, H, T, R, Q, α0, P0)
    m = numstates(T)::Int64
    p = nummeasur(Z)::Int64
    att = GrowableArray(a0)
    Ptt = GrowableArray(P0)
    params = KFParms(Z, H, T, R, Q)
    filter = KFFiltered(att, Ptt)
    smooth = KFSmoothed(Array{Float64}(1,1), Array{Float64}(1,1,1), Array{Float64}(1,1), Array{Float64}(1,1,1))
    inival = KFInitVal(a0, P0, P0, BitArray(m))
    caches = KFCache(GrowableArray(zeros(T)), GrowableArray(zeros(p)))
    CovarianceFilter(params, filter, smooth, inival, caches, Ref(0.0), Ref(1))
end

function SequentialFilter(Z::A, H::A, T::A, R::B, Q::A, a0::AZ, P0::PZ) where {A, B, AZ, PZ}
    #m, p, q = checksizes(Z, H, T, R, Q, α0, P0)  
    m = numstates(T)::Int64
    p = nummeasur(Z)::Int64
    a = GrowableArray(a0)
    P = GrowableArray(P0)
    params = KFParms(Z, H, T, R, Q)
    filter = KFFiltered(a, P)
    smooth = KFSmoothed(Array{Float64}(1,1), Array{Float64}(1,1,1), Array{Float64}(1,1), Array{Float64}(1,1,1))
    inival = KFInitVal(a0, P0, P0, BitArray(m))
    caches = KFCache(GrowableArray(zeros(T)), GrowableArray(zeros(p)))
    CovarianceFilter(params, filter, smooth, inival, caches, Ref(0.0), Ref(1))
end

"""
filter!(cf::CovarianceFilter, y::Vector{Float64})
On-line covariance filter. 
"""
function Base.filter!(cf::CovarianceFilter, y::Vector{Float64})    
    p, m, r = size(cf.p)
    a, P = currentstate(cf)
    @unpack Z, H, T, R, Q = cf.p  
    att, Ptt, v, F, F⁻¹ = filterstep(a, P, Z, H, T, R, Q, y)
    push!(cf.f.a, att)
    push!(cf.f.P, Ptt)
    push!(cf.c.F, F⁻¹)
    push!(cf.c.Y, y)
    s = 0.5*logdet(F)+v'*F⁻¹*v
    cf.loglik[] += - p/2*log(2π) - s[1]
    cf.t[] += 1
end

function Base.filter!(cf::CovarianceFilter{KFP} where KFP<:KFParms{A}, Y::Matrix{Float64}) where A<:AbstractArray
    TT, PP = size(Y)
    p, m, r = size(cf.p)
    @assert p==PP "Inconsistent dimension. Y must be (Tx$p)."
    ## Check whether the last element of cf.f.a is not undef
    @assert isassigned(cf.f.a, length(cf.f.a))
    @assert isassigned(cf.f.P, length(cf.f.P))
    # @assert isassigned(cf.c.F, length(cf.c.F))
    # @assert isassigned(cf.c.Y, length(cf.c.Y))
    offset = cf.t[]
    resize!(cf.f.a.data, TT+offset)
    resize!(cf.f.P.data, TT+offset)
    resize!(cf.c.F.data, TT+offset)
    resize!(cf.c.Y.data, TT+offset)
    @unpack Z, H, T, R, Q = cf.p      
    a, P = currentstate(cf)
    @inbounds for t = 1:TT
        y = Y[t,:]
        att, Ptt, v, F, F⁻¹ = filterstep(a, P, Z, H, T, R, Q, y)
        cf.f.a[t+offset] = att
        cf.f.P[t+offset] = Ptt
        cf.c.F[t+offset] = F⁻¹
        cf.c.Y[t+offset] = y
        s = 0.5*logdet(F)+v'*F⁻¹*v
        cf.loglik[] += - p/2*log(2π) - s[1]
        cf.t[] += 1
    end       
end

function Base.filter!(cf::CovarianceFilter{KFP} where KFP<:KFParms{A}, Y::Vector{Float64}) where A<:AbstractFloat
    TT = length(Y)
    p, m, r = size(cf.p)
    @assert p==1 "Inconsistent dimension. Y must be (Tx1)."
    ## Check whether the last element of cf.f.a is not undef
    @assert isassigned(cf.f.a, length(cf.f.a))
    @assert isassigned(cf.f.P, length(cf.f.P))
    # @assert isassigned(cf.c.F, length(cf.c.F))
    # @assert isassigned(cf.c.Y, length(cf.c.Y))
    offset = cf.t[]
    resize!(cf.f.a.data, TT+offset)
    resize!(cf.f.P.data, TT+offset)
    resize!(cf.c.F.data, TT+offset)
    resize!(cf.c.Y.data, TT+offset)
    @unpack Z, H, T, R, Q = cf.p      
    a, P = currentstate(cf)
    @inbounds for t = 1:TT
        att, Ptt, v, F, F⁻¹ = filterstep(a, P, Z, H, T, R, Q, Y[t])
        cf.f.a[t+offset] = att
        cf.f.P[t+offset] = Ptt
        cf.c.F[t+offset] = [F⁻¹]
        cf.c.Y[t+offset] = [Y[t]]
        s = 0.5*logdet(F)+v'*F⁻¹*v
        cf.loglik[] += - p/2*log(2π) - s[1]
        cf.t[] += 1
    end       
end


function Base.filter!(cf::CovarianceFilter{KFP} where KFP<:KFParms{A}, y::Float64) where A<:Float64
    a, P = currentstate(cf)
    p, m, r = size(cf.p)
    @unpack Z, H, T, R, Q = cf.p  
    att, Ptt, F⁻¹ = filterstep(Z, H, T, R, Q, y)
    push!(cf.f.a, att)
    push!(cf.f.P, Ptt)
    push!(cf.c.F, [F⁻¹])
    push!(cf.c.Y, [y])
    cf.loglik[] += - p/2*log(2π) - 0.5*abs(F)+v*v/F
    cf.t[] += 1
end

function smooth!(cf::CovarianceFilter)
    n = cf.t[]::Int64
    p, m, r = size(cf.p)
    @assert n > 1 "There is not data to smooth over"
    @unpack Z, H, T, R, Q = cf.p  
    F⁻¹ = cf.c.F
    Y   = cf.c.Y
    a   = cf.f.a
    P   = cf.f.P
    â  = Array{Float64}(m, n)
    V  = Array{Float64}(m, m, n)
    r  = Array{Float64}(p, n)
    N   = Array{Float64}(p, p, n)
    r[:,n] = 0.0
    N[:, :, n] = 0.0
    for t in n:-1:2
        v = Y[t] .- Z*a[t-1]
        L = T .- T*P[t-1]*Z'*F⁻¹[t]*Z
        r[:, t-1] = Z'*F⁻¹[t]*v .+ L'*r[:, t]
        â[:, t] = a[t-1] .+ P[t-1]*r[:,t-1]
        N[:, :, t-1] = reshape(convert(Matrix, Z'*F⁻¹[t]*Z + L'*N[:,:,t]*L), (p, p, 1))
        V[:, :, t]   = reshape(convert(Matrix, P[t-1] .- P[t-1]*N[:, :, t-1]*P[t-1]), (p, p, 1))
    end
    cf.s.r = r[:, 1:n-1]'
    cf.s.a = â'[2:end,:]
    cf.s.V = V[:, :, 2:end]
    cf.s.N = N[:, :, 1:n-1]
end

function filterstep(a, P, Z, H, T, R, Q, y)    
    v = y - Z*a
    F = Z*P*Z' .+ H    
    invF = inv(F)
    K = T*P*Z'*invF
    L = T.-K*Z
    att = T*a + K*v
    Ptt = T*P*L' .+ R*Q*R'
    (att, Ptt, v, F, invF)
end

function simulate(cf::CovarianceFilter, nout) 
    @unpack Z, H, T, R, Q = cf.p  
    p, m = size(Z) 
    Y = Array{Float64, 2}(p, nout+1)
    α = Array{Float64, 2}(m, nout+2)
    fH = chol(H)'
    fQ = chol(Q)'
    fP = chol(cf.i.P0)'
    α[:,1] = T*cf.i.a0 + R*fP*randn(m)
    for j in 1:nout+1
        Y[:, j] = Z*α[:,j] + fH*randn(p)
        α[:, j+1] = T*α[:,j] + R*fQ*randn(m)
    end
    (Y[:,1:end-1], α[:,2:end-1])
end

loglik(cf) = cf.loglik[]::Float64


include("methods.jl")

export CovarianceFilter, simulate


end # module
