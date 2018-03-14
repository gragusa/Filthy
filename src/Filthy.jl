module Filthy

using StaticArrays
using GrowableArrays
using Parameters
using ForwardDiff
using Optim

struct KFParms{Zt, Ht, Tt, Rt, Qt}
    Z::Zt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
end

mutable struct KFFiltered{AT, PT, PI}
    a::AT
    P::PT
    Pstar::PI
    Pinf::PI
    d::Array{Int,1}
    flagF::Array{Int, 1}
    hasdiffended::BitArray{1}
end


struct Masked{FREEIDX, FIXEDIDX, FREEITR, FIXEDITR, P}
    freeindexes::FREEIDX
    fixedindexes::FIXEDIDX
    freeitr::FREEITR
    fixeditr::FIXEDITR
    parent::P
    count::Array{Int64, 1}
end

struct OptimSSMPar{T, G, S}
    Z::Matrix{T}
    H::Matrix{T}
    R::Matrix{T}
    T::Matrix{T}
    Q::Matrix{T}
    a1::Vector{T}
    P1::Vector{T}
    P1inf::G
    Pstar::S
end

struct OptimSSMParDual{T, F, N}
    Z::ForwardDiff.Dual{T,F,N}
    H::ForwardDiff.Dual{T,F,N}
    R::ForwardDiff.Dual{T,F,N}
    T::ForwardDiff.Dual{T,F,N}
    Q::ForwardDiff.Dual{T,F,N}
    a1::ForwardDiff.Dual{T,F,N}
    P1::ForwardDiff.Dual{T,F,N}
end

struct SSMOptim{M, P, D}
    m::M
    p::P
    d::D
end

struct OptimSSM{M, JCACHE, HCACHE, RR, T, F, G, S, Y}
    m::M
    jacobiancache::JCACHE
    hessiancache::HCACHE
    Z::Matrix{Float64}
    H::Matrix{Float64}
    R::RR
    T::Matrix{Float64}
    Q::Matrix{Float64}
    a1::T
    P1::F
    P1inf::G
    Pstar::S
    y::Y
end

struct OptimSSMScalar{T<:AbstractFloat}
    par::Array{T, 1}
    idx::BitArray{1}
    a1::T
    P1::T
    P1inf::T
    Pstar::T
    y::Array{T, 1}
end


function KFFiltered(a, P, Pinf::Void, Q::Union{AbstractArray, AbstractFloat})
    KFFiltered(a, P, P, P, [1], BitArray{1}([0]), BitArray([1]))
end



function KFFiltered(a1, P1, P1inf::AbstractMatrix)
    ## Check positive element on the diagonal    
    ## To Do: Check that P1inf is diagonal with diagonal entry > 0
    a1′, P1′, Pstar = setupdiffusematrices(a1, P1, P1inf)
    KFFiltered(GrowableArray(a1′), GrowableArray(P1′), GrowableArray(Pstar), GrowableArray(P1inf), [1], [1],  BitArray([0]))
end

function KFFiltered(a1, P1, P1inf::AbstractFloat)
    ## Check positive element on the diagonal    
    @assert P1inf > 0 "Initial value to diffuse part should be > 0"    
    KFFiltered(GrowableArray(a1), GrowableArray(P1), GrowableArray(0.0), GrowableArray(P1inf), [1], [1],  BitArray([0]))
end

mutable struct KFSmoothed{AT, PT}
    a::AT
    V::PT
    r::AT
    N::PT
end

struct KFInitVal{AT, PT}
    a1::AT
    P1::PT
end

struct KFCache{FF, FI, DD}
    F⁻¹::FF ## Z*F⁻¹ (m x m)
    F::FI
    Y::DD   ## Data 
end

function KFCache(T::AbstractMatrix, p::Int)
    KFCache(GrowableArray(zeros(T)), GrowableArray(zeros(T)), GrowableArray(zeros(p)))
end 

function KFCache(T::AbstractFloat, p::Int)
    @assert p==1 "Something wrong"
    KFCache(GrowableArray(zero(T)), GrowableArray(zero(T)), GrowableArray(zero(eltype(T))))
end 


struct LinearStateSpace{P<:KFParms, F<:KFFiltered, S<:KFSmoothed,  I<:KFInitVal, C<:KFCache, L<:Array, N<:Base.RefValue{Int64}}
    p::P
    f::F
    s::S
    i::I
    c::C
    loglik::L
    t::N        
end

function LinearStateSpace(Z::A, H::D, T::A, R::B, Q::A, a1::AZ, P1::PZ; P1inf::Union{Void, PZ}=nothing) where {A, D, B, AZ, PZ}
    @assert isa(P1, A) "P1 must be of type $(typeof(Z))"
    if isa(Z, AbstractMatrix)
        @assert isa(a1, AbstractVector) "a1 must be of type Vector"
    elseif isa(Z, AbstractFloat)
        @assert isa(a1, AbstractFloat) "a1 must be of type AbstractFloat"
    end
    #checksizes(Z, H, T, R, Q, α0, P1)
    m = numstates(T)::Int64
    p = nummeasur(Z)::Int64
    # a = GrowableArray(a1)
    # P = GrowableArray(P1)
    params = KFParms(Z, H, T, R, Q)
    filter = KFFiltered(a1, P1, P1inf)
    smooth = KFSmoothed(Array{Float64}(1,1), Array{Float64}(1,1,1), 
                        Array{Float64}(1,1), Array{Float64}(1,1,1))
    inival = KFInitVal(a1, P1)
    caches = KFCache(T, p)
    LinearStateSpace(params, filter, smooth, inival, caches, [0.0], Ref(1))
end

#=============
Filter methods
==============#

#=-
A. Covariance Filter
-=#

function onlinefilter!(ss::LinearStateSpace, y)
    hasdiffended = first(ss.f.hasdiffended)::Bool
    onlinefilterstep!(ss, y, Val{!hasdiffended})
    ss.t[] += 1
    nothing
end

function onlinefilter_set!(ss::LinearStateSpace, y, offset::Int)
    hasdiffended = first(ss.f.hasdiffended)::Bool
    onlinefilterstep_set!(ss, y, offset, Val{!hasdiffended})
    ss.t[] += 1
    nothing
end

function Base.filter!(ss::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Vector) where A<:AbstractFloat
    (T, p, m, r, offset) = Filthy.checkfilterdataconsistency(ss, Y)
    #rY = reshape(Y, (p, T))
    Filthy.resizecontainer!(ss, T+offset)
    for j in 1:length(Y)
        onlinefilter_set!(ss, Y[j], offset)
    end
end

function Base.filter!(ss::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Matrix) where A<:AbstractArray
    (T, p, m, r, offset) = Filthy.checkfilterdataconsistency(ss, Y)
    #rY = reshape(Y, (p, T))
    Filthy.resizecontainer!(ss, T+offset)
    for j in 1:size(Y, 2)
        onlinefilter_set!(ss, Y[:,j], offset)
    end
end

function storepush!(ss, a, P, v, F, F⁻¹, Pinf, Pstar, y, flag)
    push!(ss.f.a, a)
    push!(ss.f.P, P)
    push!(ss.c.F⁻¹, F⁻¹)
    push!(ss.c.F, F)
    push!(ss.c.Y, y)
    push!(ss.f.Pinf, Pinf)
    push!(ss.f.Pstar, Pstar)
    ss.f.d[1] += 1
    push!(ss.f.flagF, flag)
    nothing
end

function storepush!(ss, a, P, v, F, F⁻¹, y)
    push!(ss.f.a, a)
    push!(ss.f.P, P)
    push!(ss.c.F⁻¹, F⁻¹)
    push!(ss.c.F, F)
    push!(ss.c.Y, y)
    nothing
end

function storeset!(ss, a, P, v, F, F⁻¹, Pinf, Pstar, y, offset, flag)
    idx = ss.t[] + offset
    ss.f.a[idx]    = a
    ss.f.P[idx]    = P
    ss.c.F⁻¹[idx]  = F⁻¹
    ss.c.F[idx]    = F
    ss.c.Y[idx]    = y
    push!(ss.f.Pinf, Pinf)
    push!(ss.f.Pstar, Pstar)
    ss.f.d[1] += 1
    push!(ss.f.flagF, flag)
    nothing
end

function storeset!(ss, a, P, v, F, F⁻¹, y, offset)
    idx = ss.t[] + offset
    ss.f.a[idx] = a
    ss.f.P[idx] = P
    ss.c.F⁻¹[idx] = F⁻¹
    ss.c.F[idx] = F
    ss.c.Y[idx] = y
    # s = loglikpart(v, F, F⁻¹)
    # p, _ = size(ss.p)
    # ss.loglik[1] += - p/2*log(2π) - s
    nothing
end


function onlinefilterstep!(ss::LinearStateSpace, y, v::Type{Val{false}})
    a, P = currentstate(ss, Val{false})
    @unpack Z, H, T, R, Q = ss.p
    a′, P′, v, F, F⁻¹ = filterstep(a, P, Z, H, T, R, Q, y)
    storepush!(ss, a′, P′, v, F, F⁻¹, y)
end

function onlinefilterstep!(ss::LinearStateSpace, y, v::Type{Val{true}})
    a, P, Pinf, Pstar = currentstate(ss, Val{true})
    @unpack Z, H, T, R, Q = ss.p
    a′, P′, v, F, F⁻¹, Pinf′, Pstar′, flag = filterstep(a, P, Pinf, Pstar, Z, H, T, R, Q, y)    
    ss.f.hasdiffended[1] = norm(Pinf′) <= 1e-08
    storepush!(ss, a′, P′, v, F, F⁻¹, Pinf′, Pstar′, y, flag)                
end

function onlinefilterstep_set!(ss::LinearStateSpace, y, offset, v::Type{Val{false}})
    a, P = currentstate(ss, Val{false})
    @unpack Z, H, T, R, Q = ss.p
    a′, P′, v, F, F⁻¹ = filterstep(a, P, Z, H, T, R, Q, y)
    storeset!(ss, a′, P′, v, F, F⁻¹, y, offset)
    updatelik!(ss, v, F, F⁻¹)
end

function onlinefilterstep_set!(ss::LinearStateSpace, y, offset, v::Type{Val{true}})
    a, P, Pinf, Pstar = currentstate(ss, Val{true})
    @unpack Z, H, T, R, Q = ss.p
    a′, P′, v, F, F⁻¹, Pinf′, Pstar′, flag = filterstep(a, P, Pinf, Pstar, Z, H, T, R, Q, y)    
    ss.f.hasdiffended[1] = norm(Pinf′) <= 1e-06
    storeset!(ss, a′, P′, v, F, F⁻¹, Pinf′, Pstar′, y, offset, flag)
    updatelik!(ss, v, F, F⁻¹, flag)
end

function filterstep(a::AA, P::PP, Z, H, T, R, Q, y) where {AA, PP}
    ## Non-diffuse step
    v = y .- Z*a
    F = Z*P*Z' .+ H    
    Finv = inv(F)
    K = T*P*Z'*Finv
    L = T.-K*Z
    a′ = T*a .+ K*v
    P′ = T*P*L' .+ R*Q*R'
    (a′, P′, v, F, Finv)
end

function filterstep(a, P, Pinf, Pstar, Z, H, T, R, Q, y)
    Finf = Z*Pinf*Z'
    (Fflag, Finfinv) = Filthy.safeinverse(Finf)    
    diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, Val{Fflag})    
end

function diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{1}})
    ## Finf is invertible
    v     = y .- Z*a
    Kinf  = T*Pinf*Z'*Finfinv
    Linf  = T - Kinf*Z 
    Fstar = Z*Pstar*Z' + H
    Kstar = (T*Pstar*Z' + Kinf*Fstar)*Finfinv
    a′     = T*a + Kinf*v
    Pinf′  = T*Pinf*Linf' 
    Pstar′ = T*Pstar*Linf' + Kinf*Finf*Kstar' + R*Q*R'
    P′     = Pstar′
    (a′, P′, v, Finf, Finfinv, Pinf′, Pstar′, 1)
end

function diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{0}})
    ## Finf is not > 0
    v     = y - Z*a        
    Fstar = Z*Pstar*Z' + H
    Fstarinv = inv(Fstar)  
    Kstar = T*Pstar*Z'*Fstarinv
    Lstar = T - Kstar*Z 
    a′     = T*a + Kstar*v
    Pinf′  = T*Pinf*T' 
    Pstar′ = T*Pstar*Lstar' + R*Q*R'
    P′     = Pstar′
    (a′, P′, v, Fstar, Fstarinv, Pinf′, Pstar′, 0)
end

function smooth!(ss::LinearStateSpace)
    n = ss.t[]::Int64
    p, m, r = size(ss.p)
    @assert n > 1 "There is not data to smooth over"
    @unpack Z, H, T, R, Q = ss.p  
    â  = Array{Float64}(m, n)
    V  = Array{Float64}(m, m, n)
    r  = Array{Float64}(p, n)
    N  = Array{Float64}(p, p, n)

    F⁻¹ = ss.c.F⁻¹
    Y  = ss.c.Y
    a  = ss.f.a
    P  = ss.f.P
    d  = ss.f.d[1]
    r[:, n] = 0.0
    N[:, :, n] = 0.0
    
    @inbounds for t in n:-1:d+1 #(d+1)
        ## Hoisting
        Finv = F⁻¹[t]
        PP = P[t-1]
        v = Y[t] .- Z*a[t-1]
        L = T .- T*P[t-1]*Z'*Finv*Z
        r[:, t-1] = Z'*Finv*v .+ L'*r[:, t]
        â[:, t] = a[t-1] .+ PP*r[:,t-1]
        N[:, :, t-1] = reshape(convert(Matrix, Z'*Finv*Z + L'*N[:,:,t]*L), (p, p, 1))
        V[:, :, t] = reshape(convert(Matrix, P[t-1] .- P[t-1]*N[:, :, t-1]*P[t-1]), (p, p, 1))
    end

    if d>=1
        rdz = Array{Float64}(p, d)
        rdo = Array{Float64}(p, d)
        Ndz = Array{Float64}(p, p, d)
        Ndo = Array{Float64}(p, p, d)
        Ndt = Array{Float64}(p, p, d)
        rdz[:, d] = r[:, d]
        rdo[:, d] = 0.0
        Ndz[:, :, d] = N[:, :, d]
        Ndo[:, :, d] = 0.0
        Ndt[:, :, d] = 0.0
    end    

    ## Diffuse step
    for t in d:-1:2
        Pinf = ss.f.Pinf[t-1]
        Pstar = ss.f.Pstar[t]
        v = Y[t] .-Z*a[t-1]
        if ss.f.flagF[t] == 1
            Finf = Z*Pinf*Z'
            Finfinv = inv(Finf)
            Kinf = T*Pinf*Z*Finfinv
            Linf = T - Kinf*Z
            Fstar = Z*Pstar*Z' + H
            Kstar = (T*Pstar*Z' - Kinf*Fstar)*Finfinv
            F♯ = Kstar'Ndz[:,:,t]*Kstar - Finfinv*Fstar*Finfinv
            rdz[:,t-1] = Linf'rdz[:, t]
            rdo[:,t-1] = Z'*(Finfinv*v - Kstar'rdz[:,t]) + Linf'*rdo[:,t]
            Ndz[:,:,t-1] = Linf'*Ndz[:,:,t]*Linf
            A = Linf'Ndz[:,:,t]*Kstar*Z
            Ndo[:,:,t-1] = Z'Finfinv*Z + Linf'*Ndo[:, :, t]*Linf - A - A'
            A = Linf'Ndo[:,:,t]*Kstar*Z
            Ndt[:,:,t-1] = Z'*F♯*Z + Linf'*Ndt[:,:,t]*Linf - A - A'
            â[:,t] = a[t-1] .+ Pstar*rdz[:,t-1] .+ Pinf*rdo[:,t-1]
            A = Pinf*Ndo[:,:,t-1]*Pstar
            B = Pinf*Ndt[:,:,t-1]*Pinf
            V[:,:,t] = Pstar - Pstar*Ndz[:,:,t-1]*Pstar - (A + A') - B 
        else
            Fstar = Z'*Pstar*Z + H
            Fstarinv = inv(Fstar)
            Lstar = T - Kstar*Z
            Kstar = T*Pstar*Z'*Fstarinv
            rdz[:,t-1] = Z'Fstarinv*v - Lstar'rdz[:, t]
            rdo[:,t-1] = T'rdo[:,t]
            Ndz[:,:,t-1] = Z'*Fstarinv*Z + Lstar'Ndz[:,:,t]*Lstar            
            Ndo[:,:,t-1] = T'Ndo[:,:,t]*Lstar
            Ndt[:,:,t-1] = T*Ndt[:,:,t]*T
            â[:,t-1] = a[t] .+ Pstar*rdz[:,t] .+ Pinf*rdz[:,t-1]
            V[:,:,t] = Pstar - Pstar*Ndz[:,:,t-1] 
        end
    end
    ss.s.r = r[:, 1:n-1]'
    ss.s.a = â'[2:end, :]
    ss.s.V = V[:, :, 2:end]
    ss.s.N = N[:, :, 1:n-1]
    nothing
end

function fastloglik(Z::AbstractMatrix, H, T, R, Q, a′, P′, Pinf′, Pstar′, Y::AbstractMatrix)
    n, p = size(Y)
    ll = zero(eltype(Z))
    j = 1; p = 1
    c = -p*n*log(2π)/2
    while (any(Pinf′ .> 0)) & (j < n)
        a′, P′, v, F, F⁻¹, Pinf′, Pstar′, flag = filterstep(a′, P′, Pinf′, Pstar′, Z, H, T, R, Q, Y[j,:])
        ll += loglik(v, F, F⁻¹, flag)
        j += 1
    end
    for t in j:n
        a′, P′, v, F, F⁻¹ = filterstep(a′, P′, Z, H, T, R, Q, Y[t, :])        
        ll += loglik(v, F, F⁻¹)
    end
    ll +c
end

function fastloglikscalar(Z::FF, H::FF, T::FF, R::FF, Q::FF, a1::G, P1::G, P1inf::G, Pstar::G, y) where {FF<:Real, G<:AbstractFloat}
    ll = zero(eltype(Z));  n = length(y)
    a = a1; P = P1; j = 1; p = 1; c = -n/2*log(2π)
    while (P1inf > 0) & (j < n)
        a, P, v, F, F⁻¹, P1inf, Pstar, flag = filterstep(a, P, P1inf, Pstar, Z, H, T, R, Q, y[j])
        ll += loglik(v, F, F⁻¹, flag)
        j += 1
    end
    for t in j:n
        a, P, v, F, F⁻¹ = filterstep(a, P, Z, H, T, R, Q, y[t])
        ll += loglik(v, F, F⁻¹)
    end
    ll + c
end


# function futil_callback(s::OptimSSM, callback::Function = identiy, theta)    
#     @unpack a1, P1, P1inf, Pstar = s
#     Z = similar(theta, size(s.Z))
#     H = similar(theta, size(s.H))
#     T = similar(theta, size(s.T))
#     R = similar(theta, size(s.R))
#     Q = similar(theta, size(s.Q))
#     ## H and Q are the chol 
#     remask!(s.m, (Z, H, T, R, Q), theta)
#     if callback!(s, theta)
#       -fastloglik(Z, H*H', T, R, Q*Q', a1, P1, P1inf, Pstar, s.y)
#     else
#       +Inf
#     end
# end


function futil(s::OptimSSM, theta)    
    @unpack a1, P1, P1inf, Pstar = s
    Z = similar(theta, size(s.Z))
    H = similar(theta, size(s.H))
    T = similar(theta, size(s.T))
    R = similar(theta, size(s.R))
    Q = similar(theta, size(s.Q))
    ## H and Q are the chol 
    remask!(s.m, (Z, H, T, R, Q), theta)    
    -fastloglik(Z, H*H', T, R, Q*Q', a1, P1, P1inf, Pstar, s.y)
end

function futil2(s::OptimSSM, theta)    
    @unpack a1, P1, P1inf, Pstar, Z, H, T, R, Q = s
    remask!(s.m, (Z, H, T, R, Q), theta)
    ## H and Q are the chol 
    ## Missed optimization: Q*Q' and H*H' could be done 
    ## in place A_mul_bt! --- adding a field to OptimSSM
    -fastloglik(Z, H*H', T, R, Q*Q', a1, P1, P1inf, Pstar, s.y)
end


function futil_scalar(s::OptimSSMScalar, theta)    
    @unpack a1, P1, P1inf, Pstar, idx, par, y = s
    x = similar(theta, length(par))
    copy!(x, par)
    x[idx] = theta
    -fastloglikscalar(x[1], exp(x[2]), x[3], x[4], exp(x[5]), a1, P1, P1inf, Pstar, y)
end

function fit(::Type{LinearStateSpace}, Z::A, CH::A, T::A, R::A, CQ::A, a1::AZ, P1::PZ, P1inf::PZ, Y, start, lower, upper) where {A<:AbstractMatrix, AZ<:AbstractVector, PZ<:AbstractMatrix}
    @assert islowertriangular(CH) "CH must be lower triangular"
    @assert islowertriangular(CQ) "CQ must be lower triangular"
    Z₀, CH₀, T₀, R₀, CQ₀ = map(obj-> convert(Matrix, obj), (Z, CH, T, R, CQ))
    mask = Masked((Z₀, CH₀, T₀, R₀, CQ₀))
    allfixed(mask) && return LinearStateSpaceModel(Z, CH*CH', T, R, CQ*CQ', a1, P1, P1inf)
    ## Setup and do maximum likelihood
    ## Copy matrices to Matrix{Float64}        
    a, P, Pstar = setupdiffusematrices(a1, P1, P1inf)
    fitobj = OptimSSM(mask, Z₀, CH₀, T₀, R₀, CQ₀, a, P, P1inf, Pstar, Y)
    f(x) = futil(fitobj, x)::Float64
    d = Optim.OnceDifferentiable(f, start, autodiff = :forward)
    out = Optim.optimize(d, start, lower, upper, Fminbox{BFGS}())
    remask!(mask, (Z₀, CH₀, T₀, R₀, CQ₀), Optim.minimizer(out))
    ZZ = SMatrix{size(Z)...}(Z₀)
    HH = SMatrix{size(CH)...}(CH₀*CH₀')
    TT = SMatrix{size(T)...}(T₀)    
    RR = SMatrix{size(R₀)...}(R₀)
    QQ = SMatrix{size(CQ)...}(CQ₀*CQ₀')
    aa = SVector{length(a1)}(fitobj.a1)
    PP = SMatrix{size(P1)...}(fitobj.P1)
    PPinf = SMatrix{size(P1)...}(fitobj.P1inf)    
    LinearStateSpace(ZZ, HH, TT, R, QQ, aa, PP, P1inf=PPinf)
end



function fit(::Type{LinearStateSpace}, Z::A, H::A, T::A, R::B, Q::A, a1::AZ, P1::PZ, P1inf::PZ, Y, start) where {A<:AbstractFloat, B, AZ<:AbstractFloat, PZ<:AbstractFloat}
    @assert !((P1inf != zero(P1inf)) && (P1inf != one(P1inf))) "P1inf must be either 1.0 or 0.0"
    Pstar = one(P1inf)
    if P1inf == zero(P1inf)
        @assert P1 > 0 "P1 and P1inf cannot be both zero"       
    elseif P1inf == one(P1inf)
        P1 = zero(P1inf)
        Pstar = zero(P1inf)
    end
    b = [Z, H, T, R, Q]
    idx = isnan.(b)
    b[idx] = start
    start[:] = log.(b[idx .& [false, true, false, false, true]])
    s = OptimSSMScalar(b, idx, a1, P1, P1inf, Pstar, y)
    f(x) = futil_scalar(s, x)
    d = Optim.OnceDifferentiable(f, start, autodiff = :forward)
    out = Optim.optimize(d, start, BFGS())   
    sol = Optim.minimizer(out)
    b[idx] = sol
    sol[:] = exp.(b[idx .& [false, true, false, false, true]])
    b[idx] = sol
    out.minimizer[:] = sol
    out.initial_x[:] = sol
    LinearStateSpace(b[1], b[2], b[3], b[4], b[5], a1, P1, P1inf=P1inf)
end


function reset!(ss::LinearStateSpace)
    ## Clean filtered and smoothed states
    resize!(ss.f.a.data, 1)
    resize!(ss.f.P.data, 1)
    resize!(ss.f.Pstar.data, 1)
    resize!(ss.f.Pinf.data, 1)
    resize!(ss.f.flagF, 1)
    resize!(ss.c.Y.data, 1)
    resize!(ss.c.F.data, 1)
    resize!(ss.c.F⁻¹.data, 1)
    ss.f.d[1] = 1    
    ss.f.hasdiffended = [false]
    ss.t[] = 1
    nothing
end

function simulate(cf::LinearStateSpace, nout) 
    @unpack Z, H, T, R, Q = cf.p  
    p, m = size(Z) 
    Y = Array{Float64, 2}(p, nout+1)
    α = Array{Float64, 2}(m, nout+2)
    fH = chol(H)'
    fQ = chol(Q)'
    fP = chol(cf.i.P1)'
    α[:,1] = T*cf.i.a1 + R*fP*randn(m)
    for j in 1:nout+1
        Y[:, j] = Z*α[:,j] + fH*randn(p)
        α[:, j+1] = T*α[:,j] + R*fQ*randn(m)
    end
    (Y[:,1:end-1], α[:,2:end-1])
end

loglik(cf) = cf.loglik[]::Float64


include("methods.jl")

export LinearStateSpace, simulate, filter!, smooth!, fit, reset!

end # module
