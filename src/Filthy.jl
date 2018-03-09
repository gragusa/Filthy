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

mutable struct KFFiltered{AT, PT, PI}
    a::AT
    P::PT
    Pstar::PI
    Pinf::PI
    d::Array{Int,1}
    isFsingular::BitArray{1}
    hasdiffended::BitArray{1}
end

function KFFiltered(a, P, Pinf::Void, Q::Union{AbstractArray, AbstractFloat})
    KFFiltered(a, P, P, P, [1], BitArray{1}([0]), BitArray([1]))
end

function KFFiltered(a, P, Pinf::AbstractMatrix, Q::AbstractMatrix)
    ## Check positive element on the diagonal    
    idx = convert(Array{Int}, [Pinf[j,j]>0 for j in 1:size(Pinf, 1)])
    Pinf_ = convert(typeof(Pinf), diagm(idx))    
    R₀ = eye(size(Pinf_,1))
    R₀[find(Pinf_.==1)] = 0.0
    R₀ = convert(typeof(Pinf), R₀)
    Pstar = GrowableArray(R₀*R₀')
    Pinf′ = GrowableArray(Pinf_)    
    KFFiltered(a, P, Pstar, Pinf′, [1], BitArray([0]),  BitArray([0]))
end

function KFFiltered(a, P, Pinf::AbstractFloat, Q::AbstractFloat)
    ## Check positive element on the diagonal    
    @assert Pinf > 0 "Initial value to diffuse part should be > 0"    
    Pstar = GrowableArray(0.0)
    Pinf′ = GrowableArray(Pinf)
    #isdiff = isdiffuse(Pinf)
    KFFiltered(a, P, Pstar, Pinf′, [1], BitArray(1),  BitArray([0]))
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

function LinearStateSpace(Z::A, H::A, T::A, R::B, Q::A, a1::AZ, P1::PZ; Pinf::Union{Void, PZ}=nothing) where {A, B, AZ, PZ}
    @assert isa(P1, A) "P1 must be of type $(typeof(Z))"
    if isa(Z, AbstractMatrix)
        @assert isa(a1, AbstractVector) "a1 must be of type Vector"
    elseif isa(Z, AbstractFloat)
        @assert isa(a1, AbstractFloat) "a1 must be of type AbstractFloat"
    end
    #checksizes(Z, H, T, R, Q, α0, P1)
    m = numstates(T)::Int64
    p = nummeasur(Z)::Int64
    a = GrowableArray(a1)
    P = GrowableArray(P1)
    params = KFParms(Z, H, T, R, Q)
    filter = KFFiltered(a, P, Pinf, Q)
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

function storepush!(ss, a, P, v, F, F⁻¹, Pinf, Pstar, y, ispd)
    # @show a
    # @show P
    # @show F⁻¹
    # @show F
    # @show y
    # @show Pinf
    # @show Pstar
    push!(ss.f.a, a)
    push!(ss.f.P, P)
    push!(ss.c.F⁻¹, F⁻¹)
    push!(ss.c.F, F)
    push!(ss.c.Y, y)
    push!(ss.f.Pinf, Pinf)
    push!(ss.f.Pstar, Pstar)
    ss.f.d[1] += 1
    push!(ss.f.isFsingular, !ispd)
    nothing
end

function storepush!(ss, a, P, v, F, F⁻¹, y)
    # @show a
    # @show P
    # @show F⁻¹
    # @show F
    # @show y
    push!(ss.f.a, a)
    push!(ss.f.P, P)
    push!(ss.c.F⁻¹, F⁻¹)
    push!(ss.c.F, F)
    push!(ss.c.Y, y)
    nothing
end

function storeset!(ss, a, P, v, F, F⁻¹, Pinf, Pstar, y, offset, ispd)
    idx = ss.t[] + offset
    ss.f.a[idx]    = a
    ss.f.P[idx]    = P
    ss.c.F⁻¹[idx]  = F⁻¹
    ss.c.F[idx]    = F
    ss.c.Y[idx]    = y
    push!(ss.f.Pinf, Pinf)
    push!(ss.f.Pstar, Pstar)
    ss.f.d[1] += 1
    push!(ss.f.isFsingular, !ispd)
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
    a′, P′, v, F, F⁻¹, Pinf′, Pstar′, ispd = filterstep(a, P, Pinf, Pstar, Z, H, T, R, Q, y)    
    ss.f.hasdiffended[1] = norm(Pinf′) <= 1e-08
    storepush!(ss, a′, P′, v, F, F⁻¹, Pinf′, Pstar′, y, ispd)                
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
    a′, P′, v, F, F⁻¹, Pinf′, Pstar′, ispd = filterstep(a, P, Pinf, Pstar, Z, H, T, R, Q, y)    
    ss.f.hasdiffended[1] = norm(Pinf′) <= 1e-06
    storeset!(ss, a′, P′, v, F, F⁻¹, Pinf′, Pstar′, y, offset, ispd)
    updatelik!(ss, v, F, F⁻¹, ispd)
end

function filterstep(a, P, Z, H, T, R, Q, y)
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
    (isFinvertible, Finfinv) = Filthy.safeinverse(Finf)    
    diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, Val{isFinvertible})    
end

function diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{true}})
    ## Finf is invertible
    v     = y .- Z*a
    Kinf  = T*Pinf*Z'*Finfinv
    Linf  = T - Kinf*Z 
    Fstar = Z*Pstar*Z' + H
    Kstar = (T*Pstar*Z' + Kinf*Fstar)*Finfinv
    a′     = T*a + Kinf*v
    Pinf′  = T*Pinf*Linf' 
    Pstar′ = T*Pstar*Linf' + Kinf*Finf*Kstar' + R*Q*R
    P′     = Pstar′
    @show Finfinv
    @show Pstar′
    @show Pinf′    
    (a′, P′, v, Finf, Finfinv, Pinf′, Pstar′, true)
end

function diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{false}})
    ## Finf is not notposdef/invertible
    v     = y - Z*a        
    Fstar = Z*Pstar*Z' + H
    Fstarinv = inv(Fstar)  
    Kstar = T*Pstar*Z'*Fstarinv
    Lstar = T - Kstar*Z 
    a′     = T*a + Kstar*v
    Pinf′  = T*Pinf*T' 
    Pstar′ = T*Pstar*Lstar' + R*Q*R
    P′     = Pstar′
    @show Pstar′
    @show Pinf′
    (a′, P′, v, Fstar, Fstarinv, Pinf′, Pstar′, false)
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
        if !ss.f.isFsingular[t]
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
end





function fastsmooth!(ss::LinearStateSpace)
    n = ss.t[]::Int64
    p, m, r = size(ss.p)
    @assert n > 1 "There is not data to smooth over"
    @unpack Z, H, T, R, Q = ss.p  
    F⁻¹ = ss.c.F⁻¹
    Y  = ss.c.Y
    a  = ss.f.a
    P  = ss.f.P
    # â  = Array{Float64}(m, n)
    # V  = Array{Float64}(m, m, n)
    â = Array{NTuple{m, Float64}}(n)
    V = Array{NTuple{m^2, Float64}}(n)
    r = zeros(p)
    N = zeros(p, p)
    r′ = zeros(p)
    N′ = zeros(p, p)
    v = Array{Float64}(p)
    L = Array{Float64}(m,m)
    d  = ss.f.d[1]

    if d>2
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
    
    for t in n:-1:d #(d+1)
        v .= Y[t] .- Z*a[t-1]
        L .= convert(Matrix, T .- T*P[t-1]*Z'*F⁻¹[t]*Z)
        r′ .= Z'*F⁻¹[t]*v .+ L'*r
        â[t] = convert(Tuple, a[t-1] .+ P[t-1]*r′)
        N′ .= convert(Matrix, Z'*F⁻¹[t]*Z + L'*N*L)
        V[t] = tuple((P[t-1] .- P[t-1]*N′*P[t-1])...)
        copy!(N, N')
        copy!(r, r′)
    end

    ## Diffuse step
    for t in (d-1):-1:2
        Pinf = ss.f.Pinf[t-1]
        Pstar = ss.f.Pstar[t]
        v = Y[t] .-Z*a[t-1]  
        if !ss.f.isFsingular[t]
            Finf = Z*Pinf*Z'
            Finfinv = inv(Finf)
            Kinf = T*Pinf*Z*Finfinv
            Linf = T - Kinf*Z
            Fstar = Z'*Pstar*Z + H
            Kstar = (T*Pstar*Z' + Kinf*Fstar)*Finfinv
            F♯ = Kstar'Ndz[:,:,t]*Kstar - Finfinv*Fstar*Finfinv
            rdz[:,t-1] = Linf'rdz[:, t]
            rdo[:,t-1] = Z'*(Finfinv*v - Kstar'rdz[:,t]) + Linf'r[:,t]
            Ndz[:,:,t-1] = Linf'*Ndz[:,:,t]*Linf
            A = Linf'Ndz[:,:,t]*Kstar*Z
            Ndo[:,:,t-1] = Z'Finfinv*Z + Linf'*Ndo[:, :, t]*Linf - A - A'
            A = Linf'Ndo[:,:,t]*Kstar*Z
            Ndt[:,:,t-1] = Z'*F♯*Z + Linf'*Ndt[:,:,t]*Linf - A - A'
            â[:,t-1] = a[t] .+ Pstar*rdz[:,t-1] .+ Pinf*rdz[:,t-1]
            A = Pinf*Ndo[:,:,t-1]*Pstar
            V[:,:,t] = Pstar - Pstar*Ndz[:,:,t-1] - A - A'
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
            â[:,t-1] = a[t] .+ Pstar*rdz[:,t-1] .+ Pinf*rdz[:,t-1]
            A = Pinf*Ndo[:,:,t-1]*Pstar
            V[:,:,t] = Pstar - Pstar*Ndz[:,:,t-1] - A - A'
        end
    end
    (â, V)
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

export LinearStateSpace, simulate


end # module
