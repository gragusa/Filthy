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
    isinitdiffuse::Bool
    hasdiffended::BitArray{1}
end

function KFFiltered(a, P, Pinf::Void, Q::Union{AbstractArray, AbstractFloat})
    KFFiltered(a, P, P, P, [1], false, BitArray([1]))
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
    isdiff = isdiffuse(Pinf_)
    KFFiltered(a, P, Pstar, Pinf′, [1], isdiff,  BitArray([0]))
end

function KFFiltered(a, P, Pinf::AbstractFloat, Q::AbstractFloat)
    ## Check positive element on the diagonal    
    @assert Pinf > 0 "Initial value to diffuse part should be > 0"    
    Pstar = GrowableArray(0.0)
    Pinf′ = GrowableArray(Pinf)
    isdiff = isdiffuse(Pinf)
    KFFiltered(a, P, Pstar, Pinf′, [1], isdiff,  BitArray([0]))
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

function storepush!(ss, a, P, v, F, F⁻¹, Pinf, Pstar, y)
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
    # s = loglikpart(v, F, F⁻¹)
    # p, _ = size(ss.p)
    # ss.loglik[1] += - p/2*log(2π) - s
    ss.f.d[1] += 1
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
    # s = loglikpart(v, F, F⁻¹)
    # p, _ = size(ss.p)
    # ss.loglik[1] += - p/2*log(2π) - s
    nothing
end

function storeset!(ss, a, P, v, F, F⁻¹, Pinf, Pstar, y, offset)
    idx = ss.t[] + offset
    ss.f.a[idx]    = a
    ss.f.P[idx]    = P
    ss.c.F⁻¹[idx]  = F⁻¹
    ss.c.F[idx]    = F
    ss.c.Y[idx]    = y
    push!(ss.f.Pinf, Pinf)
    push!(ss.f.Pstar, Pstar)
    # s = loglikpart(v, F, F⁻¹)
    # p, _ = size(ss.p)
    # ss.loglik[1] += - p/2*log(2π) - s
    ss.f.d[1] += 1
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
    ss.f.hasdiffended[1] = norm(Pinf′) <= 1e-06
    storepush!(ss, a′, P′, v, F, F⁻¹, Pinf′, Pstar′, y)                
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
    storeset!(ss, a′, P′, v, F, F⁻¹, Pinf′, Pstar′, y, offset)
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
    (a′, P′, v, Fstar, Fstarinv, Pinf′, Pstar′, false)
end










# """
# filter!(cf::LinearStateSpace, y::Vector{Float64})
# On-line covariance filter. 
# """
# function Base.filter!(cf::LinearStateSpace, y::Vector{Float64})    
#     p, m, r = size(cf.p)
#     a, P, Pinf, Pstar = currentstate(cf)
#     @unpack Z, H, T, R, Q = cf.p
#     exact = isexactfilter(Pinf)::Bool
#     a′, P′, v, F, F⁻¹, Pstar′, Pinf′, d = filterstep(a, P, Pstar, Pinf, Z, H, T, R, Q, y, Val{exact})
#     push!(cf.f.a, a′)
#     push!(cf.f.P, P′)
#     push!(cf.c.F⁻¹, F⁻¹)
#     push!(cf.c.F, F)
#     if cf.i.d == cf.t[]
#         push!(cf.f.Pinf, Pstar′)
#         push!(cf.f.Pstar, Pinf′)
#     end
#     push!(cf.c.Y, y)
#     s = 0.5*logdet(F)+v'*F⁻¹*v
#     cf.loglik[1] += - p/2*log(2π) - s[1]
#     cf.t[] += 1
# end

# function Base.filter!(cf::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Matrix{Float64}) where A<:AbstractArray
#     TT, PP = size(Y)
#     p, m, r = size(cf.p)
#     offset = cf.t[]
#     @assert p==PP "Inconsistent dimension. Y must be (Tx$p)."
#     ## Check whether the last element of cf.f.a is not undef
#     @assert isassigned(cf.f.a, length(cf.f.a))
#     @assert isassigned(cf.f.P, length(cf.f.P))
    
#     resize!(cf.f.a.data, TT+offset)
#     resize!(cf.f.P.data, TT+offset)
#     resize!(cf.c.F.data, TT+offset)
#     resize!(cf.c.Y.data, TT+offset)
#     @unpack Z, H, T, R, Q = cf.p      
#     a, P = currentstate(cf)
#     @inbounds for t = 1:TT
#         y = Y[t,:]
#         a, P, v, F, F⁻¹, Pstar, Pinf = filterstep(a, P, Z, H, T, R, Q, Y[t])
#         cf.f.a[t+offset] = a
#         cf.f.P[t+offset] = P
#         if cf.i.d[1] == t
#             cf.f.Pstar[t+offset] = Pstar
#             cf.f.Pinf[t+offset] = Pinf
#         end
#         cf.c.F⁻¹[t+offset] = [F⁻¹]
#         cf.c.F[t+offset] = [F]
#         cf.c.Y[t+offset] = [Y[t]]
#         s = 0.5*logdet(F)+v'*F⁻¹*v
#         cf.loglik[1] += - p/2*log(2π) - s[1]
#         cf.t[] += 1
#     end
# end

# function Base.filter!(cf::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Vector{Float64}) where A<:AbstractFloat
#     TT = length(Y)
#     p, m, r = size(cf.p)
#     @assert p==1 "Inconsistent dimension. Y must be (Tx1)."
#     ## Check whether the last element of cf.f.a is not undef
#     @assert isassigned(cf.f.a, length(cf.f.a))
#     @assert isassigned(cf.f.P, length(cf.f.P))
#     # @assert isassigned(cf.c.F, length(cf.c.F))
#     # @assert isassigned(cf.c.Y, length(cf.c.Y))
#     offset = cf.t[]
#     resize!(cf.f.a.data, TT+offset)
#     resize!(cf.f.P.data, TT+offset)
#     resize!(cf.c.F.data, TT+offset)
#     resize!(cf.c.Y.data, TT+offset)
#     @unpack Z, H, T, R, Q = cf.p      
#     a, P = currentstate(cf)
#     @inbounds for t = 1:TT
#         a, P, v, F, F⁻¹, Pstar, Pinf = filterstep(a, P, Z, H, T, R, Q, Y[t])        
#         cf.f.a[t+offset] = a
#         cf.f.P[t+offset] = P
#         if cf.i.d[1] == t
#             cf.f.Pstar[t+offset] = Pstar
#             cf.f.Pinf[t+offset] = Pinf
#         end
#         cf.c.F⁻¹[t+offset] = [F⁻¹]
#         cf.c.F[t+offset] = [F]
#         cf.c.Y[t+offset] = [Y[t]]
#         s = 0.5*logdet(F)+v'*F⁻¹*v
#         cf.loglik[1] += - p/2*log(2π) - s[1]
#         cf.t[] += 1
#     end       
# end


# function Base.filter!(cf::LinearStateSpace{KFP} where KFP<:KFParms{A}, y::Float64) where A<:Float64
#     a, P = currentstate(cf)
#     p, m, r = size(cf.p)
#     @unpack Z, H, T, R, Q = cf.p  
#     att, Ptt, F⁻¹ = filterstep(Z, H, T, R, Q, y)
#     push!(cf.f.a, att)
#     push!(cf.f.P, Ptt)
#     push!(cf.c.F⁻¹, [F⁻¹])
#     push!(cf.c.F, [F])
#     push!(cf.c.Y, [y])
#     cf.loglik[1] += - p/2*log(2π) - 0.5*abs(F)+v*v/F
#     cf.t[] += 1
# end

# function smooth!(cf::LinearStateSpace)
#     n = cf.t[]::Int64
#     p, m, r = size(cf.p)
#     @assert n > 1 "There is not data to smooth over"
#     @unpack Z, H, T, R, Q = cf.p  
#     F⁻¹ = cf.c.F
#     Y   = cf.c.Y
#     a   = cf.f.a
#     P   = cf.f.P
#     â  = Array{Float64}(m, n)
#     V  = Array{Float64}(m, m, n)
#     r  = Array{Float64}(p, n)
#     N   = Array{Float64}(p, p, n)
#     r[:,n] = 0.0
#     N[:, :, n] = 0.0
#     for t in n:-1:2
#         v = Y[t] .- Z*a[t-1]
#         L = T .- T*P[t-1]*Z'*F⁻¹[t]*Z
#         r[:, t-1] = Z'*F⁻¹[t]*v .+ L'*r[:, t]
#         â[:, t] = a[t-1] .+ P[t-1]*r[:,t-1]
#         N[:, :, t-1] = reshape(convert(Matrix, Z'*F⁻¹[t]*Z + L'*N[:,:,t]*L), (p, p, 1))
#         V[:, :, t]   = reshape(convert(Matrix, P[t-1] .- P[t-1]*N[:, :, t-1]*P[t-1]), (p, p, 1))
#     end
#     cf.s.r = r[:, 1:n-1]'
#     cf.s.a = â'[2:end,:]
#     cf.s.V = V[:, :, 2:end]
#     cf.s.N = N[:, :, 1:n-1]
# end









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
