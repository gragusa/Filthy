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
end

function KFFiltered(a, P, Pinf::Void, Q::AbstractArray, c)
    KFFiltered(a, P, P, P)
end

function KFFiltered(a, P, Pinf::AbstractArray, Q::AbstractArray, c)
    ## Check positive element on the diagonal    
    idx = [((diag(Pinf).>0).data .+ 0.0)...]
    Pinf_ = convert(typeof(Pinf), diagm(idx))
    R₀ = eye(size(Pinf_,1))
    for j in 1:size(Pinf_,1)
        if Pinf_[j, j]>0
            R₀[j, j] = 0.0
        end
    end
    R₀ = convert(typeof(Q), R₀)
    KFFiltered(a, P, GrowableArray(R₀*Q*R₀'), GrowableArray(Pinf_))
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
    d::Array{Int,1}
    diffuse::BitArray{1}
end

struct KFCache{FF, FI, DD}
    F⁻¹::FF ## Z*F⁻¹ (m x m)
    F::FI
    Y::DD   ## Data 
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
    #checksizes(Z, H, T, R, Q, α0, P1)
    m = numstates(T)::Int64
    p = nummeasur(Z)::Int64
    att = GrowableArray(a1)
    Ptt = GrowableArray(P1)
    params = KFParms(Z, H, T, R, Q)
    filter = KFFiltered(att, Ptt, Pinf, Q, 1.0)
    smooth = KFSmoothed(Array{Float64}(1,1), Array{Float64}(1,1,1), Array{Float64}(1,1), Array{Float64}(1,1,1))
    inival = KFInitVal(a1, P1, [0], BitArray(m))
    caches = KFCache(GrowableArray(zeros(T)), GrowableArray(zeros(p)))
    LinearStateSpace(params, filter, smooth, inival, caches, [0.0], Ref(1))
end

"""
filter!(cf::LinearStateSpace, y::Vector{Float64})
On-line covariance filter. 
"""
function Base.filter!(cf::LinearStateSpace, y::Vector{Float64})    
    p, m, r = size(cf.p)
    a, P, Pinf, Pstar = currentstate(cf)
    @unpack Z, H, T, R, Q = cf.p
    exact = isexactfilter(Pinf)::Bool
    a′, P′, v, F, F⁻¹, Pstar′, Pinf′, d = filterstep(a, P, Pstar, Pinf, Z, H, T, R, Q, y, Val{exact})
    push!(cf.f.a, a′)
    push!(cf.f.P, P′)
    push!(cf.c.F⁻¹, F⁻¹)
    push!(cf.c.F, F)
    if cf.i.d == cf.t[]
        push!(cf.f.Pinf, Pstar′)
        push!(cf.f.Pstar, Pinf′)
    end
    push!(cf.c.Y, y)
    s = 0.5*logdet(F)+v'*F⁻¹*v
    cf.loglik[1] += - p/2*log(2π) - s[1]
    cf.t[] += 1
end

function Base.filter!(cf::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Matrix{Float64}) where A<:AbstractArray
    TT, PP = size(Y)
    p, m, r = size(cf.p)
    offset = cf.t[]
    @assert p==PP "Inconsistent dimension. Y must be (Tx$p)."
    ## Check whether the last element of cf.f.a is not undef
    @assert isassigned(cf.f.a, length(cf.f.a))
    @assert isassigned(cf.f.P, length(cf.f.P))
    
    resize!(cf.f.a.data, TT+offset)
    resize!(cf.f.P.data, TT+offset)
    resize!(cf.c.F.data, TT+offset)
    resize!(cf.c.Y.data, TT+offset)
    @unpack Z, H, T, R, Q = cf.p      
    a, P = currentstate(cf)
    @inbounds for t = 1:TT
        y = Y[t,:]
        a, P, v, F, F⁻¹, Pstar, Pinf = filterstep(a, P, Z, H, T, R, Q, Y[t])
        cf.f.a[t+offset] = a
        cf.f.P[t+offset] = P
        if cf.i.d[1] == t
            cf.f.Pstar[t+offset] = Pstar
            cf.f.Pinf[t+offset] = Pinf
        end
        cf.c.F⁻¹[t+offset] = [F⁻¹]
        cf.c.F[t+offset] = [F]
        cf.c.Y[t+offset] = [Y[t]]
        s = 0.5*logdet(F)+v'*F⁻¹*v
        cf.loglik[1] += - p/2*log(2π) - s[1]
        cf.t[] += 1
    end
end

function Base.filter!(cf::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Vector{Float64}) where A<:AbstractFloat
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
        a, P, v, F, F⁻¹, Pstar, Pinf = filterstep(a, P, Z, H, T, R, Q, Y[t])        
        cf.f.a[t+offset] = a
        cf.f.P[t+offset] = P
        if cf.i.d[1] == t
            cf.f.Pstar[t+offset] = Pstar
            cf.f.Pinf[t+offset] = Pinf
        end
        cf.c.F⁻¹[t+offset] = [F⁻¹]
        cf.c.F[t+offset] = [F]
        cf.c.Y[t+offset] = [Y[t]]
        s = 0.5*logdet(F)+v'*F⁻¹*v
        cf.loglik[1] += - p/2*log(2π) - s[1]
        cf.t[] += 1
    end       
end


function Base.filter!(cf::LinearStateSpace{KFP} where KFP<:KFParms{A}, y::Float64) where A<:Float64
    a, P = currentstate(cf)
    p, m, r = size(cf.p)
    @unpack Z, H, T, R, Q = cf.p  
    att, Ptt, F⁻¹ = filterstep(Z, H, T, R, Q, y)
    push!(cf.f.a, att)
    push!(cf.f.P, Ptt)
    push!(cf.c.F⁻¹, [F⁻¹])
    push!(cf.c.F, [F])
    push!(cf.c.Y, [y])
    cf.loglik[1] += - p/2*log(2π) - 0.5*abs(F)+v*v/F
    cf.t[] += 1
end

function smooth!(cf::LinearStateSpace)
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

function safeinverse(x)
    try
        (true, inv(x))
    catch 
        #if isa(mess, Base.LinAlg.SingularException)
            (false, x)
        #else
        #    throw("")
        #end
    end
end

function filterstep(a, P, Pstar, Pinf, Z, H, T, R, Q, y, ::Type{Val{false}})    
    ## Non-diffuse step
    v = y - Z*a
    F = Z*P*Z' .+ H    
    Finv = inv(F)
    K = T*P*Z'*invF
    L = T.-K*Z
    a′ = T*a + K*v
    P′ = T*P*L' .+ R*Q*R'
    (a′, P′, v, F, Finv, P′, P′)
end

function filterstep(a, P, Pstar, Pinf, Z, H, T, R, Q, y, ::Type{Val{true}})
    ## Diffuse step
    Finf = Z*Pinf*Z'
    (isFinvertible, Finfinv) = safeinverse(Finf)    
    diffusefilterstep(a, P, Pstar, Pinf, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{isFinvertible}})    
end

function diffusefilterstep(a, P, Pstar, Pinf, Finf, Finfinv, Z, H, T, R, Q, y, Val{true})
    ## Finf is invertible
    v     = y - Z*a
    Kinf  = T*Pinf*Z'*Finfinv
    Linf  = T - Kinf*Z 
    Kstar = (T*Pstar*Z' + Kinf*Fstar)*Finfinv
    Fstar = Z*Pstar*Z' + H
    a′     = T*a + Kinf*v
    Pinf′  = T*Pinf*Linf' 
    Pstar′ = T*Pstar*Linf' - Kinf*Finf*Kstar' + R*Q*R
    P′     = Pstar′
    (a′, P′, v, Finf, Finfinv, Pstar′, Pinf′)
end

function diffusefilterstep(a, P, Pstar, Pinf, Finf, Finfinv, Z, H, T, R, Q, y, Val{false})
    ## Finf is not invertible
    v     = y - Z*a    
    Lstar = T - Kstar*Z 
    Fstar = Z*Pstar*Z' + H
    Kstar = T*Pstar*Z'*inv(Fstar)  
    a′     = T*a + Kstar*v
    Pinf′  = T*Pinf*T' 
    Pstar′ = T*Pstar*Lstar' + R*Q*R
    P′     = Pstar′
    (a′, P′, v, zeros(Finf), Inf.*ones(Finf),  Pstar′, Pinf′)
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
