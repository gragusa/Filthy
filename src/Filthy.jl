module Filthy

using StaticArrays
using GrowableArrays
using Parameters
using ForwardDiff
using MathProgBase
using LinearAlgebra

include("types.jl")
include("filter.jl")
include("mpb.jl")
include("methods.jl")


function SSMOptim(m, Z, CH, T, R, CQ, a, P, a1!, P1!, P1inf, Pstar, Y)
    t = prototypical(m)
    CS = ForwardDiff.pickchunksize(length(t))
    p = SSMOptimPar(Z, CH, T, R, CQ, a, P, a1!, P1!, P1inf, Pstar)
    jcfg = ForwardDiff.GradientConfig(nothing, t, ForwardDiff.Chunk(CS))
    hcfg = ForwardDiff.HessianConfig(nothing, t, ForwardDiff.Chunk(CS))
    M1 = map(a->Array{ForwardDiff.Dual{Nothing,Float64,CS},2}(size(a)...), (Z, CH, T, R, CQ))
    a1 = Array{ForwardDiff.Dual{Nothing,Float64,CS},1}(length(a))
    M2 = map(a->Array{ForwardDiff.Dual{Void,Float64,CS},2}(size(a)...), (P, P1inf, Pstar))
    d = SSMOptimParDual(M1..., a1, M2..., jcfg, hcfg)
    d = SSMOptim(m, p, d, Y)
    d
end

function get_parameters(s::SSMOptim, ::Type{T}) where T
    (s.p.Z::Array{T, 2},
     s.p.H::Array{T, 2},
     s.p.T::Array{T, 2},
     s.p.R::Array{T, 2},
     s.p.Q::Array{T, 2},
     s.p.a1::Array{T, 1},
     s.p.P1::Array{T, 2},
     s.p.P1inf::Array{T, 2},
     s.p.Pstar::Array{T, 2})
end

function get_parameters(s::SSMOptim, ::Type{T}) where T<:ForwardDiff.Dual
    (s.d.Z::Array{T, 2},
     s.d.H::Array{T, 2},
     s.d.T::Array{T, 2},
     s.d.R::Array{T, 2},
     s.d.Q::Array{T, 2},
     s.d.a1::Array{T, 1},
     s.d.P1::Array{T, 2},
     s.p.P1inf::Array{Float64, 2},
     s.p.Pstar::Array{Float64, 2})
end


function KFFiltered(a::T, P, Pinf::Nothing) where T<:Real
    KFFiltered([a], [P], [P], [P], [1], [0], BitArray([1]))
end

function KFFiltered(a::T, P, Pinf::Nothing) where T<:AbstractArray
    KFFiltered(GrowableArray(a), GrowableArray(P), [], [], [1], [1],  BitArray([1]))
end

function KFFiltered(a1, P1, P1inf::AbstractMatrix)
    ## Check positive element on the diagonal
    ## To Do: Check that P1inf is diagonal with diagonal entry > 0
    hasended = all(diag(P1inf) .== 0) ? 1 : 0
    a1′, P1′, Pstar = setupdiffusematrices(a1, P1, P1inf)
    KFFiltered(GrowableArray(a1′), GrowableArray(P1′), GrowableArray(Pstar), GrowableArray(P1inf), [1], [1],  BitArray([hasended]))
end

function KFFiltered(a1, P1, P1inf::AbstractFloat)
    ## Check positive element on the diagonal
    @assert P1inf > 0 "Initial value to diffuse part should be > 0"
    KFFiltered(GrowableArray(a1), GrowableArray(P1), GrowableArray(0.0), GrowableArray(P1inf), [1], [1],  BitArray([0]))
end


function KFCache(T::AbstractMatrix, p::Int, m::Int)
    KFCache(GrowableArray(zeros(p,p)),
            GrowableArray(zeros(p,p)),
            GrowableArray(zeros(p)),
            Array{Float64}(undef, p),
            Array{Float64}(undef, p),
            Array{Float64}(undef, p, p),  # C
            Array{Float64}(undef, m, p),  # M
            Array{Float64}(undef, p, p),  # ZM
            Array{Float64}(undef, m, p),  # TM
            Array{Float64}(undef, m, p),  # K
            Array{Float64}(undef, m, m),  # KZ
            Array{Float64}(undef, m, m),  # KZp
            Array{Float64}(undef, m, m),  # TPL
            Array{Float64}(undef, m, m),   # L
            Array{Float64}(undef, p),     # a′
            Array{Float64}(undef, m, m)   # P′
            )
end

function KFCache(T::AbstractFloat, p::Int, m::Int)
    @assert p==1 "Something wrong"
    KFCache(GrowableArray(zero(T)),
            GrowableArray(zero(T)),
            GrowableArray(zero(eltype(T))),
            Array{Float64}(undef, 1),
            Array{Float64}(undef, 1),
            Array{Float64}(undef, 1, 1),  # C
            Array{Float64}(undef, 1, 1),  # M
            Array{Float64}(undef, 1, 1),  # ZM
            Array{Float64}(undef, 1, 1),  # TM
            Array{Float64}(undef, 1, 1),  # K
            Array{Float64}(undef, 1, 1),  # KZ
            Array{Float64}(undef, 1, 1),  # KZp
            Array{Float64}(undef, 1, 1),  # TPL
            Array{Float64}(undef, 1, 1),  # L
            Array{Float64}(undef, 1),  # a
            Array{Float64}(undef, 1, 1)  # P
            )
end



function LinearStateSpace(Z::ZT, H::HT, T::TT, R::RT, Q::QT, a1::AZ, P1::PZ; P1inf::Union{Nothing, PZ}=nothing) where {ZT, HT, TT, RT, QT, AZ, PZ}
    @assert isa(P1, TT) "P1 must be of type $(typeof(TT))"
    if isa(Z, AbstractMatrix)
        @assert isa(a1, AbstractVector) "a1 must be of type Vector"
    elseif isa(Z, AbstractFloat)
        @assert isa(a1, AbstractFloat) "a1 must be of type AbstractFloat"
    end
    #checksizes(Z, H, T, R, Q, α0, P1)
    m = numstates(T)::Int64
    p = nummeasur(Z)::Int64
    params = KFParms(Z, H, T, R, Q)
    filter = KFFiltered(a1, P1, P1inf)
    smooth = KFSmoothed(Array{Float64}(undef, 1,1), Array{Float64}(undef, 1,1,1),
                        Array{Float64}(undef, 1,1), Array{Float64}(undef, 1,1,1))
    inival = KFInitVal(a1, P1)
    caches = KFCache(T, p, m)
    LinearStateSpace(params, filter, smooth, inival, caches, [0.0], Ref(1))
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

function ssm_obj_fun(s::SSMOptim, theta)
    Z, H, T, R, Q, a1, P1, P1inf, Pstar = get_parameters(s, eltype(theta))
    ## H and Q are the chol
    remask!(s.m, (Z, H, T, R, Q), theta)
    try
        s.p.a1!(s, a1)
        s.p.P1!(s, P1)
        fastloglik(Z, H*H', T, R, Q*Q', a1, P1, P1inf, Pstar, s.Y)
    catch
        -Inf
    end
end

function ssm_obj_fun_scalar(s::OptimSSMScalar, theta)
    @unpack a1, P1, P1inf, Pstar, idx, par, y = s
    x = similar(theta, length(par))
    copy!(x, par)
    x[idx] = theta
    -fastloglikscalar(x[1], exp(x[2]), x[3], x[4], exp(x[5]), a1, P1, P1inf, Pstar, y)
end
"""
fit(Z::A, CH::A, T::A, R::A, CQ::A, a1::AZ, P1::PZ, P1inf::PW, Y, start, lower, upper) where {A<:AbstractMatrix, AZ<:Union{Function, AbstractVector}, PZ<:Union{Function, AbstractMatrix}, PW}

"""
function fit(::Type{LinearStateSpace}, Z::A, CH::A, T::A, R::A, CQ::A, a1!::Function,
            P1!::Function, P1inf::PINF, Y, x0, lower, upper, solver = IpoptSolver()) where {A<:AbstractMatrix, PINF}
    @assert islowertriangular(CH) "CH must be lower triangular"
    @assert islowertriangular(CQ) "CQ must be lower triangular"
    Z₀, CH₀, T₀, R₀, CQ₀ = map(obj-> convert(Matrix, obj), (copy(Z), copy(CH), copy(T), copy(R), copy(CQ)))
    mask = Masked((Z₀, CH₀, T₀, R₀, CQ₀))
    allfixed(mask) && (print_with_color(:cyan, "All State Space parameters are defined."); return nothing)
    ## Setup and do maximum likelihood
    ## Copy matrices to Matrix{Float64}
    a1m = Array{eltype(Z), 1}(size(T,1))
    P1m = Array{eltype(Z), 2}(size(R, 1), size(R, 1))
    a, P, Pstar = setupdiffusematrices(a1m, P1m, P1inf)
    #fitobj = OptimSSM(mask, Z₀, CH₀, T₀, R₀, CQ₀, a, P,  P1inf, Pstar, Y)
    s = SSMOptim(mask, Z₀, CH₀, T₀, R₀, CQ₀, a, P, a1!, P1!, P1inf, Pstar, Y)
    m = MathProgBase.NonlinearModel(solver)
    MathProgBase.loadproblem!(m, length(x0), 0, lower, upper, Array{Float64}(0), Array{Float64}(0),  :Max, s)
    MathProgBase.setwarmstart!(m, x0)
    MathProgBase.optimize!(m)
    sol = MathProgBase.getsolution(m)
    remask!(mask, (Z₀, CH₀, T₀, R₀, CQ₀), sol)
    ZZ = SMatrix{size(Z)...}(Z₀)
    HH = SMatrix{size(CH)...}(CH₀*CH₀')
    TT = SMatrix{size(T)...}(T₀)
    RR = SMatrix{size(R₀)...}(R₀)
    QQ = SMatrix{size(CQ)...}(CQ₀*CQ₀')
    aa = SVector{length(a)}(s.p.a1)
    PP = SMatrix{size(P)...}(s.p.P1)
    PPinf = SMatrix{size(P)...}(s.p.P1inf)
    LinearStateSpace(ZZ, HH, TT, RR, QQ, aa, PP, P1inf=PPinf)
end

function fit!(ss::LinearStateSpace, Y, x0, lower, upper, solver)
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




export LinearStateSpace, simulate, filter!, smooth!, fit, reset!, get_parameters

end # module
