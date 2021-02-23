#=============
Filter methods
==============#

#=-
A. Entry points
-=#

function Base.filter!(ss::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::B) where {A<:AbstractArray, B<:AbstractArray}
    (T, p, m, r, offset) = Filthy.checkfilterdataconsistency(ss, Y)
    #rY = reshape(Y, (p, T))
    Filthy.resizecontainer!(ss, T+offset)
    for j in 1:size(Y, 2)
        onlinefilter_set!(ss, Y[:,j], offset)
    end
end

function Base.filter!(ss::LinearStateSpace{KFP} where KFP<:KFParms{A}, Y::Vector) where A<:AbstractFloat
    (T, p, m, r, offset) = Filthy.checkfilterdataconsistency(ss, Y)
    #rY = reshape(Y, (p, T))
    Filthy.resizecontainer!(ss, T+offset)
    for j in 1:length(Y)
        onlinefilter_set!(ss, Y[j], offset)
    end
    nothing
end

function onlinefilter_set!(ss::LinearStateSpace, y, offset::Int)
    hasdiffended = first(ss.f.hasdiffended)::Bool
    onlinefilterstep_set!(ss, y, offset, Val{!hasdiffended})
    ss.t[] += 1
    nothing
end

function onlinefilter!(ss::LinearStateSpace, y)
    hasdiffended = first(ss.f.hasdiffended)::Bool
    onlinefilterstep!(ss, y, Val{!hasdiffended})
    ss.t[] += 1
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
    #@show a′, P′, v, F, F⁻¹
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

#=-
B. Actual steps
-=#
function filterstep(a, P, Pinf, Pstar, Z, H, T, R, Q, y)
    Finf = Z*Pinf*Z'
    (Fflag, Finfinv) = Filthy.safeinverse(Finf)
    diffusefilterstep(a, P, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, Val{Fflag})
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



# #=-
# C. In-place filter step
# -=#

# # Standard filtering equation
# function filterstep!(a′, P′, v, vp, C, M, ZM, TM, K, KZ, KZp, TPL, L, a::AA, P::PP, Z, H, T, R, Q, y) where {AA, PP}
#     ## v  is (p x 1)
#     ## v′ is (p x 1)
#     ## C is (p x p)
#     ## M  is (m x p)
#     ## ZM is (p x p)
#     ## TM is (m x p)
#     ## K  is (m x p)
#     ## KZ is (m x m)
#     ## KZ′ is (m x m)
#     ## TPL is (m x m)
#     ## L  is (m x m)

#     ## [v]
#     A_mul_B!(vp, Z, a)
#     @inbounds @simd for i in eachindex(v)
#         v[i] = y[i] - vp[i]
#     end
#     ## [M]
#     A_mul_Bt!(M, P, Z)
#     ## [ZPZ'] -> ZM
#     A_mul_B!(ZM, Z, M)
#     ## [ZPZ' + H] -> C
#     @inbounds @simd for i in eachindex(ZM)
#         C[i] = ZM[i] + H[i]
#     end
#     Cinv = inv(factorize(C))
#     ## [T*M] -> TM
#     A_mul_B!(TM, T, M)
#     ## [T*M*Cinv] -> K
#     A_mul_B!(K, TM, Cinv)
#     ## [KZ] -> KZ
#     A_mul_B!(KZ, K, Z)
#     ## [T-KZ] -> L
#     @inbounds @simd for i in eachindex(L)
#         L[i] = T[i] - KZ[i]
#     end
#     ## [T*a] -> Ta
#     A_mul_B!(a′, T, a)
#     A_mul_B!(vp, K, v)
#     @inbounds @simd for i in eachindex(a′)
#         a′[i] = a′[i] + vp[i]
#     end
#     ## [TPL']
#     A_mul_B!(KZp, T, P)
#     A_mul_Bt!(KZ, KZp, L)
#     ## [RQR]
#     A_mul_B!(K, R, Q)
#     A_mul_Bt!(TM, K, R)
#     ## [TPL'+RQR] -> P
#     @inbounds @simd for i in eachindex(P′)
#         P′[i] = KZ[i] + TM[i]
#     end
#     (a′, P′, v, C, Cinv)
# end

# ## diffuse filter version
# ## F>0 and invertible
# function filterstep!(a′, P′, v, vp, C, M, ZM, TM, K, KZ, KZp, L, a::AA, P::PP, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{1}}) where {AA, PP}
#     ## v  is (p x 1)
#     ## v′ is (p x 1)
#     ## C is (p x p)
#     ## M  is (m x p)
#     ## ZM is (p x p)
#     ## TM is (m x p)
#     ## K  is (m x p)
#     ## KZ is (m x m)
#     ## KZ′ is (m x m)
#     ## L  is (m x m)

#     ## [v]
#     A_mul_B!(vp, Z, a)
#     @inbounds @simd for i in eachindex(v)
#         v[i] = y[i] - vp[i]
#     end
#     ## [M]
#     A_mul_Bt!(M, Pstar, Z)
#     ## [ZPZ'] -> ZM
#     A_mul_B!(ZM, Z, M)
#     # [ZPZ' + H] -> C (Fstar)
#     @inbounds @simd for i in eachindex(ZM)
#         C[i] = ZM[i] + H[i]
#     end
#     ## [T*M] -> TM
#     A_mul_B!(TM, T, M)
#     ## [T*M*Cinv] -> K (Kinf)
#     A_mul_B!(K, TM, Cinv)
#     ## Kstar
#     A_mul_B!(KZ, K, C)
#     @inbounds @simd for i in eachindex(Cinv)
#         Cinv[i] = TM[i] + KZ[i]
#     end
#     A_mul_B!(ZM, Cinv, Finfinv)
#     ## [KZ] -> KZ
#     A_mul_B!(KZ, K, Z)
#     ## [T-KZ] -> L
#     @inbounds @simd for i in eachindex(L)
#         L[i] = T[i] - KZ[i]
#     end

#     ## [T*a] -> Ta
#     A_mul_B!(a′, T, a)
#     A_mul_B!(vp, K, v)
#     @inbounds @simd for i in eachindex(a′)
#         a′[i] = a′[i] + vp[i]
#     end
#     ## [TPL']
#     A_mul_B!(KZp, T, Pinf)
#     A_mul_Bt!(KZ, KZp, L)  ## Pinf'
#     ## [RQR]
#     A_mul_B!(K, R, Q)
#     A_mul_Bt!(TM, K, R)
#     ## [TPstarL'+RQR] ->
#     A_mul_B!(KZp, T, Pstar)
#     A_mul_Bt!(TPL, KZp, L)


#     @inbounds @simd for i in eachindex(P′)
#         P′[i] = KZ[i] + TPL[i] + TM[i] ## Pstar'
#     end

#     (a′, P′, v, Fstar, Fstarinv, Kz, P′, 0)
# end

# function filterstep!(a′, P′, v, vp, C, M, ZM, TM, K, KZ, KZp, L, a::AA, P::PP, Pinf, Pstar, Finf, Finfinv, Z, H, T, R, Q, y, ::Type{Val{0}}) where {AA, PP}
#     ## v  is (p x 1)
#     ## v′ is (p x 1)
#     ## C is (p x p)
#     ## M  is (m x p)
#     ## ZM is (p x p)
#     ## TM is (m x p)
#     ## K  is (m x p)
#     ## KZ is (m x m)
#     ## KZ′ is (m x m)
#     ## TPL is (m x m)
#     ## L  is (m x m)

#     ## [v]
#     A_mul_B!(vp, Z, a)
#     @inbounds @simd for i in eachindex(v)
#         v[i] = y[i] - vp[i]
#     end
#     ## [M]
#     A_mul_Bt!(M, Pstar, Z)
#     ## [ZPZ'] -> ZM
#     A_mul_B!(ZM, Z, M)
#     # [ZPZ' + H] -> C (Fstar)
#     @inbounds @simd for i in eachindex(ZM)
#         C[i] = ZM[i] + H[i]
#     end
#     Fstarinv = inv(factor(C))

#     ## Kstar
#     A_mul_B!(TM, T, M)
#     ## [T*M*Cinv] -> K
#     A_mul_B!(K, TM, Fstarinv)


#     ## Lstar


#     ## [T*a] -> Ta
#     A_mul_B!(a′, T, a)
#     A_mul_B!(vp, K, v)
#     @inbounds @simd for i in eachindex(a′)
#         a′[i] = a′[i] + vp[i]
#     end
#     ## [TPL']
#     A_mul_B!(KZp, T, Pinf)
#     A_mul_Bt!(KZ, KZp, L)  ## Pinf'
#     ## [RQR]
#     A_mul_B!(K, R, Q)
#     A_mul_Bt!(TM, K, R)
#     ## [TPstarL'+RQR] ->
#     A_mul_B!(KZp, T, Pstar)
#     A_mul_Bt!(TPL, KZp, L)


#     @inbounds @simd for i in eachindex(P′)
#         P′[i] = KZ[i] + TPL[i] + TM[i] ## Pstar'
#     end

#     (a′, P′, v, Fstar, Fstarinv, Kz, P′, 0)
# end


#=-
D. Storage
-=#


function storepush!(ss, a::T, P, v, F, F⁻¹, Pinf, Pstar, y, flag) where T<:Real
    push!(ss.f.a, a...)
    push!(ss.f.P, P...)
    push!(ss.c.F⁻¹, F⁻¹)
    push!(ss.c.F, F)
    push!(ss.c.Y, y)
    push!(ss.f.Pinf, Pinf)
    push!(ss.f.Pstar, Pstar)
    ss.f.d[1] += 1
    push!(ss.f.flagF, flag)
    nothing
end


function storepush!(ss, a::T, P, v, F, F⁻¹, Pinf, Pstar, y, flag) where T<:AbstractArray
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


function storepush!(ss, a::T, P, v, F, F⁻¹, y) where T<:Real
    push!(ss.f.a, a...)
    push!(ss.f.P, P...)
    push!(ss.c.F⁻¹, F⁻¹)
    push!(ss.c.F, F)
    push!(ss.c.Y, y)
    nothing
end

function storepush!(ss, a::T, P, v, F, F⁻¹, y) where T<:AbstractArray
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
