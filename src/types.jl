struct KFParms{Zt, Ht, Tt, Rt, Qt}
    Z::Zt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
end

mutable struct KFFiltered{AT, PT, PI, PIX}
    a::AT
    P::PT
    Pstar::PI
    Pinf::PIX
    d::Array{Int,1}
    flagF::Array{Int, 1}
    hasdiffended::BitArray{1}
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

struct KFCache{FF, FI, DD, T}
    F⁻¹::FF ##  (p x p)
    F::FI   ##  (p x p)
    Y::DD   ##  (p)
    v::Array{T, 1}
    vp::Array{T, 1}
    C::Array{T, 2}
    M::Array{T, 2}
    ZM::Array{T, 2}
    TM::Array{T, 2}
    K::Array{T, 2}
    KZ::Array{T, 2}
    KZp::Array{T, 2}
    L::Array{T, 2}
    TPL::Array{T, 2}
    ap::Array{T, 1}
    Pp::Array{T, 2}
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


struct SSMOptimPar{T, V, F1, F2, M}
    Z::T
    H::T
    T::T
    R::T
    Q::T
    a1::V
    P1::T
    a1!::F1
    P1!::F2
    P1inf::M
    Pstar::M
end

struct SSMOptimParDual{T, S, M, J, H}
    Z::Array{T,2}
    H::Array{T,2}
    T::Array{T,2}
    R::Array{T,2}
    Q::Array{T,2}
    a1::Array{S,1}
    P1::Array{T,2}
    P1inf::M
    Pstar::M
    jcfg::J
    hcfg::H
end

struct SSMOptimCache{T}
    v::Array{T, 1}
    vp::Array{T, 1}
    C::Array{T, 2}
    M::Array{T, 1}
    ZM::Array{T, 1}
    TM::Array{T, 1}
    K::Array{T, 1}
    KZ::Array{T, 1}
    KZp::Array{T, 1}
    L::Array{T, 1}
end

struct SSMOptimCacheDual{T}
    v::Array{T, 1}
    vp::Array{T, 1}
    C::Array{T, 2}
    M::Array{T, 1}
    ZM::Array{T, 1}
    TM::Array{T, 1}
    K::Array{T, 1}
    KZ::Array{T, 1}
    KZp::Array{T, 1}
    L::Array{T, 1}
end

struct SSMOptim{M, P, D, F} <:MathProgBase.AbstractNLPEvaluator
    m::M
    p::P
    d::D
    Y::Matrix{F}
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

struct Masked{FREEIDX, FIXEDIDX, FREEITR, FIXEDITR, P}
    freeindexes::FREEIDX
    fixedindexes::FIXEDIDX
    freeitr::FREEITR
    fixeditr::FIXEDITR
    parent::P
    count::Array{Int64, 1}
end
