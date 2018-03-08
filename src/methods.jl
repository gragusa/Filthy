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


function updatelik!(ss, v, F, F⁻¹, ispd) 
    p, m, r = size(ss)
    s = ispd ? 0.5*(logdet(F)) : s = 0.5*(logdet(F) + v'F⁻¹*v)
    ss.loglik[1] += -p/2*log(2π) - first(s)
end

function updatelik!(ss, v, F, F⁻¹) 
    @show ss.t[]
    p, m, r = size(ss)
    s = 0.5*(logdet(F) + v'F⁻¹*v)
    ss.loglik[1] += -p/2*log(2π) - first(s)
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
        try
            (true, inv(x))
        catch 
            (false, x)
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
    msg = p == 1 ? "a Vector" : "a (Tx$p) Matrix"
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



# function checksizes(Z::P, H::P, T::P, R::P, Q::P, Y::Matrix{Float64}) where P<:AbstractArray{Float64, 2}
#     rZ, cZ = size(Z)
#     rH, cH = size(H)
#     rT, cT = size(T)
#     rR, cR = size(R)
#     rQ, cQ = size(Q)
#     ## Time series dimension -> column of Y
#     p, T = size(Y)

#     @assert p == rZ  "The dimension of Y does not agree with the dimension of Z"
#     @assert p == rH  "The dimension of Y does not agree with the dimension of H"

#     @assert rH == cH "H must be a square matrix"
#     @assert rQ == cQ "Q must be a square matrix"
#     @assert rT == cT "T must be a square matrix"

#     @assert rT == rR "The dimension of T does not agree with the dimension of R"
#     @assert rT == rQ "The dimension of T does not agree with the dimension of R"

#     @assert cZ == rT "The dimension of Z does not agree with the dimension of T"
#     @assert cR == rQ "The dimension of R does not agree with the dimension of Q"

# end
    