@noinline function currentstate(cf::LinearStateSpace) 
    idx = cf.t[]::Int
    (cf.f.a[idx], cf.f.P[idx], cf.f.Pinf[idx], cf.f.Pstar[idx])
end

isexactfilter(Pinf) = any(Pinf.data.!=0)

numstates(a::AbstractArray) = size(a,1)
numstates(a::AbstractFloat) = 1

nummeasur(a::AbstractArray) = size(a,1)
nummeasur(a::AbstractFloat) = 1


Base.size(cf::LinearStateSpace) = size(cf.p)

Base.size(p::KFParms{P}) where P<:AbstractFloat = (1,1,1)
Base.size(p::KFParms{P}) where P<:AbstractArray = (size(p.Z)..., size(p.Q, 1))





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
    