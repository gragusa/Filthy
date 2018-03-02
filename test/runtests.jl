using Filthy
using Base.Test
using StaticArrays
using BenchmarkTools

# m = 2   # Dinemsion of state
# p = 2   # Dinemsion of measurements
# q = 2   # dimension of \eta

# Tr = randn(m,m)
# Qr = randn(q,q)
# QQ = Qr'*Qr/10
# const T = SMatrix{m,m}(Tr*diagm(linspace(0.5,0.99,m))/Tr)
# const R = @SMatrix eye(m,q)
# const Q = SMatrix{q,q}(QQ)

# const Z = @SMatrix randn(p,p)
# const H = @SMatrix eye(p,p)



# cf = Filthy.CovarianceFilter(Z, H, T, R, Q, zeros(2), eye(2))
# Y, a = simulate(cf, 200)

# @btime begin
#     cf = Filthy.CovarianceFilter(Z, H, T, R, Q, zeros(2), eye(2))
#     for i in 1:200
#         filter!(cf, Y[:, i])
#     end
# end

# using Plots
# gr()

# Plots.plot(1:200, a[1,:])
# plot!(kf.s.att[1:end,1])

## Nile Data

## In the Nile data example,

## y = α + ϵ
## α' = α + η

## In terms of Matrices

## Y    = Zt αt + ϵt, ϵt ∼ N(0, Ht)
## αt+1 = Tt αt + Rt ηt, ηt ~ N(0, Qt)

nile = readcsv("Nile.csv")
Zn = fill(1.0, (1,1))
Hn = fill(15099., (1,1))
Rn = fill(1.0, (1,1))
Tn = fill(1.0, (1,1))
Qn = fill(1469.1, (1,1))

Zs = SMatrix{1,1}(Zn)
Hs = SMatrix{1,1}(Hn)
Rs = SMatrix{1,1}(Rn)
Ts = SMatrix{1,1}(Tn)
Qs = SMatrix{1,1}(Qn)

cf_scalar = Filthy.CovarianceFilter(1.0, 15099., 1.0, 1.0, 1469.1, 0.0, 10.^7.)
cf_matrix = Filthy.CovarianceFilter(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.^7., (1,1)))
cf_static = Filthy.CovarianceFilter(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.^7., (1,1))))

@btime begin 
    cf_scalar = Filthy.CovarianceFilter(1.0, 15099., 1.0, 1.0, 1469.1, 0.0, 10.^7.)
    for j in eachindex(nile)
        filter!(cf_scalar, nile[j])
    end
end

@btime begin 
    cf_static = Filthy.CovarianceFilter(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.^7., (1,1))))
    for j in eachindex(nile)
        filter!(cf_static, [nile[j]])
    end
end

@btime begin 
    cf_matrix = Filthy.CovarianceFilter(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.^7., (1,1)))
    for j in eachindex(nile)
        filter!(cf_matrix, [nile[j]])
    end
end

@test all(cf_scalar.s.att[:,1] .≈ cf_matrix.s.att[:,1])
@test all(cf_static.s.att[:,1] .≈ cf_matrix.s.att[:,1])



p = 2
m = 2
r = 2



Z = @SMatrix eye(p,p)
H = @SMatrix eye(p,p)


T = SMatrix{m,m}([0.92 0.2; -.2 -0.9])
R = I
Q = SMatrix{m,m}(eye(m))




cf = CovarianceFilter(Z, H, T, R, Q, SVector{2}(zeros(2)), SMatrix{2,2}(eye(2)))

srand(12234)
Y, a = simulate(cf, 200)

#writecsv("Y.csv", Y')

cf = CovarianceFilter(Z, H, T, R, Q, SVector{2}(zeros(2)), SMatrix{2,2}(eye(2)))
    for i in 1:200
        filter!(cf, Y[:, i])
    end
end

@test convert(Vector,  cf.s.att[10]) .≈ [0.46446643122638098244; 2.13857431622820914896]
@test convert(Vector, cf.s.att[100]) .≈ [1.20622036437758350935; -0.39600752142009676415]

tt = @btime begin
cf = CovarianceFilter(Z, H, T, R, Q, SVector{2}(zeros(2)), SMatrix{2,2}(eye(2)))
    for i in 1:200
        filter!(cf, Y[:, i])
    end
end





# using Plots
# gr()
# p1 = Plots.plot(1:size(Y,2), cf.s.att[2:end,1], color = :darkblue)
# p2 = Plots.plot(1:size(Y,2), cf.s.att[2:end,2], color = :darkblue)
# plot!(p1, a[1,:])
# plot!(p2, a[2,:])
# plot!(nile, color = :black)
# plot!(cf.s.att[2:end,1]-1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)
# plot!(cf.s.att[2:end,1]+1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)







# using Plots
# gr()
# Plots.plot(1:length(nile), cf.s.att[2:end,1], color = :darkblue)
# plot!(nile, color = :black)
# plot!(cf.s.att[2:end,1]-1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)
# plot!(cf.s.att[2:end,1]+1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)

# Plots.plot(1:length(nile), cf.s.Ptt[2:end,1], color = :darkblue)
# Plots.
