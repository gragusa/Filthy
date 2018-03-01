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
cf = Filthy.CovarianceFilter(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.^7., (1,1)))
cfs = Filthy.CovarianceFilter(1.0, 15099., 1.0, 1.0, 1469.1, 0.0, 10.^7.)

for j in eachindex(nile)
    filter!(cf, [nile[j]])
end

for j in eachindex(nile)
    filter!(cfs, nile[j])
end

@test all(cfs.s.att[:,1] .≈ cf.s.att[:,1])



# using Plots
# gr()
# Plots.plot(1:length(nile), cf.s.att[2:end,1], color = :darkblue)
# plot!(nile, color = :black)
# plot!(cf.s.att[2:end,1]-1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)
# plot!(cf.s.att[2:end,1]+1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)

# Plots.plot(1:length(nile), cf.s.Ptt[2:end,1], color = :darkblue)
# Plots.
