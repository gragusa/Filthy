using Filthy
using Test
using StaticArrays
using BenchmarkTools
using CSV
#=
Nile data
-==========================#

nile = CSV.read("Nile.csv", allowmissing=:none)

nile = Matrix(nile)


# Values from Durbin and Koopman
# Zn = fill(1.0, (1,1))
# Hn = fill(15099., (1,1))
# Rn = fill(1.0, (1,1))
# Tn = fill(1.0, (1,1))
# Qn = fill(1469.1, (1,1))

# Values from MLE (inexact, P = 1e07)
Zn = fill(1.0, (1,1))
Hn = fill(15099.82, (1,1))
Rn = fill(1.0, (1,1))
Tn = fill(1.0, (1,1))
Qn = fill(1468.487, (1,1))



Zs = SMatrix{1,1}(Zn)
Hs = SMatrix{1,1}(Hn)
Rs = SMatrix{1,1}(Rn)
Ts = SMatrix{1,1}(Tn)
Qs = SMatrix{1,1}(Qn)




#=
Constructors
=----------=#
@testset "Constructors (inexact).........." begin
    @test isa(LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1469.1, 0.0, 10.0^7), LinearStateSpace)
    @test isa(LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.0^7, (1,1))), LinearStateSpace)
    @test isa(LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), SMatrix{1,1}(fill(10.0^7, (1,1)))), LinearStateSpace)

    @test_throws AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, 0.0, fill(10.0^7, (1,1)))
    @test_throws AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], 10.0^7)
    @test_throws AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, 0.0, 10.0^7)

    @test_throws AssertionError LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), rand(1,1))
    @test_throws AssertionError LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.0^7, (1,1))))
    @test_throws AssertionError LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), rand(1,1))

    @test_throws AssertionError LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1469.1, [0.0], 10.0^7)
    @test_throws AssertionError LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1469.1, 0.0, [10.0^7])
    @test_throws AssertionError LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1469.1, [0.0], [10.0^7])
end

@testset "Constructors (exact).........." begin
    @test isa(LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, 0.0, 10.0^7, P1inf = 1.0), LinearStateSpace)
    @test isa(LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.0^7, (1,1)), P1inf = fill(1.0, (1,1))), LinearStateSpace)
    @test isa(LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), SMatrix{1,1}(fill(10.0^7, (1,1)))), LinearStateSpace)

    ss = LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), SMatrix{1,1}(fill(10.0^7, (1,1))), P1inf = SMatrix{1,1}([1.0]))
    @test isa(ss.f.Pinf[1], StaticArrays.SArray{Tuple{1,1},Float64,2,1})
    @test isa(ss.f.Pstar[1], StaticArrays.SArray{Tuple{1,1},Float64,2,1})
    @test ss.f.Pinf[1][1] == 1.0
    @test ss.f.Pstar[1][1] == 0.0

    ss = LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.0^7, (1,1)), P1inf = fill(1.0, (1,1)))
    @test isa(ss.f.Pinf[1], Array{Float64, 2})
    @test isa(ss.f.Pstar[1], Array{Float64, 2})
    @test ss.f.Pinf[1][1] == 1.0
    @test ss.f.Pstar[1][1] == 0.0

    ss = LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, 0.0, 10.0^7, Pinf = 1.0)
    @test isa(ss.f.Pinf[1], Float64)
    @test isa(ss.f.Pstar[1], Float64)
    @test ss.f.Pinf[1][1] == 1.0
    @test ss.f.Pstar[1][1] == 0.0

    # @test_throws AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, 0.0, fill(10.0^7, (1,1)))
    # @test_throws AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], 10.0^7)
    # @test_throws AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, 0.0, 10.0^7)

    # @test_throws AssertionError LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), rand(1,1))
    # @test_throws AssertionError LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.0^7, (1,1))))
    # @test_throws AssertionError LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), rand(1,1))

    # @test_throws AssertionError LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, [0.0], 10.^7.)
    # @test_throws AssertionError LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, 0.0, [10.^7.])
    # @test_throws AssertionError LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, [0.0], [10.^7.])
    # @test_throw  AssertionError LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.^7., (1,1)))
end


#=
Online version
=#

ss_scalar = LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, 0.0, 10.0^7)
for j in eachindex(nile)
    Filthy.onlinefilter!(ss_scalar, nile[j])
end

ss_matrix = LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.0^7, (1,1)))
for j in eachindex(nile)
    Filthy.onlinefilter!(ss_matrix, nile[j,:])
end

ss_static = LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), SMatrix{1,1}(fill(10.^7., (1,1))))
for j in eachindex(nile)
    Filthy.onlinefilter!(ss_static, nile[j,:])
end

@testset "Filter correctness (online)....." begin
    @test isa(ss_matrix.f.a[1], Vector)
    @test isa(ss_matrix.f.P[1], Matrix)

    @test isa(ss_scalar.f.a[1], Float64)
    @test isa(ss_scalar.f.P[1], Float64)

    @test isa(ss_static.f.a[1], StaticArray)
    @test isa(ss_static.f.P[1], StaticMatrix)

    @test ss_scalar.f.a[1] == ss_matrix.f.a[1][1]
    @test ss_scalar.f.a[10] == ss_matrix.f.a[10][1]
    @test ss_scalar.f.a[101] == ss_matrix.f.a[101][1]

    @test ss_scalar.f.a[1] == ss_static.f.a[1][1]
    @test ss_scalar.f.a[10] == ss_static.f.a[10][1]
    @test ss_scalar.f.a[101] == ss_static.f.a[101][1]

    @test ss_scalar.c.F[1]   == ss_static.c.F[1][1]
    @test ss_scalar.c.F[10]  == ss_static.c.F[10][1]
    @test ss_scalar.c.F[101] == ss_static.c.F[101][1]

    @test ss_scalar.f.a[1] == 0.0
    @test ss_scalar.f.a[10] ≈ 1171.22228
    @test ss_scalar.f.a[101] ≈ 798.3871578

    @test ss_scalar.c.F[2] ≈ 10015099.822227377445
    @test ss_scalar.c.F[11] ≈ 20635.5495
    @test ss_scalar.c.F[101] ≈ 20599.87957

end


#=
Vectorized version
=#

ss_scalar = LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, 0.0, 10.0^7.)
Filthy.filter!(ss_scalar, vec(nile))

ss_matrix = LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.0^7, (1,1)))
Filthy.filter!(ss_matrix, nile')

ss_static = LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), SMatrix{1,1}(fill(10.^7., (1,1))))
Filthy.filter!(ss_static, nile')

@testset "Filter correctness (offline) [inexact]..." begin
    @test ss_scalar_vec.f.a[1] == ss_scalar.f.a[1]
    @test ss_scalar_vec.f.a[10] == ss_scalar.f.a[10]
    @test ss_scalar_vec.f.a[100] == ss_scalar.f.a[100]
    @test ss_matrix_vec.f.a[1] == ss_matrix.f.a[1]
    @test ss_matrix_vec.f.a[10] == ss_matrix.f.a[10]
    @test ss_matrix_vec.f.a[100] == ss_matrix.f.a[100]
    @test ss_static_vec.f.a[1] == ss_static.f.a[1]
    @test ss_static_vec.f.a[10] == ss_static.f.a[10]
    @test ss_static_vec.f.a[100] == ss_static.f.a[100]
    @test ss_scalar_vec.f.a[1] == 0.0
    @test ss_scalar_vec.f.a[10] ≈ 1171.22228
    @test ss_scalar_vec.f.a[101] ≈ 798.3871578
    @test ss_scalar_vec.c.F[2] ≈ 10015099.822227377445
    @test ss_scalar_vec.c.F[11] ≈ 20635.5495
    @test ss_scalar_vec.c.F[101] ≈ 20599.87957
end



# Values from MLE (inexact, P = 1)
Zn = fill(1.0, (1,1))
Hn = fill(15098.654334841132368, (1,1))
Rn = fill(1.0, (1,1))
Tn = fill(1.0, (1,1))
Qn = fill(1469.1632513366273542, (1,1))

Zs = SMatrix{1,1}(Zn)
Hs = SMatrix{1,1}(Hn)
Rs = SMatrix{1,1}(Rn)
Ts = SMatrix{1,1}(Tn)
Qs = SMatrix{1,1}(Qn)



ss = LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SVector{1}(0.0), SMatrix{1,1}(fill(0., (1,1))), P1inf = SMatrix{1,1}([1.0]))
Filthy.filter!(ss, nile')

@testset "Filter correctness (offline) [exact]....." begin
    @test ss.f.Pinf[1][1] == 1.0
    @test ss.f.P[1][1] == 0.0
    @test ss.f.Pstar[1][1] == 0.0
    @test ss.f.Pstar[2][1] ≈ 16568.307
    @test ss.f.P[2][1][1] ≈ 16568.307
    @test ss.f.d==2
    @test ss.f.hasdiffended[1] == true
    @test_throws BoundsError ss.f.Pstar[ss.f.d+2]
    @test ss.loglik[1] ≈ -633.46456
end

#=
Smoothing
---------=#
Filthy.smooth!(ss)
@testset "State smoother" begin
    ss.s.a[1]
end



# @btime begin
#     ss_scalar = LinearStateSpace(1.0, 15099.82, 1.0, 1.0, 1468.487, 0.0, 10.^7.)
#     Filthy.filter!(ss_scalar, vec(nile))
# end



# # stat_online = @btime begin
# #     cf_static = Filthy.CovarianceFilter(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.^7., (1,1))))
# #     for j in eachindex(nile)
# #         filter!(cf_static, [nile[j]])
# #     end
# # end

# # stat_vec = @btime begin
# #     cf_static = Filthy.CovarianceFilter(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.^7., (1,1))))
# #     filter!(cf_static, nile)
# # end


# # @btime begin
# #     cf_matrix = Filthy.CovarianceFilter(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.^7., (1,1)))
# #     for j in eachindex(nile)
# #         filter!(cf_matrix, [nile[j]])
# #     end
# # end

# # @test all(cf_scalar.s.att[:,1] .≈ cf_matrix.s.att[:,1])
# # @test all(cf_static.s.att[:,1] .≈ cf_matrix.s.att[:,1])


# # using Plots
# # gr()
# # Plots.plot(1:length(nile), cf_static.f.a[2:end,1], color = :darkblue)
# # plot!(nile, color = :black)
# # plot!(cf_static.s.a, color = :red)
# # plot!(cf.s.att[2:end,1]-1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)
# # plot!(cf.s.att[2:end,1]+1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)




# p = 2
# m = 2
# r = 2

# Z = SMatrix{p,p}([0.2 0.1; 0.4 0.15])
# H = SMatrix{p,p}([1.2 0.1; 0.1 1.15])
# T = SMatrix{m,m}([0.92 0.2; -.2 -0.9])
# R = I
# Q = SMatrix{m,m}(eye(m))
# a1 = SVector{2}(zeros(2))
# P1 = SMatrix{2,2}(1000.*eye(2))
# Pinf = SMatrix{2,2}(eye(2))

# ss = Filthy.LinearStateSpace(Z, H, T, R, Q, a1, P1, Pinf = Pinf)
# Filthy.filter!(ss, Y)

# ss2 = Filthy.LinearStateSpace(Z, H, T, R, Q, a1, P1)
# Filthy.filter!(ss2, Y)



# srand(12234)
# Y, a = simulate(ss, 200)

#writecsv("Y.csv", Y')


# # for i in 1:200
# #     filter!(cf, Y[:, i])
# # end


# # @test all(convert(Vector, cf.f.a[10]) .≈ [0.46446643122638098244; 2.13857431622820914896])
# # @test all(convert(Vector, cf.f.a[100]) .≈ [1.20622036437758350935; -0.39600752142009676415])

# # cf = CovarianceFilter(Zp, Hp, Tp, Rp, Qp, SVector{2}(zeros(2)), SMatrix{2,2}(eye(2)))
# # filter!(cf, Y')
# # @test all(convert(Vector, cf.f.a[10]) .≈ [0.46446643122638098244; 2.13857431622820914896])
# # @test all(convert(Vector, cf.f.a[100]) .≈ [1.20622036437758350935; -0.39600752142009676415])

# # tt = @btime begin
# #     cf = CovarianceFilter(Zp, Hp, Tp, Rp, Qp, SVector{2}(zeros(2)), SMatrix{2,2}(eye(2)))
# #     for i in 1:size(Y,2)
# #         filter!(cf, Y[:, i])
# #     end
# # end

# # vv = @btime begin
# #     cf = CovarianceFilter(Zp, Hp, Tp, Rp, Qp, SVector{2}(zeros(2)), SMatrix{2,2}(eye(2)))
# #     filter!(cf, Y')
# # end




# # # using Plots
# # # gr()
# # # p1 = Plots.plot(1:size(Y,2), cf.s.att[2:end,1], color = :darkblue)
# # # p2 = Plots.plot(1:size(Y,2), cf.s.att[2:end,2], color = :darkblue)

# # p1 = Plots.plot(1:size(Y,2), cf.f.a[2:end,1], color = :darkblue)
# # p2 = Plots.plot(1:size(Y,2), cf.f.a[2:end,2], color = :darkblue)

# # # plot!(p1, a[1,:])
# # # plot!(p2, a[2,:])
# # # plot!(nile, color = :black)
# # # plot!(cf.s.att[2:end,1]-1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)
# # # plot!(cf.s.att[2:end,1]+1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)







# # # using Plots
# # # gr()
# # # Plots.plot(1:length(nile), cf.s.att[2:end,1], color = :darkblue)
# # # plot!(nile, color = :black)
# # # plot!(cf.s.att[2:end,1]-1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)
# # # plot!(cf.s.att[2:end,1]+1.64*sqrt.(cf.s.Ptt[2:end,1]), color = :darkblue, linestyle = :dot)

# # # Plots.plot(1:length(nile), cf.s.Ptt[2:end,1], color = :darkblue)
# # # Plots.





# # @btime begin
# #     cf = Filthy.CovarianceFilter(Z, H, T, R, Q, zeros(2), eye(2))
# #     for i in 1:200
# #         filter!(cf, Y[:, i])
# #     end
# # end

# # using Plots
# # gr()

# # Plots.plot(1:200, a[1,:])
# # plot!(kf.s.att[1:end,1])

# ## Nile Data

# ## In the Nile data example,

# ## y = α + ϵ
# ## α' = α + η

# ## In terms of Matrices

# ## Y    = Zt αt + ϵt, ϵt ∼ N(0, Ht)
# ## αt+1 = Tt αt + Rt ηt, ηt ~ N(0, Qt)



# # scalar_online = @btime begin
# #     cf_scalar = LinearStateSpace(1.0, 15099., 1.0, 1.0, 1469.1, 0.0, 10.^7.)
# #     for j in eachindex(nile)
# #         Filthy.onlinefilter!(cf_scalar, nile[j])
# #     end
# # end

# # stat_online = @btime begin
# #     cf_static = LinearStateSpace(Zs, Hs, Ts, Rs, Qs, SMatrix{1,1}([0.0]), SMatrix{1,1}(fill(10.^7., (1,1))))
# #     for j in eachindex(nile)
# #         Filthy.onlinefilter!(cf_static, nile[j,:])
# #     end
# # end

# # matrix_online = @btime begin
# #     ss_matrix = LinearStateSpace(Zn, Hn, Tn, Rn, Qn, [0.0], fill(10.^7., (1,1)))
# #     for j in eachindex(nile)
# #         Filthy.onlinefilter!(ss_matrix, nile[j,:])
# #     end
# # end
