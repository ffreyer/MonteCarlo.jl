const ZCConf = Array{Int8, 2}
const ZCDistribution = Int8[-1,1]

@with_kw_noshow struct ZCModel <: Model
    L::Int

    # user optional
    # mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t1x::Float64 = t1
    t1y::Float64 = t1
    t1z::Float64 = t1
    t2::Float64 = 1.0
    tperp::Float64 = 1.0
    mu::Float64 = 0.0
    @assert mu == 0.0 "mu is only a compatability hack"

    # non-user fields
    l::TriangularLattice = TriangularLattice(L)
    neighs::Matrix{Int} = neighbors_lookup_table(l)
    flv::Int = 2
end


ZCModel(params::Dict{Symbol, T}) where T = ZCModel(; params...)
ZCModel(params::NamedTuple) = ZCModel(; params...)

# cosmetics
import Base.summary
import Base.show
# Base.summary(model::ZCModel) = "$(model.dims)D attractive Hubbard model"
# Base.show(io::IO, model::ZCModel) = print(io, "$(model.dims)D attractive Hubbard model, L=$(model.L) ($(length(model.l)) sites)")
# Base.show(io::IO, m::MIME"text/plain", model::ZCModel) = print(io, model)




# implement `Model` interface
@inline nsites(m::ZCModel) = length(m.l)
hoppingeltype(::Type{DQMC}, m::ZCModel) = ComplexF64

# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC}, m::ZCModel, nslices::Int) = rand(ZCDistribution, nsites(m), nslices)


function hopping_matrix(mc::DQMC, m::ZCModel)
    # 2N for spin flip
    N = length(m.l)
    l = m.l
    T = zeros(ComplexF64, 2N, 2N)

    for src in 1:N
        # t_perp
        T[src, src + N] += m.tperp
        T[src + N, src] += m.tperp

        # t'
        for trg in l.ext_neighs[:, src]
            l.isAsite[src] != l.isAsite[trg] && error(
                "$src and $trg are on different sublattices, t2"
            )
            # up-up block
            T[trg, src] += m.t2
            # down down block
            T[trg + N, src + N] += -m.t2
        end

        # σ_x
        for trg in l.neighs[1:3:end, src]
            l.isAsite[src] == l.isAsite[trg] && error(
                "$src and $trg are on the same sublattice, σx"
            )
            T[trg, src] += m.t1x
            T[trg + N, src + N] += -m.t1x
        end

        # σ_y
        for trg in l.neighs[2:3:end, src]
            l.isAsite[src] == l.isAsite[trg] && error(
                "$src and $trg are on the same sublattice, σy"
            )
            if l.isAsite[src]
                T[trg, src] += m.t1y * 1im
                T[trg + N, src + N] += -m.t1y * 1im
            else
                T[trg, src] += m.t1y * -1im
                T[trg + N, src + N] += -m.t1y * -1im
            end
        end

        # σ_z
        for trg in l.neighs[3:3:end, src]
            l.isAsite[src] != l.isAsite[trg] && error(
                "$src and $trg are on different sublattices, σz"
            )
            if l.isAsite[src]
                T[trg, src] += m.t1z
                T[trg + N, src + N] += -m.t1z
            else
                T[trg, src] += -m.t1z
                T[trg + N, src + N] += m.t1z
            end
        end
    end

    return T
end


"""
Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline function interaction_matrix_exp!(mc::DQMC, m::ZCModel,
            result::Matrix, conf::ZCConf, slice::Int, power::Float64=1.)
    dtau = mc.p.delta_tau
    # TODO optimize
    # compute this only once
    lambda = acosh(exp(0.5m.U * dtau))
    result .= Diagonal([
        (exp.(sign(power) * lambda * conf[:,slice]))...,
        (exp.(-sign(power) * lambda * conf[:,slice]))...
    ])
    nothing
end


@inline @fastmath @inbounds function propose_local(
        mc::DQMC, m::ZCModel, i::Int, slice::Int, conf::ZCConf
    )

    N = nsites(m)
    G = mc.s.greens
    Δτ = mc.p.delta_tau
    # TODO optimize
    # compute this only once
    α = acosh(exp(0.5Δτ * m.U))

    # TODO optimize
    # unroll matmult?
    ΔE_Boson = -2.0α * conf[i, slice]
    Δ = [exp(ΔE_Boson) - 1 0.0; 0.0 exp(-ΔE_Boson) - 1]
    R = I + Δ * (I - G[i:N:end, i:N:end])
    # Calculate det of 2x2 Matrix
    # det() vs unrolled: 206ns -> 2.28ns
    detratio = R[1, 1] * R[2, 2] - R[1, 2] * R[2, 1]

    return detratio, 0.0, (R, Δ)
end

@inline @inbounds @fastmath function accept_local!(
        mc::DQMC, m::ZCModel, i::Int, slice::Int, conf::ZCConf,
        delta, detratio, _::Float64
    )

    N = nsites(m)
    G = mc.s.greens
    R, Δ = delta

    # TODO optimize
    # BLAS?

    # inverting R in-place, using that R is 2x2
    # speed up: 470ns -> 2.6ns, see matrix_inverse.jl
    inv_div = 1.0 / detratio
    R[1, 2] = -R[1, 2] * inv_div
    R[2, 1] = -R[2, 1] * inv_div
    x = R[1, 1]
    R[1, 1] = R[2, 2] * inv_div
    R[2, 2] = x * inv_div

    IG = -G[:, i:N:end]
    IG[i, 1] += 1.0
    IG[i+N, 2] += 1.0

    R = R * Δ
    IG = IG * R
    # mc.s.greens_temp = IG * (R * Δ) * G[i:N:end, :]
    mul!(mc.s.greens_temp, IG, G[i:N:end, :])
    G .-= mc.s.greens_temp

    conf[i, slice] *= -1

    nothing
end



@inline function energy_boson(mc::DQMC, m::ZCModel, hsfield::ZCConf)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end

include("measurements.jl")