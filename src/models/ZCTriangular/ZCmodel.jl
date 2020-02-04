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
    t1y::ComplexF64 = t1
    t1z::Float64 = t1
    t2::Float64 = 1.0
    tperp::Float64 = 1.0
    delta::Float64 = 0.0
    mu::Float64 = 0.0

    # temporary stuff to avoid allocations
    Gslice::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 2, 2(L^2))
    IG::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 2(L^2), 2)
    IGR::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 2(L^2), 2)
    # IG::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 2, 2(L^2))
    # IGR::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 2, 2(L^2))
    R::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 2, 2)
    #@assert mu == 0.0 "mu is only a compatability hack"

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
Base.summary(m::ZCModel) = "$(m.L) × $(m.L) ZC Model"
Base.show(io::IO, m::ZCModel) = print(io, "$(m.L) × $(m.L) ZC Model")
function Base.show(io::IO, ::MIME"text/plain", m::ZCModel)
    print(io, "ZCModel (Triangular Lattice, $(m.L) × $(m.L) Sites)\n")
    println(io, "\tH = t σⁱ (c↑c↑ - c↓c↓)          | t  = $(m.t1) (Nearest Neighbor hopping) ($(m.t1x), $(m.t1y), $(m.t1z))")
    println(io, "\t  + t' (c↑c↑ - c↓c↓)            | t' = $(m.t2) (Third Nearest Neighbor hopping)")
    println(io, "\t  + t⟂ (c↑c↓ + c↓c↑)            | t⟂ = $(m.tperp) (Interlayer hopping)")
    println(io, "\t  + μ (n↑ - n↓)                 | μ  = $(m.mu) (chemical potential)")
    println(io, "\t  + Δ σᶻ (n↑ - n↓)              | Δ  = $(m.delta)")
    print(io, "\t  - U (n↑ - 0.5)(n↓ - 0.5)      | U  = $(m.U) (repulsive Coulomb interaction)")
end



# implement `Model` interface
@inline nsites(m::ZCModel) = length(m.l)
hoppingeltype(::Type{DQMC}, m::ZCModel) = ComplexF64

# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC}, m::ZCModel, nslices::Int) = rand(ZCDistribution, nsites(m), nslices)


@bm function hopping_matrix(mc::DQMC, m::ZCModel)
    # 2N for spin flip
    N = length(m.l)
    l = m.l
    T = zeros(ComplexF64, 2N, 2N)

    for src in 1:N
        # mu
        T[src, src] += m.mu
        T[src+N, src+N] -= m.mu

        # Δ
        if l.isAsite[src]
            T[src, src] += m.delta
            T[src+N, src+N] -= m.delta
        else
            T[src, src] -= m.delta
            T[src+N, src+N] += m.delta
        end
        # if l.isAsite[src]
        #     T[src, src] += m.delta
        #     T[src+N, src+N] += m.delta
        # else
        #     T[src, src] -= m.delta
        #     T[src+N, src+N] -= m.delta
        # end

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
@inline @bm function interaction_matrix_exp!(mc::DQMC, m::ZCModel,
            result::Matrix, conf::ZCConf, slice::Int, power::Float64=1.)

    # TODO maybe optimize this? ~25ns
    dtau = mc.p.delta_tau
    lambda = acosh(exp(0.5m.U * dtau))

    result .= zero(eltype(result))
    N = div(size(result, 1), 2)
    @inbounds for i in 1:N
        result[i, i] = exp(sign(power) * lambda * conf[i, slice])
    end
    @inbounds for i in N+1:2N
        result[i, i] = exp(-sign(power) * lambda * conf[i-N, slice])
    end

    nothing
end


@inline @fastmath @inbounds @bm function propose_local(
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

@inline @inbounds @fastmath @bm function accept_local!(
        mc::DQMC, m::ZCModel, i::Int, slice::Int, conf::ZCConf,
        delta, detratio, _::Float64
    )

    @timeit_debug "accept_local (pre)" begin
        N = nsites(m)
        G = mc.s.greens
        R, Δ = delta
    end

    # TODO optimize
    # BLAS?

    # inverting R in-place, using that R is 2x2
    # speed up: 470ns -> 2.6ns, see matrix_inverse.jl
    # blazingly fast
    @timeit_debug "accept_local (inversion)" begin
        inv_div = 1.0 / detratio
        R[1, 2] = -R[1, 2] * inv_div
        R[2, 1] = -R[2, 1] * inv_div
        x = R[1, 1]
        R[1, 1] = R[2, 2] * inv_div
        R[2, 2] = x * inv_div
    end

    # decently fast
    @timeit_debug "accept_local (m.IG, m.R)" begin
        @views mul!(m.IG, G[:, i:N:end], -1.0)
        @views m.IG[i, 1] += 1.0
        @views m.IG[i+N, 2] += 1.0
        mul!(m.R, R, Δ)
        mul!(m.IGR, m.IG, m.R)
    end

    # TODO SSSLLOOOWWW
    @timeit_debug "accept_local (finalize computation)" begin
        # mc.s.greens_temp = IG * (R * Δ) * G[i:N:end, :]
        mul!(mc.s.greens_temp, m.IGR, G[i:N:end, :])

        G .-= mc.s.greens_temp
        # mul!(G, mc.s.greens_temp, -1.0)

        conf[i, slice] *= -1
    end

    # @timeit_debug "accept_local (finalize computation)" begin
    #     # mc.s.greens_temp = IG * (R * Δ) * G[i:N:end, :]
    #     mul!(mc.s.greens_temp, m.IG, G[i:N:end, :])
    #     # @views copyto!(m.Gslice, G[i:N:end, :])
    #
    #     mc.s.tmp .= G .- mc.s.greens_temp
    #     # @. G -=
    #     # mul!(G, mc.s.greens_temp, -1.0)
    #
    #     conf[i, slice] *= -1
    # end

    # if !(mc.s.tmp ≈ G)
    #     display(mc.s.tmp .- G)
    #     error("REEEE")
    # end

    nothing
end



@inline @bm function energy_boson(mc::DQMC, m::ZCModel, hsfield::ZCConf)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end

#     save_model(filename, model, entryname)
#
# Save (minimal) information necessary to reconstruct the given `model` in a
# jld-file `filename` under group `entryname`.
#
# By default the full model object is saved. When saving a simulation, the
# entryname defaults to `MC/Model`.
function save_model(file::JLD.JldFile, model::ZCModel, entryname::String)
    @info "saving"
    write(file, entryname * "/VERSION", 2)
    write(file, entryname * "/type", typeof(model))
    write(file, entryname * "/data/L", model.L)
    write(file, entryname * "/data/U", model.U)
    write(file, entryname * "/data/t1", model.t1)
    write(file, entryname * "/data/t2", model.t2)
    write(file, entryname * "/data/tperp", model.tperp)
    write(file, entryname * "/data/mu", model.mu)
    nothing
end

#     load_model(data, ::Type{Model})
#
# Loads a model from a given `data` dictionary produced by `JLD.load(filename)`.
# The second argument can be used for dispatch between different models.
function load_model(data, ::Type{ZCModel})
    if data["VERSION"] == 0
        return data["data"]
    elseif data["VERSION"] in (1, 2)
        data["VERSION"] == 1 && (@warn "mu not in saved data")
        return ZCModel(
            L = data["data"]["L"],
            U = data["data"]["U"],
            t1 = data["data"]["t1"],
            t2 = data["data"]["t2"],
            tperp = data["data"]["tperp"],
            mu = get(data["data"], "mu", 0.0)
        )
    else
        V = data["VERSION"]
        error("VERSION not recognized ($V)")
    end
end

include("measurements.jl")