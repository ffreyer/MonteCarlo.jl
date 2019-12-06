const ZrSiSConf = Array{Int8, 2}
const ZrSiSDistribution = Int8[-1,1]

@with_kw_noshow struct ZrSiSModel <: Model
    L::Int

    # user optional
    # mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t::Float64 = 1.0
    delta::Float64 = 1.1
    tperp::Float64 = 1.0
    mu::Float64 = 0.0
    @assert mu == 0.0 "mu is only a compatability hack"

    # non-user fields
    l::SquareLattice = SquareLattice(L)
    neighs::Matrix{Int} = neighbors_lookup_table(l)
    flv::Int = 2
end


ZrSiSModel(params::Dict{Symbol, T}) where T = ZrSiSModel(; params...)
ZrSiSModel(params::NamedTuple) = ZrSiSModel(; params...)

# cosmetics
import Base.summary
import Base.show
# Base.summary(model::ZrSiSModel) = "$(model.dims)D attractive Hubbard model"
# Base.show(io::IO, model::ZrSiSModel) = print(io, "$(model.dims)D attractive Hubbard model, L=$(model.L) ($(length(model.l)) sites)")
# Base.show(io::IO, m::MIME"text/plain", model::ZrSiSModel) = print(io, model)




# implement `Model` interface
@inline nsites(m::ZrSiSModel) = 2length(m.l)
hoppingeltype(::Type{DQMC}, m::ZrSiSModel) = Float64

# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC}, m::ZrSiSModel, nslices::Int) = rand(ZrSiSDistribution, nsites(m), nslices)


function hopping_matrix(mc::DQMC, m::ZrSiSModel)
    # 2N for spin flip
    N = length(m.l)
    l = m.l
    T = zeros(ComplexF64, 4N, 4N)

    for src in 1:N
        for trg in l.neighs[:, src]
            # A up
            T[trg, src] += m.t
            # A down
            T[trg+N, src+N] += m.t
            # B up
            T[trg+2N, src+2N] -= m.t
            # B down
            T[trg+3N, src+3N] -= m.t

            # diagonal part
            T[src, src] += m.delta
            T[src+N, src+N] += m.delta
            T[src+2N, src+2N] -= m.delta
            T[src+3N, src+3N] -= m.delta
        end
    end

    return T
end


"""
Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline function interaction_matrix_exp!(mc::DQMC, m::ZrSiSModel,
            result::Matrix, conf::ZrSiSConf, slice::Int, power::Float64=1.)
    dtau = mc.p.delta_tau
    # TODO optimize
    # compute this only once
    lambda = acosh(exp(0.5m.U * dtau))
    # (n_A - 0.5)(n_B - 0.5)
    # result .= Diagonal([
    #     (exp.(sign(power) * lambda * conf[:,slice]))...,
    #     (exp.(-sign(power) * lambda * conf[:,slice]))...
    # ])
    N = length(m.l)
    result .= Diagonal([
        exp.(sign(power) * lambda * conf[1:N, slice])...,
        exp.(-sign(power) * lambda * conf[N+1:end, slice])...,
        exp.(-sign(power) * lambda * conf[1:N, slice])...,
        exp.(sign(power) * lambda * conf[N+1:end, slice])...
    ])
    nothing
end


@inline @fastmath @inbounds function propose_local(
        mc::DQMC, m::ZrSiSModel, i::Int, slice::Int, conf::ZrSiSConf
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
        mc::DQMC, m::ZrSiSModel, i::Int, slice::Int, conf::ZrSiSConf,
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



@inline function energy_boson(mc::DQMC, m::ZrSiSModel, hsfield::ZrSiSConf)
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
function save_model(file::JLD.JldFile, model::ZrSiSModel, entryname::String)
    @info "saving"
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(model))
    write(file, entryname * "/data/L", model.L)
    write(file, entryname * "/data/U", model.U)
    write(file, entryname * "/data/t1", model.t1)
    write(file, entryname * "/data/t2", model.t2)
    write(file, entryname * "/data/tperp", model.tperp)
    nothing
end

#     load_model(data, ::Type{Model})
#
# Loads a model from a given `data` dictionary produced by `JLD.load(filename)`.
# The second argument can be used for dispatch between different models.
function load_model(data, ::Type{ZrSiSModel})
    if data["VERSION"] == 0
        return data["data"]
    elseif data["VERSION"] == 1
        return ZrSiSModel(
            L = data["data"]["L"],
            U = data["data"]["U"],
            t1 = data["data"]["t1"],
            t2 = data["data"]["t2"],
            tperp = data["data"]["tperp"]
        )
    else
        V = data["VERSION"]
        error("VERSION not recognized ($V)")
    end
end

# include("measurements.jl")
