const ZCConf = Array{Int8, 2}
const ZCDistribution = Int8[-1,1]

@with_kw_noshow struct ZCModel <: Model
    L::Int

    # user optional
    mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."
    t1::Float64 = 1.0
    t2::Float64 = 1.0
    tperp::Float64 = 1.0

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


# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC}, m::ZCModel, nslices::Int) = rand(ZCDistribution, nsites(m), nslices)


function hopping_matrix(mc::DQMC, m::ZCModel)
    N = length(m.l)
    l = m.l
    T = zeros(ComplexF64, 2N, 2N)

    # Nearest neighbor hoppings
    @inbounds @views begin
        for src in 1:N
            T[src, src] -= m.mu
            T[src+N, src+N] -= m.mu

            # spin up <- spin down
            T[src, src+N] += m.tperp
            # spin down <- spin up, sign from h.c.
            T[src+N, src] += m.tperp

            # sigma_z:  up -> up with +,  down -> down with -, H_down comes with -
            trg = l.neighs[3, src]
            T[trg, src] += m.t1
            T[trg+N, src+N] += m.t1
            # h.c.
            T[src, trg] += m.t1
            T[src+N, trg+N] += m.t1

            # sigma_x:  no sign, switch up and down, H_down gives -
            trg = l.neighs[1, src]
            T[trg+N, src] += m.t1
            T[trg, src+N] += -m.t1
            # h.c.
            T[src+N, trg] += m.t1
            T[src, trg+N] += -m.t1

            # sigma_y:  -i up -> down, i down -> up, H_down -
            trg = l.neighs[2, src]
            T[trg+N, src] += -1im * m.t1
            T[trg, src+N] += -1im * m.t1
            # h.c
            T[src+N, trg] += -1im * m.t1
            T[src, trg+N] += -1im * m.t1

            for nb in 1:3
                trg = l.ext_neighs[nb, src]
                T[trg, src] += m.t2
                T[trg+N, src+N] += -m.t2
                # h.c.
                T[src, trg] += m.t2
                T[src+N, trg+N] += -m.t2
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
    lambda = acosh(exp(m.U * dtau/2))
    # result .= spdiagm(0 => exp.(sign(power) * lambda * conf[:,slice]))
    result .= spdiagm(0 => [
        (exp.(sign(power) * lambda * conf[:,slice]))...,
        (exp.(sign(power) * lambda * conf[:,slice]))...
    ])
    nothing
end


@inline function propose_local(mc::DQMC, m::ZCModel, i::Int, slice::Int, conf::ZCConf)
    N = length(m.l)
    G = mc.s.greens
    Δτ = mc.p.delta_tau
    α = acosh(exp(0.5Δτ * m.U))

    ΔE_Boson = -2.0α * conf[i, slice]
    Δ = exp(ΔE_Boson) - 1
    R = I + Δ * (I - G[i:N:end, i:N:end])
    detratio = det(R)

    return detratio, ΔE_Boson, (R, Δ)
end

@inline function accept_local!(mc::DQMC, m::ZCModel, i::Int, slice::Int, conf::ZCConf, delta, detratio, ΔE_boson::Float64)
    N = length(m.l)
    G = mc.s.greens
    R, Δ = delta

    # TODO optimize
    # BLAS?
    @inbounds begin
        IG = -G[:, i:N:end]
        IG[i, 1] += 1.0
        IG[i+N, 2] += 1.0
        mc.s.greens_temp = IG * inv(R) * Δ * G[i:N:end, :]
        G .-= mc.s.greens_temp

        conf[i, slice] *= -1
    end

    nothing
end



@inline function energy_boson(mc::DQMC, m::ZCModel, hsfield::ZCConf)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end

include("observables.jl")
