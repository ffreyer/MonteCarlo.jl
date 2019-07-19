const HubbardConf = Array{Int8, 2}
const HubbardDistribution = Int8[-1,1]

@with_kw_noshow mutable struct KaneMeleModel{C<:AbstractCubicLattice} <: Model
    # user mandatory
    L::Int

    # user optional
    mu::Float64 = 0.0
    lambda::Float64 = 1.0
    t::Float64 = 1.0
    U::Float64 = 1.0
    @assert U >= 0.0 "U must be positive"

    # non-user fields
    l::C = HoneycombLattice(L)
    flv::Int = 1 # 2 spins, but symmetric
end



"""
    KaneMeleModel(kwargs::Dict{String, Any})

"""
function KaneMeleModel(kwargs::Dict{String, Any})
    symbol_dict = Dict([Symbol(k) => v for (k, v) in kwargs])
    KaneMeleModel(; symbol_dict...)
end

# cosmetics
import Base.summary
import Base.show
Base.summary(model::KaneMeleModel) = "2D Kane Mele Model"
function Base.show(io::IO, model::KaneMeleModel)
    print(io, "2D Kane Mele Model, L=$(model.L) ($(model.l.sites) sites)")
end
Base.show(io::IO, m::MIME"text/plain", model::KaneMeleModel) = print(io, model)

# methods
"""
    energy(m::KaneMeleModel, hsfield)

Calculate bosonic part of the energy for configuration `hsfield`.
"""
@inline function energy_boson(m::KaneMeleModel, hsfield)
    dtau = mc.p.delta_tau
    alpha = acosh(exp(m.U * dtau/2))
    return alpha * sum(hsfield)
end

import Base.rand
"""
    rand(mc::DQMC, m::KaneMeleModel)

Draw random HS field configuration.
"""
@inline function rand(mc::DQMC, m::KaneMeleModel)
    rand(HubbardDistribution, m.l.sites, mc.p.slices)
end

"""
    conftype(::Type{DQMC}, m::KaneMeleModel)

Returns the type of a (Hubbard-Stratonovich field) configuration of the Kane
Mele model.
"""
@inline conftype(::Type{DQMC}, m::KaneMeleModel) = KaneMeleHubbardConf


"""
    greenseltype(::Type{DQMC}, m::KaneMeleModel)

Returns the element type of the Green's function.
"""
@inline greenseltype(::Type{DQMC}, m::KaneMeleModel) = Float64

"""
    propose_local(dqmc, model::KaneMeleModel, i::Int, slice, conf, E_boson::Float64) -> detratio, delta_E_boson, delta

Propose a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
"""
@inline function propose_local(mc::DQMC, m::KaneMeleModel, i::Int, slice::Int, conf, E_boson::Float64)
    # see for example dos Santos (2002)
    greens = mc.s.greens
    dtau = mc.p.delta_tau
    alpha = acosh(exp(m.U * dtau/2))

    delta_E_boson = -2. * alpha * conf[i, slice]
    gamma = exp(delta_E_boson) - 1
    detratio = (1 + gamma * (1 - greens[i,i]))^2 # squared because of two spin sectors.

    return detratio, delta_E_boson, gamma
end

"""
    accept_local(mc::DQMC, m::KaneMeleModel, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)

Accept a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
Arguments `delta`, `detratio` and `delta_E_boson` correspond to output of `propose_local()`
for that flip.
"""
@inline function accept_local!(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, delta, detratio, delta_E_boson::Float64)
    greens = mc.s.greens
    gamma = delta

    u = -greens[:, i]
    u[i] += 1.
    # OPT: speed check, maybe @views/@inbounds
    greens .-= kron(u * 1. /(1 + gamma * u[i]), transpose(gamma * greens[i, :]))
    conf[i, slice] *= -1
    nothing
end


"""
    interaction_matrix_exp!(mc::DQMC, m::KaneMeleModel, result::Matrix, conf, slice::Int, power::Float64=1.) -> nothing

Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline function interaction_matrix_exp!(mc::DQMC, m::KaneMeleModel,
            result::Matrix, conf, slice::Int, power::Float64=1.)
    dtau = mc.p.delta_tau
    alpha = acosh(exp(m.U * dtau/2))
    result .= spdiagm(0 => exp.(sign(power) * alpha * conf[:,slice]))
    nothing
end

"""
	hopping_matrix(mc::DQMC, m::KaneMeleModel)

Calculates the hopping matrix \$T_{i, j}\$ where \$i, j\$ are
site indices.
"""
function hopping_matrix(mc::DQMC, m::KaneMeleModel)
    N = m.l.sites
    neighs = m.l.neighs # row = up, right, down, left; col = siteidx
    NNNs = m.l.NNNs # row = up, right, down, left; col = siteidx

    T = diagm(0 => fill(-m.mu, N))

    # Nearest neighbor hoppings
    @inbounds @views begin
        for src in 1:N
            for nb in 1:size(neighs,1)
                trg = neighs[nb,src]
                T[trg,src] += -m.t
                # T[trg+N,src+N] += -m.t
            end
        end
    end

    # Next Nearest neighbor hoppings
    # Term: iλ ∑_NNN(i, j) c_i† dot(σ⃗, d⃗_ik × d⃗_jk) c_j
    # where d are lattice vectors (directions i -> k <- j, k a NN of i, j)
    # and we assume the result of the cross product to be normalized,
    # i.e. we pull lattice related prefactors into λ
    @inbounds @views begin
        for i in 1:2m.l.L, j in 1:2m.l.L
            src = m.l.lattice[i, j]
            for nb in 1:size(NNNs,1)
                # cross product gives ±e_z, alternating clockwise and w/ sublattice
                # total sign is:
                cpsign = iseven(i+j+nb) ? -1.0 : 1.0
                # sign change due to i (h.c.):
                hcsign = nb > 3 ? -1.0 : 1.0
                trg = neighs[nb,src]
                T[trg,src] += hcsign * cpsign * m.lambda
                # T[trg+N,src+N] -= hcsign * cpsign * m.lambda
            end
        end
    end

    return T
end

# include("observables.jl")
