@with_kw_noshow mutable struct KaneMeleModel{C<:AbstractCubicLattice} <: Model
    # user mandatory
    L::Int

    # user optional
    mu::Float64 = 0.0
    lambda::Float64 = 1.0
    t::Float64 = 1.0

    # non-user fields
    l::C = HoneycombLattice(L)
    flv::Int = 2
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
    # dtau = mc.p.delta_tau
    # lambda = acosh(exp(m.U * dtau/2))
    # return lambda * sum(hsfield)
    throw(ErrorException(
        "There is no bosonic energy for the Kane-Mele model. " *
        "(It can be solved without running a DQMC simulations.)"
    ))
end

import Base.rand
"""
    rand(mc::DQMC, m::KaneMeleModel)

Draw random HS field configuration.
"""
@inline function rand(mc::DQMC, m::KaneMeleModel)
    # rand(HubbardDistribution, m.l.sites, mc.p.slices)
    @warn(
        "There is no Hubbard Stratonovich field for the Kane-Mele model. " *
        "(It can be solved without running a DQMC simulations.)"
    )
    Int8[]
end

"""
    conftype(::Type{DQMC}, m::KaneMeleModel)

Returns the type of a (Hubbard-Stratonovich field) configuration of the Kane
Mele model.
"""
@inline function conftype(::Type{DQMC}, m::KaneMeleModel)
    @warn(
        "There is no configuration type for the Kane-Mele model. " *
        "(It can be solved without running a DQMC simulations.)"
    )
    Array{Int8, 1}
end

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
    throw(ErrorException(
        "There is no local update defined for the Kane-Mele model. " *
        "(It can be solved without running a DQMC simulations.)"
    ))
end

"""
    accept_local(mc::DQMC, m::KaneMeleModel, i::Int, slice::Int, conf, delta, detratio, delta_E_boson)

Accept a local HS field flip at site `i` and imaginary time slice `slice` of current configuration `conf`.
Arguments `delta`, `detratio` and `delta_E_boson` correspond to output of `propose_local()`
for that flip.
"""
@inline function accept_local!(mc::DQMC, m::HubbardModelAttractive, i::Int, slice::Int, conf::HubbardConf, delta, detratio, delta_E_boson::Float64)
    throw(ErrorException(
        "There is no local update defined for for the Kane-Mele model. " *
        "(It can be solved without running a DQMC simulations.)"
    ))
end


"""
    interaction_matrix_exp!(mc::DQMC, m::KaneMeleModel, result::Matrix, conf, slice::Int, power::Float64=1.) -> nothing

Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline function interaction_matrix_exp!(mc::DQMC, m::KaneMeleModel,
            result::Matrix, conf, slice::Int, power::Float64=1.)
    throw(ErrorException(
        "There is no interaction in the Kane-Mele model. " *
        "(It can be solved without running a DQMC simulations.)"
    ))
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

    T = diagm(0 => fill(-m.mu, 2N))

    # Nearest neighbor hoppings
    @inbounds @views begin
        for src in 1:N
            for nb in 1:size(neighs,1)
                trg = neighs[nb,src]
                T[trg,src] += -m.t
                T[trg+N,src+N] += -m.t
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
                T[trg+N,src+N] -= hcsign * cpsign * m.lambda
            end
        end
    end

    return T
end

# include("observables.jl")
