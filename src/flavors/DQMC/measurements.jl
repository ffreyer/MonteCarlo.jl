################################################################################
### General DQMC Measurements
################################################################################



"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.
"""
struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function GreensMeasurement(mc::DQMC, model)
    o = LightObservable(
        LogBinner(zeros(eltype(mc.s.greens), size(mc.s.greens))),
        "Equal-times Green's function",
        "Observables.jld",
        "Equal-times Green's function"
    )
    GreensMeasurement{typeof(o)}(o)
end
function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens(mc))
end



"""
    BosonEnergyMeasurement(mc::DQMC, model)

Measures the bosnic energy of the given DQMC simulation and model.

Note that this measurement requires `energy_boson(mc, model, conf)` to be
implemented for the specific `model`.
"""
struct BosonEnergyMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function BosonEnergyMeasurement(mc::DQMC, model)
    o = LightObservable(Float64, name="Bosonic Energy", alloc=1_000_000)
    BosonEnergyMeasurement{typeof(o)}(o)
end
function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end


function default_measurements(mc::DQMC, model)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end



################################################################################
### Spin 1/2 Measurements
################################################################################



abstract type SpinOneHalfMeasurement <: AbstractMeasurement end

# Abuse prepare! to verify requirements
function prepare!(m::SpinOneHalfMeasurement, mc::DQMC, model)
    model.flv != 2 && throw(AssertionError(
        "A spin 1/2 measurement requires two (spin) flavors of fermions, but " *
        "the given model has $(model.flv)."
    ))
end
finish!(::SpinOneHalfMeasurement, ::DQMC, model) = nothing



"""
    MagnetizationMeasurement(mc::DQMC, model)

Measures:
* `m1x`, `m1y`, `m1z`: the average onsite magnetization in x, y, or z direction
* `m2x`, `m2y`, `m2z`: the average spin density correlation between any two sites
"""
struct MagnetizationMeasurement{
        OT1x <: AbstractObservable,
        OT1y <: AbstractObservable,
        OT1z <: AbstractObservable,

        OT2x <: AbstractObservable,
        OT2y <: AbstractObservable,
        OT2z <: AbstractObservable,
    } <: SpinOneHalfMeasurement

    m1x::OT1x
    m1y::OT1y
    m1z::OT1z

    m2x::OT2x
    m2y::OT2y
    m2z::OT2z

    # m_total::OT3
end
function MagnetizationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex(T)

    # Magnetizations
    m1x = LightObservable(
        LogBinner([zero(T) for _ in 1:N]),
        "Magnetization x", "Observables.jld", "Magnetization x"
    )
    m1y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N]),
        "Magnetization y", "Observables.jld", "Magnetization y"
    )
    m1z = LightObservable(
        LogBinner([zero(T) for _ in 1:N]),
        "Magnetization z", "Observables.jld", "Magnetization z"
    )

    # Spin density correlation
    m2x = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation x", "Observables.jld", "Spin Density Correlation x"
    )
    m2y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation y", "Observables.jld", "Spin Density Correlation y"
    )
    m2z = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation z", "Observables.jld", "Spin Density Correlation z"
    )


    # This can be calculated after the simulation
    # m_total = LightObservable(
    #     LogBinner([zero(promote_type(T, Ty)) for _ in 1:N]),
    #     "Local Moment", "Observables.jld", "Local Moment" # TODO What is this?
    # )
    MagnetizationMeasurement(
        m1x, m1y, m1z,
        m2x, m2y, m2z
        # m_total
    )
end
function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc)

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    # Magnetization
    # c_{i, up}^† c_{i, down} + c_{i, down}^† c_{i, up}
    mx = [- G[i+N, i] - G[i, i+N]           for i in 1:N]
    # -i [c_{i, up}^† c_{i, down} - c_{i, down}^† c_{i, up}]
    my = [-1im * (G[i, i+N] - G[i+N, i])    for i in 1:N]
    # c_{i, up}^† c_{i, up} - c_{i, down}^† c_{i, down}
    mz = [G[i+N, i+N] - G[i, i]             for i in 1:N]
    push!(m.m1x, mx)
    push!(m.m1y, my)
    push!(m.m1z, mz)

    # Spin Density Correlation
    # = m_i^x * m_j^x
    m2x = mx * mx'
    m2y = my * my'
    m2z = mz * mz'
    push!(m.m2x, m2x)
    push!(m.m2y, m2y)
    push!(m.m2z, m2z)

    # # Local Moment (?)
    # # ⟨m_i^2⟩ = ⟨(m_i^x)^2 + (m_i^y)^2 + (m_i^z)^2⟩
    # m_total = [m2x[i, i] + m2y[i, i] + m2z[i, i] for i in 1:N]
    # push!(m.m_total, m_total)

    # # S = zero(eltype(m.m2y))
    # q = given
    # for i in 1:N, j in 1:N
    #     # TODO: should i, j be lattice indices (multiple dimensions)?
    #     S += exp(1im * q * (i - j)) * (
    #         m1x[i] * m1x[j] + m1y[i] * m1y[j] + m1z[i] * m1z[j]
    #     ) / 4
    # end
    # S /= N
end
function save(m::MagnetizationMeasurement, filename)
    saveobs(m.m1x, filename)
    saveobs(m.m1y, filename)
    saveobs(m.m1z, filename)
    saveobs(m.m2x, filename)
    saveobs(m.m2y, filename)
    saveobs(m.m2z, filename)
end



################################################################################
### Fermion Measurements
################################################################################



abstract type FermionMeasurement <: AbstractMeasurement end

prepare!(::FermionMeasurement, ::DQMC, model) = nothing
finish!(::FermionMeasurement, ::DQMC, model) = nothing


# struct BosonEnergyMeasurement{OT <: AbstractObservable} <: FermionMeasurement
#     obs::OT
# end
# function BosonEnergyMeasurement(mc::DQMC, model)
#     N = nsites(model)
#     T = greenseltype(mc, model)
#
#     # Magnetizations
#     n = LightObservable(
#         LogBinner([zero(T) for _ in 1:N]),
#         "Particle number", "Observables.jld", "Particle number"
#     )
#     BosonEnergyMeasurement(n)
# end
# function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
#     N = nsites(model)
#     G = greens(mc)
#
#     # G[1:N,    1:N]    up -> up section
#     # G[N+1:N,  1:N]    down -> up section
#     # ...
#     # G[i, j] = c_i c_j^†
#
#     n = [2 - G[i, i] - G[i+N, i+N] for _ in 1:N]
#     push!(m.n, n)
#
# end
