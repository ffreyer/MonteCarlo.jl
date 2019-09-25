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



"""
    MagnetizationMeasurement(mc::DQMC, model)

Measures:
* `x`, `y`, `z`: the average onsite magnetization in x, y, or z direction
"""
struct MagnetizationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
end
function MagnetizationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

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

    MagnetizationMeasurement(m1x, m1y, m1z)
end
function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc)
    IG = I - G

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
    push!(m.x, mx)
    push!(m.y, my)
    push!(m.z, mz)
end



"""
    SpinDensityCorrelationMeasurement(mc::DQMC, model)

Measures:
* `x`, `y`, `z`: the average spin density correlation between any two sites
"""
struct SpinDensityCorrelationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
end
function MagnetizationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

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

    MagnetizationMeasurement(x, y, z)
end
function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    # NOTE
    # these maybe wrong, maybe IG -> G
    # Spin Density Correlation
    m2x = zeros(eltype(G), N, N)
    m2y = zeros(eltype(G), N, N)
    m2z = zeros(eltype(G), N, N)
    for i in 1:N, j in 1:N
        m2x[i, j] = (
            IG[i+N, i] * IG[j+N, j] + IG[j+N, i] * G[i+N, j] +
            IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
            IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] +
            IG[i, i+N] * IG[j, j+N] + IG[j, i+N] * G[i, j+N]
        )
        m2y[i, j] = (
            - IG[i+N, i] * IG[j+N, j] - IG[j+N, i] * G[i+N, j] +
              IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
              IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] -
              IG[i, i+N] * IG[j, j+N] - IG[j, i+N] * G[i, j+N]
        )
        m2z[i, j] = (
            IG[i, i] * IG[j, j] + IG[j, i] * G[i, j] -
            IG[i, i] * IG[j+N, j+N] - IG[j+N, i] * G[i+N, j] -
            IG[i+N, i+N] * IG[j, j] - IG[j, i+N] * G[i, j+N] +
            IG[i+N, i+N] * IG[j+N, j+N] + IG[j+N, i+N] * G[i+N, j+N]
        )
    end
    push!(m.x, m2x)
    push!(m.y, m2y)
    push!(m.z, m2z)
end



struct PairingCorrelationMeasurement{
        OT <: AbstractObservable,
        T
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::Matrix{T}
end
function PairingCorrelationMeasurement(mc::DQMC, model)
    T = eltype(mc.s.greens)
    N = nsites(model)

    obs = LightObservable(
        LogBinner(zeros(T, N, N)),
        "Equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "Equal time pairing correlation matrix"
    )
    temp = zeros(T, N, N)

    PairingCorrelationMeasurement(obs, temp)
end
function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc)
    N = nsites(model)
    push!(
        m.obs,
        G[1:N, 1:N] .* G[N+1:2N, N+1:2N] - G[1:N, N+1:2N] .* G[N+1:2N, 1:N]
    )
end




################################################################################
### Fermion Measurements
################################################################################



abstract type FermionMeasurement <: AbstractMeasurement end


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
