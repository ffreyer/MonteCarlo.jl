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
        "G"
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



"""
    ChargeDensityCorrelationMeasurement(mc::DQMC, model)

Measures the charge density correlation matrix `⟨nᵢnⱼ⟩`.
"""
struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable
    } <: AbstractMeasurement
    obs::OT
    temp::Matrix
end
function ChargeDensityCorrelationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    obs = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Charge density wave correlations", "Observables.jld", "CDC"
    )
    ChargeDensityCorrelationMeasurement(obs, [zero(T) for _ in 1:N, __ in 1:N])
end
function measure!(m::ChargeDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    # TODO
    # implement spinflavors(model)
    # then get N from size(model.l) / spinflavors(model) ?
    N = nsites(model)
    flv = model.flv
    G = greens(mc)
    IG = I - G
    m.temp .= zero(eltype(m.temp))
    for f1 in 0:flv-1, f2 in 0:flv-1
        for i in 1:N, j in 1:N
            m.temp[i, j] += IG[i + f1*N, i + f1*N] * IG[j + f2*N, j + f2*N] +
                            IG[j + f2*N, i + f1*N] *  G[i + f1*N, j + f2*N]
        end
    end
    push!(m.obs, m.temp)
end


"""
    CurrentDensityCorrelationMeasurement(mc::DQMC, model)

Measures the current density correlation `Λₓₓ(i, τ) = ⟨jₓ(i, τ) jₓ(0, 0)⟩`.
"""
struct CurrentDensityCorrelationMeasurement{
        OT <: AbstractObservable
    } <: AbstractMeasurement
    obs::OT
    temp::Matrix
end
function CurrentDensityCorrelationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    obs = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Charge density wave correlations", "Observables.jld", "CDC"
    )
    CurrentDensityCorrelationMeasurement(obs, [zero(T) for _ in 1:N, __ in 1:N])
end
function measure!(m::CurrentDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    # TODO
    # implement spinflavors(model)
    # then get N from size(model.l) / spinflavors(model) ?
    N = nsites(model)
    flv = model.flv
    G = greens(mc)
    IG = I - G
    m.temp .= zero(eltype(m.temp))
    for f1 in 0:flv-1, f2 in 0:flv-1
        for i in 1:N, j in 1:N
            m.temp[i, j] += begin
                IG[i + f1*N, j + f1*N] * IG[i + f2*N, j + f2*N] +
                IG[i + f2*N, j + f1*N] *  G[i + f1*N, j + f2*N] -

                IG[j + f1*N, i + f1*N] * IG[i + f2*N, j + f2*N] -
                IG[i + f2*N, i + f1*N] *  G[j + f1*N, j + f2*N] -

                IG[i + f1*N, j + f1*N] * IG[j + f2*N, i + f2*N] -
                IG[j + f2*N, j + f1*N] *  G[i + f1*N, i + f2*N] +

                IG[j + f1*N, i + f1*N] * IG[j + f2*N, i + f2*N] +
                IG[j + f2*N, i + f1*N] *  G[j + f1*N, i + f2*N]
            end
        end
    end
    push!(m.obs, m.temp)
end


################################################################################
### Spin 1/2 Measurements
################################################################################



abstract type SpinOneHalfMeasurement <: AbstractMeasurement end

# Abuse prepare! to verify requirements
function prepare!(m::SpinOneHalfMeasurement, mc::DQMC, model)
    model.flv != 2 && throw(AssertionError(
        "A spin 1/2 measurement ($(typeof(m))) requires two (spin) flavors of fermions, but " *
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
        "Magnetization x", "Observables.jld", "Mx"
    )
    m1y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N]),
        "Magnetization y", "Observables.jld", "My"
    )
    m1z = LightObservable(
        LogBinner([zero(T) for _ in 1:N]),
        "Magnetization z", "Observables.jld", "Mz"
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
function SpinDensityCorrelationMeasurement(mc::DQMC, model)
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

    # Spin density correlation
    m2x = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation x", "Observables.jld", "sdc-x"
    )
    m2y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation y", "Observables.jld", "sdc-y"
    )
    m2z = LightObservable(
        LogBinner([zero(T) for _ in 1:N, __ in 1:N]),
        "Spin Density Correlation z", "Observables.jld", "sdc-z"
    )

    SpinDensityCorrelationMeasurement(m2x, m2y, m2z)
end
function measure!(m::SpinDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
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



"""
    PairingCorrelationMeasurement(mc::DQMC, model)

Measures the s-wave equal-time pairing correlation matrix (`.mat`) and its uniform
Fourier transform (`.uniform_fourier`).
"""
struct PairingCorrelationMeasurement{
        OT1 <: AbstractObservable,
        OT2 <: AbstractObservable,
        T
    } <: SpinOneHalfMeasurement
    mat::OT1
    uniform_fourier::OT2
    temp::Matrix{T}
end
function PairingCorrelationMeasurement(mc::DQMC, model)
    T = eltype(mc.s.greens)
    N = nsites(model)

    obs1 = LightObservable(
        LogBinner(zeros(T, N, N)),
        "Equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "etpc-s"
    )
    obs2 = LightObservable(
        LogBinner(T),
        "Uniform Fourier tranforms of equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "etpc-s Fourier"
    )
    temp = zeros(T, N, N)

    PairingCorrelationMeasurement(obs1, obs2, temp)
end
function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc)
    N = nsites(model)
    m.temp .= G[1:N, 1:N] .* G[N+1:2N, N+1:2N] - G[1:N, N+1:2N] .* G[N+1:2N, 1:N]
    push!(m.mat, m.temp)
    push!(m.uniform_fourier, sum(m.temp) / N)
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
