function default_measurements(mc::DQMC, model)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end



################################################################################
### General DQMC Measurements
################################################################################



"""
    GreensMeasurement(mc::DQMC, model)

Measures the equal time Greens function of the given DQMC simulation and model.

The mean of this measurement corresponds to the expectation value of the Greens
function for the full partition function, i.e. including fermionic and bosonic
(auxiliary field) degrees of freedom.
"""
struct GreensMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function GreensMeasurement(mc::DQMC, model)
    o = LightObservable(
        LogBinner(zeros(eltype(mc.s.greens), size(mc.s.greens)), capacity=1_000_000),
        "Equal-times Green's function",
        "Observables.jld",
        "G"
    )
    GreensMeasurement{typeof(o)}(o)
end
@bm function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens(mc))
end
function save_measurement(file::JLD.JldFile, m::GreensMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function load_measurement(data, ::Type{T}) where T <: GreensMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
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
@bm function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end
function save_measurement(file::JLD.JldFile, m::BosonEnergyMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function load_measurement(data, ::Type{T}) where T <: BosonEnergyMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
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


#
function get_lattice_shape(model::Model)
    try
        return size(model.l)
    catch e
        @warn "Failed to get lattice shape. Using flat array of sites instead."
        @warn e
        return nsites(model)
    end
end



@doc raw"""
    ChargeDensityCorrelationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the charge density correlation
matrix `⟨nᵢnⱼ⟩`.

The mean of this measurement corresponds to the expectation value of the charge
density correlation matrix for the full partition function, i.e. including
fermionic and bosonic (auxiliary field) degrees of freedom.


The fermionic expectation value is computed via Wick's theorem.
```math
      \langle n_i n_j \rangle
    = \langle (n_{i, \uparrow} + n_{i, \downarrow}) (n_{j, \uparrow} +
    n_{j, \downarrow})\rangle
    = \langle n_{i, \uparrow} n_{j, \uparrow} \rangle +
      \langle n_{i, \uparrow} n_{j, \downarrow} \rangle +
      \langle n_{i, \downarrow} n_{j, \uparrow} \rangle +
      \langle n_{i, \downarrow} n_{j, \downarrow} \rangle
    = ... + \langle c_{i, \uparrow}^\dagger c_{i, \uparrow}
        c_{j, \downarrow}^\dagger c_{j, \downarrow} \rangle + ...
    = ... + \langle c_{i, \uparrow}^\dagger c_{i, \uparrow} \rangle
      \langle c_{j, \downarrow}^\dagger c_{j, \downarrow} \rangle +
      \langle c_{i, \uparrow}^\dagger c_{j, \downarrow} \rangle
      \langle c_{i, \uparrow} c_{j, \downarrow}^\dagger \rangle + ...
    = ... + (I - G)_{ii}^{\uparrow\uparrow} (I - G)_{jj}^{\downarrow\downarrow} +
      (I-G)_{ji}^{\downarrow\uparrow} G_{ij}^{\uparrow\downarrow} + ...
```
"""
struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
end
function ChargeDensityCorrelationMeasurement(mc::DQMC, model; shape=get_lattice_shape(model))
    N = nsites(model)
    T = eltype(mc.s.greens)
    obs = LightObservable(
        LogBinner(reshape([zero(T) for _ in 1:N], shape)),
        "Charge density wave correlations", "Observables.jld", "CDC"
    )
    ChargeDensityCorrelationMeasurement(obs, reshape([zero(T) for _ in 1:N], shape))
end
@bm function measure!(m::ChargeDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G
    m.temp .= zero(eltype(m.temp))

    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += begin
                # ⟨n↑n↑⟩
                IG[i, i] * IG[j, j] +
                IG[j, i] *  G[i, j] +
                # ⟨n↑n↓⟩
                IG[i, i] * IG[j + N, j + N] +
                IG[j + N, i] *  G[i, j + N] +
                # ⟨n↓n↑⟩
                IG[i + N, i + N] * IG[j, j] +
                IG[j, i + N] *  G[i + N, j] +
                # ⟨n↓n↓⟩
                IG[i + N, i + N] * IG[j + N, j + N] +
                IG[j + N, i + N] *  G[i + N, j + N]
            end
        end
    end

    push!(m.obs, m.temp / N)
end
function save_measurement(
        file::JLD.JldFile,
        m::ChargeDensityCorrelationMeasurement,
        entryname::String
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    write(file, entryname * "/temp_dim", size(m.temp))
    write(file, entryname * "/temp_type", typeof(m.temp))
    nothing
end
function load_measurement(
        data, ::Type{T}
    ) where T <: ChargeDensityCorrelationMeasurement

    @assert data["VERSION"] == 1
    data["type"](data["obs"], data["temp_type"](undef, data["temp_size"]...))
end



"""
    MagnetizationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the magnetization
`M_x = ⟨c_{i, ↑}^† c_{i, ↓} + h.c.⟩` in x-,
`M_y = -i⟨c_{i, ↑}^† c_{i, ↓} - h.c.⟩` in y- and
`M_z = ⟨n_{i, ↑} - n_{i, ↓}⟩` in z-direction.

The mean of this measurement corresponds to the expectation value of the x/y/z
magnetization for the full partition function, i.e. including fermionic and
bosonic (auxiliary field) degrees of freedom.

Note:

The Magnetization in x/y/z direction can be accessed via fields `x`, `y` and `z`.
"""
struct MagnetizationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: AbstractArray
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
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

    MagnetizationMeasurement(m1x, m1y, m1z, [zero(T) for _ in 1:N])
end
@bm function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    # Magnetization
    # c_{i, up}^† c_{i, down} + c_{i, down}^† c_{i, up}
    # mx = [- G[i+N, i] - G[i, i+N]           for i in 1:N]
    map!(i -> -G[i+N, i] - G[i, i+N], m.temp, 1:N)
    push!(m.x, m.temp)

    # -i [c_{i, up}^† c_{i, down} - c_{i, down}^† c_{i, up}]
    # my = [-1im * (G[i, i+N] - G[i+N, i])    for i in 1:N]
    map!(i -> -1im *(G[i+N, i] - G[i, i+N]), m.temp, 1:N)
    push!(m.y, m.temp)
    # c_{i, up}^† c_{i, up} - c_{i, down}^† c_{i, down}
    # mz = [G[i+N, i+N] - G[i, i]             for i in 1:N]
    map!(i -> G[i+N, i+N] - G[i, i], m.temp, 1:N)
    push!(m.z, m.temp)
end
function save_measurement(
        file::JLD.JldFile,
        m::MagnetizationMeasurement,
        entryname::String
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs_x", m.x)
    write(file, entryname * "/obs_y", m.y)
    write(file, entryname * "/obs_z", m.z)
    nothing
end
function load_measurement(
        data, ::Type{T}
    ) where T <: MagnetizationMeasurement

    @assert data["VERSION"] == 1
    data["type"](data["obs_x"], data["obs_y"], data["obs_z"])
end



"""
    SpinDensityCorrelationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the spin density correlation matrix
`SDC_x = ⟨(c_{i, ↑}^† c_{i, ↓} + h.c.) (c_{j, ↑}^† c_{j, ↓} + h.c.)⟩` in x-,
`SDC_y = -⟨(c_{i, ↑}^† c_{i, ↓} - h.c.) (c_{j, ↑}^† c_{j, ↓} - h.c.)⟩` in y- and
`SDC_z = ⟨(n_{i, ↑} - n_{i, ↓}) (n_{j, ↑} - n_{j, ↓})⟩` in z-direction.

The mean of this measurement corresponds to the expectation value of the x/y/z
spin density correlation matrix for the full partition function, i.e. including
fermionic and bosonic (auxiliary field) degrees of freedom.

Note:

The spin density correlation matrix in x/y/z direction can be accessed via fields `x`,
`y` and `z`.
"""
struct SpinDensityCorrelationMeasurement{
        OTx <: AbstractObservable,
        OTy <: AbstractObservable,
        OTz <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
end
function SpinDensityCorrelationMeasurement(mc::DQMC, model; shape=get_lattice_shape(model))
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

    # Spin density correlation
    sdc2x = LightObservable(
        LogBinner(reshape([zero(T) for _ in 1:N], shape)),
        "Spin Density Correlation x", "Observables.jld", "sdc-x"
    )
    sdc2y = LightObservable(
        LogBinner(reshape([zero(Ty) for _ in 1:N], shape)),
        "Spin Density Correlation y", "Observables.jld", "sdc-y"
    )
    sdc2z = LightObservable(
        LogBinner(reshape([zero(T) for _ in 1:N], shape)),
        "Spin Density Correlation z", "Observables.jld", "sdc-z"
    )

    SpinDensityCorrelationMeasurement(sdc2x, sdc2y, sdc2z, reshape([zero(T) for _ in 1:N], shape))
end
@bm function measure!(m::SpinDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†


    # Spin Density Correlation
    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += (
                IG[i+N, i] * IG[j+N, j] + IG[j+N, i] * G[i+N, j] +
                IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
                IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] +
                IG[i, i+N] * IG[j, j+N] + IG[j, i+N] * G[i, j+N]
            )
        end
    end
    push!(m.x, m.temp / N)

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += (
                - IG[i+N, i] * IG[j+N, j] - IG[j+N, i] * G[i+N, j] +
                  IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
                  IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] -
                  IG[i, i+N] * IG[j, j+N] - IG[j, i+N] * G[i, j+N]
            )
        end
    end
    push!(m.y, m.temp / N)

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += (
                IG[i, i] * IG[j, j] + IG[j, i] * G[i, j] -
                IG[i, i] * IG[j+N, j+N] - IG[j+N, i] * G[i, j+N] -
                IG[i+N, i+N] * IG[j, j] - IG[j, i+N] * G[i+N, j] +
                IG[i+N, i+N] * IG[j+N, j+N] + IG[j+N, i+N] * G[i+N, j+N]
            )
        end
    end
    push!(m.z, m.temp / N)
end
function save_measurement(
        file::JLD.JldFile,
        m::SpinDensityCorrelationMeasurement,
        entryname::String
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs_x", m.x)
    write(file, entryname * "/obs_y", m.y)
    write(file, entryname * "/obs_z", m.z)
    nothing
end
function load_measurement(
        data, ::Type{T}
    ) where T <: SpinDensityCorrelationMeasurement

    @assert data["VERSION"] == 1
    data["type"](data["obs_x"], data["obs_y"], data["obs_z"])
end



"""
    PairingCorrelationMeasurement(mc::DQMC, model)

Measures the fermionic expectation value of the s-wave pairing correlation.

We define `Δᵢ = c_{i, ↑} c_{i, ↓}` s the pair-field operator and `Pᵢⱼ = ⟨ΔᵢΔⱼ^†⟩`
as the s-wave pairing correlation matrix. `Pᵢⱼ` can be accesed via the field
`mat` and its site-average via the field `uniform_fourier`.
"""
struct PairingCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
end
function PairingCorrelationMeasurement(mc::DQMC, model; shape=get_lattice_shape(model))
    T = eltype(mc.s.greens)
    N = nsites(model)

    obs1 = LightObservable(
        LogBinner(reshape(zeros(T, N), shape)),
        "Equal time pairing correlation matrix (s-wave)",
        "observables.jld",
        "etpc-s"
    )

    PairingCorrelationMeasurement(obs1, reshape(zeros(T, N), shape))
end
@bm function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc, model)
    N = nsites(model)
    # Pᵢⱼ = ⟨ΔᵢΔⱼ^†⟩
    #     = ⟨c_{i, ↑} c_{i, ↓} c_{j, ↓}^† c_{j, ↑}^†⟩
    #     = ⟨c_{i, ↑} c_{j, ↑}^†⟩ ⟨c_{i, ↓} c_{j, ↓}^†⟩ -
    #       ⟨c_{i, ↑} c_{j, ↓}^†⟩ ⟨c_{i, ↓} c_{j, ↑}^†⟩
    # m.temp .= G[1:N, 1:N] .* G[N+1:2N, N+1:2N] - G[1:N, N+1:2N] .* G[N+1:2N, 1:N]

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += G[i, j] * G[i+N, j+N] - G[i, j+N] * G[i+N, j]
        end
    end

    push!(m.obs, m.temp / N)
end

"""
    uniform_fourier(M, dqmc)
    uniform_fourier(M, N)

Computes the uniform Fourier transform of matrix `M` in a system with `N` sites.
"""
uniform_fourier(M::AbstractArray, mc::DQMC) = sum(M) / nsites(mc.model)
uniform_fourier(M::AbstractArray, N::Integer) = sum(M) / N


struct UniformFourierWrapped{T <: AbstractObservable}
    obs::T
end
"""
    uniform_fourier(m::AbstractMeasurement[, field::Symbol])
    uniform_fourier(obs::AbstractObservable)

Wraps an observable with a `UniformFourierWrapped`.
Calling `mean` (`var`, etc) on a wrapped observable returns the `mean` (`var`,
etc) of the uniform Fourier transform of that observable.

`mean(uniform_fourier(m))` is equivalent to
`uniform_fourier(mean(m.obs), nsites(model))` where `obs` may differ between
measurements.
"""
uniform_fourier(m::PairingCorrelationMeasurement) = UniformFourierWrapped(m.obs)
uniform_fourier(m::ChargeDensityCorrelationMeasurement) = UniformFourierWrapped(m.obs)
function uniform_fourier(m::AbstractMeasurement, field::Symbol)
    UniformFourierWrapped(getfield(m, field))
end
uniform_fourier(obs::AbstractObservable) = UniformFourierWrapped(obs)

# Wrappers for Statistics functions
MonteCarloObservable.mean(x::UniformFourierWrapped) = _uniform_fourier(mean(x.obs))
MonteCarloObservable.var(x::UniformFourierWrapped) = _uniform_fourier(var(x.obs))
MonteCarloObservable.varN(x::UniformFourierWrapped) = _uniform_fourier(varN(x.obs))
MonteCarloObservable.std(x::UniformFourierWrapped) = _uniform_fourier(std(x.obs))
MonteCarloObservable.std_error(x::UniformFourierWrapped) = _uniform_fourier(std_error(x.obs))
MonteCarloObservable.all_vars(x::UniformFourierWrapped) = _uniform_fourier.(all_vars(x.obs))
MonteCarloObservable.all_varNs(x::UniformFourierWrapped) = _uniform_fourier.(all_varNs(x.obs))
# Autocorrelation time should not be averaged...
MonteCarloObservable.tau(x::UniformFourierWrapped) = maximum(tau(x.obs))
MonteCarloObservable.all_taus(x::UniformFourierWrapped) = maximum.(all_varNs(x.obs))
_uniform_fourier(M::AbstractArray) = sum(M) / length(M)




################################################################################
### Greens functions
################################################################################


# NOTE: Likely broken
# """
#     greens(mc::DQMC, slice1, slice2)
#
# Computes the unequal time Green's function G(τ₁, τ₂) where τ₁ ≥ τ₂.
# """
# function greens(mc::DQMC, slice1::Int64, slice2::Int64)
#     slice1 == slice2 && return greens(mc, slice1)
#     N = nslices(mc)
#
#     @assert slice1 > slice2
#
#     G = greens(mc, slice2+1)
#     copyto!(mc.s.Ul, I)
#     mc.s.Dl .= 1
#     copyto!(mc.s.Tl, I)
#
#     for i in slice2+1:slice1
#         multiply_slice_matrix_left!(mc, mc.model, i, mc.s.Ul)
#         if i % mc.p.safe_mult == 0
#             rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
#             mc.s.U, mc.s.Dl, mc.s.T = udt(mc.s.Ul)
#             mul!(mc.s.tmp, mc.s.T, mc.s.Tl)
#             copyto!(mc.s.Ul, mc.s.U)
#             copyto!(mc.s.Tl, mc.s.tmp)
#         end
#     end
#     # Finalize product Ul Dl Tl G
#     rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
#     rmul!(mc.s.Ul, mc.s.Tl)
#     lmul!(mc.s.Ul, G)
#     return G
# end


"""
    greens(mc::DQMC, slice)

Computes the equal-time Greens function G(τ) = G(τ, τ) for an arbitary
imaginary time τ. The time is given through a time-slice index `slice` where
`τ = Δτ ⋅ slice`.

The elements of the Greens function depends on (the implementation of) the
model. For the attractive Hubbard model the Greens function is a (N sites × N
sites) matrix corresponding to to the fermion expectation value
`G_{ij} = ⟨c_{i, σ} c_{j, σ}^†⟩`.
"""
greens(mc::DQMC, slice::Int64) = copy(_greens!(mc, slice, mc.s.greens_temp, mc.s.tmp))
function _greens!(mc::DQMC, slice::Int64,greens_out::Matrix, temp::Matrix)
    cur_slice = current_slice(mc)
    N = nslices(mc)
    if cur_slice == slice
        greens_out .= mc.s.greens
        return _greens!(mc, greens_out, temp)
    end

    # use slice matrix multiplications if it's within safe_mult boundaries
    d = let
        x = slice - cur_slice
        y = N - abs(x)
        abs(x) < y ? x : sign(-x) * y
    end
    if abs(d) < div(mc.p.safe_mult, 2)
        greens_out .= mc.s.greens
        if d > 0 # forward in time
            for s in cur_slice:cur_slice+d-1
                multiply_slice_matrix_left!(
                    mc, mc.model, mod1(s, N), greens_out
                )
                multiply_slice_matrix_inv_right!(
                    mc, mc.model, mod1(s, N), greens_out
                )
            end
        else # backward in time
            for s in cur_slice-1:-1:cur_slice+d
                multiply_slice_matrix_inv_left!(
                    mc, mc.model, mod1(s, N), greens_out
                )
                multiply_slice_matrix_right!(
                    mc, mc.model, mod1(s, N), greens_out
                )
            end
        end
        return _greens!(mc, greens_out, temp)
    end

    # Otherwise we need to explicitly recalculate stuff
    # We use these as udt "stack"
    copyto!(mc.s.Ul, I)
    mc.s.Dl .= 1
    copyto!(mc.s.Tl, I)
    copyto!(mc.s.Ur, I)
    mc.s.Dr .= 1
    copyto!(mc.s.Tr, I)

    for i in 1:slice-1
        multiply_slice_matrix_left!(mc, mc.model, i, mc.s.Ul)
        if i % mc.p.safe_mult == 0
            rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
            mc.s.U, mc.s.Dl, mc.s.T = udt(mc.s.Ul)
            mul!(mc.s.tmp, mc.s.T, mc.s.Tl)
            copyto!(mc.s.Ul, mc.s.U)
            copyto!(mc.s.Tl, mc.s.tmp)
        end
    end
    # Finalize product and UDT decomposition
    rmul!(mc.s.Ul, Diagonal(mc.s.Dl))
    mc.s.U, mc.s.Dl, mc.s.T = udt(mc.s.Ul)
    mul!(mc.s.tmp, mc.s.T, mc.s.Tl)
    copyto!(mc.s.Ul, mc.s.U)
    copyto!(mc.s.Tl, mc.s.tmp)

    for i in N:-1:slice
        multiply_daggered_slice_matrix_left!(mc, mc.model, i, mc.s.Ur)
        if i % mc.p.safe_mult == 0
            rmul!(mc.s.Ur, Diagonal(mc.s.Dr))
            mc.s.U, mc.s.Dr, mc.s.T = udt(mc.s.Ur)
            mul!(mc.s.tmp, mc.s.T, mc.s.Tr)
            copyto!(mc.s.Ur, mc.s.U)
            copyto!(mc.s.Tr, mc.s.tmp)
        end
    end
    rmul!(mc.s.Ur, Diagonal(mc.s.Dr))
    mc.s.U, mc.s.Dr, mc.s.T = udt(mc.s.Ur)
    mul!(mc.s.tmp, mc.s.T, mc.s.Tr)
    copyto!(mc.s.Ur, mc.s.U)
    copyto!(mc.s.Tr, mc.s.tmp)

    # slightly modified calculate_greens()
    mc.s.U, mc.s.D, mc.s.T = udt_inv_one_plus(
        UDT(mc.s.Ul, mc.s.Dl, mc.s.Tl),
        UDT(mc.s.Ur, mc.s.Dr, mc.s.Tr),
        tmp = mc.s.U, tmp2 = mc.s.T, tmp3 = mc.s.tmp,
        internaluse = true
    )
    mul!(mc.s.tmp, mc.s.U, Diagonal(mc.s.D))
    mul!(greens_out, mc.s.tmp, mc.s.T)
    return _greens!(mc, greens_out, temp)
end


greens(mc::DQMC, model::Model) = greens(mc)




# Ported from crstnbr/dqmc

# unequal time greens function
@enum Direction begin
    LEFT
    RIGHT
end

mutable struct UnequalTimeGreensFunction{G} <: AbstractMeasurement
    Gt0::Vector{Matrix{G}}
    G0t::Vector{Matrix{G}}

    BT0Inv_u_stack::Vector{Matrix{G}}
    BT0Inv_d_stack::Vector{Vector{Float64}}
    BT0Inv_t_stack::Vector{Matrix{G}}

    BBetaT_u_stack::Vector{Matrix{G}}
    BBetaT_d_stack::Vector{Vector{Float64}}
    BBetaT_t_stack::Vector{Matrix{G}}

    BT0_u_stack::Vector{Matrix{G}}
    BT0_d_stack::Vector{Vector{Float64}}
    BT0_t_stack::Vector{Matrix{G}}

    BBetaTInv_u_stack::Vector{Matrix{G}}
    BBetaTInv_d_stack::Vector{Vector{Float64}}
    BBetaTInv_t_stack::Vector{Matrix{G}}
end



function UnequalTimeGreensFunction(mc::DQMC, model)
    M = mc.p.slices
    N = nsites(model)
    Nflv = N*model.flv
    G = geltype(mc)

    nranges = length(mc.s.ranges)

    UnequalTimeGreensFunction(
        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:M],
        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:M],

        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],
        Vector{Float64}[zeros(Float64, Nflv) for _ in 1:nranges],
        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],

        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],
        Vector{Float64}[zeros(Float64, Nflv) for _ in 1:nranges],
        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],

        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],
        Vector{Float64}[zeros(Float64, Nflv) for _ in 1:nranges],
        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],

        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges],
        Vector{Float64}[zeros(Float64, Nflv) for _ in 1:nranges],
        Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:nranges]
    )
end

function deallocate!(m::UnequalTimeGreensFunction)
    m.BT0Inv_u_stack = Matrix{G}[]
    m.BT0Inv_d_stack = Vector{Float64}[]
    m.BT0Inv_t_stack = Matrix{G}[]
    m.BBetaT_u_stack = Matrix{G}[]
    m.BBetaT_d_stack = Vector{Float64}[]
    m.BBetaT_t_stack = Matrix{G}[]
    m.BT0_u_stack = Matrix{G}[]
    m.BT0_d_stack = Vector{Float64}[]
    m.BT0_t_stack = Matrix{G}[]
    m.BBetaTInv_u_stack = Matrix{G}[]
    m.BBetaTInv_d_stack = Vector{Float64}[]
    m.BBetaTInv_t_stack = Matrix{G}[]

    println("Deallocated UDT stacks memory of TDGF measurement.")
    nothing
end

"""
Returns the actual memory usage of fields related to `measure_tdgfs` (in MB).
"""
function memory_usage_tdgfs(m::UnequalTimeGreensFunction)
    s = x -> Base.summarysize(x) / 1024 / 1024 # size in MB

    mem = 0.

    mem += s(m.Gt0)
    mem += s(m.G0t)

    mem += s(m.BT0Inv_u_stack)
    mem += s(m.BT0Inv_d_stack)
    mem += s(m.BT0Inv_t_stack)
    mem += s(m.BBetaT_u_stack)
    mem += s(m.BBetaT_d_stack)
    mem += s(m.BBetaT_t_stack)
    mem += s(m.BT0_u_stack)
    mem += s(m.BT0_d_stack)
    mem += s(m.BT0_t_stack)
    mem += s(m.BBetaTInv_u_stack)
    mem += s(m.BBetaTInv_d_stack)
    mem += s(m.BBetaTInv_t_stack)

    return mem
end


"""
        estimate_memory_usage_tdgfs(L, beta; flv=4, safe_mult=10, delta_tau=0.1)
Estimates the expected memory usage of fields related to `measure_tdgfs` (in MB).
"""
function estimate_memory_usage_tdgfs(L, beta; flv=4, safe_mult=10, delta_tau=0.1)
    N = L^2
    M = Int(beta / delta_tau)

    gfmem = (N * flv)^2 * 128 / 8 # gfmem in bytes

    mem = 2 * gfmem * M # Gt0, G0t
    mem += 8 * gfmem * M / safe_mult # u and t stacks
    mem += 4 * (N * 64 / 8) # d_stack

    return round(mem / 1024 / 1024, digits=1)
end

"""
Estimates the expected memory usage of fields related to `measure_tdgfs` (in MB).
"""
function estimate_memory_usage_tdgfs(mc::DQMC)
    estimate_memory_usage_tdgfs(
        mc.p.L, mc.p.beta;
        flv = mc.p.flv,
        safe_mult = mc.p.safe_mult,
        delta_tau = mc.p.delta_tau
    )
end


# TODO


function measure!(m::UnequalTimeGreensFunction, mc::DQMC, model, i::Int64)
    G = geltype(mc)
    M = mc.p.slices
    safe_mult = mc.p.safe_mult

    Gt0 = m.Gt0
    G0t = m.G0t

    BT0Inv_u_stack = m.BT0Inv_u_stack
    BT0Inv_d_stack = m.BT0Inv_d_stack
    BT0Inv_t_stack = m.BT0Inv_t_stack
    BBetaT_u_stack = m.BBetaT_u_stack
    BBetaT_d_stack = m.BBetaT_d_stack
    BBetaT_t_stack = m.BBetaT_t_stack
    BT0_u_stack = m.BT0_u_stack
    BT0_d_stack = m.BT0_d_stack
    BT0_t_stack = m.BT0_t_stack
    BBetaTInv_u_stack = m.BBetaTInv_u_stack
    BBetaTInv_d_stack = m.BBetaTInv_d_stack
    BBetaTInv_t_stack = m.BBetaTInv_t_stack


    # ---- first, calculate Gt0 and G0t only at safe_mult slices
    # right mult (Gt0)
    calc_Bchain_udts!(mc, BT0Inv_u_stack, BT0Inv_d_stack, BT0Inv_t_stack, invert=true, dir=LEFT);
    calc_Bchain_udts!(mc, BBetaT_u_stack, BBetaT_d_stack, BBetaT_t_stack, invert=false, dir=RIGHT);

    # left mult (G0t)
    calc_Bchain_udts!(mc, BT0_u_stack, BT0_d_stack, BT0_t_stack, invert=false, dir=LEFT);
    calc_Bchain_udts!(mc, BBetaTInv_u_stack, BBetaTInv_d_stack, BBetaTInv_t_stack, invert=true, dir=RIGHT);


    safe_mult_taus = 1:safe_mult:mc.p.slices
    @inbounds for i in 1:length(safe_mult_taus) # i = ith safe mult time slice
        tau = safe_mult_taus[i] # tau = tauth (overall) time slice
        if i != 1
            # TODO: temporary matrices
            # Gt0
            inv_sum_loh!(
                Gt0[tau],
                UDT(BT0Inv_u_stack[i-1], BT0Inv_d_stack[i-1], BT0Inv_t_stack[i-1]),
                UDT(BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i])
            ) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
            # effective_greens2greens!(mc, Gt0[tau])
            _greens!(mc, Gt0[tau], mc.s.greens_temp)

            # G0t
            inv_sum_loh!(
                G0t[tau],
                UDT(BT0_u_stack[i-1], BT0_d_stack[i-1], BT0_t_stack[i-1]),
                UDT(BBetaTInv_u_stack[i], BBetaTInv_d_stack[i], BBetaTInv_t_stack[i])
            ) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
            # effective_greens2greens!(mc, G0t[tau])
            _greens!(mc, G0t[tau], mc.s.greens_temp)
        else
            # TODO: temporary matrices
            # Gt0
            inv_one_plus_loh!(
                Gt0[tau],
                UDT(BBetaT_u_stack[1], BBetaT_d_stack[1], BBetaT_t_stack[1])
            )
            _greens!(mc, Gt0[tau], mc.s.greens_temp)
            # effective_greens2greens!(mc, Gt0[tau])

            # G0t
            inv_one_plus_loh!(
                G0t[tau],
                UDT(BBetaTInv_u_stack[1], BBetaTInv_d_stack[1], BBetaTInv_t_stack[1])
            )
            _greens!(mc, G0t[tau], mc.s.greens_temp)
            # effective_greens2greens!(mc, G0t[tau]) # TODO: check analytically that we can still do this
        end
    end

    # ---- fill time slices between safe_mult slices
    fill_tdgf!(mc, Gt0, G0t)

    @inbounds for i in 1:M
        G0t[i] .*= -1 # minus sign
    end

    nothing
end


"""
Calculate UDTs at safe_mult time slices of
dir = LEFT:
inv=false:    B(tau, 1) = B(tau) * B(tau-1) * ... * B(1)                     # mult left, 1:tau
inv=true:     [B(tau, 1)]^-1 = B(1)^-1 * B(2)^-1 * ... B(tau)^-1             # mult inv right, 1:tau
udv[i] = from 1 to mc.s.ranges[i][end]
dir = RIGHT:
inv=false:    B(beta, tau) = B(beta) * B(beta-1) * ... * B(tau)              # mult right, beta:tau
inv=true:     [B(beta, tau)]^-1 = B(tau)^-1 * B(tau+1)^-1 * ... B(beta)^-1   # mult inv left, beta:tau
udv[i] = from mc.s.ranges[i][1] to mc.p.slices (beta)
"""
function calc_Bchain_udts!(mc::DQMC, u_stack, d_stack, t_stack; invert::Bool=false, dir::Direction=LEFT)
    G = geltype(mc)
    flv = mc.model.flv
    N = mc.model.l.sites
    curr_U_or_T = mc.s.curr_U
    ranges = mc.s.ranges

    rightmult = false
    ((dir == RIGHT && !invert) || (dir == LEFT && invert)) && (rightmult = true)

    # @show rightmult

    rng = 1:length(ranges)
    dir == RIGHT && (rng = reverse(rng))

    # Calculate udv[i], given udv[i-1]
    @inbounds for (i, rngidx) in enumerate(rng)

        if i == 1
            copyto!(curr_U_or_T, I)
        else
            if !rightmult
                copyto!(curr_U_or_T, u_stack[i-1])
            else
                copyto!(curr_U_or_T, t_stack[i-1])
            end
        end

        slice_range = dir == RIGHT ? reverse(ranges[rngidx]) : ranges[rngidx]

        for slice in slice_range
            if invert == false
                if dir == LEFT
                    multiply_slice_matrix_left!(mc, mc.model, slice, curr_U_or_T)
                else
                    # rightmult
                    multiply_slice_matrix_right!(mc, mc.model, slice, curr_U_or_T)
                end
            else
                if dir == LEFT
                    # rightmult
                    multiply_slice_matrix_inv_right!(mc, mc.model, slice, curr_U_or_T)
                else
                    multiply_slice_matrix_inv_left!(mc, mc.model, slice, curr_U_or_T)
                end
            end
        end

        if i != 1
            if !rightmult
                rmul!(curr_U_or_T, Diagonal(d_stack[i-1]))
            else
                lmul!(Diagonal(d_stack[i-1]), curr_U_or_T)
            end
        end

        if !rightmult
            # u_stack[i], T = decompose_udt!(curr_U_or_T, d_stack[i])
            u_stack[i], d_stack[i], T = udt!(curr_U_or_T)
        else
            # U, t_stack[i] = decompose_udt!(curr_U_or_T, d_stack[i])
            U, d_stack[i], t_stack[i] = udt!(curr_U_or_T)
        end

        if i == 1
            if !rightmult
                mul!(t_stack[i], T, I)
            else
                mul!(u_stack[i], I, U)
            end
        else
            if !rightmult
                mul!(t_stack[i], T, t_stack[i-1])
            else
                mul!(u_stack[i], u_stack[i-1], U)
            end
        end
    end

    if dir == RIGHT
        reverse!(u_stack), reverse!(d_stack), reverse!(t_stack)
    end

    nothing
end



# Given Gt0 and G0t at safe mult slices (mc.s.ranges[i][1])
# propagate to all other slices.
function fill_tdgf!(mc, Gt0, G0t)
    safe_mult = mc.p.safe_mult
    M = mc.p.slices

    safe_mult_taus = 1:safe_mult:M
    @inbounds for tau in 1:M
        (tau in safe_mult_taus) && continue # skip safe mult taus

        Gt0[tau] .= Gt0[tau-1] # copy
        multiply_slice_matrix_left!(mc, mc.model, tau, Gt0[tau])

        G0t[tau] .= G0t[tau-1] # copy
        multiply_slice_matrix_inv_right!(mc, mc.model, tau, G0t[tau])
    end

    nothing
end










# Calculate "G(tau, 0)", i.e. G(slice,1) as G(slice,1) = [B(slice, 1)^-1 + B(beta, slice)]^-1 which is equal to B(slice,1)G(1)
function calc_tdgf_direct(mc::DQMC, slice::Int, safe_mult::Int=mc.p.safe_mult; scalettar=true)
    if slice != 1
        Ul, Dl, Tl = calc_Bchain_inv(mc, 1, slice-1, safe_mult)
    else
        Ul, Dl, Tl = mc.s.eye_full, mc.s.ones_vec, mc.s.eye_full
    end

    if slice != mc.p.slices
        Ur, Dr, Tr = calc_Bchain(mc, slice, mc.p.slices, safe_mult)
    else
        Ur, Dr, Tr = mc.s.eye_full, mc.s.ones_vec, mc.s.eye_full
    end

    # time displace
    if !scalettar
        U, D, T = udt_inv_sum(UDT(Ul, Dl, Tl), UDT(Ur, Dr, Tr))
    else
        U, D, T = udt_inv_sum_loh(UDT(Ul, Dl, Tl), UDT(Ur, Dr, Tr))
    end
    effective_greens2greens!(mc, U, T)

    rmul!(U, Diagonal(D))
    return U*T
end


function effective_greens2greens!(mc::DQMC_CBFalse, U::AbstractMatrix, T::AbstractMatrix)
    eTminus = mc.s.hopping_matrix_exp
    eTplus = mc.s.hopping_matrix_exp_inv

    T = T * eTminus
    U = eTplus * U
    nothing
end
function effective_greens2greens!(mc::DQMC_CBTrue, U::AbstractMatrix, T::AbstractMatrix)
    chkr_hop_half_minus = mc.s.chkr_hop_half
    chkr_hop_half_plus = mc.s.chkr_hop_half_inv
    n_groups = mc.l.n_groups
    tmp = mc.s.tmp

    @inbounds @views begin
        for i in reverse(1:n_groups)
            mul!(tmp, T, chkr_hop_half_minus[i])
            T .= tmp
        end
        for i in reverse(1:n_groups)
            mul!(tmp, chkr_hop_half_plus[i], U)
            U .= tmp
        end
    end
    nothing
end
function save_measurement(
        file::JLD.JldFile,
        m::PairingCorrelationMeasurement,
        entryname::String
    )
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/mat_obs", m.mat)
    write(file, entryname * "/num_obs", m.uniform_fourier)
    write(file, entryname * "/temp_type", typeof(m.temp))
    write(file, entryname * "/temp_size", size(m.temp))
    nothing
end
function load_measurement(
        data, ::Type{T}
    ) where T <: PairingCorrelationMeasurement

    @assert data["VERSION"] == 1
    data["type"](
        data["mat_obs"],
        data["num_obs"],
        data["temp_type"](undef, data["temp_size"]...)
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
