function default_measurements(mc::DQMC, model)
    Dict(
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model)
    )
end

"""
    _default_capacity(mc::DQMC)

Returns 100_000 if the configuration measurement is not available or empty, and
the length of the configuration measurement if it is not.

This should be useful for `run` - `setup measurements` - `replay` workflow, 
returning the exact capacity needed.
"""
function _default_capacity(mc::DQMC)
    k = if isdefined(mc, :measurements)
        findfirst(v -> v isa ConfigurationMeasurement, mc.measurements)
    else
        nothing
    end
    if k === nothing
        return 100_000
    else
        N = length(mc.measurements[k].obs)
        return N == 0 ? 100_000 : N
    end
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
function GreensMeasurement(mc::DQMC, model; capacity=_default_capacity(mc))
    T = greenseltype(DQMC, model)
    N = model.flv * length(lattice(model))
    o = LightObservable(
        LogBinner(zeros(T, (N, N)), capacity=capacity),
        "Equal-times Green's function",
        "Observables.jld",
        "G"
    )
    GreensMeasurement{typeof(o)}(o)
end
@bm function measure!(m::GreensMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, greens(mc))
end
function save_measurement(file::JLDFile, m::GreensMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: GreensMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end


"""
    BosonEnergyMeasurement(mc::DQMC, model)

Measures the bosonic energy of the given DQMC simulation and model.

Note that this measurement requires `energy_boson(mc, model, conf)` to be
implemented for the specific `model`.
"""
struct BosonEnergyMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function BosonEnergyMeasurement(mc::DQMC, model; capacity=_default_capacity(mc))
    o = LightObservable(Float64, name="Bosonic Energy", alloc=capacity)
    BosonEnergyMeasurement{typeof(o)}(o)
end
@bm function measure!(m::BosonEnergyMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, energy_boson(mc, model, conf(mc)))
end
function save_measurement(file::JLDFile, m::BosonEnergyMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: BosonEnergyMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end


"""
    OccupationMeasurement(mc::DQMC, ::Model[; capacity=_default_capacity(mc)])

Measures the occupation per state (e.g. site and spin).

The mean of this measurement corresponds to the expectation value of the 
occupation number for the full partition function, i.e. including fermionic and 
bosonic (auxiliary field) degrees of freedom.
"""
struct OccupationMeasurement{OT <: AbstractObservable} <: AbstractMeasurement
    obs::OT
end
function OccupationMeasurement(mc::DQMC, model; capacity=_default_capacity(mc))
    N = model.flv * length(lattice(model))
    o = LightObservable(
        LogBinner(zeros(Float64, N), capacity=capacity),
        "State resolved Occupation",
        "Observables.jld",
        "Occ"
    )
    OccupationMeasurement{typeof(o)}(o)
end
function OccupationMeasurement(m::GreensMeasurement{<: LightObservable}; capacity=capacity(m.obs.B))
    N = size(m.obs)[1]
    o = LightObservable(
        LogBinner(zeros(Float64, N), capacity=capacity),
        "State resolved Occupation",
        "Observables.jld",
        "Occ"
    )

    N = min(length(o.B.count), length(m.obs.B.count))
    m.obs.B.count[N] > 1 && throw(OverflowError("OccupationMeasurement does not have enough capacity!"))
    N = findlast(>(1), m.obs.B.count)
    N === nothing && return OccupationMeasurement{typeof(o)}(o)
    
    for i in 1:N
        o.B.x_sum[i]  .= diag(m.obs.B.x_sum[i])
        o.B.x2_sum[i] .= diag(m.obs.B.x2_sum[i])
        o.B.count[i]   = m.obs.B.count[i]
        o.B.compressors[i].value .= diag(m.obs.B.compressors[i].value)
        o.B.compressors[i].switch = m.obs.B.compressors[i].switch
    end
    OccupationMeasurement{typeof(o)}(o)
end
@bm function measure!(m::OccupationMeasurement, mc::DQMC, model, i::Int64)
    push!(m.obs, diag(greens(mc)))
end
function save_measurement(file::JLDFile, m::OccupationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", typeof(m))
    write(file, entryname * "/obs", m.obs)
    nothing
end
function _load(data, ::Type{T}) where T <: OccupationMeasurement
    @assert data["VERSION"] == 1
    data["type"](data["obs"])
end



################################################################################
### Utility
################################################################################



_get_shape(model) = (length(lattice(model)),)
_get_shape(mask::RawMask) = (mask.nsites, mask.nsites)
_get_shape(mask::DistanceMask) = length(mask)

# m is the measurement for potential dispatch
function mask_kernel!(m, mask::RawMask, IG, G, kernel::Function, output)
    for i in 1:size(mask, 1)
        for j in 1:size(mask, 2)
            output[i, j] = kernel(IG, G, i, j)
        end
    end
    output
end
function mask_kernel!(m, mask::DistanceMask, IG, G, kernel::Function, output)
    output .= zero(eltype(output))
    for (dir_idx, src, trg) in getorder(mask)
        output[dir_idx] += kernel(IG, G, src, trg)
    end
    output ./= dirlengths(mask)
    output
end

mask(m::AbstractMeasurement) = m.mask



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
    ChargeDensityCorrelationMeasurement(mc::DQMC, model[; mask, capacity])

Measures the fermionic expectation value of the charge density correlation
matrix `⟨nᵢnⱼ⟩`.

The mean of this measurement corresponds to the expectation value of the charge
density correlation matrix for the full partition function, i.e. including
fermionic and bosonic (auxiliary field) degrees of freedom.
"""
struct ChargeDensityCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array,
        MT <: AbstractMask
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
    mask::MT
end
function ChargeDensityCorrelationMeasurement(
        mc::DQMC, model; 
        mask=DistanceMask(lattice(model)), capacity=_default_capacity(mc)
    )
    T = greenseltype(DQMC, model)
    obs = LightObservable(
        LogBinner(zeros(T, _get_shape(mask)), capacity=capacity),
        "Charge density wave correlations", "Observables.jld", "CDC"
    )
    temp = zeros(T, _get_shape(mask))
    ChargeDensityCorrelationMeasurement(obs, temp, mask)
end
function measure!(m::ChargeDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc, model)
    IG = I - G

    mask_kernel!(m, m.mask, IG, G, _cdc_kernel, m.temp)

    push!(m.obs, m.temp)
end
function _cdc_kernel(IG, G, i, j)
    N = div(size(IG, 1), 2)
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
        AT <: AbstractArray,
        # ATy <: AbstractArray
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
    # tempy::ATy
end

function MagnetizationMeasurement(mc::DQMC, model; capacity=_default_capacity(mc))
    N = length(lattice(model))
    T = greenseltype(DQMC, model)
    Ty = T <: Complex ? T : Complex{T}

    # Magnetizations
    m1x = LightObservable(
        LogBinner([zero(T) for _ in 1:N], capacity=capacity),
        "Magnetization x", "Observables.jld", "Mx"
    )
    m1y = LightObservable(
        LogBinner([zero(Ty) for _ in 1:N], capacity=capacity),
        "Magnetization y", "Observables.jld", "My"
    )
    m1z = LightObservable(
        LogBinner([zero(T) for _ in 1:N], capacity=capacity),
        "Magnetization z", "Observables.jld", "Mz"
    )

    MagnetizationMeasurement(
        m1x, m1y, m1z, [zero(T) for _ in 1:N], [zero(Ty) for _ in 1:N]
    )
end
function measure!(m::MagnetizationMeasurement, mc::DQMC, model, i::Int64)
    N = length(lattice(model))
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
    map!(i -> (G[i+N, i] - G[i, i+N]), m.tempy, 1:N)
    push!(m.y, m.tempy)

    # c_{i, up}^† c_{i, up} - c_{i, down}^† c_{i, down}
    # mz = [G[i+N, i+N] - G[i, i]             for i in 1:N]
    map!(i -> G[i+N, i+N] - G[i, i], m.temp, 1:N)
    push!(m.z, m.temp)
end
function save_measurement(file::JLDFile, m::MagnetizationMeasurement, entryname::String)
    write(file, entryname * "/VERSION", 1)
    write(file, entryname * "/type", MagnetizationMeasurement)
    write(file, entryname * "/x", m.x)
    write(file, entryname * "/y", m.y)
    write(file, entryname * "/z", m.z)
    nothing
end
# function _load(data, ::Type{T}) where T <: MagnetizationMeasurement
#     @assert data["VERSION"] == 1
#     x = data["x"]
#     y = data["y"]
#     z = data["z"]
#     temp = similar(x.B.x_sum[1])
#     tempy = similar(y.B.x_sum[1])
#     data["type"](x, y, z, temp, tempy)
# end


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
        AT <: Array,
        MT <: AbstractMask
    } <: SpinOneHalfMeasurement

    x::OTx
    y::OTy
    z::OTz
    temp::AT
    mask::MT
end
function SpinDensityCorrelationMeasurement(
        mc::DQMC, model; 
        mask=DistanceMask(lattice(model)), capacity=_default_capacity(mc)
    )
    T = greenseltype(DQMC, model)
    Ty = T <: Complex ? T : Complex{T}

    # Spin density correlation
    sdc2x = LightObservable(
        LogBinner(zeros(T, _get_shape(mask)), capacity=capacity),
        "Spin Density Correlation x", "Observables.jld", "sdc-x"
    )
    sdc2y = LightObservable(
        LogBinner(zeros(Ty, _get_shape(mask)), capacity=capacity),
        "Spin Density Correlation y", "Observables.jld", "sdc-y"
    )
    sdc2z = LightObservable(
        LogBinner(zeros(T, _get_shape(mask)), capacity=capacity),
        "Spin Density Correlation z", "Observables.jld", "sdc-z"
    )
    temp = zeros(T, _get_shape(mask))
    SpinDensityCorrelationMeasurement(sdc2x, sdc2y, sdc2z, temp, mask)
end
function measure!(m::SpinDensityCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc, model)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†


    # Spin Density Correlation
    mask_kernel!(m, m.mask, IG, G, _sdc_x_kernel, m.temp)
    push!(m.x, m.temp)

    mask_kernel!(m, m.mask, IG, G, _sdc_y_kernel, m.temp)
    push!(m.y, m.temp)

    mask_kernel!(m, m.mask, IG, G, _sdc_z_kernel, m.temp)
    push!(m.z, m.temp)
end
function _sdc_x_kernel(IG, G, i, j)
    N = div(size(IG, 1), 2)
    IG[i+N, i] * IG[j+N, j] + IG[j+N, i] * G[i+N, j] +
    IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
    IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] +
    IG[i, i+N] * IG[j, j+N] + IG[j, i+N] * G[i, j+N]
end
function _sdc_y_kernel(IG, G, i, j)
    N = div(size(IG, 1), 2)
    - IG[i+N, i] * IG[j+N, j] - IG[j+N, i] * G[i+N, j] +
      IG[i+N, i] * IG[j, j+N] + IG[j, i] * G[i+N, j+N] +
      IG[i, i+N] * IG[j+N, j] + IG[j+N, i+N] * G[i, j] -
      IG[i, i+N] * IG[j, j+N] - IG[j, i+N] * G[i, j+N]
end
function _sdc_z_kernel(IG, G, i, j)
    N = div(size(IG, 1), 2)
    IG[i, i] * IG[j, j] + IG[j, i] * G[i, j] -
    IG[i, i] * IG[j+N, j+N] - IG[j+N, i] * G[i, j+N] -
    IG[i+N, i+N] * IG[j, j] - IG[j, i+N] * G[i+N, j] +
    IG[i+N, i+N] * IG[j+N, j+N] + IG[j+N, i+N] * G[i+N, j+N]
end


# TODO
# Add Symmetry-mask to further compress this?
"""
    PairingCorrelationMeasurement(mc::DQMC, model[; mask=DistanceMask(lattice(model))])

Measures the fermionic expectation value of generic pairing correlations.

We define `Δᵢ = c_{i, ↑} c_{i+di, ↓}` s the pair-field operator and
`P = ⟨ΔᵢΔⱼ^†⟩` as the pairing correlation matrix. Currently the mask must be a 
`DistanceMask`, as a `RawMask` has too high memory requirements. The measurement
is indexed via directional indices `P[di, dj]` which correspond to 
`directions(mask)[di]`. 

See also:
* [`mean`](@ref), [`std_error`](@ref), [`var`](@ref)
* [`RawMask`](@ref), [`DistanceMask`](@ref), [`directions`](@ref)
* [`uniform_fourier`](@ref), [`structure_factor`](@ref)
"""
struct PairingCorrelationMeasurement{
        OT <: AbstractObservable,
        AT <: Array,
        MT <: DistanceMask
    } <: SpinOneHalfMeasurement
    obs::OT
    temp::AT
    mask::MT
end
function PairingCorrelationMeasurement(
        mc::DQMC, model; 
        mask=DistanceMask(lattice(model)), capacity=_default_capacity(mc)
    )
    mask isa RawMask && @error(
        "The Pairing Correlation Measurement will be extremely large with a RawMask!" *
        " (Estimate: $(ceil(Int64, log2(capacity))*3*length(lattice(model))^4*8 / 1024 / 1024)MB)"
    )
    T = greenseltype(DQMC, model)
    shape = _get_shape(mask)

    obs1 = LightObservable(
        LogBinner(zeros(T, shape), capacity=capacity),
        "Equal time pairing correlation matrix",
        "observables.jld",
        "etpc-s"
    )
    temp = zeros(T, shape)
    PairingCorrelationMeasurement(obs1, temp, mask)
end
function measure!(m::PairingCorrelationMeasurement, mc::DQMC, model, i::Int64)
    G = greens(mc, model)
    # Pᵢⱼ = ⟨ΔᵢΔⱼ^†⟩
    #     = ⟨c_{i, ↑} c_{i+d, ↓} c_{j+d, ↓}^† c_{j, ↑}^†⟩
    #     = ⟨c_{i, ↑} c_{j, ↑}^†⟩ ⟨c_{i+d, ↓} c_{j+d, ↓}^†⟩ -
    #       ⟨c_{i, ↑} c_{j+d, ↓}^†⟩ ⟨c_{i+d, ↓} c_{j, ↑}^†⟩

    # Doesn't require IG
    mask_kernel!(m, m.mask, G, G, _pc_s_wave_kernel, m.temp)
    push!(m.obs, m.temp)
end

# m is the measurement for potential dispatch
# function mask_kernel!(
#         m::PairingCorrelationMeasurement,
#         mask::RawMask, IG, G, kernel::Function, output
#     )
#     for src1 in 1:size(mask, 1), src2 in 1:size(mask, 1)
#         for trg1 in 1:size(mask, 2), trg2 in 1:size(mask, 2)
#             output[src1, src2, trg1, trg2] += kernel(IG, G, src1, src2, trg1, trg2)
#         end
#     end
#     output
# end
function mask_kernel!(
        m::PairingCorrelationMeasurement,
        mask::DistanceMask, IG, G, kernel::Function, output
    )
    output .= zero(eltype(output))
    # Sum of Δ_v(r_1) Δ_v^†(r_2) over r_1, r_2 (synced direction) 
    for (dir_idx, src1, trg1) in getorder(mask)
        for (src2, trg2) in getdirorder(mask, dir_idx)
            output[dir_idx] += kernel(IG, G, src1, src2, trg1, trg2)
        end
    end
    output ./= dirlengths(mask).^2
    output
end
function _pc_s_wave_kernel(IG, G, src1, src2, trg1, trg2)
    N = div(size(IG, 1), 2)
    G[src1, src2] * G[trg1+N, trg2+N] - G[src1, trg2+N] * G[trg1+N, src2]
end
# function save_measurement(file::JLDFile, m::PairingCorrelationMeasurement, entryname::String)
#     write(file, entryname * "/VERSION", 1)
#     write(file, entryname * "/type", PairingCorrelationMeasurement)
#     B = m.obs.B

#     write(file, entryname * "/obs", m.obs)
#     nothing
# end
# function _load(data, ::Type{T}) where T <: PairingCorrelationMeasurement
#     @assert data["VERSION"] == 1
#     x = data["x"]
#     y = data["y"]
#     z = data["z"]
#     temp = similar(x.B.x_sum[1])
#     tempy = similar(y.B.x_sum[1])
#     data["type"](x, y, z, temp, tempy)
# end