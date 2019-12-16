const TestConf = Array{Int8, 2}
const TestDistribution = Int8[-1,1]

@with_kw_noshow struct TestModel <: Model
    L::Int

    # user optional
    # mu::Float64 = 0.0
    U::Float64 = 1.0
    @assert U >= 0. "U must be positive."

    # non-user fields
    l::TriangularLattice = TriangularLattice(L) # hopping matrix is random, so lattice irrelevant
    neighs::Matrix{Int} = neighbors_lookup_table(l)
    flv::Int = 2
end


TestModel(params::Dict{Symbol, T}) where T = TestModel(; params...)
TestModel(params::NamedTuple) = TestModel(; params...)

# cosmetics
import Base.summary
import Base.show
# Base.summary(model::TestModel) = "$(model.dims)D attractive Hubbard model"
# Base.show(io::IO, model::TestModel) = print(io, "$(model.dims)D attractive Hubbard model, L=$(model.L) ($(length(model.l)) sites)")
# Base.show(io::IO, m::MIME"text/plain", model::TestModel) = print(io, model)




# implement `Model` interface
@inline nsites(m::TestModel) = length(m.l)
hoppingeltype(::Type{DQMC}, m::TestModel) = ComplexF64

# implement `DQMC` interface: mandatory
@inline Base.rand(::Type{DQMC}, m::TestModel, nslices::Int) = rand(TestDistribution, nsites(m), nslices)


function hopping_matrix(mc::DQMC, m::TestModel)
    # 2N for spin flip
    N = length(m.l)
    T = zeros(ComplexF64, 2N, 2N)

    # Generate T_0
    S = 2.0rand(ComplexF64, N, N) .- (1+1im)
    T_0 = S + S'
    T[1:N, 1:N] .= T_0
    T[N+1:end, N+1:end] .= -T_0
    @info "Is T_0 symmetric?          $(issymmetric(T_0))"
    @info "Is T_0 pos. semi-definite? $(try all(>=(0), eigvals(T_0)) catch e; e end)"
    @info "Is T_0 neg. semi-definite? $(try all(<=(0), eigvals(T_0)) catch e; e end)"
    @info "T_0 .>= 0.0?               $(try all(>=(0), T_0) catch e; e end)"
    @info "T_0 .<= 0.0?               $(try all(<=(0), T_0) catch e; e end)"
    @info "Is T_0 hermitian?          $(ishermitian(T_0))"

    # Failed:
    # S = 2rand(ComplexF64, N, N) .- 1
    # > T_0 = S
    # > T_0 = S + tranpose(S)


    # Generate T_perp
    S = 2rand(N, N) .- 1.0
    T_perp = S + S' #+ N * I)
    T[1:N, N+1:end] .= T_perp
    T[N+1:end, 1:N] .= T_perp
    @info "Is T⟂ symmetric?          $(issymmetric(T_perp))"
    @info "Is T⟂ pos. semi-definite? $(all(>=(0), eigvals(T_perp)))"
    @info "Is T⟂ neg. semi-definite? $(all(<=(0), eigvals(T_perp)))"
    @info "T⟂ .>= 0.0?               $(all(>=(0), T_perp))"
    @info "T⟂ .<= 0.0?               $(all(<=(0), T_perp))"
    @info "Is T⟂ hermitian?          $(ishermitian(T_perp))"

    return T
end


"""
Calculate the interaction matrix exponential `expV = exp(- power * delta_tau * V(slice))`
and store it in `result::Matrix`.

This is a performance critical method.
"""
@inline function interaction_matrix_exp!(mc::DQMC, m::TestModel,
            result::Matrix, conf::TestConf, slice::Int, power::Float64=1.)
    # TODO optimize
    # compute this only once
    dtau = mc.p.delta_tau
    lambda = acosh(exp(0.5m.U * dtau))

    result .= Diagonal(vcat(
        exp.(sign(power) * lambda * conf[:, slice]),
        exp.(-sign(power) * lambda * conf[:, slice])
    ))
    nothing
end


@inline @fastmath @inbounds function propose_local(
        mc::DQMC, m::TestModel, i::Int, slice::Int, conf::TestConf
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
        mc::DQMC, m::TestModel, i::Int, slice::Int, conf::TestConf,
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



@inline function energy_boson(mc::DQMC, m::TestModel, hsfield::TestConf)
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
function save_model(file::JLD.JldFile, model::TestModel, entryname::String)
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
function load_model(data, ::Type{TestModel})
    if data["VERSION"] == 0
        return data["data"]
    elseif data["VERSION"] == 1
        return TestModel(
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
