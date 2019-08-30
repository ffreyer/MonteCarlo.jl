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
            T[src+N, src] -= m.tperp

            # sigma_z:  up -> up with +,  down -> down with -
            trg = l.neighs[3, src]
            T[trg, src] += m.t1
            T[trg+N, src+N] += -m.t1
            # h.c.
            T[src, trg] -= m.t1
            T[src+N, trg+N] -= -m.t1

            # sigma_x:  no sign, switch up and down
            trg = l.neighs[1, src]
            T[trg+N, src] += m.t1
            T[trg, src+N] += m.t1
            # h.c.
            T[src+N, trg] -= m.t1
            T[src, trg+N] -= m.t1

            # sigma_y:  -i up -> down, i down -> up
            trg = l.neighs[2, src]
            T[trg+N, src] += -1im * m.t1
            T[trg, src+N] +=  1im * m.t1
            # h.c
            T[src+N, trg] -= -1im * m.t1
            T[src, trg+N] -=  1im * m.t1

            for nb in 1:3
                trg = l.ext_neighs[nb, src]
                T[trg, src] += m.t2
                T[trg+N, src+N] += m.t2
                # h.c.
                T[src, trg] -= m.t2
                T[src+N, trg+N] -= m.t2
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

    IG = -G[i:N:end, :]
    IG[1, i] += 1.0
    IG[2, i+N] += 1.0
    A = G[:, i:N:end] * inv(R) * Δ * IG
    G .-= A
    conf[i, slice] *= -1

    nothing
end



@inline function energy_boson(mc::DQMC, m::ZCModel, hsfield::ZCConf)
    dtau = mc.p.delta_tau
    lambda = acosh(exp(m.U * dtau/2))
    return lambda * sum(hsfield)
end

include("observables.jl")


#=
@inline function propose_local(mc::DQMC, m::ZCModel, i::Int, slice::Int, conf::ZCConf)
    N = length(m.l)
    G = greens(mc)
    Δτ = mc.p.delta_tau
    α = acosh(exp(0.5Δτ * m.U))

    # NOTE should this have a flavour index?
    ΔE_Boson = -2.0α * conf[i, slice]
    # this is for repulsive
    # where σ = (n↑ - n↓)
    # Δ_up = exp(ΔE_Boson) - 1
    # Δ_down = exp(-ΔE_Boson) - 1
    # we need (n↑ + n↓ - 1)
    # how does that work?
    # Δ_up = exp(ΔE_Boson) - 1
    # Δ_down = exp(ΔE_Boson) - 1
    # NOTE tested against carstens eV1 * eV2 - I
    # NOTE interaction_matrix_exp may still be wrong
    Δ = exp(ΔE_Boson) - 1
    # A = similar(mc.s.eV)
    # B = similar(mc.s.eV)
    # interaction_matrix_exp!(mc, m, A, conf, slice, -1.0)
    # new_conf = conf
    # new_conf[i, slice] *= -1.0
    # interaction_matrix_exp!(mc, m, B, new_conf, slice, 1.0)
    # delta = A * B - I
    # Δ_mat = zeros(eltype(delta), size(delta)...)
    # Δ_mat[i, i] = Δ
    # Δ_mat[i+N, i+N] = Δ
    # if !(Δ_mat ≈ delta)
    #     println("Δ")
    #     println(Δ)
    #     println("Δ_mat")
    #     println([Δ_mat[k, k] for k in 1:2N])
    #     println("delta")
    #     println([delta[k, k] for k in 1:2N])
    #     println("diff")
    #     println([round(abs((delta .- Δ_mat)[k, k]), digits=6) for k in 1:2N])
    #     Base.error(":(")
    # end

    # R_up = (1 + Δ_up * (1 - G[i, i]))
    # R_down = (1 + Δ_down * (1 - G[i+N, i+N]))
    R = eltype(G)[
        (1 + Δ * (1 - G[i, i]))         (Δ * G[i, i+N]);
            (Δ * G[i+N, i])         (1 + Δ * (1 - G[i+N, i+N]))
    ]
    detratio = det(R)
    # detratio = R_up * R_down

    if abs(detratio) > 1e9
        # @error "R = $detratio\nR_s = ($R_up, $R_down)\nG_s = ($(G[i, i]), $(G[i+N, i+N]))\nΔ_s = ($Δ_up, $Δ_down)\n$ΔE_Boson"
        @error "R = $detratio\n\nG_s = $(G[i:N:2N, i:N:2N])\nΔ_s = ($Δ)\n$ΔE_Boson"
        # G_old = deepcopy(G)
        # calculate_greens(mc)
        # G_new = greens(mc)
        # if G_old ≈ G_new
        #     @info "Greens correct?"
        # else
        #     ΔG = G_new - G_old
        #     for i in 1:size(ΔG, 1)
        #         for j in 1:size(ΔG, 2)
        #             x = ΔG[i, j]
        #             c = if abs(x) > 1e5; :red
        #             elseif abs(x) > 1e3; :yellow
        #             elseif abs(x) > 1e1; :blue
        #             else                 :default end
        #             printstyled((@sprintf "%0.2f + %0.2fim  " real(x) imag(x)), color=c)
        #         end
        #         println()
        #     end
        #     println()
        #     display(G_old)
        #     println()
        #     display(G_new)
        # end
    elseif abs(detratio) > 1e6
        # @warn "R = $detratio\nR_s = ($R_up, $R_down)\nG_s = ($(G[i, i]), $(G[i+N, i+N]))\nΔ_s = ($Δ_up, $Δ_down)\n$ΔE_Boson"
        @warn "R = $detratio\n\nG_s = $(G[i:N:2N, i:N:2N])\nΔ_s = ($Δ)\n$ΔE_Boson"
    elseif abs(detratio) > 1e3
        # @info "R = $detratio\nR_s = ($R_up, $R_down)\nG_s = ($(G[i, i]), $(G[i+N, i+N]))\nΔ_s = ($Δ_up, $Δ_down)\n$ΔE_Boson"
        @info "R = $detratio\n\nG_s = $(G[i:N:2N, i:N:2N])\nΔ_s = ($Δ)\n$ΔE_Boson"
    end

    return detratio, ΔE_Boson, (Δ, R)
    # return detratio, ΔE_Boson, (R_up, R_down, Δ_up, Δ_down)
end
=#
