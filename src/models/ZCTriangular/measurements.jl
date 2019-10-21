struct SuperconductivityMeasurement{
        OT <: AbstractObservable,
        T
    } <: SpinOneHalfMeasurement
    s_wave::OT
    ext_s_wave::OT
    dxy_wave::OT
    dx2_y2_wave::OT
    f_wave::OT
    py_wave::OT
    px_wave::OT

    temp1::Matrix{T}
    temp2::Matrix{T}
    temp3::Matrix{T}
    temp4::Matrix{T}
    temp5::Matrix{T}
    temp6::Matrix{T}
end
function SuperconductivityMeasurement(mc::DQMC, model)
    T = eltype(mc.s.greens)
    N = nsites(model)

    s_wave      = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "s-wave equal time pairing correlation", "Observables.jld", "etpc-s"
    )
    ext_s_wave      = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "extended s-wave equal time pairing correlation", "Observables.jld", "etpc-se"
    )
    dxy_wave    = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "d_xy-wave equal time pairing correlation", "Observables.jld", "etpc-dxy"
    )
    dx2_y2_wave = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "d_{x²-y²}-wave equal time pairing correlation", "Observables.jld", "etpc-dx2y2"
    )
    f_wave      = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "f-wave equal time pairing correlation", "Observables.jld", "etpc-f"
    )
    py_wave     = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "p_y-wave equal time pairing correlation", "Observables.jld", "etpc-py"
    )
    px_wave     = LightObservable(
        LogBinner(zeros(T, N, N), capacity=1_000_000),
        "p_x-wave equal time pairing correlation", "Observables.jld", "etpc-px"
    )

    SuperconductivityMeasurement(
        s_wave, ext_s_wave, dxy_wave, dx2_y2_wave, f_wave, py_wave, px_wave,
        zeros(T, N, N), zeros(T, N, N), zeros(T, N, N),
        zeros(T, N, N), zeros(T, N, N), zeros(T, N, N)
    )
end
function measure!(m::SuperconductivityMeasurement, mc::DQMC, model, i::Int64)
    # Equal time pairing correlation
    # Assumptions:
    # - neighs are always ordered the same, i.e. all neighs[1, :] point in the
    #   same direction
    # - the upper left block of G is spin up - spin up, the lower right spin down
    #   spin down
    G = greens(mc)
    IG = I - G
    N = nsites(model)

    # see 10.1103/PhysRevB.72.134513
    # f[i, :] are the prefactors for [s, dxy, dx2-y2, f, py, px][i]
    f = [
         1  1  1  1  1  1;
         0 -1  1  0 -1  1;
         2 -1 -1  2 -1 -1;
        -1  1 -1  1 -1  1;
         0  1  1  0 -1 -1;
        -2 -1  1  2  1 -1
    ]

    for i in 1:N, j in 1:N
        temp = zeros(eltype(G), 6)
        for (k, ip) in enumerate(model.l.neighs[:, i])
            for (l, jp) in enumerate(model.l.neighs[:, j])
                # x = prefactor(i, dir) * prefactor(j, dir) *
                temp .+= f[:, k] .* f[:, l] * (
                    G[i, j] * G[ip+N, jp+N] - G[i, jp+N] * G[ip+N, j]
                )
            end
        end
        m.temp1[i, j] = temp[1]
        m.temp2[i, j] = temp[2]
        m.temp3[i, j] = temp[3]
        m.temp4[i, j] = temp[4]
        m.temp5[i, j] = temp[5]
        m.temp6[i, j] = temp[6]
    end
    push!(m.s_wave, G[1:N, 1:N] .* G[N+1:2N, N+1:2N] - G[1:N, N+1:2N] .* G[N+1:2N, 1:N])
    push!(m.ext_s_wave, m.temp1)
    push!(m.dxy_wave, m.temp2)
    push!(m.dx2_y2_wave, m.temp3)
    push!(m.f_wave, m.temp4)
    push!(m.py_wave, m.temp5)
    push!(m.px_wave, m.temp6)
end


# # nah
# struct ChiralityMeasurement{
#         OT <: AbstractObservable
#     } <: SpinOneHalfMeasurement
#     triplets::Vector{Vector{Int64}}
#     obs::OT
# end
# function ChiralityMeasurement(mc::DQMC, model::ZCModel)
#     NN = model.l.neighs
#     # Generate triplets of sites which from a triangle.
#     # each triplet rotates the same way
#     triplets = [
#         [src, NN[mod1(4+i, 6), src], NN[mod1(6+i, 6), NN[mod1(4+i, 6), src]]]
#         for src in 1:nsites(model) for i in 0:1
#     ]
#     T = eltype(mc.s.greens)
#     obs = LightObservable(
#         LogBinner(zeros(T <: Complex ? T : Complex{T}, length(triplets))),
#         "Plaquette Chirality", "Observables.jld", "Plaquette Chirality"
#     )
#     ChiralityMeasurement(triplets, obs)
# end
# function measure!(m::ChiralityMeasurement, mc::DQMC, model, i)
#     N = nsites(model)
#     G = greens(mc)
#     values = map(m.triplets) do t
#         i1, i2, i3 = t
#         spins = [[
#             -G[i+N, i] - G[i, i+N],
#             -1im * (G[i, i+N] - G[i+N, i]),
#             G[i+N, i+N] - G[i, i]
#         ] for i in t]
#         dot(cross(spins[1], spins[2]), spins[3]) / 8
#     end
#     push!(m.obs, values)
#     nothing
# end



function default_measurements(mc::DQMC, model::ZCModel)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model),
        :Magnetization => MagnetizationMeasurement(mc, model),
        :Superconductivity => SuperconductivityMeasurement(mc, model)
        # :Chirality => ChiralityMeasurement(mc, model)
    )
end
