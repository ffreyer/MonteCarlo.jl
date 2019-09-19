struct SuperconductivityMeasurement{
        OT <: AbstractObservable,
        T
    } <: SpinOneHalfMeasurement
    s_wave::OT
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
function SuperconductivityMeasurement(mc::DQMC, model::ZCModel)
    T = eltype(mc.s.greens)
    N = nsites(model)

    s_wave      = LightObservable(
        LogBinner(zeros(T, N, N)),
        "s-wave equal time pairing correlation", "Observables.jld", "s-wave equal time pairing correlation"
    )
    dxy_wave    = LightObservable(
        LogBinner(zeros(T, N, N)),
        "d_xy-wave equal time pairing correlation", "Observables.jld", "d_xy-wave equal time pairing correlation"
    )
    dx2_y2_wave = LightObservable(
        LogBinner(zeros(T, N, N)),
        "d_{x²-y²}-wave equal time pairing correlation", "Observables.jld", "d_{x²-y²}-wave equal time pairing correlation"
    )
    f_wave      = LightObservable(
        LogBinner(zeros(T, N, N)),
        "f-wave equal time pairing correlation", "Observables.jld", "f-wave equal time pairing correlation"
    )
    py_wave     = LightObservable(
        LogBinner(zeros(T, N, N)),
        "p_y-wave equal time pairing correlation", "Observables.jld", "p_y-wave equal time pairing correlation"
    )
    px_wave     = LightObservable(
        LogBinner(zeros(T, N, N)),
        "p_x-wave equal time pairing correlation", "Observables.jld", "p_x-wave equal time pairing correlation"
    )

    SuperconductivityMeasurement(
        s_wave, dxy_wave, dx2_y2_wave, f_wave, py_wave, px_wave,
        zeros(T, N, N), zeros(T, N, N), zeros(T, N, N),
        zeros(T, N, N), zeros(T, N, N), zeros(T, N, N)
    )
end
function measure!(m::SuperconductivityMeasurement, mc::DQMC, model::ZCModel, i::Int64)
    # Equal time pairing correlation
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
        temp1 = zeros(eltype(G), 6)
        temp2 = zeros(eltype(G), 6)
        for (k, ip) in enumerate(model.l.neighs[:, i])
            for (l, jp) in enumerate(model.l.neighs[:, j])
                temp1 .+= f[:, k] .* f[:, l] * IG[jp + N, ip + N]
                temp2 .+= f[:, k] .* f[:, l] * IG[ip + N, jp + N]
            end
        end
        m.temp1[i, j] = -0.25 * (IG[j, i] * temp1[1] + temp2[1] * IG[i, j])
        m.temp2[i, j] = -0.25 * (IG[j, i] * temp1[2] + temp2[2] * IG[i, j])
        m.temp3[i, j] = -0.25 * (IG[j, i] * temp1[3] + temp2[3] * IG[i, j])
        m.temp4[i, j] = -0.25 * (IG[j, i] * temp1[4] + temp2[4] * IG[i, j])
        m.temp5[i, j] = -0.25 * (IG[j, i] * temp1[5] + temp2[5] * IG[i, j])
        m.temp6[i, j] = -0.25 * (IG[j, i] * temp1[6] + temp2[6] * IG[i, j])
    end
    push!(m.s_wave, m.temp1)
    push!(m.dxy_wave, m.temp2)
    push!(m.dx2_y2_wave, m.temp3)
    push!(m.f_wave, m.temp4)
    push!(m.py_wave, m.temp5)
    push!(m.px_wave, m.temp6)
end
function save(m::SuperconductivityMeasurement, filename)
    saveobs(m.s_wave, filename)
    saveobs(m.dxy_wave, filename)
    saveobs(m.dx2_y2_wave, filename)
    saveobs(m.f_wave, filename)
    saveobs(m.py_wave, filename)
    saveobs(m.px_wave, filename)
end



struct ChiralityMeasurement{
        OT <: AbstractObservable
    } <: SpinOneHalfMeasurement
    triplets::Vector{Vector{Int64}}
    obs::OT
end
function ChiralityMeasurement(mc::DQMC, model::ZCModel)
    NN = model.l.neighs
    # Generate triplets of sites which from a triangle.
    # each triplet rotates the same way
    triplets = [
        [src, NN[mod1(4+i, 6), src], NN[mod1(6+i, 6), NN[mod1(4+i, 6), src]]]
        for src in 1:nsites(model) for i in 0:1
    ]
    T = eltype(mc.s.greens)
    obs = LightObservable(
        LogBinner(zeros(T <: Complex ? T : Complex{T}, length(triplets))),
        "Plaquette Chirality", "Observables.jld", "Plaquette Chirality"
    )
    ChiralityMeasurement(triplets, obs)
end
function measure!(m::ChiralityMeasurement, mc::DQMC, model, i)
    N = nsites(model)
    G = greens(mc)
    values = map(m.triplets) do t
        i1, i2, i3 = t
        spins = [[
            -G[i+N, i] - G[i, i+N],
            -1im * (G[i, i+N] - G[i+N, i]),
            G[i+N, i+N] - G[i, i]
        ] for i in t]
        dot(cross(spins[1], spins[2]), spins[3]) / 8
    end
    push!(m.obs, values)
    nothing
end



function default_measurements(mc::DQMC, model::ZCModel)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model),
        :Magnetization => MagnetizationMeasurement(mc, model),
        :Superconductivity => SuperconductivityMeasurement(mc, model),
        :Chirality => ChiralityMeasurement(mc, model)
    )
end
