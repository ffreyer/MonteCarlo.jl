# NOTE
# - fairly confident in pairing correlation. Did crosscheck for s-wave, but
#   never anything else
# - not confident at all in CDC and SDC - these are an educated guess based of PC

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

    temp::Matrix{T}
end
function SuperconductivityMeasurement(mc::DQMC, model, shape=get_lattice_shape(model))
    T = eltype(mc.s.greens)
    N = nsites(model)

    s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "s-wave equal time pairing correlation", "Observables.jld", "etpc-s"
    )
    ext_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "extended s-wave equal time pairing correlation", "Observables.jld", "etpc-se"
    )
    dxy_wave    = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_xy-wave equal time pairing correlation", "Observables.jld", "etpc-dxy"
    )
    dx2_y2_wave = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_{x²-y²}-wave equal time pairing correlation", "Observables.jld", "etpc-dx2y2"
    )
    f_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "f-wave equal time pairing correlation", "Observables.jld", "etpc-f"
    )
    py_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_y-wave equal time pairing correlation", "Observables.jld", "etpc-py"
    )
    px_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_x-wave equal time pairing correlation", "Observables.jld", "etpc-px"
    )

    SuperconductivityMeasurement(
        s_wave, ext_s_wave, dxy_wave, dx2_y2_wave, f_wave, py_wave, px_wave,
        reshape(zeros(T, N), shape)
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

    m.temp .= zero(eltype(m.temp))
    for i in 1:N
        for delta in 0:N-1
            j = mod1(i + delta, N)
            m.temp[delta+1] += G[i, j] * G[i+N, j+N] - G[i, j+N] * G[i+N, j]
        end
    end
    push!(m.s_wave, m.temp / N)

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
    obs = [m.ext_s_wave, m.dxy_wave, m.dx2_y2_wave, m.f_wave, m.py_wave, m.px_wave]
    for sym in 1:6
        m.temp .= zero(eltype(m.temp))
        for i in 1:N
            for delta in 0:N-1
                j = mod1(i + delta, N)
                for (k, ip) in enumerate(model.l.neighs[:, i])
                    for (l, jp) in enumerate(model.l.neighs[:, j])
                        # x = prefactor(i, dir) * prefactor(j, dir) *
                        m.temp[delta+1] += f[sym, k] .* f[sym, l] * (
                            G[i, j] * G[ip+N, jp+N] - G[i, jp+N] * G[ip+N, j]
                        )
                    end
                end
            end
        end
        push!(obs[sym], m.temp / N)
    end
end


struct SpinDensityCorrelationMeasurement2{
        OT <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement

    x_s_wave::OT
    x_ext_s_wave::OT
    x_dxy_wave::OT
    x_dx2_y2_wave::OT
    x_f_wave::OT
    x_py_wave::OT
    x_px_wave::OT

    y_s_wave::OT
    y_ext_s_wave::OT
    y_dxy_wave::OT
    y_dx2_y2_wave::OT
    y_f_wave::OT
    y_py_wave::OT
    y_px_wave::OT

    z_s_wave::OT
    z_ext_s_wave::OT
    z_dxy_wave::OT
    z_dx2_y2_wave::OT
    z_f_wave::OT
    z_py_wave::OT
    z_px_wave::OT

    temp::AT
end
function SpinDensityCorrelationMeasurement2(mc::DQMC, model; shape=get_lattice_shape(model))
    N = nsites(model)
    T = eltype(mc.s.greens)
    Ty = T <: Complex ? T : Complex{T}

    # Spin density correlation
    sdc_x_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "s-wave equal time spin denisty correlation (x)", "Observables.jld", "sdc_x_s"
    )
    sdc_x_ext_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "extended s-wave equal time spin denisty correlation (x)", "Observables.jld", "sdc_x_se"
    )
    sdc_x_dxy_wave    = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_xy-wave equal time spin denisty correlation (x)", "Observables.jld", "sdc_x_dxy"
    )
    sdc_x_dx2_y2_wave = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_{x²-y²}-wave equal time spin denisty correlation (x)", "Observables.jld", "etpcdx2y2"
    )
    sdc_x_f_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "f-wave equal time spin denisty correlation (x)", "Observables.jld", "sdc_x_f"
    )
    sdc_x_py_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_y-wave equal time spin denisty correlation (x)", "Observables.jld", "sdc_x_py"
    )
    sdc_x_px_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_x-wave equal time spin denisty correlation (x)", "Observables.jld", "sdc_x_px"
    )


    sdc_y_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "s-wave equal time spin denisty correlation (y)", "Observables.jld", "sdc_y_s"
    )
    sdc_y_ext_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "extended s-wave equal time spin denisty correlation (y)", "Observables.jld", "sdc_y_se"
    )
    sdc_y_dxy_wave    = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_xy-wave equal time spin denisty correlation (y)", "Observables.jld", "sdc_y_dxy"
    )
    sdc_y_dx2_y2_wave = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_{x²-y²}-wave equal time spin denisty correlation (y)", "Observables.jld", "etpcdx2y2"
    )
    sdc_y_f_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "f-wave equal time spin denisty correlation (y)", "Observables.jld", "sdc_y_f"
    )
    sdc_y_py_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_y-wave equal time spin denisty correlation (y)", "Observables.jld", "sdc_y_py"
    )
    sdc_y_px_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_x-wave equal time spin denisty correlation (y)", "Observables.jld", "sdc_y_px"
    )


    sdc_z_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "s-wave equal time spin density correlation (z)", "Observables.jld", "sdc_z_s"
    )
    sdc_z_ext_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "extended s-wave equal time spin density correlation (z)", "Observables.jld", "sdc_z_se"
    )
    sdc_z_dxy_wave    = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_xy-wave equal time spin density correlation (z)", "Observables.jld", "sdc_z_dxy"
    )
    sdc_z_dx2_y2_wave = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_{x²-y²}-wave equal time spin density correlation (z)", "Observables.jld", "etpcdx2y2"
    )
    sdc_z_f_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "f-wave equal time spin density correlation (z)", "Observables.jld", "sdc_z_f"
    )
    sdc_z_py_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_y-wave equal time spin density correlation (z)", "Observables.jld", "sdc_z_py"
    )
    sdc_z_px_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_x-wave equal time spin density correlation (z)", "Observables.jld", "sdc_z_px"
    )


    SpinDensityCorrelationMeasurement2(
        sdc_x_s_wave, sdc_x_ext_s_wave, sdc_x_dxy_wave, sdc_x_dx2_y2_wave, sdc_x_f_wave, sdc_x_py_wave, sdc_x_px_wave,
        sdc_y_s_wave, sdc_y_ext_s_wave, sdc_y_dxy_wave, sdc_y_dx2_y2_wave, sdc_y_f_wave, sdc_y_py_wave, sdc_y_px_wave,
        sdc_z_s_wave, sdc_z_ext_s_wave, sdc_z_dxy_wave, sdc_z_dx2_y2_wave, sdc_z_f_wave, sdc_z_py_wave, sdc_z_px_wave,
        reshape([zero(T) for _ in 1:N], shape)
    )
end
@bm function measure!(m::SpinDensityCorrelationMeasurement2, mc::DQMC, model, i::Int64)
    N = nsites(model)
    G = greens(mc, model)
    IG = I - G

    # G[1:N,    1:N]    up -> up section
    # G[N+1:N,  1:N]    down -> up section
    # ...
    # G[i, j] = c_i c_j^†

    obs = [
        [m.x_ext_s_wave, m.x_dxy_wave, m.x_dx2_y2_wave, m.x_f_wave, m.x_py_wave, m.x_px_wave],
        [m.y_ext_s_wave, m.y_dxy_wave, m.y_dx2_y2_wave, m.y_f_wave, m.y_py_wave, m.y_px_wave],
        [m.z_ext_s_wave, m.z_dxy_wave, m.z_dx2_y2_wave, m.z_f_wave, m.z_py_wave, m.z_px_wave]
    ]
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


    # Spin Density Correlation  --[X]--
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
    push!(m.x_s_wave, m.temp / N)

    for sym in 1:6
        m.temp .= zero(eltype(m.temp))
        for i in 1:N
            for delta in 0:N-1
                j = mod1(i + delta, N)
                for (k, ip) in enumerate(model.l.neighs[:, i])
                    for (l, jp) in enumerate(model.l.neighs[:, j])
                        m.temp[delta+1] += f[sym, k] .* f[sym, l] * (
                            IG[ip+N, i] * IG[jp+N, j] + IG[jp+N, i] * G[ip+N, j] +
                            IG[ip+N, i] * IG[jp, j+N] + IG[jp, i] * G[ip+N, j+N] +
                            IG[ip, i+N] * IG[jp+N, j] + IG[jp+N, i+N] * G[ip, j] +
                            IG[ip, i+N] * IG[jp, j+N] + IG[jp, i+N] * G[ip, j+N]
                        )
                    end
                end
            end
        end
        push!(obs[1][sym], m.temp / N)
    end


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
    push!(m.y_s_wave, m.temp / N)

    for sym in 1:6
        m.temp .= zero(eltype(m.temp))
        for i in 1:N
            for delta in 0:N-1
                j = mod1(i + delta, N)
                for (k, ip) in enumerate(model.l.neighs[:, i])
                    for (l, jp) in enumerate(model.l.neighs[:, j])
                        m.temp[delta+1] += f[sym, k] .* f[sym, l] * (
                            - IG[ip+N, i] * IG[jp+N, j] - IG[jp+N, i] * G[ip+N, j] +
                              IG[ip+N, i] * IG[jp, j+N] + IG[jp, i] * G[ip+N, j+N] +
                              IG[ip, i+N] * IG[jp+N, j] + IG[jp+N, i+N] * G[ip, j] -
                              IG[ip, i+N] * IG[jp, j+N] - IG[jp, i+N] * G[ip, j+N]
                        )
                    end
                end
            end
        end
        push!(obs[2][sym], m.temp / N)
    end


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
    push!(m.z_s_wave, m.temp / N)

    for sym in 1:6
        m.temp .= zero(eltype(m.temp))
        for i in 1:N
            for delta in 0:N-1
                j = mod1(i + delta, N)
                for (k, ip) in enumerate(model.l.neighs[:, i])
                    for (l, jp) in enumerate(model.l.neighs[:, j])
                        m.temp[delta+1] += f[sym, k] .* f[sym, l] * (
                            IG[ip, i] * IG[jp, j] + IG[jp, i] * G[ip, j] -
                            IG[ip, i] * IG[jp+N, j+N] - IG[jp+N, i] * G[ip, j+N] -
                            IG[ip+N, i+N] * IG[jp, j] - IG[jp, i+N] * G[ip+N, j] +
                            IG[ip+N, i+N] * IG[jp+N, j+N] + IG[jp+N, i+N] * G[ip+N, j+N]
                        )
                    end
                end
            end
        end
        push!(obs[3][sym], m.temp / N)
    end
end


struct ChargeDensityCorrelationMeasurement2{
        OT <: AbstractObservable,
        AT <: Array
    } <: SpinOneHalfMeasurement
    s_wave::OT
    ext_s_wave::OT
    dxy_wave::OT
    dx2_y2_wave::OT
    f_wave::OT
    py_wave::OT
    px_wave::OT
    temp::AT
end
function ChargeDensityCorrelationMeasurement2(mc::DQMC, model; shape=get_lattice_shape(model))
    N = nsites(model)
    T = eltype(mc.s.greens)

    s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "s-wave equal time charge density correlation", "Observables.jld", "cdc_s"
    )
    ext_s_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "extended s-wave equal time charge density correlation", "Observables.jld", "cdc_se"
    )
    dxy_wave    = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_xy-wave equal time charge density correlation", "Observables.jld", "cdc_dxy"
    )
    dx2_y2_wave = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "d_{x²-y²}-wave equal time charge density correlation", "Observables.jld", "etpcdx2y2"
    )
    f_wave      = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "f-wave equal time charge density correlation", "Observables.jld", "cdc_f"
    )
    py_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_y-wave equal time charge density correlation", "Observables.jld", "cdc_py"
    )
    px_wave     = LightObservable(
        LogBinner(reshape(zeros(T, N), shape), capacity=1_000_000),
        "p_x-wave equal time charge density correlation", "Observables.jld", "cdc_px"
    )

    ChargeDensityCorrelationMeasurement2(
        s_wave, ext_s_wave, dxy_wave, dx2_y2_wave, f_wave, py_wave, px_wave,
        reshape([zero(T) for _ in 1:N], shape)
    )
end
@bm function measure!(m::ChargeDensityCorrelationMeasurement2, mc::DQMC, model, i::Int64)
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
    push!(m.s_wave, m.temp / N)

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
    obs = [m.ext_s_wave, m.dxy_wave, m.dx2_y2_wave, m.f_wave, m.py_wave, m.px_wave]
    for sym in 1:6
        m.temp .= zero(eltype(m.temp))
        for i in 1:N
            for delta in 0:N-1
                j = mod1(i + delta, N)
                for (k, ip) in enumerate(model.l.neighs[:, i])
                    for (l, jp) in enumerate(model.l.neighs[:, j])
                        m.temp[delta+1] += f[sym, k] .* f[sym, l] * (
                            # ⟨n↑n↑⟩
                            IG[ip, i] * IG[jp, j] +
                            IG[jp, i] *  G[ip, j] +
                            # ⟨n↑n↓⟩
                            IG[ip, i] * IG[jp + N, j + N] +
                            IG[jp + N, i] *  G[ip, j + N] +
                            # ⟨n↓n↑⟩
                            IG[ip + N, i + N] * IG[jp, j] +
                            IG[jp, i + N] *  G[ip + N, j] +
                            # ⟨n↓n↓⟩
                            IG[ip + N, i + N] * IG[jp + N, j + N] +
                            IG[jp + N, i + N] *  G[ip + N, j + N]
                        )
                    end
                end
            end
        end
        push!(obs[sym], m.temp / N)
    end

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
