const p = "temp_dir"
isdir(p) || mkdir(p)
rm.(joinpath.(p, readdir(p)))

@testset "MC" begin
    model = IsingModel(dims=2, L=2)
    mc = MC(model, beta=0.66, thermalization=33, sweeps=123)
    run!(mc, verbose=false)
    MonteCarlo.save("$p/testfile.jld", mc)
    x = MonteCarlo.load("$p/testfile.jld")
    rm("$p/testfile.jld")

    # Repeat these tests once with x being replayed rather than loaded
    replay_done = false
    @label all_checks

    for f in fieldnames(typeof(mc.p))
        @test getfield(mc.p, f) == getfield(x.p, f)
    end
    @test mc.conf == x.conf
    @test mc.model.L == x.model.L
    @test mc.model.dims == x.model.dims
    for f in fieldnames(typeof(mc.model.l))
        @test getfield(mc.model.l, f) == getfield(x.model.l, f)
    end
    @test mc.model.neighs == x.model.neighs
    @test mc.model.energy[] == x.model.energy[]
    for (k, v) in mc.thermalization_measurements
        for f in fieldnames(typeof(v))
            @test getfield(v, f) == getfield(x.thermalization_measurements[k], f)
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            @test getfield(v, f) == getfield(x.measurements[k], f)
        end
    end

    # Check everything again with x being a replayed simulation
    if !replay_done
        replay!(x)
        replay_done = true
        @goto all_checks
    end

    # Test resume
    model = IsingModel(dims=2, L=10)
    mc = MC(model, beta=1.0)
    run!(mc, verbose=false, thermalization=1, sweeps=2mc.p.measure_rate+1)
    mc = MC(model, beta=1.0, thermalization=50, sweeps=100_000, measure_rate=100)
    t = time()
    run!(mc, verbose=false)
    t = time() - t

    mc = MC(model, beta=1.0, thermalization=50, sweeps=300_000, measure_rate=100)
    state = run!(
        mc, verbose=false,
        safe_before = now() + Millisecond(round(Int, 1000t)),
        grace_period = Millisecond(0),
        resumable_filename = "$p/resumable_testfile.jld"
    )
    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) > 1

    # Force resume! to compile
    resume!(
        "$p/resumable_testfile.jld",
        safe_before=now(),
        grace_period=Millisecond(0),
        resumable_filename = "$p/temp.jld",
        verbose=false
    )
    rm("$p/temp.jld")
    mc, state = resume!(
        "$p/resumable_testfile.jld",
        verbose=false,
        safe_before = now() + Millisecond(round(Int, 1100t)),
        grace_period = Millisecond(0),
        force_overwrite = true,
        resumable_filename = "$p/resumable_testfile.jld"
    )
    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) > 1
    @test length(readdir(p)) == 1

    mc, state = resume!("$p/resumable_testfile.jld", verbose=false)
    @test state == true
    @test all(x in timeseries(mc.measurements[:conf].obs) for x in ts)
    rm("$p/resumable_testfile.jld")
end


rm.(joinpath.(p, readdir(p)))

@testset "DQMC" begin
    model = HubbardModelAttractive(dims=2, L=4, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=21, sweeps=117, measure_rate = 1)
    t = time()
    run!(mc, verbose=false)
    t = time() - t
    MonteCarlo.save("$p/testfile.jld", mc)
    x = MonteCarlo.load("$p/testfile.jld")
    rm("$p/testfile.jld")

    # Repeat these tests once with x being replayed rather than loaded
    replay_done = false
    @label all_checks

    for f in fieldnames(typeof(mc.p))
        @test getfield(mc.p, f) == getfield(x.p, f)
    end
    @test mc.conf == x.conf
    @test mc.model.dims == x.model.dims
    @test mc.model.L == x.model.L
    @test mc.model.mu == x.model.mu
    @test mc.model.t == x.model.t
    @test mc.model.U == x.model.U
    for f in fieldnames(typeof(mc.model.l))
        @test getfield(mc.model.l, f) == getfield(x.model.l, f)
    end
    @test mc.model.neighs == x.model.neighs
    @test mc.model.flv == x.model.flv
    for (k, v) in mc.thermalization_measurements
        for f in fieldnames(typeof(v))
            if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LightObservable in MonteCarloObservable
                @test getfield(v, f).B == getfield(x.measurements[k], f).B
            else
                @test getfield(v, f) == getfield(x.measurements[k], f)
            end
        end
    end
    for (k, v) in mc.measurements
        for f in fieldnames(typeof(v))
            if getfield(v, f) isa LightObservable
                # TODO
                # implement == for LightObservable in MonteCarloObservable
                @test getfield(v, f).B == getfield(x.measurements[k], f).B
            else
                @test getfield(v, f) == getfield(x.measurements[k], f)
            end
        end
    end

    # Check everything again with x being a replayed simulation
    if !replay_done
        replay!(x)
        replay_done = true
        @goto all_checks
    end


    # Test resume
    model = HubbardModelAttractive(dims=2, L=2, t = 1.7, U = 5.5)
    mc = DQMC(model, beta=1.0, thermalization=500, sweeps=1000, measure_rate=1)
    t = time()
    run!(mc, verbose=false)
    t = time() - t
    mc = DQMC(model, beta=1.0, thermalization=500, sweeps=3000, measure_rate=1)
    state = run!(
        mc, verbose=false,
        safe_before = now() + Millisecond(round(Int, 950t)),
        grace_period = Millisecond(0),
        resumable_filename = "$p/resumable_testfile.jld"
    )
    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) > 1

    mc, state = resume!(
        "$p/resumable_testfile.jld",
        verbose=false,
        safe_before = now() + Millisecond(round(Int, 1100t)),
        grace_period = Millisecond(0),
        force_overwrite = true,
        resumable_filename = "$p/resumable_testfile.jld"
    )
    @test state == false
    ts = deepcopy(timeseries(mc.measurements[:conf].obs))
    @assert length(ts) > 1
    @test length(readdir(p)) == 1

    mc, state = resume!("$p/resumable_testfile.jld", verbose=false)
    @test state == true
    @test all(x in timeseries(mc.measurements[:conf].obs) for x in ts)
    rm("$p/resumable_testfile.jld")
end

isdir(p) && rm(p, recursive=true)