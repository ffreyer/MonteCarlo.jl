using MonteCarlo, MonteCarloObservable
using Test
using Random
using MonteCarlo: @bm, TimerOutputs


@testset "All Tests" begin
    @testset "Utilities" begin
        @bm function test1(x, y)
            sleep(x+y)
        end
        @bm test2(x, y) = sleep(x+y)
        function test3(x, y)
            TimerOutputs.@timeit_debug "test3" begin sleep(x+y) end
        end
        test4(x, y) = TimerOutputs.@timeit_debug "test4" begin sleep(x+y) end

        TimerOutputs.enable_debug_timings(Main)
        x, y = 0.005, 0.005
        MonteCarlo.test1(x, y)
        MonteCarlo.test2(x, y)
        test3(x, y)
        test4(x, y)
        TimerOutputs.reset_timer!()
        for _ in 1:10
            MonteCarlo.test1(x, y)
            MonteCarlo.test2(x, y)
            test3(x, y)
            test4(x, y)
        end
        TimerOutputs.disable_debug_timings(Main)

        to = TimerOutputs.DEFAULT_TIMER
        t1 = TimerOutputs.time(to["test1"])
        t2 = TimerOutputs.time(to["test2"])
        t3 = TimerOutputs.time(to["test3"])
        t4 = TimerOutputs.time(to["test4"])

        @test t1 ≈ t2 rtol=0.01
        @test t2 ≈ t3 rtol=0.01
        @test t3 ≈ t4 rtol=0.01
    end

    @testset "Lattices" begin
        include("lattices.jl")
    end

    @testset "Models" begin
        include("modeltests_IsingModel.jl")
        include("modeltests_HubbardModelAttractive.jl")
    end

    @testset "Flavors" begin
        # include("flavortests_MC.jl")
        include("flavortests_DQMC.jl")
    end

    @testset "Measurements" begin
        include("measurements.jl")
    end

    @testset "Intergration tests" begin
        include("integration_tests.jl")
    end

    @testset "Exact Diagonalization" begin
        include("ED/ED_tests.jl")
    end
end
