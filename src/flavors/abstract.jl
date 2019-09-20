# abstract monte carlo flavor definition
"""
Abstract definition of a Monte Carlo flavor.
"""
abstract type MonteCarloFlavor end

# MonteCarloFlavor interface: mandatory
"""
    init!(mc)

Initialize the Monte Carlo simulation. Has an alias function `reset!`.
"""
init!(mc::MonteCarloFlavor) = error("MonteCarloFlavor $(typeof(mc)) doesn't implement `init!`!")

"""
    run!(mc)

Run the Monte Carlo Simulation.
"""
run!(mc::MonteCarloFlavor) = error("MonteCarloFlavor $(typeof(mc)) doesn't implement `run!`!")


"""
    reset!(mc::MonteCarloFlavor)

Resets the Monte Carlo simulation `mc`.
Previously set parameters will be retained.
"""
reset!(mc::MonteCarloFlavor) = init!(mc) # convenience mapping
