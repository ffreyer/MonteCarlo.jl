var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Documentation-1",
    "page": "Home",
    "title": "Documentation",
    "category": "section",
    "text": "TODO"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install the package execute the following command in the REPL:Pkg.clone(\"https://github.com/crstnbr/MonteCarlo.jl\")Afterwards, you can use MonteCarlo.jl like any other package installed with Pkg.add():using MonteCarloTo obtain the latest version of the package just do Pkg.update() or specifically Pkg.update(\"MonteCarlo\")."
},

{
    "location": "manual/gettingstarted.html#",
    "page": "Getting started",
    "title": "Getting started",
    "category": "page",
    "text": ""
},

{
    "location": "manual/gettingstarted.html#Manual-1",
    "page": "Getting started",
    "title": "Manual",
    "category": "section",
    "text": ""
},

{
    "location": "manual/gettingstarted.html#Installation-/-Updating-1",
    "page": "Getting started",
    "title": "Installation / Updating",
    "category": "section",
    "text": "To install the package execute the following command in the REPL:Pkg.clone(\"https://github.com/crstnbr/MonteCarlo.jl\")To obtain the latest version of the package just do Pkg.update() or specifically Pkg.update(\"MonteCarlo\")."
},

{
    "location": "manual/gettingstarted.html#Example-1",
    "page": "Getting started",
    "title": "Example",
    "category": "section",
    "text": "This is a simple demontration of how to perform a classical Monte Carlo simulation of the 2D Ising model:# load packages\nusing MonteCarlo, MonteCarloObservable\n\n# load your model\nm = IsingModel(dims=2, L=8, β=0.35);\nobservables(m) # what observables do exist for that model?\n\n# choose a Monte Carlo flavor and run the simulation\nmc = MC(m);\nrun!(mc, sweeps=1000, thermalization=1000, verbose=false);\n\n# analyze results\nm = mc.obs[\"m\"] # magnetization\nmean(m)\nstd(m) # one-sigma error\n\n# create standard plots\nhist(m)\nplot(m)(Image: ) (Image: )"
},

{
    "location": "manual/examples.html#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "manual/examples.html#Examples-1",
    "page": "Examples",
    "title": "Examples",
    "category": "section",
    "text": ""
},

{
    "location": "manual/examples.html#D-Ising-model-1",
    "page": "Examples",
    "title": "2D Ising model",
    "category": "section",
    "text": "Results: (Image: )Code:using MonteCarlo, Distributions, PyPlot, DataFrames, JLD\n\nTdist = Normal(MonteCarlo.IsingTc, .64)\nn_Ts = 2^8\nTs = sort!(rand(Tdist, n_Ts))\nTs = Ts[Ts.>=1.2]\nTs = Ts[Ts.<=3.8]\ntherm = 10^4\nsweeps = 10^3\n\ndf = DataFrame(L=Int[], T=Float64[], M=Float64[], χ=Float64[], E=Float64[], C_V=Float64[])\n\nfor L in 2.^[3, 4, 5, 6]\n	println(\"L = \", L)\n	for (i, T) in enumerate(Ts)\n		println(\"\\t T = \", T)\n		β = 1/T\n		model = IsingModel(dims=2, L=L, β=β)\n		mc = MC(model)\n		obs = run!(mc, sweeps=sweeps, thermalization=therm, verbose=false)\n		push!(df, [L, T, mean(obs[\"m\"]), mean(obs[\"χ\"]), mean(obs[\"e\"]), mean(obs[\"C\"])])\n	end\n	flush(STDOUT)\nend\n\nsort!(df, cols = [:L, :T])\n@save \"ising2d.jld\" df\n\n# plot results together\ngrps = groupby(df, :L)\nfig, ax = subplots(2,2, figsize=(12,8))\nfor g in grps\n	L = g[:L][1]\n	ax[1][:plot](g[:T], g[:E], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[2][:plot](g[:T], g[:C_V], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[3][:plot](g[:T], g[:M], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\n	ax[4][:plot](g[:T], g[:χ], \"o\", markeredgecolor=\"black\", label=\"L=$L\")\nend\nax[1][:legend](loc=\"best\")\nax[1][:set_ylabel](\"Energy\")\nax[1][:set_xlabel](\"Temperature\")\n\nax[2][:set_ylabel](\"Specific heat\")\nax[2][:set_xlabel](\"Temperature\")\nax[2][:axvline](x=MonteCarlo.IsingTc, color=\"black\", linestyle=\"dashed\", label=\"\\$ T_c \\$\")\nax[2][:legend](loc=\"best\")\n\nax[3][:set_ylabel](\"Magnetization\")\nax[3][:set_xlabel](\"Temperature\")\nx = linspace(1.2, MonteCarlo.IsingTc, 100)\ny = (1-sinh.(2.0 ./ (x)).^(-4)).^(1/8)\nax[3][:plot](x,y, \"k--\", label=\"exact\")\nax[3][:plot](linspace(MonteCarlo.IsingTc, 3.8, 100), zeros(100), \"k--\")\nax[3][:legend](loc=\"best\")\n\nax[4][:set_ylabel](\"Susceptibility χ\")\nax[4][:set_xlabel](\"Temperature\")\nax[4][:axvline](x=MonteCarlo.IsingTc, color=\"black\", linestyle=\"dashed\", label=\"\\$ T_c \\$\")\nax[4][:legend](loc=\"best\")\ntight_layout()\nsavefig(\"ising2d.pdf\")"
},

{
    "location": "manual/custommodels.html#",
    "page": "Custom models",
    "title": "Custom models",
    "category": "page",
    "text": ""
},

{
    "location": "manual/custommodels.html#Custom-models-1",
    "page": "Custom models",
    "title": "Custom models",
    "category": "section",
    "text": "Although MonteCarlo.jl already ships with famous models, foremost the Ising and Hubbard models, a key idea of the design of the package is to have a (rather) well defined interface between models and Monte Carlo flavors. This way it should be easy for you to implement your own physical model. Sometimes examples say more than words, so feel encouraged to have a look at the implementations of the above mentioned models."
},

{
    "location": "manual/custommodels.html#Mandatory-fields-and-methods-1",
    "page": "Custom models",
    "title": "Mandatory fields and methods",
    "category": "section",
    "text": "Any concrete model type, let's call it MyModel in the following, must be a subtype of the abstract type MonteCarlo.Model. Internally it must have at least the following fields:β::Float64: temperature (depends on MC flavor if this will actually be used)\nl::Lattice: any MonteCarlo.LatticeFurthermore it must implement the following methods:conftype(m::Model): type of a configuration\nenergy(m::Model, conf): energy of configuration\nrand(m::Model): random configuration\npropose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi: propose local move\naccept_local(m::Model, i::Int, conf, E::Float64): accept a local move"
},

{
    "location": "methods/general.html#",
    "page": "General",
    "title": "General",
    "category": "page",
    "text": ""
},

{
    "location": "methods/general.html#Methods:-General-1",
    "page": "General",
    "title": "Methods: General",
    "category": "section",
    "text": "Below you find all general exports."
},

{
    "location": "methods/general.html#Index-1",
    "page": "General",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"general.md\"]"
},

{
    "location": "methods/general.html#MonteCarlo.init!-Union{Tuple{MonteCarlo.MC{#s83,S} where #s83<:MonteCarlo.Model}, Tuple{S}} where S",
    "page": "General",
    "title": "MonteCarlo.init!",
    "category": "Method",
    "text": "init!(mc::MC[; seed::Real=-1])\n\nInitialize the classical Monte Carlo simulation mc. If seed !=- 1 the random generator will be initialized with srand(seed).\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.run!-Union{Tuple{MonteCarlo.MC{#s12,S} where #s12<:MonteCarlo.Model}, Tuple{S}} where S",
    "page": "General",
    "title": "MonteCarlo.run!",
    "category": "Method",
    "text": "run!(mc::MC[; verbose::Bool=true, sweeps::Int, thermalization::Int])\n\nRuns the given classical Monte Carlo simulation mc. Progress will be printed to STDOUT if verborse=true (default).\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.MC",
    "page": "General",
    "title": "MonteCarlo.MC",
    "category": "Type",
    "text": "Classical Monte Carlo simulation\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.MC-Union{Tuple{M}, Tuple{M}} where M<:MonteCarlo.Model",
    "page": "General",
    "title": "MonteCarlo.MC",
    "category": "Method",
    "text": "MC(m::M) where M<:Model\n\nCreate a classical Monte Carlo simulation for model m with default parameters.\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.IsingModel",
    "page": "General",
    "title": "MonteCarlo.IsingModel",
    "category": "Type",
    "text": "Famous Ising model on a cubic lattice.\n\n\n\n"
},

{
    "location": "methods/general.html#MonteCarlo.IsingModel-Tuple{Int64,Int64,Float64}",
    "page": "General",
    "title": "MonteCarlo.IsingModel",
    "category": "Method",
    "text": "IsingModel(dims::Int, L::Int, β::Float64)\nIsingModel(; dims::Int=2, L::Int=8, β::Float64=1.0)\nIsingModel(; dims::Int=2, L::Int=8, T::Float64=1.0)\n\nCreate Ising model on dims-dimensional cubic lattice with linear system size L and inverse temperature β.\n\n\n\n"
},

{
    "location": "methods/general.html#Documentation-1",
    "page": "General",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nPrivate = false\nOrder   = [:function, :type]\nPages = [\"mc.jl\", \"IsingModel.jl\"]"
},

{
    "location": "methods/models.html#",
    "page": "Models",
    "title": "Models",
    "category": "page",
    "text": ""
},

{
    "location": "methods/models.html#Methods:-Models-1",
    "page": "Models",
    "title": "Methods: Models",
    "category": "section",
    "text": "Below you find all methods that any particular implementation of the abstract type MonteCarlo.Model must implement."
},

{
    "location": "methods/models.html#Index-1",
    "page": "Models",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"models.md\"]"
},

{
    "location": "methods/models.html#MonteCarlo.observables-Tuple{MonteCarlo.Model}",
    "page": "Models",
    "title": "MonteCarlo.observables",
    "category": "Method",
    "text": "observables(m::Model)\n\nGet all observables defined for a given model.\n\nReturns a Dict{String, String} where values are the observables names and keys are short versions of those names. The keys can be used to collect correponding observable objects from a Monte Carlo simulation, e.g. like mc.obs[key].\n\n\n\n"
},

{
    "location": "methods/models.html#Base.Random.rand-Tuple{MonteCarlo.Model}",
    "page": "Models",
    "title": "Base.Random.rand",
    "category": "Method",
    "text": "rand(m::Model)\n\nDraw random configuration.\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.accept_local!-Tuple{MonteCarlo.Model,Int64,Any,Float64,Any,Float64}",
    "page": "Models",
    "title": "MonteCarlo.accept_local!",
    "category": "Method",
    "text": "accept_local(m::Model, i::Int, conf, E::Float64)\n\nAccept a local move for site i of current configuration conf with energy E. Arguments Δi and ΔE correspond to output of propose_local() for that local move.\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.conftype-Tuple{MonteCarlo.Model}",
    "page": "Models",
    "title": "MonteCarlo.conftype",
    "category": "Method",
    "text": "conftype(m::Model)\n\nReturns the type of a configuration.\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.energy-Tuple{MonteCarlo.Model,Any}",
    "page": "Models",
    "title": "MonteCarlo.energy",
    "category": "Method",
    "text": "energy(m::Model, conf)\n\nCalculate energy of configuration conf for Model m.\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.propose_local-Tuple{MonteCarlo.Model,Int64,Any,Float64}",
    "page": "Models",
    "title": "MonteCarlo.propose_local",
    "category": "Method",
    "text": "propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi\n\nPropose a local move for element i of current configuration conf with energy E. Returns the local move Δi = new[i] - conf[i] and energy difference ΔE = E_new - E_old.\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.reset!-Tuple{MonteCarlo.MonteCarloFlavor}",
    "page": "Models",
    "title": "MonteCarlo.reset!",
    "category": "Method",
    "text": "reset!(mc::MonteCarloFlavor)\n\nResets the Monte Carlo simulation mc. Previously set parameters will be retained.\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.CubicLattice",
    "page": "Models",
    "title": "MonteCarlo.CubicLattice",
    "category": "Type",
    "text": "Abstract cubic lattice.\n\n1D -> Chain\n2D -> SquareLattice\nND -> NCubeLattice\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.Lattice",
    "page": "Models",
    "title": "MonteCarlo.Lattice",
    "category": "Type",
    "text": "Abstract definition of a lattice. Necessary fields depend on Monte Carlo flavor. However, any concrete Lattice type should have at least the following fields:\n\n- `sites`: number of lattice sites\n- `neighs::Matrix{Int}`: neighbor matrix (row = neighbors, col = siteidx)\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.Model",
    "page": "Models",
    "title": "MonteCarlo.Model",
    "category": "Type",
    "text": "Abstract definition of a model. A concrete model type must have two fields:\n\n- `β::Float64`: temperature (depends on MC flavor if this will actually be used)\n- `l::Lattice`: any [Lattice](@ref)\n\nA concrete model must implement the following methods:\n\n- `conftype(m::Model)`: type of a configuration\n- `energy(m::Model, conf)`: energy of configuration\n- `rand(m::Model)`: random configuration\n- `propose_local(m::Model, i::Int, conf, E::Float64) -> ΔE, Δi`: propose local move\n- `accept_local(m::Model, i::Int, conf, E::Float64)`: accept a local move\n\n\n\n"
},

{
    "location": "methods/models.html#MonteCarlo.MonteCarloFlavor",
    "page": "Models",
    "title": "MonteCarlo.MonteCarloFlavor",
    "category": "Type",
    "text": "Abstract definition of a Monte Carlo flavor.\n\nA concrete monte carlo flavor must implement the following methods:\n\n- `init!(mc)`: initialize the simulation without overriding parameters (will also automatically be available as `reset!`)\n- `run!(mc)`: run the simulation\n\n\n\n"
},

{
    "location": "methods/models.html#Documentation-1",
    "page": "Models",
    "title": "Documentation",
    "category": "section",
    "text": "Modules = [MonteCarlo]\nOrder   = [:function, :type]\nPages = [\"abstract_types.jl\", \"abstract_functions.jl\"]"
},

]}