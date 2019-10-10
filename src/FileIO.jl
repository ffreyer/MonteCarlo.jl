# Saving and loading happens in a nested fashion:
# save(filename, mc) calls
#   save_mc(filename, mc) calls
#       save_model(filename, mc.model) calls
#           save_lattice(filename, mc.lattice)
#       save_measurements(filename, mc) calls
#           save_measurement(filename, measurement)
#
# loading follows the same structure
# Each level (beyond the outermost save) has an `entryname::String` and a
# `Val(VERSION)`.


function save(filename, mc::MonteCarloFlavor; force_overwrite=false, allow_rename=true)
    @assert endswith(filename, ".jld")

    # handle ranming and overwriting
    isfile(filename) && !force_overwrite && !allow_rename && throw(ErrorException(
        "Cannot save because \"$filename\" already exists. Consider setting " *
        "`allow_reanme = true` to adjust the filename or `force_overwrite = true`" *
        " to overwrite the file."
    ))
    if isfile(filename) && !force_overwrite && allow_rename
        while isfile(filename)
            # those map to 0-9, A-Z, a-z
            x = rand([(48:57)..., (65:90)..., (97:122)...])
            s = string(Char(x))
            filename = filename[1:end-4] * s * ".jld"
        end
    end

    mode = isfile(filename) ? "r+" : "w"
    jldopen(filename, mode) do f
        write(f, "VERSION", 1)
    end
    save_mc(filename, mc, "MC")
end

function load(filename)
    data = JLD.load(filename)
    @assert data["VERSION"] == 1
    load_mc(data)
end
load_mc(data) = load_mc(data["MC"], data["MC"]["type"])



"""
    save_model(filename, model, entryname)

Save (minimal) information necessary to reconstruct the given `model` in a
jld-file `filename` under group `entryname`.

By default the full model object is saved. When saving a simulation, the
entryname defaults to `Model`.
"""
function save_model(filename::String, model, entryname::String)
    mode = isfile(filename) ? "r+" : "w"
    jldopen(filename, mode) do f
        write(f, entryname * "/VERSION", 0)
        write(f, entryname * "/type", typeof(model))
        write(f, entryname * "/data", model)
    end
    nothing
end

"""
    load_model(data)

Loads a model from a given `data` dictionary produced by `JLD.load(filename)`.
"""
function load_model(data, ::DataType)
    @assert data["VERSION"] == 0
    data["data"]
end


"""
    save_lattice(filename, lattice, entryname)

Save (minimal) information necessary to reconstruct the given `lattice` in a
jld-file `filename` under group `entryname`.

By default the full lattice object is saved. When saving a simulation, the
entryname defaults to `Lattice`.
"""
function save_lattice(filename::String, lattice::AbstractLattice, entryname::String)
    mode = isfile(filename) ? "r+" : "w"
    jldopen(filename, mode) do f
        write(f, entryname * "/VERSION", 0)
        write(f, entryname * "/type", typeof(lattice))
        write(f, entryname * "/data", lattice)
    end
    nothing
end

"""
    load_lattice(data)

Loads a lattice from a given `data` dictionary produced by `JLD.load(filename)`.
"""
function load_lattice(data, ::DataType)
    @assert data["VERSION"] == 0
    data["data"]
end
