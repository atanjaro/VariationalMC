"""

    SimulationInfo

A type containing information about the simulation, including the location data is written to,
the simulation ID, and MPI process ID, and whether this simulation started a new simulation or resumed
a previous simulation.

"""
mutable struct SimulationInfo

    # filepath to where data directory will live
    filepath::String

    # prefix of data directory name
    datafolder_prefix::String

    # data directory name
    datafolder_name::String

    # data directory including filepath
    datafolder::String

    # process ID number (for MPI)
    pID::Int

    # simulation ID number
    sID::Int

    # whether previous simulation is being resumed or a new one is begininning
    resuming::Bool
end


"""

    SimulationInfo( ; datafolder_prefix::String, filepath::String = ".", sID::Int=0, pID::Int=0 )

Creates an instance of the SimulationInfo type.

"""
function SimulationInfo(; datafolder_prefix::String, filepath::String = ".", sID::Int=0, pID::Int=0)

    # initialize data folder names
    datafolder_name = @sprintf "%s-%d" datafolder_prefix sID
    datafolder = joinpath(filepath, datafolder_name)

    # if null data folder id given, determine data name and id
    if sID==0
        while isdir(datafolder) || sID==0
            sID += 1
            datafolder_name = @sprintf "%s-%d" datafolder_prefix sID
            datafolder = joinpath(filepath, datafolder_name)
        end
    end

    # if directory already exists then must be resuming simulation
    resuming = isdir(datafolder)

    return SimulationInfo(filepath, datafolder_prefix, datafolder_name, datafolder, pID, sID, resuming)
end


# print struct info as TOML format
function Base.show(io::IO, ::MIME"text/plain", sim_info::SimulationInfo)

    @printf io "[simulation_info]\n\n"
    @printf io "name  = \"%s\"\n" sim_info.datafolder_prefix
    @printf io "sID   = %d\n" sim_info.sID
    @printf io "pID   = %d\n" sim_info.pID
    @printf io "julia_version     = \"%s\"" VERSION

    return nothing
end


"""

    save_simulation_info( sim_info::SimulationInfo, additional_info = nothing )

Save the contents `sim_info` to a TOML file, and add an optional additional table to the
output file based on the contents of a dictionary `additional_info`.

"""
function save_simulation_info(sim_info::SimulationInfo, additional_info = nothing)

    (; datafolder, pID, sID) = sim_info
    fn = @sprintf "simulation_info_pID%d_sID%d.toml" pID sID
    open(joinpath(datafolder, fn), "w") do fout
        show(fout, "text/plain", sim_info)
        if !isnothing(additional_info)
            @printf fout "\n\n"
            TOML.print(fout, Dict("additional_info" => additional_info), sorted = true)
        end
    end

    return nothing
end


"""

    initialize_datafolder( sim_info::SimulationInfo )

Initalize `sim_info.datafolder` directory if it does not already exist.

"""
function initialize_datafolder(sim_info::SimulationInfo)

    (; pID, datafolder, resuming) = sim_info

    # if main process and starting new simulation (not resuming an existing simulation)
    if iszero(pID) && !resuming

        # make data folder diretory
        mkdir(datafolder)
    end

    return nothing
end


function model_summary(;
        n̄::Float64,
        nup::Int,
        ndn::Int,
        simulation_info::SimulationInfo, 
        model_geometry::ModelGeometry, 
        tight_binding_model::TightBindingModel, 
        parameters::Tuple)

    # if process ID is 1
    if iszero(simulation_info.pID)

        # construct full filename, including filepath
        fn = joinpath(simulation_info.datafolder, "model_summary.toml")

        # open file to write to
        open(fn, "w") do fout
            # write n̄
            @printf fout "n̄ = %.6f\n\n" n̄
            # write nup
            @printf fout "nup = %.6f\n\n" nup
            # write ndn
            @printf fout "ndn = %d\n\n" ndn

            # write model geometry out to file
            show(fout, "text/plain", model_geometry)

            # write tight-binding model to file assuming spin symmetry
            show(fout, MIME("text/plain"), tight_binding_model)

            # write various parameters to file
            for parameter in parameters
                show(fout, "text/plain", interaction)
            end
        end


    end
end