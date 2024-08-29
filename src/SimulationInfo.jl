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

    # simulation ID number
    sID::Int

    # processor ID number
    pID::Int

    # whether the simulation is resuming
    resuming::Bool
end


"""

    SimulationInfo( )

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