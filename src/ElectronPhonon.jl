"""

    PhononParameters

A type defining quantities related to optical phonons.

"""
struct PhononParameters
    # phonon frequency
    Ω::AbstractFloat

    # phonon mass
    M::AbstractFloat

    # microscopic coupling constant
    α::AbstractFloat
end


"""

    HolsteinModel

A type defining quantities related to a Holstein model of electon-phonon coupling.

"""
struct HolsteinModel
    # phonon parameters
    phonon_parameters::PhononParameters

    # fugacity
    μₚₕ::AbstractFloat

    # phonon number
    Nₚₕ::Int

    # phonon density configuration
    phconfig::Vector{Int}
end


"""

    SSHModel

A type defining quantities related to an SSH model of electon-phonon coupling.

"""
struct SSHModel
    # phonon parameters
    phonon_parameters::PhononParameters

    # phonon location
    loc::AbstractString

    # fugacity
    z::Vector{AbstractFloat}

    # phonon displacement configuration 
    phconfig::Matrix{AbstractFloat}
end



"""

    initialize_phonon_parameters( Ω::AbstractFloat, M::AbstractFloat, α::AbstractFloat )

Initializes an instance of the PhononParameters type

"""
function initialize_phonon_parameters(Ω::AbstractFloat, M::AbstractFloat, α::AbstractFloat)
    return PhononParameters(Ω, M, α)
end


"""

    initialize_electron_phonon_model( phonon_parameters::PhononParameters, μₚₕ::AbstractFloat )

Given generic phonon parameters and initial fugacity, returns an instance of the HolsteinModel type.

"""
function initialize_electron_phonon_model(μₚₕ::AbstractFloat, phonon_parameters::PhononParameters, model_geometry::ModelGeometry)
     # intialize initial phonon configuration
     phconfig = generate_initial_phonon_density_configuration(model_geometry)

     # initial number of phonons
     Nₚₕ = 0

     return HolsteinModel(phonon_parameters, μₚₕ, Nₚₕ, phconfig)
end


"""

    initialize_electron_phonon_model( phonon_parameters::PhononParameters, loc::AbstractString )

Given generic phonon parameters and phonon location, returns an instance of the SSHModel type.

"""
function initialize_electron_phonon_model(loc::AbstractString, z_x::AbstractFloat, z_y::AbstractFloat, phonon_parameters::PhononParameters, model_geometry::ModelGeometry)
    # lattice dimensions
    dims = size(model_geometry.lattice.L)[1]

    # intialize initial phonon configuration
    phconfig = generate_initial_phonon_displacement_configuration(loc, model_geometry)

    # standard deviation of the equilibrium distribution of a quantum harmonic oscillator
    ΔX = sqrt(0.5)

    # add initial random displacements
    for i in eachindex(phconfig)
        x₀ = rand(rng) * ΔX
        phconfig[i] += x₀
    end

    # initialize fugacity
    z = AbstractFloat[]
    push!(z, z_x)
    push!(z, z_y)

    return SSHModel(phonon_parameters, loc, z, phconfig)
end

