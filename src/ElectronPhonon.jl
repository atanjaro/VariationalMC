struct PhononMode
    # orbital
    ν::Int

    # phonon mass
    M::Float64

    # phonon frequency
    Ω::Float64
end

mutable struct HolsteinCoupling{D}
    # phonon mode of coupling
    phonon_mode::Int

    # displacement vector to density phonon mode is coupled to
    bond::Bond{D}

    # linear (X) coupling coefficient
    α::Float64

    # quadratic (X²) coupling coefficient
    α2::Float64

    # phonon fugacity
    μₚₕ::Float64

    # phonon density configuration
    phconfig::Vector{Int}
end


mutable struct SSHCoupling{D}
    # phonon modes getting coupled
    phonon_modes::NTuple{2,Int}

    # bond/hopping associated with bond
    bond::Bond{D}

    # linear SSH coupling
    α::Float64

    # quadratic SSH coupling
    α2::Float64

    # phonon fugacities
    z::Vector{AbstractFloat}

    # phonon displacement configuration 
    phconfig::Matrix{AbstractFloat}
end

mutable struct ElectronPhononModel
    # phonon modes
    phonon_modes::Vector{PhononMode}

    # holstein couplings
    holstein_couplings::Vector{HolsteinCoupling}

    # SSH couplings
    ssh_couplings::Vector{SSHCoupling}
end

# initialize a null electron-phonon model
function ElectronPhononModel(tight_binding_model)
    if isnothing(tight_binding_model)
        error("Tight-binding model improperly specified.")
    end

    phonon_modes = PhononMode[]
    holstein_couplings = HolsteinCoupling{Int}[]
    ssh_couplings = SSHCoupling{Int}[]

    return ElectronPhononModel(phonon_modes, holstein_couplings, ssh_couplings)
end

# add phonon mode
function add_phonon_mode!(electron_phonon_model::ElectronPhononModel, phonon_mode::PhononMode)
    # record phonon mode
    push!(electron_phonon_model.phonon_modes, phonon_mode)

    return length(electron_phonon_model.phonon_modes)
end


function add_holstein_coupling!(electron_phonon_model::ElectronPhononModel, holstein_coupling::HolsteinCoupling) 
    # get the phonon mode getting coupled to
    phonon_modes = electron_phonon_model.phonon_modes
    phonon_mode = phonon_modes[holstein_coupling.phonon_mode]

    # get the bond associated with holstein coupling
    holstein_bond = holstein_coupling.bond

    # make sure the initial bond orbital matches the orbital species of the phonon mode
    @assert phonon_mode.orbital == holstein_bond.orbitals[1]

    # record the holstein coupling
    holstein_couplings_up = electron_phonon_model.holstein_couplings
    push!(holstein_couplings, holstein_coupling)

    return length(holstein_couplings)
end

function add_ssh_coupling!(electron_phonon_model::ElectronPhononModel, tight_binding_model::TightBindingModel, ssh_coupling::SSHCoupling)

    phonon_modes = electron_phonon_model.phonon_modes
    ssh_couplings = electron_phonon_model.ssh_couplings
    tbm_bonds = tight_binding_model_up.t_bonds
    ssh_bond = ssh_coupling.bond

    # get initial and final phonon modes that are coupled
    phonon_mode_init = phonon_modes[ssh_coupling.phonon_modes[1]]
    phonon_mode_final = phonon_modes[ssh_coupling.phonon_modes[2]]

    # make sure a hopping already exists in the tight binding model for the ssh coupling
    @assert ssh_bond in tbm_bonds

    # make the the staring and ending orbitals of the ssh bond match the orbital species of the phonon modes getting coupled
    @assert ssh_bond.orbitals[1] == phonon_mode_init.orbital
    @assert ssh_bond.orbitals[2] == phonon_mode_final.orbital

    # record the ssh_bond
    push!(ssh_couplings, ssh_coupling)

    return length(ssh_couplings)
end

# """

#     HolsteinModel

# A type defining quantities related to a Holstein model of electon-phonon coupling.

# """
# mutable struct HolsteinModel
#     # phonon parameters
#     phonon_parameters::PhononParameters

#     # fugacity
#     μₚₕ::AbstractFloat

#     # phonon number
#     Nₚₕ::Int

#     # phonon density configuration
#     phconfig::Vector{Int}
# end


# """

#     SSHModel

# A type defining quantities related to an SSH model of electon-phonon coupling.

# """
# mutable struct SSHModel
#     # phonon parameters
#     phonon_parameters::PhononParameters

#     # phonon location
#     loc::AbstractString

#     # fugacities
#     z::Vector{AbstractFloat}

#     # phonon displacement configuration 
#     phconfig::Matrix{AbstractFloat}
# end



"""

    initialize_electron_phonon_model(Ω::AbstractFloat,  phonon_parameters::PhononParameters, μₚₕ::AbstractFloat )

Given generic phonon parameters and initial fugacity, returns an instance of the HolsteinModel type.

"""
function initialize_electron_phonon_model(Ω::AbstractFloat, M::AbstractFloat, α::AbstractFloat, μₚₕ::AbstractFloat, phonon_parameters::PhononParameters, model_geometry::ModelGeometry)
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
function initialize_electron_phonon_model(Ω::AbstractFloat, M::AbstractFloat, α::AbstractFloat, loc::AbstractString, z_x::AbstractFloat, z_y::AbstractFloat, phonon_parameters::PhononParameters, model_geometry::ModelGeometry)
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

"""

    update_electron_phonon_model(  )

After a Metropolis update, updates phonon configurations and parameters.

"""
function update_electron_phonon_model!(holstein_model::HolsteinModel, phconfig::Vector{Int}, model_geometry::ModelGeometry)
    N = model_geometry.lattice.N

    # update total phonon number
    Nₚₕ = 0
    for i in 1:N
        n_ph = get_phonon_occupation(i, phconfig)
        Nₚₕ += n_ph
    end

    # update phconfig
    holstein_model.phconfig = phconfig

    # update total phonon number
    holstein_model.Nₚₕ = Nₚₕ

    return nothing
end

# TODO: maybe put in a phonon module (Phonon.jl) to contain handling of the coherent states and such?