# ## PHONON PARAMETERS ##
# # phonon mass
# M = 1.0

# # phonon frequency
# Ω = 1.0

# # microscopic coupling constant
# α = 0.5

# # initial phonon fugacity
# μₚₕ = 0.0
# ## PHONON PARAMETERS ##

# ## ELECTRON PHONON ##
# #  initialize null electron-phonon model
# electron_phonon_model = ElectronPhonon(tight_binding_model);

# # define dispersionless phonon mode to live on each site (optical phonon)
# phonon = PhononMode(1, M, Ω);

# # add the phonon definition to the electon-phonon model
# phonon_ID = add_phonon_mode!(electron_phonon_model, phonon);

# # define onsite Holstein coupling between electrons and local dispersionless phonon
# holstein_coupling = HolsteinCoupling(phonon_ID, bond_x, α, 0.0, μₚₕ) # TODO: remove fugacity definition from type

# # define optical SSH coupling
# # Defines total effective hopping amplitude given by t_eff = t-α⋅(Xᵢ₊₁-Xᵢ)
# ssh_coupling = SSHCoupling((phonon_ID, phonon_ID), bond_x, α, 0.0, [0.0,0.0]) # TODO: remove fugacity definition from type

# # add Holstein coupling defintion
# holstein_coupling_ID = add_holstein_coupling!(electron_phonon_model, holstein_coupling) 

# # add SSH coupling definition
# ssh_coupling_ID = add_ssh_coupling!(electron_phonon_model, ssh_coupling)

# # initialize electron-phonon parameters
# electron_phonon_parameters = ElectronPhononParameters()
# ## ELECTRON PHONON ##


"""

    PhononMode

A type defining quantities related to a phonon mode. 

"""
struct PhononMode
    # orbital
    ν::Int

    # phonon mass
    M::Float64

    # phonon frequency
    Ω::Float64
end


"""

    HolsteinCoupling

A type defining quantities related to Holstein-type electron-phonon coupling.

"""
struct HolsteinCoupling{D}
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
end


"""

    SSHCoupling

A type defining quantities related to SSH-type electron-phonon coupling. 

"""
struct SSHCoupling{D}
    # phonon modes getting coupled
    phonon_modes::NTuple{2,Int}

    # bond/hopping associated with bond
    bond::Bond{D}

    # linear SSH coupling
    α::Float64

    # quadratic SSH coupling
    α2::Float64

    # phonon fugacities
    z::Vector{Float64}
end


"""

    ElectronPhonon

A type defining quantities related to an electron-phonon model.

"""
struct ElectronPhonon
    # phonon modes
    phonon_modes::Vector{PhononMode}

    # holstein couplings
    holstein_couplings::Vector{HolsteinCoupling}

    # SSH couplings
    ssh_couplings::Vector{SSHCoupling}
end


"""

    ElectronPhonon( tight_binding_model::TightBindingModel )

Constructor for the ElectronPhonon type.

"""
function ElectronPhonon(tight_binding_model::TightBindingModel)
    if isnothing(tight_binding_model)
        error("Tight-binding model improperly specified.")
    end

    phonon_modes = PhononMode[]
    holstein_couplings = HolsteinCoupling[]
    ssh_couplings = SSHCoupling[]

    return ElectronPhonon(phonon_modes, holstein_couplings, ssh_couplings)
end


"""

    add_phonon_mode!( electron_phonon_model::ElectronPhonon, phonon_mode::PhononMode )

Gven an initialized electron-phonon model, adds a defined phonon mode.

"""
function add_phonon_mode!(electron_phonon_model::ElectronPhonon, phonon_mode::PhononMode)
    # record phonon mode
    push!(electron_phonon_model.phonon_modes, phonon_mode)

    return length(electron_phonon_model.phonon_modes)
end


"""

    add_holstein_coupling!( electron_phonon_model::ElectronPhonon, holstein_coupling::HolsteinCoupling )

Given an initialized electron-phonon model, adds a Holstein-type electon-phonon coupling. 

"""
function add_holstein_coupling!(electron_phonon_model::ElectronPhonon, holstein_coupling::HolsteinCoupling) 
    # get the phonon mode getting coupled to
    phonon_modes = electron_phonon_model.phonon_modes
    phonon_mode = phonon_modes[holstein_coupling.phonon_mode]

    # get the bond associated with holstein coupling
    holstein_bond = holstein_coupling.bond

    # make sure the initial bond orbital matches the orbital species of the phonon mode
    @assert phonon_mode.ν == holstein_bond.orbitals[1]

    # record the holstein coupling
    holstein_couplings = electron_phonon_model.holstein_couplings
    push!(holstein_couplings, holstein_coupling)

    return length(holstein_couplings)
end


"""

    add_ssh_coupling!( electron_phonon_model::ElectronPhonon, tight_binding_model::TightBindingModel, ssh_coupling::SSHCoupling )

Given an initialized electron-phonon model, adds an SSH-type electron-phonon coupling. 

"""
function add_ssh_coupling!(electron_phonon_model::ElectronPhonon, ssh_coupling::SSHCoupling)
    phonon_modes = electron_phonon_model.phonon_modes
    ssh_couplings = electron_phonon_model.ssh_couplings
    ssh_bond = ssh_coupling.bond

    # get initial and final phonon modes that are coupled
    phonon_mode_init = phonon_modes[ssh_coupling.phonon_modes[1]]
    phonon_mode_final = phonon_modes[ssh_coupling.phonon_modes[2]]

    # make the the staring and ending orbitals of the ssh bond match the orbital species of the phonon modes getting coupled
    @assert ssh_bond.orbitals[1] == phonon_mode_init.ν
    @assert ssh_bond.orbitals[2] == phonon_mode_final.ν

    # record the ssh_bond
    push!(ssh_couplings, ssh_coupling)

    return length(ssh_couplings)
end


"""
    PhononParameters

A type defining quantities related to phonon parameters.

"""
struct PhononParameters
    # number of types of phonon modes
    nphonon::Int

    # number of phonon modes
    Nphonon::Int

    # phonon masses
    M::Vector{Float64}

    # phonon frequencies
    Ω::Vector{Float64}
end


"""

    PhononParameters( model_geometry::ModelGeometry, electron_phonon_model::ElectronPhonon )

Constructor for the PhononParameters type. 

"""
function PhononParameters(model_geometry::ModelGeometry, electron_phonon_model::ElectronPhonon)
    lattice = model_geometry.lattice
    unit_cell = model_geometry.unit_cell

    # total number of sites
    N = lattice.N

    # get phonon mode definitions
    phonon_modes = electron_phonon_model.phonon_modes

    # get the number of phonon mode definitions
    nphonon = length(phonon_modes)

    # get the total number of phonon modes in the lattice
    Nphonon = nphonon * N

    # allocate array of masses for each phonon mode
    Ms = zeros(Nphonon)

    # allocate array of phonon frequncies for each phonon mode
    Ωs = zeros(Nphonon)

    # allocate phonon_to_site
    phonon_to_site = zeros(Int, Nphonon)

    # allocate site_to_phonons
    site_to_phonons = [Int[] for i in 1:Nsites]
    
    # iterate over phonon modes
    phonon = 0 # phonon counter
    for nph in 1:phonon
        # get phonon mode
        phonon_mode = phonon_modes[nph]

        # get orbital
        orbital = phonon_mode.ν

        for n in 1:N
            # increment phonon counter
            phonon += 1

            # get site associated with phonon mode
            site = loc_to_site(n, orbital, unit_cell)

            # record phonon ==> site
            phonon_to_site[phonon] = site

            # record site ==> phonon
            push!(site_to_phonons[site], phonon)

            # assign phonon mass
            Ms[phonon] = phonon_mode.M

            # assign phonon frequency
            Ωs[phonon] = phonon_mode.Ω
        end
    end

    return PhononParameters(nphonon, Nphonon, Ms, Ωs)
end


struct HolsteinParameters
    # number of type of Holstein couplings
    nholstein::Int

    # number of Holstein couplings
    Nholstein::Int

    # linear coupling
    α::Vector{Float64}

    # quadratic coupling
    α2::Vector{Float64}

    # neighbor table for couplings where first row is the site the phonon the lives on,
    # and the second row is the site whose density the phonon mode is coupling to
    neighbor_table::Matrix{Int}

    # map coupling to phonon
    coupling_to_phonon::Vector{Int}

    # map phonon to coupling
    phonon_to_coupling::Vector{Vector{Int}}
end


function HolsteinParameters(model_geometry::ModelGeometry, electron_phonon_model::ElectronPhonon)
    lattice = model_geometry.lattice
    unit_cell = model_geometry.unit_cell
    phonon_modes = electron_phonon_model.phonon_modes
    holstein_couplings = electron_phonon_model.holstein_couplings

    # number Holstein coupling definitions
    nholstein = length(holstein_couplings)

    # get number of types of phonon models
    nphonon = length(phonon_modes)

    # get the number of unit cells in the lattice
    N = lattice.N

    # total number of holstein couplings
    Nholstein = N * nholstein 

    # get total number of phonon modes
    Nphonon = N * nphonon

    # build the neighbor table for the holstein couplings
    holstein_bonds = [holstein_coupling.bond for holstein_coupling in holstein_couplings]
    neighbor_table = build_neighbor_table(holstein_bonds, unit_cell, lattice)

    # allocate arrays for Holstein coupling parameters
    αs = zeros(Float64, Nholstein)
    α2s = zeros(Float64, Nholstein)

    # allocate arrays mapping Holstein coupling to phonon in lattice
    coupling_to_phonon = zeros(Int, Nholstein)

    # iterate over Holstein coupling defintitions
    holstein_counter = 0 # holstein coupling counter
    for hc in 1:nholstein
        # get the holstein coupling definition
        holstein_coupling = holstein_couplings[hc]

        # get the phonon mode definition/ID associated with Holstein coupling
        phonon_mode = holstein_coupling.phonon_mode

        # iterate over unit cells
        for n in 1:N
            # increment Holstein coupling counter
            holstein_counter += 1

            # get the phonon mode getting coupled to
            phonon = N * (phonon_mode - 1) + n

            # record the phonon mode associated with the coupling
            coupling_to_phonon[holstein_counter] = phonon

            # initialize coupling parameters
            αs[holstein_counter]  = holstein_coupling.α
            α2s[holstein_counter] = holstein_coupling.α2
        end
    end

    # construct phonon to coupling map
    phonon_to_coupling = Vector{Int}[]
    for phonon in 1:Nphonon
        push!(phonon_to_coupling, findall(i -> i==phonon, coupling_to_phonon))
    end

    return HolsteinParameters(nholstein, Nholstein, α, α2, neighbor_table, coupling_to_phonon, phonon_to_coupling)
end


struct SSHParameters
   # number of types of SSH couplings
   nssh::Int

   # number of SSH couplings in lattice
   Nssh::Int

   # linear coupling
   α::Vector{Float64}

   # quadratic coupling
   α2::Vector{Float64}

   # SSH neighbor table
   neighbor_table::Matrix{Int}

   # map SSH coupling to phonon mode
   coupling_to_phonon::Matrix{Int}

   # initial phonon to coupling
   init_phonon_to_coupling::Vector{Vector{Int}}

   # final phonon to coupling
   final_phonon_to_coupling::Vector{Vector{Int}}

   # map hopping in bare tight binding model to SSH coupling
   hopping_to_couplings::Vector{Vector{Int}}
   
   # map coupling to bare hopping in tight binding model
   coupling_to_hopping::Vector{Int}
end


function SSHParameters(model_geometry::ModelGeometry, electron_phonon_model::ElectronPhonon, tight_binding_model::TightBindingModel )
    ssh_couplings = electron_phonon_model.ssh_couplings
    phonon_modes = electron_phonon_model.phonon_modes
    lattice = model_geometry.lattice
    unit_cell = model_geometry.unit_cell

    # number of SSH coupling definitions
    nssh = length(ssh_couplings)

    # get number of types of phonon models
    nphonon = length(phonon_modes)

    # get the number of unit cells in the lattice
    N = lattice.N

    # total number of holstein couplings
    Nssh = N * nssh

    # get total number of phonon modes
    Nphonon = N * nphonon

    # get all the ssh bonds
    ssh_bonds = [ssh_coupling.bond for ssh_coupling in ssh_couplings]

    # get the bare tight binding model hopping neighbor table
    hopping_neighbor_table = build_neighbor_table(ssh_bonds, unit_cell, lattice)

    # get the total number of hoppings in lattice
    Nhoppings = size(hopping_neighbor_table,2)

    # allocate arrays of ssh coupling parameters
    αs  = zeros(Float64, Nssh)
    α2s = zeros(Float64, Nssh)

    # allocate mapping arrays
    coupling_to_phonon   = zeros(Int, 2, Nssh)
    coupling_to_hopping  = zeros(Int, Nssh)
    hopping_to_couplings = [Int[] for _ in 1:Nhoppings]

    # construct neighbor table for ssh couplings
    ssh_neighbor_table = build_neighbor_table(ssh_bonds, unit_cell, lattice)

    # iterate over ssh coupling definitions
    ssh_counter = 0 # ssh coupling counter
    for sc in 1:nssh
        # get the ssh coupling definition
        ssh_coupling = ssh_couplings[sc]

        # get the pair of phonon mode definitions assoicated with ssh coupling
        phonon_mode_i = ssh_coupling.phonon_modes[1]
        phonon_mode_f = ssh_coupling.phonon_modes[2]

        # get the bond id associated with the ssh coupling
        ssh_bond_ID = ssh_coupling.bond_id

    end


end


# struct ElectronPhononParameters
#     # holstein parameters
#     holstein_parameters::HolsteinParameters

#     # ssh parameters
#     ssh_parameters::SSHParameters

#     # phonon density configuration
#     N_phconfig::Vector{Int}   

#     # phonon displacement configuration
#     X_phconfig::Matrix{Float}
# end

# # add initial random displacements
    # ΔX = sqrt(0.5)
    # for i in eachindex(X_phconfig)
    #     x₀ = rand(rng) * ΔX
    #     X_phconfig[i] += x₀
    # end


# #####




#   # allocate phonon density configuration
#   N_phconfig = zeros(N)

#   # allocated phonon displacement configuration
#   X_phconfig = zeros(Nphonon, N)
# #####

##################################################### DEPRECATED FUNCTIONS #####################################################

# """

#     initialize_electron_phonon_model(Ω::AbstractFloat,  phonon_parameters::PhononParameters, μₚₕ::AbstractFloat )

# Given generic phonon parameters and initial fugacity, returns an instance of the HolsteinModel type.

# """
# function initialize_electron_phonon_model(Ω::AbstractFloat, M::AbstractFloat, α::AbstractFloat, μₚₕ::AbstractFloat, phonon_parameters::PhononParameters, model_geometry::ModelGeometry)
#      # intialize initial phonon configuration
#      phconfig = generate_initial_phonon_density_configuration(model_geometry)

#      # initial number of phonons
#      Nₚₕ = 0

#      return HolsteinModel(phonon_parameters, μₚₕ, Nₚₕ, phconfig)
# end


# """

#     initialize_electron_phonon_model( phonon_parameters::PhononParameters, loc::AbstractString )

# Given generic phonon parameters and phonon location, returns an instance of the SSHModel type.

# """
# function initialize_electron_phonon_model(Ω::AbstractFloat, M::AbstractFloat, α::AbstractFloat, loc::AbstractString, z_x::AbstractFloat, z_y::AbstractFloat, phonon_parameters::PhononParameters, model_geometry::ModelGeometry)
#     # lattice dimensions
#     dims = size(model_geometry.lattice.L)[1]

#     # intialize initial phonon configuration
#     phconfig = generate_initial_phonon_displacement_configuration(loc, model_geometry)

#     # standard deviation of the equilibrium distribution of a quantum harmonic oscillator
#     ΔX = sqrt(0.5)

#     # add initial random displacements
#     for i in eachindex(phconfig)
#         x₀ = rand(rng) * ΔX
#         phconfig[i] += x₀
#     end

#     # initialize fugacity
#     z = AbstractFloat[]
#     push!(z, z_x)
#     push!(z, z_y)

#     return SSHModel(phonon_parameters, loc, z, phconfig)
# end

# """

#     update_electron_phonon_model(  )

# After a Metropolis update, updates phonon configurations and parameters.

# """
# function update_electron_phonon_model!(holstein_model::HolsteinModel, phconfig::Vector{Int}, model_geometry::ModelGeometry)
#     N = model_geometry.lattice.N

#     # update total phonon number
#     Nₚₕ = 0
#     for i in 1:N
#         n_ph = get_phonon_occupation(i, phconfig)
#         Nₚₕ += n_ph
#     end

#     # update phconfig
#     holstein_model.phconfig = phconfig

#     # update total phonon number
#     holstein_model.Nₚₕ = Nₚₕ

#     return nothing
# end

# # TODO: maybe put in a phonon module (Phonon.jl) to contain handling of the coherent states and such?