using LatticeUtilities
using Random
using LinearAlgebra
using DelimitedFiles
using BenchmarkTools
using Profile
using OrderedCollections
using CSV
using DataFrames
using DataStructures
using Printf
using JLD2

# files to include
include("Hamiltonian.jl")
include("Jastrow.jl")
include("ParticleConfiguration.jl")
include("Markov.jl")
include("Utilities.jl")
include("Greens.jl")
include("Hessian.jl")
include("SimulationInfo.jl")
include("Measurements.jl")


#############################
## DEFINE MODEL PARAMETERS ##
#############################

# Define the size of the lattice
Lx = 4
Ly = 4

# Specify number of particles in the Canonical Ensemble
#   - If initial density is given, code will automatically calculate number of electrons
#     accounting for particle-hole transformation. 
#     Note: the density must be commensurate with the size of the lattice.
#
#   - Particle numbers can also be specified directly, with total particle number and total 
#     electron number calculated automatically by the code.

# Define initial density
n̄ = 1.0

# # Define electron numbers
# nup = 42
# ndn = 54

# Nearest neighbor hopping amplitude
t = 1.0

# Next nearest neighbor hopping amplitude
tp = 0.0

# Onsite Hubbard repulsion
U = 0.5

# (BCS) chemical potential
μ_BCS = 0.0

# Phonon density fugacity
μₚₕ = 0.0

# Phonon displacement fugacity
z_x = 0.01
z_y = 0.01

# Phonon frequency
Ω = 1.0

# Microscopic electron-phonon coupling
g = 1.0

# Microscopic electron phonon coupling
α = g * sqrt(2 * Ω)

# # Dimensionless electron-phonon coupling (g definition)
# λ = (2 * g^2) / (Ω * 8)

# # Dimensionless electron-phonon coupling (α defintion)
# λ = α^2 / (Ω^2 * 8)

#######################################
##      VARIATIONAL PARAMETERS       ##
#######################################

# Whether to read-in initial determinantal parameters
# This will read-in parameters from the determinantal parameter output file of VariationalMC
readin_detpars = false
path_to_detpars = "/path/to/determinantal/parameters/"

# Whether to read-in initial Jastrow parameters
# This will read-in parameters from the Jastrow parameter output file of VariationalMC
readin_jpars = false
path_to_jpars = "/path/to/jastrow/parameters/"

# Parameters to be optimized and initial value(s)
parameters_to_optimize = ["Δs", "μ_BCS"]                              # s-wave (BCS) order parameter
parameter_values = [[0.01], [μ_BCS]]                                 
pht = true

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δcs", "Δss"]                               # charge and spin stripe order parameters
# parameter_values = [fill(0.01, Lx), fill(0.01, Lx)]       
# pht = false

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δd", "Δa", "μ_BCS"]                        # uniform d-wave order parameter
# parameter_values = [[0.05], [0.03], [μ_BCS]]                          
# pht = true

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δa"]                                       # antiferromagnetic (Neél) order parameter
# parameter_values = [[0.01]]                                            
# pht = false

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δc"]                                       # charge density wave order parameter
# parameter_values = [[0.1]]                                             
# pht = false

# Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δs", "μₚₕ"]                                # s-wave (BCS) order parameter + optical phonons
# parameter_values = [[0.01], [μₚₕ]]                      
# pht = true          

# Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δcs", "Δss", "μₚₕ"]                         # charge and spin stripe order parameters + optical phonons
# parameter_values = [[0.01], [0.01], [μₚₕ]]               
# pht = false         

# Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δs", "z_x", "z_y"]                                # s-wave (BCS) order parameter + optical phonons
# parameter_values = [[0.01], [0.01], [0.01]]                      
# pht = true    

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δcs", "Δss", "z_x", "z_y"]                         # charge and spin stripe order parameters + optical phonons
# parameter_values = [[0.01], [0.01], [z_x], [z_y]]               
# pht = false         


###################
## FILE HANDLING ##
###################

# specify filepath
filepath = "."

# simulation ID
sID = 1

# Construct the foldername the data will be written
# Note that the simulation ID `sID`` will be appended to this foldername as `*-sID`
datafolder_prefix = @sprintf "hubbard_square_U%.2f_n%.2f_Lx%d_Ly%d_swave" U n̄ Lx Ly 

# # Construct the foldername the data will be written
# # Note that the simulation ID `sID`` will be appended to this foldername as `*-sID`
# datafolder_prefix = @sprintf "hubbard_square_U%.2f_nup%.0f_ndn%.0f_Lx%d_Ly%d_udwave" U nup ndn Lx Ly 

# Initialize an instance of the SimulationInfo type
# This type helps keep track of where data will be written
simulation_info = SimulationInfo(
                filepath = filepath, 
                datafolder_prefix = datafolder_prefix,
                sID = sID
)

# Initialize the directory the data will be written to
initialize_datafolder(simulation_info)


##################################
## DEFINE SIMULATION PARAMETERS ##
##################################

# Initialize random seed
seed = abs(rand(Int))

# Initialize random number generator
rng = Xoshiro(seed)
       
# Number of minimization/optimization updates
N_opts = 3000

# Optimization bin size
opt_bin_size = 6000

# Number of simulation updates 
N_updates = 10000

# Number of simulation bins
N_bins = 100

# Simulation bin size
bin_size = div(N_updates, N_bins)

# Number of MC cycles until measurement
mc_meas_freq = 300

# Number of iterations until check for numerical stability 
n_stab = 500

# Maximum allowed error in the equal-time Green's function
δW = 1e-3

# Maximum allowed error in the T vector
δT = 1e-3

# Stabilization factor for Stochastic Reconfiguration
η = 1e-4      

# Optimization rate for Stochastic Reconfiguration
dt = 0.03        

# # Whether to output to terminal during runtime
# verbose = true

# Debugging flag
# This will output print statements to the terminal during runtime
debug = true

# Initialize additional simulation information dictionary
additional_info = Dict(
    "δW" => δW,
    "δT" => δT,
    "total_time" => 0.0,
    "simulation_time" => 0.0,
    "optimization_time" => 0.0,
    "N_opts" => N_opts,
    "N_updates" => N_updates,
    "N_bins" => N_bins,
    "bin_size" => bin_size,
    "dt" => dt,
    "seed" => seed,
    "n_bar" => n̄,
    "global_energy" => 0.0,
    "μ_BCS" => 0.0
)


##################
## DEFINE MODEL ##
##################

# # Chain unit cell
# unit_cell = UnitCell(lattice_vecs = [[1.0]],
#                             basis_vecs   = [[0.0]])

# Square unit cell
unit_cell = UnitCell([[1.0,0.0], [0.0,1.0]],           # lattice vectors
                               [[0.0,0.0]])            # basis vectors 

# # Build a chain
# lattice = Lattice([Lx],[true])

# Build a square lattice
lattice = Lattice([Lx, Ly],[true,true])

# whether to use twist averaged boundary conditions
tabc = false

# number of twist angles
N_θ = 800

# # Define nearest neighbor bonds
# bond = Bond(orbitals = (1,1), displacement = [1])

# Define nearest neighbor bonds
bond_x = Bond((1,1), [1,0])
bond_y = Bond((1,1), [0,1])

# # Define next nearest neighbor bonds
# bond_xy = Bond((1,1), [1,1])
# bond_yx = Bond((1,1), [1,-1])

# Collect all bond definitions
# bonds = [[bond]]

# Collect all bond definitions
bonds = [[bond_x, bond_y]]

# Define model geometry
model_geometry = ModelGeometry(unit_cell,lattice, bonds)

# Define non-interacting tight binding model
tight_binding_model = TightBindingModel([t,tp],μ_BCS)

# Initialize determinantal parameters
determinantal_parameters = initialize_determinantal_parameters(parameters_to_optimize, parameter_values)

# Get particle numbers 
# Use this if an initial density is sepcified
(Np, Ne, nup, ndn) = get_particle_numbers(n̄)

# # Get particle density 
# # Use this if initial particle number if specified
# (density, Np, Ne) = get_particle_density(nup, ndn)
    
###########################
## SET-UP VMC SIMULATION ##
###########################

# Construct mean-field Hamiltonian and variational operators
(H_mf, V) = build_mean_field_hamiltonian(tight_binding_model, determinantal_parameters)

# Initialize trial state and initial particle configuration
(D, pconfig, κ,  ε, ε₀, M, Uₑ) = build_determinantal_state(H_mf)  

# Initialize variational parameter matrices
A = get_Ak_matrices(V, Uₑ, ε, model_geometry)

# Initialize equal-time Green's function (W matrix)
W = get_equal_greens(M, D)

# # Initialize phonon parameters
# phonon_parameters = initialize_phonon_parameters(Ω, 1.0, α)

# Initialize model for electron-phonon coupling
# holstein = initialize_electron_phonon_model(μₚₕ, phonon_parameters, model_geometry)
# bond_ssh = initialize_electron_phonon_model("bond", z₀_x, z₀_y, phonon_parameters, model_geometry)
# optical_ssh = initialize_electron_phonon_model("onsite", z_x, z_y, phonon_parameters, model_geometry)

# Construct electron density-density Jastrow factor
jastrow = build_jastrow_factor("e-den-den", model_geometry, pconfig, pht, rng, readin_jpars)

# # Construct electron spin-spin Jastrow factor
# jastrow_spn = build_jastrow_factor("e-spn-spn", model_geometry, pconfig, pht, rng, readin_jpars)

# # Construct electron spin-spin Jastrow factor
# jastrow_eph = build_jastrow_factor("eph-den-den", model_geometry, pconfig, phconfig, pht, rng, readin_jpars)

# Initialize all variational parameters to be optimized
variational_parameters = VariationalParameters(determinantal_parameters, jastrow)

#############################
## INITIALIZE MEASUREMENTS ##
#############################

# Initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(model_geometry, variational_parameters, N_opts, opt_bin_size, N_bins, bin_size)

# Initialize the sub-directories to which the various measurements will be written
initialize_measurement_directories(simulation_info, measurement_container)


##################################
## PERFORM OPTIMIZATION UPDATES ##
##################################

# Start time for optimization
t_start_opt = time()

# Iterate over number of optimization updates
for bin in 1:N_opts

    if debug
        println("Performing optimization step: $bin")
    end

    # Iterate over size of optimization bins
    for n in 1:opt_bin_size

        # Perform local update to fermionic degrees of freedom
        (pconfig, κ, jastrow, W, D) = local_fermion_update!(W, D, model_geometry, jastrow, pconfig, κ, rng, n, n_stab, mc_meas_freq-100)

        # # Perform local update to bosonic degrees of freedom
        # (bosonic_acceptance_rate, phconfig, jastrow) = local_boson_update!(phconfig, model_geometry, rng)

        # Make measurements
        make_measurements!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, κ, Np, W, A, opt_bin_size)

        # Write the average measurements for the current bin to file
        write_measurements!(measurement_container, simulation_info, bin)
    end

    # Perform Stochastic Reconfiguration
    # This change was made so that a full bin is populated and the average for that bin is used in 
    # calculating the force vector and Hessian matrix
    sr_update!(measurement_container, determinantal_parameters, jastrow, η, dt, N_opts)

    if debug
        println("Ending optimization step")
    end
end

# End time for optimization
t_end_opt = time()


#################################
## PERFORM SIMULATION UPDATES  ##
#################################

# Start time for simulation
t_start_sim = time()

# Iterate over the number of bins, i.e. the number of measurements will be dumped to file.
for bin in 1:N_bins

    if debug
        println("Populating bin $bin")
    end

    # Iterate over the number of simulation updates performed in the current bin.
    for n in 1:bin_size
        # Perform local update to fermionic degrees of freedom
        (pconfig, κ, jastrow, W, D) = local_fermion_update!(W, D, Ne, model_geometry, jastrow, pconfig, κ, rng, n, n_stab)

        # Make measurements in the current bin
        make_measurements!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, κ, Np, W, A, bin_size)
    end

end

# End time for simulation
t_end_sim = time()

# Record optmization runtime
additional_info["optimization_time"] += t_end_opt - t_start_opt

# Record simulation runtime
additional_info["simulation_time"] += t_end_sim - t_start_sim

# Record total runtime
addition_info["total_time"] += additional_info["optimization_time"] + additional_info["simulation_time"]


# Write simulation information to file
# save_simulation_info(simulation_info, additional_info)

# Process measurements
# process_measurements(simulation_info.datafolder, 20)


