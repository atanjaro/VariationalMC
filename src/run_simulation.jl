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

# define the size of the lattice
Lx = 4
Ly = 4

# define initial electron density
n̄ = 1.0

# number of particles
# nup = 8
# ndn = 8

# nearest neighbor hopping
t = 1.0

# next nearest neighbor hopping
tp = 0.0

# Hubbard-U
U = 0.5

# chemical potential
μ_BCS = 0.0

# # phonon chemical potential (fugacity)
# μₚₕ = 0.01

# # phonon frequency
# Ω = 1.0

# # microscopic electron-phonon coupling
# g = 1.0

# # microscopic electron phonon coupling
# α = g * sqrt(2 * Ω)

# # dimensionless electron-phonon coupling (g definition)
# λ = (2 * g^2) / (Ω * 8)

# # dimensionless electron-phonon coupling (α defintion)
# λ = α^2 / (Ω^2 * 8)

#######################################
##      VARIATIONAL PARAMETERS       ##
#######################################

# whether to read-in initial determinantal parameters
readin_detpars = false
path_to_detpars = "/path/to/determinantal/parameters/"

# whether to read-in initial Jastrow parameters
readin_jpars = false
path_to_jpars = "/Users/xzt/Documents/VariationalMC/src/jastrow_out.csv"

# parameters to be optimized and initial value
parameters_to_optimize = ["Δs"]                      # BCS wavefunction
parameter_values = [0.01]                            # pht = true

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δcs", "Δss"]            # stripe wavefunction
# parameter_values = [0.01, 0.01]                    # pht = false

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δd", "Δa"]              # uniform d-wave wavefunction
# parameter_values = [0.01, 0.01]                    # pht = true

# # parameters to be optimized and initial value
# parameters_to_optimize = ["Δa"]                    # AFM (Neél) wavefunction
# parameter_values = [0.01]                          # pht = false

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δc"]                    # CDW wavefunction
# parameter_values = [0.0.1]                         # pht = false

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δs", "μₚₕ"]              # BCS wavefunction + phonons
# parameter_values = [0.01, μₚₕ]                      # pht = true          

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δcs", "Δss", "μₚₕ"]      # stripe wavefunction + phonons
# parameter_values = [0.01, 0.01, μₚₕ]                # pht = false        


###################
## FILE HANDLING ##
###################

# specify filepath
filepath = "."

# simulation ID
sID = 1

# construct the foldername the data will be written to
# note that the simulation ID `sID`` will be appended to this foldername as `*-sID`
datafolder_prefix = @sprintf "hubbard_square_U%.2f_n%.2f_Lx%d_Ly%d_swf" U n̄ Lx Ly 

# initialize an instance of the SimulationInfo type
# this type helps keep track of where data will be written to
simulation_info = SimulationInfo(
                filepath = filepath, 
                datafolder_prefix = datafolder_prefix,
                sID = sID
)

# initialize the directory the data will be written to.
initialize_datafolder(simulation_info)


##################################
## DEFINE SIMULATION PARAMETERS ##
##################################

# whether model is particle-hole transformed 
pht = true
       
# initialize random seed
seed = abs(rand(Int))

# initialize random number generator
rng = Xoshiro(seed)

# number of optimization updates
N_opts = 100 #3000

# optimization bin size
opt_bin_size = 100 #6000

# number of simulation updates 
N_updates = 100    #10000

# number of simulation bins
N_bins = 100

# simulation bin size
bin_size = div(N_updates, N_bins)

# number of MC cycles until measurement
mc_meas_freq = 10 #300

# number of iterations until check for numerical stability 
n_stab = 500

# Maximum allowed error in the equal-time Green's function
δW = 1e-3

# Maximum allowed error in the T vector
δT = 1e-3

# SR stabilization factor
η = 1e-4      

# SR optimization rate
dt = 0.03        

# whether to output to terminal during runtime
verbose = true

# debugging (this will be removed later)
debug = false

# whether to output matrices to file
write = false

# initialize additional simulation information dictionary
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

# # chain unit cell
# unit_cell = UnitCell(lattice_vecs = [[1.0]],
#                             basis_vecs   = [[0.0]])

# square unit cell
unit_cell = UnitCell([[1.0,0.0], [0.0,1.0]],           # lattice vectors
                               [[0.0,0.0]])            # basis vectors 

# # build a chain
# lattice = Lattice([Lx],[true])

# build square lattice
lattice = Lattice([Lx, Ly],[true,true])

# define nearest neighbor bonds
bond_x = Bond((1,1), [1,0])
bond_y = Bond((1,1), [0,1])
# define next nearest neighbor bonds
bond_xy = Bond((1,1), [1,1])
bond_yx = Bond((1,1), [1,-1])

# # define nearest enighbor bonds
# bond = Bond(orbitals = (1,1), displacement = [1])

# vector of all bonds
# bonds = [[bond]]
bonds = [[bond_x, bond_y], [bond_xy, bond_yx]]

# define model geometry
model_geometry = ModelGeometry(unit_cell,lattice, bonds)

# define non-interacting tight binding model
tight_binding_model = TightBindingModel([t,tp],μ_BCS)

# initialize determinantal parameters
determinantal_parameters = initialize_determinantal_parameters(parameters_to_optimize, parameter_values)

# get particle numbers (use if initial density is specified)
(Np, Ne, nup, ndn) = get_particle_numbers(n̄)

# get particle density (use if initial particle number if specified)
# density, Np, Ne = get_particle_density(nup, ndn)
    
###########################
## SET-UP VMC SIMULATION ##
###########################

# construct mean-field Hamiltonian and return variational operators
(H_mf, V) = build_mean_field_hamiltonian(tight_binding_model, determinantal_parameters)

# initialize Slater determinant state and initial particle configuration
(D, pconfig, ε, ε₀, M, Uₑ) = build_determinantal_state(H_mf)  

# initialize variational parameter matrices
A = get_Ak_matrices(V, Uₑ, ε, model_geometry)

# initialize equal-time Green's function (W matrix)
W = get_equal_greens(M, D)

# construct electron density-density Jastrow factor
jastrow = build_jastrow_factor("e-den-den", model_geometry, pconfig, pht, rng, readin_jpars)

# # construct electron spin-spin Jastrow factor
# jastrow = build_jastrow_factor("e-spn-spn", model_geometry, pconfig, pht, rng, readin_jpars)

# initialize all variational parameters to be optimized
variational_parameters = VariationalParameters(determinantal_parameters, jastrow)

#############################
## INITIALIZE MEASUREMENTS ##
#############################

# initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(model_geometry, variational_parameters, N_opts, opt_bin_size, N_bins, bin_size)

# initialize the sub-directories to which the various measurements will be written
initialize_measurement_directories(simulation_info, measurement_container)


##################################
## PERFORM OPTIMIZATION UPDATES ##
##################################

# start time for optimization
t_start_opt = time()

# iterate over number of optimization updates
for bin in 1:N_opts

    # iterate over size of optimization bins
    for n in 1:opt_bin_size

        # perform local update to fermionic degrees of freedom
        (pconfig, jastrow, W, D) = local_fermion_update!(W, D, model_geometry, jastrow, pconfig, rng, n, n_stab, mc_meas_freq)

        # # perform local update to bosonic degrees of freedom
        # (bosonic_acceptance_rate, phconfig, jastrow) = local_boson_update!(phconfig, model_geometry, rng)

        # make measurements
        make_measurements!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, Np, W, A)

        # perform Stochastic Reconfiguration
        sr_update!(measurement_container, determinantal_parameters, jastrow, η, dt)

        # write the average measurements for the current bin to file.
        write_measurements!(measurement_container, simulation_info, bin)
    end
end

# end time for optimization
t_end_opt = time()


#######################################################
## PERFORM SIIMULATION UPDATES AND MAKE MEASUREMENTS ##
#######################################################

# start time for simulation
t_start_sim = time()

# Iterate over the number of bins, i.e. the number of measurements will be dumped to file.
for bin in 1:N_bins

    if verbose
        println("Populating bin $bin")
    end

    # Iterate over the number of simulation updates performed in the current bin.
    for n in 1:bin_size
        # perform local update to fermionic dofs
        (acceptance_rate, pconfig, jastrow, W, D) = local_fermion_update!(W, D, Ne, model_geometry, jastrow, pconfig, rng, n, n_stab)

        # make measurements in the current bin
        make_measurements!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, Np, W, A)
    end

    # perform stochastic reconfiguration
    measurement_container = sr_update!(measurement_container, determinantal_parameters, jastrow, η, dt)

    # # Write the average measurements for the current bin to file.
    # write_measurements!(
    #         measurement_container = measurement_container,
    #         model_geometry = model_geometry,
    #         bin = bin,
    #         bin_size = bin_size
    # )
end

# end time for simulation
t_end_sim = time()

# record optmization runtime
additional_info["optimization_time"] += t_end_opt - t_start_opt

# record simulation runtime
additional_info["simulation_time"] += t_end_sim - t_start_sim

# record total runtime
addition_info["total_time"] += additional_info["optimization_time"] + additional_info["simulation_time"]


# write simulation information to file
# save_simulation_info(simulation_info, additional_info)

# process measurements
# process_measurements(simulation_info.datafolder, 20)


