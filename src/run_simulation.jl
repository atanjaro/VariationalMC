using LatticeUtilities
using Random
using LinearAlgebra
using DelimitedFiles
using BenchmarkTools
using Profile
using Distributions
using OrderedCollections
using CSV
using DataFrames
using DataStructures
using Distributions

# files to include
include("Hamiltonian.jl")
include("ParticleConfiguration.jl")
include("Jastrow.jl")
include("Markov.jl")
include("Utilities.jl")
include("Greens.jl")
include("Hessian.jl")
include("Measurements.jl")

#############################
## DEFINE MODEL PARAMETERS ##
#############################

# define the size of the lattice
Lx = 2
Ly = 1

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

# (BCS) chemical potential
μ_BCS = 0.0

# phonon fugacity
# μₚₕ = 0.0

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
parameters_to_optimize = ["Δs"]        # BCS wavefunction
parameter_values = [0.3]                 # pht = true

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δcs", "Δss"]       # stripe wavefunction
# parameter_values = [0.3, 0.3]                 # pht = false

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δd", "Δa"]         # uniform d-wave wavefunction
# parameter_values = [0.3, 0.3]                 # pht = true

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δa"]               # AFM (Neél) wavefunction
# parameter_values = [0.3]                      # pht = false

# parameters to be optimized and initial value
# parameters_to_optimize = ["Δc"]               # CDW wavefunction
# parameter_values = [0.3]                      # pht = false

##################################
## DEFINE SIMULATION PARAMETERS ##
##################################

# whether model is particle-hole transformed 
pht = true
       
# initialize random seed
seed = abs(rand(Int))

# initialize random number generator
rng = Xoshiro(seed)

# number of equilibration/thermalization updates (really N_equil × Np)
N_equil = 100

# number of optimization/simulation updates (really N_updates × Np)
N_updates = 300

# number of bins
N_bins = 100

# bin size
bin_size = div(N_updates, N_bins)

# number of iterations until check for numerical stability 
n_stab = 500

# Maximum allowed error in the equal-time Green's function
δW = 1e-3

# Maximum allowed error in the T vector
δT = 1e-3

# SR stabilization factor
η = 1e-4      # 10⁻⁴ is probably good for the Hubbard model

# initial SR optimization rate
dt = 0.1        # dt must be of sufficient size such that convergence is rapid and the algorithm remains stable

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
    "time" => 0.0,
    "N_equil" => N_equil,
    "N_updates" => N_updates,
    "N_bins" => N_bins,
    "bin_size" => bin_size,
    "fermionic_local_acceptance_rate" => 0.0,
    "initial_dt" => dt,
    "final_dt" => 0.0,
    "seed" => seed,
    "n_bar" => n̄,
    "global_energy" => 0.0,
    "μ_BCS" => 0.0,
    "Δs" => 0.0
)

##################
## DEFINE MODEL ##
##################

# square unit cell
unit_cell = UnitCell([[1.0,0.0], [0.0,1.0]],           # lattice vectors
                               [[0.0,0.0]])            # basis vectors 

# build square lattice
lattice = Lattice([Lx,Ly],[true,true])

# define nearest neighbor bonds
bond_x = Bond((1,1), [1,0])
bond_y = Bond((1,1), [0,1])
# define next nearest neighbor bonds
bond_xy = Bond((1,1), [1,1])
bond_yx = Bond((1,1), [1,-1])

# vector of 2D bonds
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
jastrow = build_jastrow_factor("e-den-den", model_geometry, pconfig, pht, readin_jpars)

# initialize all variational parameters to be optimized
variational_parameters = all_vpars(determinantal_parameters, jastrow)

#############################
## INITIALIZE MEASUREMENTS ##
#############################

# initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(model_geometry, variational_parameters, Np, N_equil, N_bins, bin_size)

# initialize energy measurements
initialize_measurements!(measurement_container, "energy")

###########################################
## PERFORM BURNIN/THERMALIZATION UPDATES ##
###########################################

# start time for simulation
t_start = time()
if verbose
    println("|| START OF VMC SIMULATION ||")
end

# Iterate over equilibration/thermalization updates.
for n in 1:N_equil
    if verbose
        println("|| BEGIN EQUILIBRATION ||")
    end

    # perform local update to fermionic dofs
    (local_acceptance_rate, pconfig, jastrow, W, D) = local_fermion_update!(W, D, Ne, model_geometry, jastrow, pconfig, rng, n, n_stab)

    # record acceptance rate
    additional_info["fermionic_local_acceptance_rate"] += local_acceptance_rate

    if verbose
        println("|| END EQUILIBRATION ||")
    end

end


#######################################################
## PERFORM OPTMIZATION UPDATES AND MAKE MEASUREMENTS ##
#######################################################

if verbose
    println("|| BEGIN OPTIMIZATION ||")
end

# Iterate over the number of bins, i.e. the number of measurements will be dumped to file.
for bin in 1:N_bins

    if verbose
        println("Populating bin $bin")
    end

    # Iterate over the number of optimizations/updates performed in the current bin.
    for n in 1:bin_size
        # perform local update to fermionic dofs
        (acceptance_rate, pconfig, jastrow, W, D) = local_fermion_update!(W, D, Ne, model_geometry, jastrow, pconfig, rng, n, n_stab)

        # record acceptance rate
        additional_info["fermionic_local_acceptance_rate"] += acceptance_rate

        # perform stochastic reconfiguration
        measurement_container = sr_update!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, Np, W, A, η, dt, n, bin)
    end

    # # Write the average measurements for the current bin to file.
    # write_measurements!(
    #         measurement_container = measurement_container,
    #         model_geometry = model_geometry,
    #         bin = bin,
    #         bin_size = bin_size
    # )
end

if verbose
    println("|| END OPTIMIZATION ||")
end

# end time for simulation
t_end = time()
if verbose
    println("|| END OF VMC SIMULATION ||")
end

# record simulation runtime
additional_info["time"] += t_end - t_start

# write simulation information to file
# save_simulation_info(simulation_info, additional_info)

# process measurements
# process_measurements(simulation_info.datafolder, 20)



