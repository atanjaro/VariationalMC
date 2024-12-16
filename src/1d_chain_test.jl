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
using Revise

# files to include
include("Hamiltonian.jl")
include("ElectronPhonon.jl")
include("Jastrow.jl")
include("ParticleConfiguration.jl")
include("Markov.jl")
include("Utilities.jl")
include("Greens.jl")
include("Hessian.jl")
include("SimulationInfo.jl")
include("Measurements.jl")


##
## This is a test script for a Hubbard model 4 site chain at half-filling
##


# Define the size of the lattice
Lx = 4
Ly = 1

# Define electron density
n̄ = 1.0

# # Define electron numbers
# nup = 5
# ndn = 5

# Nearest neighbor hopping amplitude
t = 1.0

# Next nearest neighbor hopping amplitude
tp = 0.0

# Onsite Hubbard repulsion
U = 1.0

# (BCS) chemical potential
μ_BCS = 0.0

# BCS (s-wave) pairing
Δs = 0.1

# antiferromagnetic order parameter
Δa = 0.001

# Parameters to be optimized and initial value(s)
parameters_to_optimize = ["Δs", "μ_BCS"]                              # s-wave (BCS) order parameter
parameter_values = [[Δs],[μ_BCS]]                                 
pht = true

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δa"]                                       # antiferromagnetic (Neél) order parameter
# parameter_values = [[Δa]]                                            
# pht = false

# specify filepath
filepath = "."

# simulation ID
sID = 1


# Construct the foldername the data will be written
# whose structure is: "modelname_geometry_U_n_Lx_Ly_param1_param2_..._paramN"
# Note that the simulation ID `sID`` will be appended to this foldername as `*-sID`
param_names = convert_par_name(parameters_to_optimize)
datafolder_prefix = @sprintf "hubbard_chain_U%.2f_n%.2f_Lx%d_Ly%d_" U n̄ Lx Ly 
datafolder_prefix = datafolder_prefix * param_names

# # Construct the foldername the data will be written
# # whose structure is: "modelname_geometry_U_nup_ndn_Lx_Ly_param1_param2_..._paramN"
# # Note that the simulation ID `sID`` will be appended to this foldername as `*-sID`
# param_names = convert_par_name(parameters_to_optimize)
# datafolder_prefix = @sprintf "hubbard_chain_U%.2f_nup%.0f_ndn%.0f_Lx%d_Ly%d_" U nup ndn Lx Ly 
# datafolder_prefix = datafolder_prefix * param_names

# Initialize an instance of the SimulationInfo type
# This type helps keep track of where data will be written
simulation_info = SimulationInfo(
                filepath = filepath, 
                datafolder_prefix = datafolder_prefix,
                sID = sID
)

# Initialize the directory the data will be written to
initialize_datafolder(simulation_info)

# random seed
seed = abs(rand(Int)) # 1829519153600081228 #

# Initialize random number generator
rng = Xoshiro(seed)
       
# Number of minimization/optimization updates
N_opts = 100

# Optimization bin size
opt_bin_size = 100

# Number of simulation updates 
N_updates = 100

# Number of simulation bins
N_bins = 10

# Simulation bin size
bin_size = div(N_updates, N_bins)

# Number of MC cycles until measurement
mc_meas_freq = 100

# number of steps until numerical stability is performed (null)
n_stab = 50

# Stabilization factor for Stochastic Reconfiguration
η = 1e-4      

# Optimization rate for Stochastic Reconfiguration
dt = 0.03        

# Debugging flag
# This will output print statements to the terminal during runtime
debug = true

# Chain unit cell
unit_cell = UnitCell(lattice_vecs = [[1.0]],
                            basis_vecs   = [[0.0]]);

# Build a chain
lattice = Lattice([Lx],[true]);

# Define nearest neighbor bonds
bond_x = Bond(orbitals = (1,1), displacement = [1]);

# Collect all bond definitions
bonds = [[bond_x]];

# Define model geometry
model_geometry = ModelGeometry(unit_cell,lattice, bonds);

# Define non-interacting tight binding model
tight_binding_model = TightBindingModel([t,tp],μ_BCS);

# Initialize determinantal parameters
determinantal_parameters = initialize_determinantal_parameters(parameters_to_optimize, parameter_values);

# # Get particle density 
# # Use this if initial particle number if specified
# (density, Np, Ne) = get_particle_density(nup, ndn)

# Get particle numbers 
# Use this if an initial density is sepcified
(Np, Ne, nup, ndn) = get_particle_numbers(n̄);

# Construct mean-field Hamiltonian and variational operators
(H_mf, V) = build_mean_field_hamiltonian(tight_binding_model, determinantal_parameters);

# Initialize trial state and initial particle configuration
(D, pconfig, κ,  ε, ε₀, M, Uₑ) = build_determinantal_state(H_mf);  

# Initialize variational parameter matrices
A = get_Ak_matrices(V, Uₑ, ε, model_geometry);           # DEBUG

# Initialize equal-time Green's function (W matrix)
W = get_equal_greens(M, D);                              # DEBUG

# Construct electron density-density Jastrow factor
jastrow_den = build_jastrow_factor("e-den-den", model_geometry, pconfig, pht, rng, false);

# Initialize all variational parameters to be optimized
variational_parameters = VariationalParameters(determinantal_parameters, jastrow_den);

# Initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(model_geometry, variational_parameters, N_opts, opt_bin_size, N_bins, bin_size);

# Initialize the sub-directories to which the various measurements will be written
initialize_measurement_directories(simulation_info, measurement_container);



##
## OPTIMIZATION TEST
##


## There seems to be never be any change in the derivatives associated with determinantal parameters. 
## As a result, it is like the parameter is never optimized, unlike the Jastrow parameters. Will have to take a closer look.

# Here are some vector 'bins' for storing data during this test
energy_bin = []
dblocc_bin = []
parameter_bin = []


bin = 1

for n in 1:opt_bin_size
    # local fermion update for mc_meas_freq equilibration steps
    (pconfig, κ, jastrow_den, W, D) = local_fermion_update!(W, D, model_geometry, jastrow_den, pconfig, κ, rng, n, n_stab, mc_meas_freq)

    make_measurements!(measurement_container,determinantal_parameters, jastrow_den, model_geometry, 
                                tight_binding_model, pconfig, κ, Np, W, A)
end

# this is the deconstructed SR updating scheme

S = get_hessian_matrix(measurement_container, opt_bin_size)
f = get_force_vector(measurement_container, opt_bin_size)

δvpars =  parameter_gradient(S, f, η)

# new varitaional parameters
vpars = all_vpars(determinantal_parameters, jastrow_den)
vpars += dt * δvpars

# push back Jastrow parameters
update_jastrow!(jastrow_den, vpars)     # BUG: when the Jastrow parameters are pushed back,
                                             # the wrong parameters are being pushed to the wrong entries in the jpar_map

# push back determinantal_parameters
update_detpars!(determinantal_parameters, vpars)




# after opt_bin_size iterations, begin SR update
# retrive value of this_bin_sum, averaging by opt_bin_size
# use derivative and energy measurements to build S and f
# solve
# update parameters 
# overwrite this_bin_sum of each measruement container












