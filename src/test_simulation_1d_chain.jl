using LatticeUtilities
using Random
using LinearAlgebra
using DelimitedFiles
using Profile
using OrderedCollections
using CSV
using DataFrames
using DataStructures
using Printf
using JLD2
using Revise
# files to include
include("Hamiltonian.jl");
include("ParticleConfiguration.jl");
include("Wavefunction.jl");
include("Jastrow.jl");
include("Markov.jl");
include("Utilities.jl");
include("Optimizer.jl");
include("SimulationInfo.jl");
include("Measurements.jl");
# include("ElectronPhonon.jl");


###########################################
##          LATTICE PARAMETERS           ##
###########################################
# define the size of the lattice
Lx = 4;
Ly = 1;

# chain unit cell
unit_cell = UnitCell(lattice_vecs = [[1.0]],
                            basis_vecs   = [[0.0]]);

# build the lattice
lattice = Lattice([Lx],[true]);

# define nearest neighbor bonds
bond_x = Bond(orbitals = (1,1), displacement = [1]);

# # define next nearest neighbor bonds
# bond_xp = Bond(orbitals = (1,1), displacement = [2]);

# collect all bond definitions
bonds = [[bond_x]];     # bonds are organized into [[nearest],[next-nearest]] #,[bond_xp]

# define model geometry
model_geometry = ModelGeometry(unit_cell,lattice, bonds);


#########################################
##          MODEL PARAMETERS           ##
#########################################
# nearest neighbor hopping amplitude
t = 1.0;

# next nearest neighbor hopping amplitude
tp = 0.0;

# (BCS) chemical potential 
μ_BCS = 0.0;

# onsite Hubbard repulsion
U = 1.0;

# antiferromagnetic (Neél) order parameter
Δa = 0.1;

# s-wave pairing (BCS)
Δs = 0.1;

# Parameters to be optimized and initial value(s)
parameters_to_optimize = ["Δa"];   
parameter_values = [[Δa]];     

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δs", "μ_BCS"];   
# parameter_values = [[Δs],[μ_BCS]];  

# whether model is particle-hole transformed
pht = false;

# define electron density
n̄ = 1.0;

# # define electron numbers     
# nup = 2
# ndn = 2

# Get particle numbers 
(Np, Ne, nup, ndn) = get_particle_numbers(n̄);       # Use this if an initial density is specified

# # Get particle density 
# (density, Np, Ne) = get_particle_density(nup, ndn);   # Use this if initial particle numbers are specified

# define non-interacting tight binding model
tight_binding_model = TightBindingModel([t, tp], μ_BCS);

# initialize determinantal parameters
determinantal_parameters = initialize_determinantal_parameters(parameters_to_optimize, parameter_values);


######################################
##          FILE HANDLING           ##
######################################
# specify filepath
filepath = ".";

# simulation ID
sID = 1;

# construct the foldername the data will be written
param_names = convert_par_name(parameters_to_optimize);
datafolder_prefix = @sprintf "hubbard_chain_U%.2f_n%.2f_Lx%d_Ly%d_" U n̄ Lx Ly ;
datafolder_prefix = datafolder_prefix * param_names;

# initialize an instance of the SimulationInfo type
simulation_info = SimulationInfo(
                filepath = filepath, 
                datafolder_prefix = datafolder_prefix,
                sID = sID
);

# initialize the directory the data will be written to
initialize_datafolder(simulation_info);


##############################################
##          SIMULATION PARAMETERS           ##
##############################################

# random seed
seed = abs(rand(Int)) #8947935343182193186 # 

# initialize random number generator
rng = Xoshiro(seed);

# number of equilibration/thermalization steps (formerly, mc_meas_freq)
N_equil = 1000;

# number of minimization/optimization updates
N_opts = 500;

# optimization bin size
opt_bin_size = 3000;

# number of simulation updates 
N_updates = 100;

# number of simulation bins
N_bins = 100;

# simulation bin size
bin_size = div(N_updates, N_bins);

# number of steps until numerical stabilization is performed 
n_stab_W = 50;
n_stab_T= 50;

# maximum allowed error in the equal-time Green's function
δW = 1e-3;

# maximum allowed error in the Jastrow T vector
δT = 1e-3;

# stabilization factor for Stochastic Reconfiguration
η = 1e-4;    

# optimization rate for Stochastic Reconfiguration
dt = 0.03;   # 0.03      

# whether debug statements are printed 
debug = false;

# # Initialize additional simulation information dictionary
# additional_info = Dict(
#     "δW" => δW,
#     "δT" => δT,
#     "total_time" => 0.0,
#     "simulation_time" => 0.0,
#     "optimization_time" => 0.0,
#     "N_opts" => N_opts,
#     "N_updates" => N_updates,
#     "N_bins" => N_bins,
#     "bin_size" => bin_size,
#     "dt" => dt,
#     "seed" => seed,
#     "n_bar" => n̄,
#     "global_energy" => 0.0,
#     "μ_BCS" => 0.0
# )


##############################################
##          SET-UP VMC SIMULATION           ##
##############################################

# initialize determinantal wavefunction
detwf = build_determinantal_wavefunction(tight_binding_model, determinantal_parameters, 
                                        Ne, nup, ndn, model_geometry, rng);

# initialize (density) Jastrow factor
jastrow = build_jastrow_factor("e-den-den", detwf, model_geometry, pht, rng);

# # initialize spin Jastrow factor
# sjastrow = build_jastrow_factor("e-spn-spn", detwf, model_geometry, pht, rng);

# Initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(N_opts, opt_bin_size, N_bins, bin_size,
                                                        determinantal_parameters, jastrow, 
                                                        model_geometry);

# Initialize the sub-directories to which the various measurements will be written
initialize_measurement_directories(simulation_info, measurement_container);


# DEBUG
energy_bin = Float64[];
dblocc_bin = Float64[];
param_bin = [];
global_acceptance_rate = 0.0;


#############################################
##          OPTIMIZATION UPDATES           ##
#############################################
start_time = time()
for bin in 1:N_opts
    # equilibrate/thermalize the system   
    for step in 1:N_equil 
         # perform local update to electronic degrees of freedom
        (acceptance_rate, detwf, jastrow) = local_fermion_update!(detwf, jastrow, Ne, 
                                                                    model_geometry, pht, δW, δT, rng)

        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate
    end

    # perform measurements for optimization
    for n in 1:opt_bin_size
        # perform local update to electronic degrees of freedom
        (acceptance_rate, detwf, jastrow) = local_fermion_update!(detwf, jastrow, Ne, 
                                                                    model_geometry, pht, δW, δT, rng) # the mc_meas_freq variable will have
                                                                                                      # to be removed from this function
        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate
        
        # make basic measurements
        make_measurements!(measurement_container, detwf, tight_binding_model, 
                            determinantal_parameters, jastrow, model_geometry, Ne, pht)
    end
    # perform update to variational parameters
    stochastic_reconfiguration!(measurement_container, determinantal_parameters, 
                                jastrow, η, dt, bin_size)

    # write measurements (to file)
    write_measurements!(measurement_container, energy_bin, dblocc_bin, param_bin)
end     
end_time = time()

# time for optimization 
opt_time = end_time - start_time



## BEGIN TESTING
using Plots
using LaTeXStrings

# collect Δa
deltaa = [v[1] for v in param_bin]
# collect Jastrow pseudopotentials
vij_1 = [v[2] for v in param_bin]
vij_2 = [v[3] for v in param_bin]

# collect Δs
deltas = [v[1] for v in param_bin]
# collect Δs
mus = [v[2] for v in param_bin]
# collect Jastrow pseudopotentials
vij_1 = [v[3] for v in param_bin]
vij_2 = [v[4] for v in param_bin]

# plot energy per site
scatter(1:500, energy_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"E/N", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,500))

# plot double occupancy
scatter(1:500, dblocc_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"D", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,500), ylims=(0,0.5))

# plot AFM parameter
scatter(1:500, deltaa/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\Delta_a", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,500))

# plot Jastrow parameters
scatter(1:500, vij_1/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        label=L"v_{ij}^1", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,500)) 
scatter!(1:500, vij_2/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        label=L"v_{ij}^2", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,500))
## END TESTING

###########################################
##          SIMULATION UPDATES           ##
###########################################
start_time = time()
for bin in 1:N_updates
    # equilibrate/thermalize the system   
    for step in 1:N_equil 
        (acceptance_rate, detwf, jastrow) = local_fermion_update!(detwf, jastrow, Ne, 
                                                                    model_geometry, pht, δW, δT, rng)
        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate
    end
    for n in 1:bin_size
        acceptance_rate, detwf, jastrow = local_fermion_update!(detwf, jastrow, Ne, 
                                                                model_geometry, pht, δW, δT, rng)
        
        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate

        # make basic measurements
        make_measurements!(measurement_container, detwf, tight_binding_model, 
                            determinantal_parameters, jastrow, model_geometry, Ne, pht)
    end
    # write measurements (to file)
    write_measurements!(measurement_container, simulation_info, 
                        energy_bin, dblocc_bin, param_bin)
end
end_time = time()

# time for simulation
sim_time = end_time - start_time

# total VMC time
total_time = opt_time + sim_time