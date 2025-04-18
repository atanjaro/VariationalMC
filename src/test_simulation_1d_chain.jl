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
include("Greens.jl")
include("Jastrow.jl");
include("Markov.jl");
include("Utilities.jl");
include("Optimizer.jl");
include("SimulationInfo.jl");
include("Measurements.jl");
# include("ElectronPhonon.jl");

# # Open the file for writing
# io = open("simulation_output_chain_L4_U4_deltaa_Nopts500_optbinsize1000_dt0p1.txt", "w")

# # Redirect stdout to the file
# redirect_stdout(io)

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

# define next nearest neighbor bonds
bond_xp = Bond(orbitals = (1,1), displacement = [2]);

# collect all bond definitions
bonds = [[bond_x], [bond_xp]];     # bonds are organized into [[nearest],[next-nearest]] 

# define model geometry
model_geometry = ModelGeometry(unit_cell,lattice, bonds);


#########################################
##          MODEL PARAMETERS           ##
#########################################
# nearest neighbor hopping amplitude
t = 1.0;

# next nearest neighbor hopping amplitude
tp = 0.0;

# onsite Hubbard repulsion
U = 4.0;

# (BCS) chemical potential 
μ = 0.0;

# s-wave pairing
Δ_0 = 0.1;

# d-wave pairing
Δ_d = 0.0;

# antiferromagnetic (Neél) order parameter
Δ_afm = 0.1;

# uniform charge-density-wave order parameter
Δ_cdw = 0.0;

# site dependent hole density
Δ_shd = fill(0.0, Lx);     

# site dependent spin density
Δ_sds = fill(0.0, Lx);

# Select which parameters will be optimized
optimize = ["Δ_afm"];  
# optimize = ["μ", "Δ_0"];   

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
tight_binding_model = TightBindingModel(t, tp);

# initialize determinantal parameters
determinantal_parameters = DeterminantalParameters(μ, Δ_0, Δ_d, Δ_afm, Δ_cdw, Δ_shd, Δ_sds, optimize);


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
seed = 3152596382106499424 #abs(rand(Int)) #
println("seed = ", seed)

# initialize random number generator
rng = Xoshiro(seed);

# number of equilibration/thermalization steps (formerly, mc_meas_freq)
N_equil = 3000;

# number of minimization/optimization updates
N_opts = 500;

# optimization bin size
opt_bin_size = 1000;

# number of simulation updates 
N_updates = 100;

# number of simulation bins
N_bins = 100;

# simulation bin size
bin_size = div(N_updates, N_bins);

# number of steps until numerical stabilization is performed 
n_stab_W = 1;
n_stab_T= 1;

# maximum allowed error in the equal-time Green's function
δW = 1e-3;

# maximum allowed error in the Jastrow T vector
δT = 1e-3;

# stabilization factor for Stochastic Reconfiguration
η = 1e-4;    

# optimization rate for Stochastic Reconfiguration
dt = 0.1;       

# whether debug statements are printed 
debug = true;

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


# TESTING AND DEBUG
energy_bin = Float64[];
dblocc_bin = Float64[];
param_bin = [];
global_acceptance_rate = 0.0;


println("Starting simulation...")
#############################################
##          OPTIMIZATION UPDATES           ##
#############################################
opt_start_time = time();
for bin in 1:N_opts
    println("Currently in bin: ", bin)

    println("Thermalizing system...")
    # equilibrate/thermalize the system   
    for step in 1:N_equil 
         # perform local update to electronic degrees of freedom
        (acceptance_rate, detwf, jastrow) = local_fermion_update!(detwf, jastrow, Ne, 
                                                                    model_geometry, pht, δW, δT, rng)

        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate
    end
    println("System thermalized!")

    # perform measurements for optimization
    for n in 1:opt_bin_size
        println("Starting Monte Carlo step = ", n)

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
opt_end_time = time();

# time for optimization 
opt_time = opt_end_time - opt_start_time;
println("Optimization completed in $(opt_time) seconds.")

# Close the file and restore stdout
close(io)

## BEGIN TESTING
using Plots
using LaTeXStrings

# collect Δa
deltaa = [v[1] for v in param_bin]
# collect Jastrow pseudopotentials
vij_1 = [v[2] for v in param_bin]
vij_2 = [v[3] for v in param_bin]

# # collect Δs
# deltas = [v[1] for v in param_bin]
# # collect Δs
# mus = [v[2] for v in param_bin]
# # collect Jastrow pseudopotentials
# vij_1 = [v[3] for v in param_bin]
# vij_2 = [v[4] for v in param_bin]

# write data to file
df_erg = DataFrame(A = collect(1:N_opts), B = energy_bin/opt_bin_size)
df_dblocc = DataFrame(A = collect(1:N_opts), B = dblocc_bin/opt_bin_size)
# df_afm = DataFrame(A = collect(1:N_opts), B = deltaa/opt_bin_size)
df_s = DataFrame(A = collect(1:N_opts), B = deltas/opt_bin_size)
df_vij = DataFrame(A = collect(1:N_opts), B = vij_1/opt_bin_size, C = vij_2/opt_bin_size)
df_mus = DataFrame(A = collect(1:N_opts), B = mus/opt_bin_size)
CSV.write("chain_L4_U4_deltas_Nopts500_optbinsize1000_dt0p1_energy_bins.csv", df_erg)
CSV.write("chain_L4_U4_deltas_Nopts500_optbinsize1000_dt0p1_dblocc_bins.csv", df_dblocc)
CSV.write("chain_L4_U4_deltas_Nopts500_optbinsize1000_dt0p1_deltas_bins.csv", df_s)
CSV.write("chain_L4_U4_deltas_Nopts500_optbinsize1000_dt0p1_mus_bins.csv", df_mus)
CSV.write("chain_L4_U4_deltas_Nopts500_optbinsize1000_dt0p1_vij_bins.csv", df_vij)

# plot energy per site
scatter(1:N_opts, energy_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"E/N", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts))

# plot double occupancy
scatter(1:N_opts, dblocc_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"D", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts), ylims=(0,0.5))

# plot AFM parameter
scatter(1:N_opts, deltaa/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\Delta_a", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts))

# # plot s-wave parameter
# scatter(1:N_opts, deltas/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
# legend=false, xlabel="Optimization steps", ylabel=L"\Delta_s", tickfontsize=14, guidefontsize=14, legendfontsize=14,
# xlims=(0,N_opts))

# plot Jastrow parameters
scatter(1:N_opts, vij_1/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        label=L"v_{ij}^1", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts)) 
scatter!(1:N_opts, vij_2/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        label=L"v_{ij}^2", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts))

# # plot mu parameter
# scatter(1:N_opts, mus/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
#         legend=false, xlabel="Optimization steps", ylabel=L"\mu_{\mathrm{BCS}}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
#         xlims=(0,N_opts))
## END TESTING






# ###########################################
# ##          SIMULATION UPDATES           ##
# ###########################################
# start_time = time()
# for bin in 1:N_updates
#     # equilibrate/thermalize the system   
#     for step in 1:N_equil 
#         (acceptance_rate, detwf, jastrow) = local_fermion_update!(detwf, jastrow, Ne, 
#                                                                     model_geometry, pht, δW, δT, rng)
#         # record acceptance rate                                                        
#         global_acceptance_rate += acceptance_rate
#     end
#     for n in 1:bin_size
#         acceptance_rate, detwf, jastrow = local_fermion_update!(detwf, jastrow, Ne, 
#                                                                 model_geometry, pht, δW, δT, rng)
        
#         # record acceptance rate                                                        
#         global_acceptance_rate += acceptance_rate

#         # make basic measurements
#         make_measurements!(measurement_container, detwf, tight_binding_model, 
#                             determinantal_parameters, jastrow, model_geometry, Ne, pht)
#     end
#     # write measurements (to file)
#     write_measurements!(measurement_container, simulation_info, 
#                         energy_bin, dblocc_bin, param_bin)
# end
# end_time = time()

# time for simulation
sim_time = end_time - start_time

# total VMC time
total_time = opt_time + sim_time