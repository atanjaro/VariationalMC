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

# FOR DATA PROCESSING TEST
using Plots
using LaTeXStrings

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
# preamble = "chain_L4_U1_mu0.0_afm0.0_cdw0.0_swave0.0_pht_opt_swave_mu_dt0.1"
# io = open("simulation_output_" * preamble * ".txt", "w")

# # Redirect stdout to the file
# redirect_stdout(io)

###########################################
##          LATTICE PARAMETERS           ##
###########################################
# define the size of the lattice
Lx = 4;
Ly = 4;

# square unit cell
unit_cell = UnitCell([[1.0,0.0], [0.0,1.0]],           # lattice vectors
                               [[0.0,0.0]])            # basis vectors 

# build the lattice
lattice = Lattice([Lx, Ly], [true, true]);

# define nearest neighbor bonds
bond_x = Bond((1,1), [1,0])
bond_y = Bond((1,1), [0,1])

# define next nearest neighbor bonds
bond_xp = Bond((1,1), [1,1])
bond_yp = Bond((1,1), [1,-1])

# collect all bond definitions
bonds = [[bond_x, bond_y], [bond_xp, bond_yp]];     # bonds are organized into [[nearest],[next-nearest]] 

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
U = 1.0;

# use to read-in initial parameters from file
path_to_parameter_file = "/Users/xzt/Documents/VariationalMC/src/vpar_out.dat"

# minimum value of each variational parameter (this is to avoid open shell issues)
minabs_vpar = 1e-4;

# select which parameters will be optimized
optimize = (
    # (BCS) chemical potential
    μ = false,
    # onsite (s-wave)
    Δ_0 = false,
    # d-wave
    Δ_d = false,
    # spin-z (AFM)
    Δ_afm = false,
    # charge density 
    Δ_cdw = false,
    # site-dependent charge (stripe)
    Δ_sdc = false,
    # site-dependent spin (stripe)
    Δ_sds = false,
    # density Jastrow 
    djastrow = false,
    # spin Jastrow
    sjastrow = false
)

# whether model is particle-hole transformed
pht = true;

# define electron density
n̄ = 1.0;

# # define electron numbers     
# nup = 2
# ndn = 2

# Get particle numbers 
(Np, Ne, nup, ndn) = get_particle_numbers(n̄);       # Use this if an initial density is specified

# # Get particle density 
# (density, Np, Ne) = get_particle_density(nup, ndn);   # Use this if initial particle numbers are specified

######################################
##          FILE HANDLING           ##
######################################
# specify filepath
filepath = ".";

# simulation ID
sID = 1;

# construct the foldername the data will be written
df_prefix = @sprintf "hubbard_square_U%.2f_n%.2f_Lx%d_Ly%d_opt" U n̄ Lx Ly;

# append parameters to the foldername
datafolder_prefix = create_datafolder_prefix(optimize, df_prefix)

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
seed = abs(rand(Int)) 
println("seed = ", seed)

# initialize random number generator
rng = Xoshiro(seed);

# number of equilibration/thermalization steps 
N_equil = 300;

# number of minimization/optimization updates
N_opts = 1000;

# optimization bin size
opt_bin_size = 1000;

# number of simulation updates 
N_updates = 1000;

# number of simulation bins
N_bins = 100;

# simulation bin size
bin_size = div(N_updates, N_bins);

# number of steps until numerical stabilization is performed 
n_stab_W = 50;
n_stab_T = 50;

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

# define non-interacting tight binding model
tight_binding_model = TightBindingModel(t, tp);

# # initialize determinantal parameters
# determinantal_parameters = DeterminantalParameters(optimize, tight_binding_model, 
#                                                     model_geometry, minabs_vpar, Ne, pht);

# initialize determinantal parameters from file
determinantal_parameters =  DeterminantalParameters(optimize, model_geometry, pht, 
                                                        path_to_parameter_file);


# # initialize (density) Jastrow factor
# jastrow = build_jastrow_factor("e-den-den", detwf, model_geometry, pht, rng);

# # initialize spin Jastrow factor
# sjastrow = build_jastrow_factor("e-spn-spn", detwf, model_geometry, pht, rng);

# Initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(N_opts, opt_bin_size, N_bins, bin_size,
                                                        determinantal_parameters,
                                                        model_geometry);

# Initialize the sub-directories to which the various measurements will be written
initialize_measurement_directories(simulation_info, measurement_container);


# TESTING AND DEBUG
energy_bin = Float64[];
dblocc_bin = Float64[];
param_bin = [];
global_acceptance_rate = 0.0;


#############################################
##          OPTIMIZATION UPDATES           ##
#############################################
println("Starting optimization...")
opt_start_time = time();

# particle configuration cache
pconfig_cache = nothing

for bin in 1:N_opts
    # initialize determinantal wavefunction
    detwf = get_determinantal_wavefunction(tight_binding_model, determinantal_parameters, 
                                            optimize, Ne, nup, ndn, model_geometry, rng,
                                            pconfig_cache);   
    println("Currently in bin: ", bin)

    println("Thermalizing system...")
    # equilibrate/thermalize the system   
    for step in 1:N_equil 
         # perform local update to electronic degrees of freedom
        (acceptance_rate, detwf) = local_fermion_update!(detwf, Ne, model_geometry, δW, rng);

        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate;
    end
    println("System thermalized!")

    # perform measurements for optimization
    for n in 1:opt_bin_size
        println("Starting Monte Carlo step = ", n)

        # perform local update to electronic degrees of freedom
        (acceptance_rate, detwf) = local_fermion_update!(detwf, Ne, model_geometry, δW, rng); 
                                                                                                    
        # record acceptance rate                                                        
        global_acceptance_rate += acceptance_rate;
        
        # make basic measurements
        make_measurements!(measurement_container, detwf, tight_binding_model, 
                            determinantal_parameters, optimize, model_geometry, Ne, pht);
    end

    # save particle configuration from the current bin
    pconfig_cahce = detwf.pconfig

    # perform update to variational parameters
    stochastic_reconfiguration!(measurement_container, 
                                determinantal_parameters, η, dt, opt_bin_size)

    # write measurements (to file)
    write_measurements!(measurement_container, energy_bin, dblocc_bin, param_bin)
end     
opt_end_time = time();

# time for optimization 
opt_time = opt_end_time - opt_start_time;
println("Optimization completed in $(opt_time) seconds.")


###########################################
##          SIMULATION UPDATES           ##
###########################################
println("Starting simulation...")
sim_start_time = time();

# SKIP simulation for now
# for bin in 1:N_updates
#     # initialize determinantal wavefunction
#     detwf = get_determinantal_wavefunction(tight_binding_model, determinantal_parameters, 
#                                             optimize, Ne, nup, ndn, model_geometry, rng,
#                                             pconfig_cache);   
#     println("Currently in bin: ", bin)

#     println("Thermalizing system...")
#     # equilibrate/thermalize the system   
#     for step in 1:N_equil 
#          # perform local update to electronic degrees of freedom
#         (acceptance_rate, detwf) = local_fermion_update!(detwf, Ne, model_geometry, δW, rng);

#         # record acceptance rate                                                        
#         global_acceptance_rate += acceptance_rate;
#     end
#     println("System thermalized!")

#     # perform measurements for optimization
#     for n in 1:bin_size
#         println("Starting Monte Carlo step = ", n)

#         # perform local update to electronic degrees of freedom
#         (acceptance_rate, detwf) = local_fermion_update!(detwf, Ne, model_geometry, δW, rng); 
                                                                                                    
#         # record acceptance rate                                                        
#         global_acceptance_rate += acceptance_rate;
        
#         # make basic measurements
#         make_measurements!(measurement_container, detwf, tight_binding_model, 
#                             determinantal_parameters, optimize, model_geometry, Ne, pht);
#     end

#     # save particle configuration from the current bin
#     pconfig_cahce = detwf.pconfig

#     # write measurements (to file)
#     write_measurements!(measurement_container, energy_bin, dblocc_bin, param_bin)
# end
sim_end_time = time();

# time for simulation
sim_time = sim_end_time - sim_start_time;
println("Simulation completed in $(sim_time) seconds.")

# total VMC time
total_time = opt_time + sim_time;
println("Total VMC time = $(total_time) seconds.")

# Close the file and restore stdout
close(io)


## BEGIN DATA PROCESSING TESTS  ##

# # PARAMETER SET 1 (MU, AFM, CDW)
# deltam = [v[1] for v in param_bin]
# deltaa = [v[2] for v in param_bin]
# deltac = [v[3] for v in param_bin]

# PARAMETER SET 2 (MU, SWAVE, AFM, CDW)
mu = [v[1] for v in param_bin]
deltas = [v[2] for v in param_bin]
deltaa = [v[3] for v in param_bin]
deltac = [v[4] for v in param_bin]


# # collect energy results
# N = 4
# E_N = (energy_bin[1:N_opts]/opt_bin_size + energy_bin[1:N_updates]/bin_size)/4

# QUICK PLOTTING
# exact energy from ED (U = 1)
energy_exact_U1 = -0.8352119

# plot energy per site
energy_plt = scatter(1:N_opts, energy_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"E/N", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts), ylims=(-1.05, 0))
# hline!([energy_exact_U1], linestyle=:dash, color=:black, linewidth=2)
savefig(energy_plt, "energy_" * preamble * ".png")

# plot double occupancy
dblocc_plt = scatter(1:N_opts, dblocc_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"D", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts), ylims=(0,0.5))
savefig(dblocc_plt, "dblocc_" * preamble * ".png")

# plot AFM parameter
dafm_plt = scatter(1:N_opts, deltaa/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\Delta_{\mathrm{AFM}}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts), ylims=(0, 0.001))
savefig(dafm_plt, "dafm_" * preamble * ".png")

dcdw_plt = scatter(1:N_opts, deltac/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\Delta_{\mathrm{CDW}}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts))
savefig(dcdw_plt, "dcdw_" * preamble * ".png")

mu_plt = scatter(1:N_opts, mu/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\mu_{\mathrm{BCS}}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts))
savefig(mu_plt, "mu_" * preamble * ".png")

ds_plt = scatter(1:N_opts, deltas/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\Delta_{0}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,N_opts),ylims=(0.00001,0.0001))
savefig(ds_plt, "ds_" * preamble * ".png")

# # plot s-wave parameter
# scatter(1:N_opts, deltas/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
# legend=false, xlabel="Optimization steps", ylabel=L"\Delta_s", tickfontsize=14, guidefontsize=14, legendfontsize=14,
# xlims=(0,N_opts))

# # plot Jastrow parameters
# scatter(1:N_opts, vij_1/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
#         label=L"v_{ij}^1", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
#         xlims=(0,N_opts)) 
# scatter!(1:N_opts, vij_2/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
#         label=L"v_{ij}^2", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
#         xlims=(0,N_opts))

# # plot mu parameter
# scatter(1:N_opts, mus/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
#         legend=false, xlabel="Optimization steps", ylabel=L"\mu_{\mathrm{BCS}}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
#         xlims=(0,N_opts))


# WRITE DATA TO FILE
# energy
df_erg = DataFrame(A = collect(1:N_opts), B = energy_bin/opt_bin_size)
CSV.write("energy_" * preamble * ".csv", df_erg)

# double occupancy
df_dblocc = DataFrame(A = collect(1:N_opts), B = dblocc_bin/opt_bin_size)
CSV.write("dblocc_" * preamble * ".csv", df_dblocc)

# parameters
df_afm = DataFrame(A = collect(1:N_opts), B = deltaa/opt_bin_size)
CSV.write("deltaAFM_" * preamble * ".csv", df_afm)

df_cdw = DataFrame(A = collect(1:N_opts), B = deltac/opt_bin_size)
CSV.write("deltaCDW_" * preamble * ".csv", df_cdw)

df_mu = DataFrame(A = collect(1:N_opts), B = mu/opt_bin_size)
CSV.write("mu_" * preamble * ".csv", df_mu)

df_s = DataFrame(A = collect(1:N_opts), B = deltas/opt_bin_size)
CSV.write("deltaS_" * preamble * ".csv", df_mu)

# # BCS parameters
# df_s = DataFrame(A = collect(1:N_opts), B = deltas/opt_bin_size)
# CSV.write("deltaS_" * preamble * ".csv", df_s)
# df_mus = DataFrame(A = collect(1:N_opts), B = mus/opt_bin_size)
# CSV.write("mus_" * preamble * ".csv", df_mus)

# # Jastrow parameters
# df_vij = DataFrame(A = collect(1:N_opts), B = vij_1/opt_bin_size, C = vij_2/opt_bin_size)
# CSV.write("vij_" * preamble * ".csv", df_vij)