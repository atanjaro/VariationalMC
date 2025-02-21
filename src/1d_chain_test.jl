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

# make some plots for testing purposes
using Plots
using LaTeXStrings

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
Lx = 6
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
U = 8.0

# chemical potential (BCS)
μ_BCS = 0.0

# s-wave pairing (BCS)
Δs = 0.1

# antiferromagnetic order parameter
Δa = 0.1

# # Parameters to be optimized and initial value(s)
# parameters_to_optimize = ["Δs", "μ_BCS"]                              # s-wave (BCS) order parameter
# parameter_values = [[Δs],[μ_BCS]]                                 
# pht = true

# Parameters to be optimized and initial value(s)
parameters_to_optimize = ["Δa"]                                       # antiferromagnetic (Neél) order parameter
parameter_values = [[Δa]]                                            
pht = false

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
seed = 4249684800071112050 #6431290886380112751 #abs(rand(Int)) # 7778951059202733546

# Initialize random number generator
rng = Xoshiro(seed)
       
# Number of minimization/optimization updates
N_opts = 1000

# Optimization bin size
opt_bin_size = 100

# Number of simulation updates 
N_updates = 1000

# Number of simulation bins
N_bins = 100

# Simulation bin size
bin_size = div(N_updates, N_bins)

# Number of MC cycles until measurement
mc_meas_freq = 300

# number of steps until numerical stability is performed 
n_stab = 50

# Maximum allowed error in the equal-time Green's function
δW = 1e-3

# Maximum allowed error in the T vector
δT = 1e-3

# Stabilization factor for Stochastic Reconfiguration
η = 1e-4   # 1e-4   

# Optimization rate for Stochastic Reconfiguration
dt = 0.1 # 0.03      

# Debugging 
debug = false

# Verbose
verbose = false

# Chain unit cell
unit_cell = UnitCell(lattice_vecs = [[1.0]],
                            basis_vecs   = [[0.0]]);

# Build a chain
lattice = Lattice([Lx],[true]);

# Define nearest neighbor bonds
bond_x = Bond(orbitals = (1,1), displacement = [1]);

# # Define next nearest neighbor bonds
# bond_xp = Bond(orbitals = (1,1), displacement = [2]);

# Collect all bond definitions
bonds = [[bond_x]];     # ,[bond_xp]

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
(D, pconfig, κ,  ε, ε₀, M, U_int) = build_determinantal_state(H_mf);  

# Initialize variational parameter matrices
A = get_Ak_matrices(V, U_int, ε, model_geometry);           

# Initialize equal-time Green's function (W matrix)
W = get_equal_greens(M, D);                              

# Construct electron density-density Jastrow factor
jastrow = build_jastrow_factor("e-den-den", model_geometry, pconfig, pht, rng, false);

# # Initialize measurement container for VMC measurements
# measurement_container = initialize_measurement_container(model_geometry::ModelGeometry, determinantal_parameters::DeterminantalParameters, 
#                                                             jastrow::Jastrow, N_opts, opt_bin_size, N_bins, bin_size);

# Initialize measurement container for VMC measurements
measurement_container = initialize_measurement_container(model_geometry::ModelGeometry, determinantal_parameters::DeterminantalParameters, 
                                                            N_opts, opt_bin_size, N_bins, bin_size);

# Initialize the sub-directories to which the various measurements will be written
initialize_measurement_directories(simulation_info, measurement_container);


# Here are some vector 'bins' for storing data during this test
# These will be replaced by the data being written to file instead (with proper statistics reported)
energy_bin = Float64[]
dblocc_bin = Float64[]
param_bin = []

acceptance_rate = 0.0

# optimization updates

for bin in 1:N_opts
    for n in 1:opt_bin_size
        # # perform local fermion update for a certain number of equilibration steps
        # (local_acceptance_rate, W, D, pconfig, κ) = local_fermion_update!(W, D, model_geometry, jastrow, pconfig, 
        #                                                                     κ, rng, n_stab, mc_meas_freq)    

        # perform local fermion update for a certain number of equilibration steps
        (local_acceptance_rate, W, D, pconfig, κ) = local_fermion_update!(W, D, model_geometry, pconfig, 
                                                                            κ, rng, n_stab, mc_meas_freq)    

        acceptance_rate += local_acceptance_rate

        # # make basic measurements
        # make_measurements!(measurement_container,determinantal_parameters, jastrow, model_geometry, 
        #                     tight_binding_model, pconfig, κ, Np, W, A)

        # make basic measurements
        make_measurements!(measurement_container,determinantal_parameters, model_geometry, 
                            tight_binding_model, pconfig, κ, Np, W, A)
    end

    # # perform Stochastic Reconfiguration
    # sr_update!(measurement_container, determinantal_parameters, jastrow, η, dt, opt_bin_size)

    # perform Stochastic Reconfiguration
    sr_update!(measurement_container, determinantal_parameters, η, dt, opt_bin_size)

    # write all measurements to file
    write_measurements!(measurement_container, simulation_info, energy_bin, dblocc_bin, param_bin, debug)                                                                                                             
end


deltaa = [v[1] for v in param_bin]
# vij_1 = [v[2] for v in param_bin]
# vij_2 = [v[3] for v in param_bin]

deltas = [v[1] for v in param_bin]
mus = [v[2] for v in param_bin]
vij_1 = [v[2] for v in param_bin]
vij_2 = [v[3] for v in param_bin]


# # write the "bins" to file
# df = DataFrame(Value = energy_bin)
# # Write to CSV
# CSV.write("energy_bins.csv", df)

# df = DataFrame(Value = deltas)
# # Write to CSV
# CSV.write("vpar_deltas.csv", df)

# df = DataFrame(Value = mus)
# # Write to CSV
# CSV.write("vpar_mus.csv", df)

# df = DataFrame(Value = vij_1)
# # Write to CSV
# CSV.write("jpar_vij_1.csv", df)

# df = DataFrame(Value = vij_2)
# # Write to CSV
# CSV.write("jpar_vij_2.csv", df)

# df = DataFrame(Value = dblocc_bin)
# # Write to CSV
# CSV.write("dblocc.csv", df)

# plot energy per site as a function of optimization steps
scatter(1:1000, energy_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"E/N", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,1000), ylims=(-5,5)) #

# plot energy per site as a function of optimization steps
scatter(1:1000, dblocc_bin/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"D", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,1000), ylims=(0,0.5))

# plot determinantal parameter(s)
scatter(1:1000, deltaa/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        legend=false, xlabel="Optimization steps", ylabel=L"\Delta_a", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,1000)) #, ylims=(0,2)

scatter(1:1000, deltas/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
    legend=false, xlabel="Optimization steps", ylabel=L"\Delta_a", tickfontsize=14, guidefontsize=14, legendfontsize=14,
    xlims=(0,1000)) #

scatter(1:1000, mus/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
    legend=false, xlabel="Optimization steps", ylabel=L"\Delta_a", tickfontsize=14, guidefontsize=14, legendfontsize=14,
    xlims=(0,1000)) #

# plot Jastrow parameters
scatter(1:1000, vij_1/opt_bin_size, marker=:circle, color=:blue, markersize=5, markerstrokewidth=0,
        label=L"v_{ij}^1", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,1000)) #, ylims=(-1,2)
scatter!(1:1000, vij_2/opt_bin_size, marker=:square, color=:red, markersize=5, markerstrokewidth=0,
        label=L"v_{ij}^2", xlabel="Optimization steps", ylabel=L"v_{ij}", tickfontsize=14, guidefontsize=14, legendfontsize=14,
        xlims=(0,1000))#, ylims=(-1,2)


energy_data = energy_bin/opt_bin_size
energy_mean = sum(energy_bin[200:1000]/opt_bin_size)/800
subset_data = energy_bin[200:1000]/opt_bin_size

using Jackknife
using Statistics


function jackknife_error(data)
    N = length(data)
    mean_full = mean(data)

    # Compute jackknife estimates by leaving one element out at a time
    means_leave_one_out = [mean(vcat(data[1:i-1], data[i+1:end])) for i in 1:N]

    # Compute jackknife variance
    variance_jk = (N - 1) / N * sum((means_leave_one_out .- mean_full) .^ 2)

    return sqrt(variance_jk)  # Jackknife standard error
end

jackknife_err = jackknife_error(subset_data)




# simulation updates
for bin in 1:N_updates
    for n in 1:N_bins
        # perform local fermion update for a certain number of equilibration steps
        (pconfig, κ, jastrow_den, W, D) = local_fermion_update!(W, D, model_geometry, jastrow_den, pconfig, κ, rng, n, n_stab, mc_meas_freq)

         # make basic measurements
        make_measurements!(measurement_container,determinantal_parameters, jastrow_den, model_geometry, 
                            tight_binding_model, pconfig, κ, Np, W, A)
    end

    # write all measurements to file
    write_measurements!(measurement_container, simulation_info, energy_bin, dblocc_bin, param_bin, debug)                                                                                                             
end














