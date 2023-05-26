using LatticeUtilities
using Random
using LinearAlgebra
using Test
using DelimitedFiles
using BenchmarkTools

# files to include
include("Parameters.jl")
include("Hamiltonian.jl")
include("ParticleConfiguration.jl")
include("Jastrow.jl")


# define model geometry
model_geometry = ModelGeometry(unit_cell,lattice)

# define particle density
Np, Ne, nup, ndn = get_particle_numbers(n̄)
# (density, Np, Ne) = get_particle_density(nup, ndn)

# define non-interacting tight binding model
tight_binding_model = TightBindingModel([t,tp],μ)

# define variational parameters
variational_parameters = VariationalParameters(["Δs"], [Δs], [opt_s])
    
# construct mean-field Hamiltonian
H_mf = build_mean_field_hamiltonian()
# writedlm("H_mf.csv", H_mf)  # TODO: move to this to function, make write=true with flag

# initialize Slater determinant state and initial particle configuration
(D, pconfig, ε₀, M, U) = build_slater_determinant()  

# construct Jastrow factors
(Tvec, jpar_matrix, num_jpars) = get_jastrow_factor()






