using LatticeUtilities
using Random
using LinearAlgebra
using Test
using DelimitedFiles
using BenchmarkTools

# files to include
# include("Parameters.jl")
include("Hamiltonian.jl")
include("ParticleConfiguration.jl")

# top level function to run simulation
function run_simulation()

    # set particle number
    nup = 2
    ndn = 2

    # system size
    Lx = 2
    Ly = 2

    # nearest-neighbor hopping amplitude
    t = 1.0

    # next nearest-neighbor hopping amplitude
    tp = 0.0

    # hubbard U
    U = 0.5

    # chemical potential
    μ = 0.0
    opt_μ = false               # whether parameter is to be optimized

    # s-wave order parameter
    Δs = 0.3                    
    opt_s  = true               # whether parameter is to be optimized    

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # initialize random seed
    seed = abs(rand(Int))

    # initialize random number generator
    rng = Xoshiro(seed)
 
    # whether model is particle-hole transformed
    pht = true
 
    # whether to output during runtime (not reccommended)
    verbose = true

    ##################
    ## DEFINE MODEL ##
    ##################

    # square unit cell
    unit_cell = UnitCell([[1.0,0.0], [0.0,1.0]],           # lattice vectors
                            [[0.0,0.0]])                   # basis vectors 

    # build square lattice
    lattice = Lattice([Lx,Ly],[true,true])

    # define nearest neighbor bonds
    bond_x = Bond((1,1), [1,0])
    bond_y = Bond((1,1), [0,1])

    # define next nearest neighbor bonds
    bond_xy = Bond((1,1), [1,1])
    bond_yx = Bond((1,1), [1,-1])

    # store 2D bonds
    bonds = [[bond_x, bond_y], [bond_xy, bond_yx]]

    ####################################################
    ## INITIALIZE MODEL PARAMETERS FOR FINITE LATTICE ##
    ####################################################

    # define model geometry
    model_geometry = ModelGeometry(unit_cell,lattice)

    # define particle density
    (density, Np, Ne) = get_particle_density(nup, ndn)

    # define non-interacting tight binding model
    tight_binding_model = TightBindingModel([t,tp],μ)

    # define variational parameters
    variational_parameters = VariationalParameters(["Δs"], [Δs], [opt_s])

    ##################################################
    ## SET-UP & INITIALIZE VMC SIMULATION FRAMEWORK ##
    ##################################################
    
    # construct mean-field Hamiltonian
    H_mf = build_mean_field_hamiltonian()
    # writedlm("H_mf.csv", H_mf)  # TODO: move to this to function, make write=true with flag

    # generate random intial particle configuration
    pconfig = generate_initial_configuration()   # TODO: use this to occupy initial wf states

    # construct Slater determinant state
    (D, ε₀, M, U) = build_slater_determinant(H_mf)  # TODO: verify that initial configuration is not singular
                                                 #       i.e. ⟨x|Φ⟩ ≂̸ 0

    # construct Jastrow factors

    return nothing
end

# run simulation
# run_simulation()