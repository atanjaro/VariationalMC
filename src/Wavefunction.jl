"""

    DeterminantalWavefunction( W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, M::Matrix{ComplexF64}
                                U_int::Matrix{ComplexF64}, A::Vector{Any}, ε::Vector{Float64}, pconfig::Vector{Int64} )

A type defining quantities related to a determinantal wavefunction.

"""
mutable struct DeterminantalWavefunction
    # equal-time Green's function
    W::Matrix{ComplexF64};

    # Slater matrix
    D::Matrix{ComplexF64};
    
    # M matrix
    M::Matrix{ComplexF64};
    
    # U matrix (that diagonalizes H)
    U_int::Matrix{ComplexF64};
    
    # variational parameter matrices
    A::Vector{Any};
    
    # initial energies
    ε::Vector{Float64};

    # particle configuration
    pconfig::Vector{Int64};

    # number of W matrix quick updates
    nq_updates_W::Int64

    # number of T vector quick updates
    nq_updates_T::Int64
end


"""

    build_determinantal_wavefunction(tight_binding_model::TightBindingModel, 
                                    determinantal_parameters::DeterminantalParameters, 
                                    Ne::Int64, nup::Int64, ndn::Int64,
                                    model_geometry::ModelGeometry, rng::Xoshiro)::DeterminantalWavefunction

Constructs a variational wavefunction based on parameters given by the tight binding model and determinantal parameters. 
Returns an instances of the DeterminantalWavefunction type.                                    

"""
function build_determinantal_wavefunction(tight_binding_model::TightBindingModel, 
                                        determinantal_parameters::DeterminantalParameters, 
                                        optimize::NamedTuple, Ne::Int64, nup::Int64, ndn::Int64, 
                                        model_geometry::ModelGeometry, rng::Xoshiro)::DeterminantalWavefunction
    # number of lattice sites
    N = model_geometry.lattice.N;

    # build auxiliary (mean-field) Hamiltonian and variational operators
    (H, V) = build_auxiliary_hamiltonian(tight_binding_model, determinantal_parameters, optimize, model_geometry, pht);

    # diagonalize Hamiltonian
    (ε, U_int) = diagonalize(H);

    if is_openshell(ε, Ne)
        debug && println("   WARNING: Open shell detected!")
    else
        debug && println("   Forming shell...")
    end

    # initialize variational parameter matrices
    A = get_variational_matrices(V, U_int, ε, model_geometry);

    # get M matrix
    M = Matrix{ComplexF64}(view(U_int, 1:size(U_int,1), 1:Ne));

    # initialize Slater matrix
    D = zeros(ComplexF64, Ne, Ne);

    # initialize W matrix
    W = zeros(ComplexF64, 2*N, Ne);

    # generate a random particle configuration
    pconfig = generate_initial_fermion_configuration(nup, ndn, model_geometry, rng);

    # initialize equal-time Green's function and Slater matrix
    overlap = initialize_equal_time_greens!(W, D, M, pconfig, Ne); 

    while overlap == false
        debug && println("Wavefunction::build_determinantal_wavefunction() : ")
        debug && println("configuration does not have ")
        debug && println("an overlap with the determinantal wavefunction => ")
        debug && println("generating a new configuration")

        # re-generate a random particle configuration
        pconfig = generate_initial_fermion_configuration(nup, ndn, model_geometry, rng);

        # initialize Slater matrix
        D = zeros(ComplexF64, Ne, Ne);

        # re-initialize equal-time Green's function and Slater matrix
        overlap = initialize_equal_time_greens!(W, D, M, pconfig, Ne);
    end

    # intialize quick updating tracker
    nq_updates_W = 0 
    nq_updates_T = 0

    return DeterminantalWavefunction(W, D, M, U_int, A, ε, pconfig, nq_updates_W, nq_updates_T);
end


##################################################### DEPRECATED FUNCTIONS #####################################################
# """

#     build_determinantal_state( H_mf::Matrix{ComplexF64} ) 

# Constructs a Slater determinant state matrix D in the the many-particle configuration basis, 
# reduced matrix M, and initial particle energies ε₀ from a random initial configuration. 
# Ensures that generated initial configuration is not singular. 

# """
# function build_determinantal_state(H_mf::Matrix{ComplexF64}, Ne, nup, ndn, model_geometry, rng)
#     # diagonalize Hamiltonian
#     ε, U_int = diagonalize(H_mf);

#     # check for open shell configuration
#     if is_openshell(ε, Ne)
#         debug && println("WARNING! Open shell detected")
#         # exit(1)
#     else
#         debug && println("Generating shell...")
#     end

#     # store energies
#     ε₀ = ε[1:Ne];

#     if debug
#         print("Initial energies = ", ε₀)
#     end

#     # store M matrix
#     M = Matrix{ComplexF64}(view(U_int, 1:size(U_int,1), 1:Ne));

#     # build Slater determinant
#     build_time1 = time();
#     while true
#         # generate random starting configuration
#         pconfig = generate_initial_fermion_configuration(nup, ndn, model_geometry, rng);

#         # store Slater matrix
#         config_indices = findall(x -> 1 ≤ x ≤ Ne, pconfig);
#         D = M[config_indices, :];

#         # check that starting configuration is not singular
#         if check_overlap(D)
#             build_time2 = time();
#             # # store particle positions
#             # κ = get_particle_positions(pconfig, model_geometry, Ne);

#             # calculate equal-time Green's function
#             W = get_equal_greens(M, D);

#             if debug
#                 println("")
#                 println("Initial configuration: ", pconfig)
#             end

#             println("Time until valid config: ", build_time2 - build_time1)

#             return W, D, pconfig, ε, ε₀, M, U_int;
#         end
#     end    
# end

# """

#     build_determinantal_state( H_mf::Matrix{AbstractFloat}, init_pconfig::Vector{Float64} ) 

# Constructs a Slater determinant state matrix D in the the many-particle configuration basis, 
# reduced matrix M, and initial particle energies ε₀ from a known particle configuration.
# Ensures that generated initial configuration is not singular. 

# """
# function build_determinantal_state(H_mf::Matrix{ComplexF64}, init_pconfig::Vector{Float64})
#     # diagonalize Hamiltonian
#     ε, U_int = diagonalize(H_mf)

#     # check for open shell configuration
#     if is_openshell(ε, Ne)
#         debug && println("WARNING! Open shell detected")
#         exit(1)
#     else
#         debug && println("Generating shell...")
#     end

#     # store energies
#     ε₀ = ε[1:Ne]

#     if debug
#         print("Initial energies = ", ε₀)
#     end

#     # Store M matrix
#     M = hcat(U[:,1:Ne])

#     # build Slater determinant
#     while true
#         pconfig = init_pconfig
#         config_indices = findall(x -> x == 1, pconfig)
#         D = M[config_indices, :]

#         # check that starting configuration is not singular
#         if is_invertible(D) 
#             if debug
#                 println("")
#                 println("Initial configuration: ", pconfig)
#             end

#             return D, pconfig, ε, ε₀, M, U_int
#         end
#     end    
# end