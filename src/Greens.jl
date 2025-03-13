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
end


"""

    build_determinantal_wavefunction(tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                                        Ne::Int64, nup::Int64, ndn::Int64, model_geometry::ModelGeometry, rng::Xoshiro)

Constructs a variational wavefunction based on parameters given by the tight binding model and determinantal parameters. 
Returns an instances of the DeterminantalWavefunction type.                                    

"""
function build_determinantal_wavefunction(tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                                        Ne::Int64, nup::Int64, ndn::Int64, model_geometry::ModelGeometry, rng::Xoshiro)
    # number of lattice sites
    N = model_geometry.lattice.N;

    # build auxiliary (mean-field) Hamiltonian and variational operators
    (H, V) = build_mean_field_hamiltonian(tight_binding_model, determinantal_parameters);

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
        debug && println("Greens::build_determinantal_wavefunction() : ")
        debug && println("configuration does not have ")
        debug && println("an overlap with the determinantal wavefunction => ")
        debug && println("generating a new configuration")

        # re-generate a random particle configuration
        pconfig = generate_initial_fermion_configuration(nup, ndn, model_geometry, rng);

        # re-initialize equal-time Green's function and Slater matrix
        overlap = initialize_equal_time_greens!(W, D, M, pconfig, Ne);
    end

    return DeterminantalWavefunction(W, D, M, U_int, A, ε, pconfig);
end


"""

    initialize_equal_time_greens( W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, 
                                        M::Matrix{ComplexF64}, pconfig::Vector{Int64}, N::Int64, Ne::Int64 )
    
Computes the equal-time Green's function.

"""
function initialize_equal_time_greens!(W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, 
                                        M::Matrix{ComplexF64}, pconfig::Vector{Int64}, Ne::Int64)
    # get indices from the particle configuration
    config_indices = [findfirst(==(i), pconfig) for i in 1:Ne];

    # get Slater matrix
    D .= M[config_indices, :];

    if abs(det(D)) < 1e-12 * size(D, 1) 
        debug && println("Greens::initialize_equal_time_greens() : state has no")
        debug && println("overlap with the determinantal wavefunction, ")
        debug && println("D = ")
        debug && display(D)
        debug && println("determinant of D = ", det(D))

        return false;
    else        
        # LU decomposition of D'
        lu_decomp = lu(D')
        
        # calculate the equal-time Green's function
        W .= transpose(lu_decomp \ transpose(M))

        return true;
    end            
end

# function initialize_equal_time_greens!(W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, 
#                                         M::Matrix{ComplexF64}, pconfig::Vector{Int64}, N::Int64, Ne::Int64)
#     # get indices from the particle configuration
#     config_indices = findall(x -> 1 ≤ x ≤ Ne, pconfig);

#     # get Slater matrix
#     D .= M[config_indices, :];

#     if abs(det(D)) < 1e-12 * size(D, 1) 
#         debug && println("Greens::initialize_equal_time_greens() : state has no")
#         debug && println("overlap with the determinantal wavefunction, ")
#         debug && println("D = ")
#         debug && display(D)

#         return false;
#     else        
#         # calculate the equal-time Green's function
#         for i in 1:2*N  # nrows
#             for j in 1:Ne # ncols
#                 sum = ComplexF64(0.0, 0.0);
#                 for k in 1:Ne # ncols
#                     sum += M[i, k] * D[k, j];
#                 end
#                 W[i, j] = sum;
#             end
#         end

#         return true;
#     end            
# end


"""
    DEBUG
"""
function check_order_magnitude_jump(W::AbstractMatrix{ComplexF64})
    abs_W = abs.(W)  # Compute magnitudes of complex elements
    max_val = maximum(abs_W)  # Get the maximum magnitude

    # Ensure elements do not exceed 10^1
    if max_val > 10
        error("Order of magnitude jump detected in W! Max value found = $max_val, which exceeds 10.")
    end
end


"""

    rank1_update!( markov_move::MarkovMove, detwf::DeterminantalWavefunction ) 
    
Performs in-place rank-1 update of the equal-time Green's function. 

"""
function rank1_update!(markov_move::MarkovMove, detwf::DeterminantalWavefunction)
    debug && println("Greens::rank1_update!() : performing rank-1 update of W!")

    # particle 
    β = markov_move.particle

    # final site of the hopping particle
    l = markov_move.l

    # get lth row of the Green's function
    rₗ  = detwf.W[l, :]  # rₗ = view(detwf.W, l, :)

    # subtract 1 from the βth element
    rₗ[β] -= 1.0 

    # get the βth column of the Green's function
    cᵦ = detwf.W[:, β]  

    # Perform rank-1 update
    detwf.W -= (cᵦ / detwf.W[l, β]) * rₗ' 

    check_order_magnitude_jump(detwf.W)

    return nothing
end


"""

    recalculate_equal_time_greens( W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, 
                                        M::Matrix{ComplexF64}, pconfig::Vector{Int64}, N::Int64, Ne::Int64 )
    
Recomputes the equal-time Green's function.

"""
function recalculate_equal_time_greens!(Wᵣ::Matrix{ComplexF64}, Dᵣ::Matrix{ComplexF64}, 
                                        M::Matrix{ComplexF64}, pconfig::Vector{Int64}, Ne::Int64)
    # get indices from the particle configuration
    config_indices = [findfirst(==(i), pconfig) for i in 1:Ne]; 

    # get Slater matrix
    Dᵣ .= M[config_indices, :];

    # LU decomposition of D'
    lu_decomp = lu(Dᵣ')

    # calculate the equal-time Green's function
    Wᵣ .= transpose(lu_decomp \ transpose(M))

    return true;      
end

# function recalculate_equal_time_greens!(W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, 
#                                         M::Matrix{ComplexF64}, pconfig::Vector{Int64}, N::Int64, Ne::Int64)
#     # get indices from the particle configuration
#     config_indices = findall(x -> 1 ≤ x ≤ Ne, pconfig);

#     # get Slater matrix
#     D .= M[config_indices, :];

#     # calculate the equal-time Green's function
#     for i in 1:2*N  # nrows
#         for j in 1:Ne # ncols
#             sum = ComplexF64(0.0, 0.0);
#             for k in 1:Ne # ncols
#                 sum += M[i, k] * D[k, j];
#             end
#             W[i, j] = sum;
#         end
#     end

#     return true;      
# end



"""

    check_deviation!( detwf::DeterminantalWavefunction, δW::Float64, Ne::Int64, model_geometry::ModelGeometry )::Nothing
    
Checks floating point error accumulation in the equal-time Green's function and if ΔW < δW, then the 
recalculated Green's function Wᵣ replaces the updated Green's function Wᵤ as well as the Slater matrix D
for the current configuration.

"""
function check_deviation!(detwf::DeterminantalWavefunction, δW::Float64, Ne::Int64, model_geometry::ModelGeometry)::Nothing
    # number of lattice sites
    N = model_geometry.lattice.N

    # re-initialize W matrix
    Wᵣ = zeros(ComplexF64, 2*N, Ne);

    # re-initialize Slater matrix
    Dᵣ = zeros(ComplexF64, Ne, Ne);

    # re-calculate the Green's function from scratch
    recalculate_equal_time_greens!(Wᵣ, Dᵣ, detwf.M, detwf.pconfig, Ne);
    
    # Difference in updated Green's function and recalculated Green's function
    difference = detwf.W .- Wᵣ;

    # Sum the absolute differences and the recalculated Green's function elements
    diff_sum = sum(abs.(difference));
    W_sum = sum(abs.(Wᵣ));

    # condition for recalculation
    ΔW = sqrt(diff_sum / W_sum);

    debug && println("Greens::check_deviation!() : deviation goal for matrix")

    if ΔW > δW
        debug && println("W not met!")
        debug && println("Greens::check_deviation!() : updated W = ")
        debug && display(detwf.W);
        debug && println("Greens::check_deviation!() : exact W = ")
        debug && display(Wᵣ);

        detwf.W = Wᵣ;

        check_order_magnitude_jump(detwf.W)

        return nothing;
    else
        debug && println("W met! Green's function is stable")

        return nothing;
    end  
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