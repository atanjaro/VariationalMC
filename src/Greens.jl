"""

    build_determinantal_state( H_mf::Matrix{ComplexF64} ) 

Constructs a Slater determinant state matrix D in the the many-particle configuration basis, 
reduced matrix M, and initial particle energies ε₀ from a random initial configuration. 
Ensures that generated initial configuration is not singular. 

"""
function build_determinantal_state(H_mf::Matrix{ComplexF64})
    # diagonalize Hamiltonian
    ε, Uₑ = diagonalize(H_mf);

    # check for open shell configuration
    if is_openshell(ε, Ne)
        println("WARNING! Open shell detected")
        # exit(1)
    else
        println("Generating shell...")
    end

    # store energies
    ε₀ = ε[1:Ne];

    if debug
        print("Initial energies = ", ε₀)
    end

    # store M matrix
    M = Matrix{ComplexF64}(view(Uₑ, 1:size(Uₑ,1), 1:Ne));

    # build Slater determinant
    while true
        pconfig = generate_initial_fermion_configuration();

        κ = get_particle_positions(pconfig, model_geometry, Ne);

        config_indices = findall(x -> x == 1, pconfig);
        D = M[config_indices, :];

        # check that starting configuration is not singular
        if is_invertible(D) 

            if debug
                println("")
                println("Initial configuration: ", pconfig)
            end

            return D, pconfig, κ, ε, ε₀, M, Uₑ;
        end
    end    
end


"""

    build_determinantal_state( H_mf::Matrix{AbstractFloat}, init_pconfig::Vector{Float64} ) 

Constructs a Slater determinant state matrix D in the the many-particle configuration basis, 
reduced matrix M, and initial particle energies ε₀ from a known particle configuration.
Ensures that generated initial configuration is not singular. 

"""
function build_determinantal_state(H_mf::Matrix{ComplexF64}, init_pconfig::Vector{Float64})
    # diagonalize Hamiltonian
    ε, Uₑ = diagonalize(H_mf)

    # check for open shell configuration
    if is_openshell(ε, Ne)
        debug && println("WARNING! Open shell detected")
        exit(1)
    else
        debug && println("Generating shell...")
    end

    # store energies
    ε₀ = ε[1:Ne]

    # Store M matrix
    M = hcat(U[:,1:Ne])

    # build Slater determinant
    while true
        pconfig = init_pconfig
        config_indices = findall(x -> x == 1, pconfig)
        D = M[config_indices, :]

        # check that starting configuration is not singular
        if is_invertible(D) 

            return D, pconfig, ε, ε₀, M, Uₑ
        end
    end    
end


"""

    get_equal_greens( M::Matrix{ComplexF64}, D::Matrix{ComplexF64} )
    
Computes the equal-time Green's function by solving DᵀWᵀ = Mᵀ using full pivot LU decomposition.

"""
function get_equal_greens(M::Matrix{ComplexF64}, D::Matrix{ComplexF64})
    debug && println("Getting equal-time Green's function...")

    # perform LU decomposition on D'
    lu_fact = lu(D')

    # solve for Wt using forward and backward substitution
    Wt = lu_fact \ M'  

    # transpose the result back to the original shape
    W = Wt'

    # convert back to a regular matrix
    W = Matrix(W)

    debug && println("W = ", W)

    return W                
end


"""

    update_equal_greens!( local_acceptance::LocalAcceptance, W::Matrix{ComplexF64} ) 
    
Perform in-place update of the equal-time Green's function. 

"""
function update_equal_greens(local_acceptance::LocalAcceptance, W::Matrix{ComplexF64})
    # final site of the hopping particle
    l = local_acceptance.fsite

    # particle number
    β = local_acceptance.particle

    # get lth row of the Green's function
    rₗ = W[l, :]

    # subtract 1 from the βth element of 
    rₗ[β] -=1

    # get the βth column of the Green's function
    cᵦ = W[:,β]

    # perform rank-1 update
    W -= cᵦ * rₗ' / W[l, β]

    # # perform rank-1 update (bugged)
    # BLAS.ger!(-1.0 / W[l, β], cᵦ, rₗ, W)

    return W
end


"""

    recalc_equal_greens( W::Matrix{ComplexF64}, δW::Float64, D::Matrix{Float64}, pconfig::Vector{Int64} ) 
    
Checks floating point error accumulation in the equal-time Green's function and if ΔW < δW, then the 
recalculated Green's function Wᵣ replaces the updated Green's function Wᵤ as well as the Slater matrix D
for the current configuration.

"""
function recalc_equal_greens(W::Matrix{ComplexF64}, δW::Float64, D::Matrix{ComplexF64}, pconfig::Vector{Int64})

    if debug
        println("Checking Green's function...")
    end

    # recalculate D for current configuration
    if debug
        println("Recalculating Slater matrix...")
    end
    Dᵣ = M[findall(x -> x == 1, pconfig), :];
    
    # Recalculate Green's function from scratch
    Wᵣ = get_equal_greens(M, Dᵣ);
    
    # Difference in updated Green's function and recalculated Green's function
    difference = W .- Wᵣ;

    # Sum the absolute differences and the recalculated Green's function elements
    diff_sum = sum(abs.(difference));
    W_sum = sum(abs.(Wᵣ));

    # condition for recalculation
    ΔW = sqrt(diff_sum / W_sum);

    if ΔW > δW
        debug && println("WARNING! Green's function has been recalculated: ΔW = ", ΔW, " > δW = ", δW)
        return Wᵣ, Dᵣ;
    else
        debug && println("Green's function is stable: ΔW = ", ΔW, " < δW = ", δW)
        return W, D;
    end  
end