"""
    build_determinantal_state( H_mf::Matrix{AbstractFloat} ) 

Returns initial energies ε₀, matrix M, and Slater matrix D in the many-particle configuration 
basis.

"""
function build_determinantal_state(H_mf)
    # Diagonalize Hamiltonian
    ε, Uₑ = diagonalize(H_mf);

    # Check for open shell configuration
    if is_openshell(ε, Ne)
        debug && println("WARNING! Open shell detected")
        # exit(1)
    else
        debug && println("Generating shell...")
    end

    # Store energies
    ε₀ = ε[1:Ne];

    if debug
        print("Initial energies = ", ε₀)
    end

    # Store M matrix
    M = hcat(Uₑ[:,1:Ne]);

    # Build Slater determinant
    while true
        pconfig = generate_initial_fermion_configuration();

        κ = get_particle_positions(pconfig, model_geometry, Ne);

        config_indices = findall(x -> x == 1, pconfig);
        D = M[config_indices, :];

        # Check that starting configuration is not singular
        if is_invertible(D) 

            if debug
                println("Initial configuration: ", pconfig)
            end

            return D, pconfig, κ, ε, ε₀, M, Uₑ;
        end
    end    
end


"""
    build_determinantal_state( H_mf::Matrix{AbstractFloat}, init_pconfig ) 

Given a initial particle configuration, returns initial energies ε₀, matrix M, and Slater matrix D in 
the many-particle configuration basis.

"""
function build_determinantal_state(H_mf, init_pconfig)
    # Diagonalize Hamiltonian
    ε, Uₑ = diagonalize(H_mf)

    # Check for open shell configuration
    if is_openshell(ε, Ne)
        debug && println("WARNING! Open shell detected")
        exit(1)
    else
        debug && println("Generating shell...")
    end

    # Store energies
    ε₀ = ε[1:Ne]

    # Store M matrix
    M = hcat(U[:,1:Ne])

    # Build Slater determinant
    while true
        pconfig = init_pconfig
        config_indices = findall(x -> x == 1, pconfig)
        D = M[config_indices, :]

        # Check that starting configuration is not singular
        if is_invertible(D) 

            return D, pconfig, ε, ε₀, M, Uₑ
        end
    end    
end


"""
    get_equal_greens( M::Matrix{Float64}, D::Matrix{Float64} )::Matrix{Float64}
    
Returns the equal-time Green's function by solving DᵀWᵀ = Mᵀ using full pivot LU decomposition.

"""
function get_equal_greens(M::Matrix{ComplexF64}, D::Matrix{ComplexF64})
    debug && println("Getting equal-time Green's function...")

    # perform the linear solve directly
    Wt = D' \ M';     

    # transpose the result back to the original shape
    W = Wt';

    debug && print("W = $W")
 
    return W;                
end          


"""
    update_equal_greens!( local_acceptance::LocalAcceptance, W ) 
    
Perform in-place update of the equal-time Green's function. 

"""
function update_equal_greens!(local_acceptance::LocalAcceptance, W)
    # convert W back into a regular matrix
    W = copy(W)

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

    # perform rank 1 update of the Green's function
    BLAS.ger!(-1.0 / W[l,β], cᵦ, rₗ, W)

    return nothing
end


"""
    recalc_equal_greens( Wᵤ::Matrix{Float64}, δW::Float64, D::Matrix{Float64}, pconfig::Vector{Int64} ) 
    
Checks floating point error accumulation in the equal-time Green's function and if ΔW < δW, then the 
recalculated Green's function Wᵣ replaces the updated Green's function Wᵤ as well as the D matrix
for the current configuration.

"""
function recalc_equal_greens(Wᵤ::Matrix{Float64}, δW::Float64, D::Matrix{Float64}, pconfig::Vector{Int64})
    # L = model_geometry.lattice.N
    # Ne = size(Wᵤ, 2)  # Assuming Ne is the number of columns in Wᵤ

    # reconstruct the Hamiltonian to reconstruct D

    # recalculate D for current configuration
    Dᵣ = M[findall(x -> x == 1, pconfig), :]
    
    # Recalculate Green's function from scratch
    Wᵣ = get_equal_greens(M, Dᵣ)
    
    # Difference in updated Green's function and recalculated Green's function
    diff = Wᵤ .- Wᵣ

    # Sum the absolute differences and the recalculated Green's function elements
    diff_sum = sum(abs.(diff))
    W_sum = sum(abs.(Wᵣ))

    ΔW = sqrt(diff_sum / W_sum)

    if ΔW > δW
        debug && println("WARNING! Green's function has been recalculated: ΔW = ", ΔW, " > δW = ", δW)
        return Wᵣ, Dᵣ
    else
        debug && println("Green's function is stable: ΔW = ", ΔW, " < δW = ", δW)
        return Wᵤ, D
    end  
end
