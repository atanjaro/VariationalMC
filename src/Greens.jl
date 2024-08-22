"""
    build_determinantal_state( H_mf::Matrix{AbstractFloat} ) 

Returns initial energies ε₀, matrix M, and Slater matrix D in the many-particle configuration 
basis.

"""
function build_determinantal_state(H_mf)
    # Diagonalize Hamiltonian
    ε, U = diagonalize(H_mf)

    # Check for open shell configuration
    if is_openshell(ε, Np)
        verbose && println("WARNING! Open shell detected")
    else
        verbose && println("Generating shell...")
    end

    # Store energies
    ε₀ = ε[1:Np]

    # Store M matrix
    M = hcat(U[:,1:Np])

    # Build Slater determinant
    # in case that there is no finite overlap...
    # max_configs = (n, k) -> div(prod(n:-1:(n-k+1)), factorial(k))
    # max_attempts = max_configs(model_geometry.lattice.N, nup)
    # attempt = 0
    while true
        pconfig = generate_initial_fermion_configuration()
        config_indices = findall(x -> x == 1, pconfig)
        D = M[config_indices, :]

        # Check that starting configuration is not singular
        if is_invertible(D) 
            # Write matrices to file if needed
            if write
                writedlm("H_mf.csv", H_mf)
                writedlm("D.csv", D)
                writedlm("M.csv", M)
                writedlm("U.csv", U)
            end

            return D, pconfig, ε, ε₀, M, U
        end
        # # Increment attempt counter
        # attempt += 1
    end    
end


"""
    get_equal_greens( M::Matrix{Float64}, D::Matrix{Float64} )::Matrix{Float64}
    
Returns the equal-time Green's function by solving DᵀWᵀ = Mᵀ using full pivot LU decomposition.

"""
function get_equal_greens(M::Matrix{Float64}, D::Matrix{Float64})::Matrix{Float64}
    verbose && println("Getting equal-time Green's function...")

    # perform the linear solve directly
    Wt = D' \ M'     

    # transpose the result back to the original shape
    W = Wt'

    debug && println("W = $W")
 
    return W                
end          


"""
    update_equal_greens!( local_acceptance::LocalAcceptance, W::Matrix{Float64} ) 
    
Perform in-place update of the equal-time Green's function. 

"""
function update_equal_greens!(local_acceptance::LocalAcceptance, W::Matrix{Float64})
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
    # Np = size(Wᵤ, 2)  # Assuming Np is the number of columns in Wᵤ

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
        verbose && println("WARNING! Green's function has been recalculated: ΔW = ", ΔW, " > δW = ", δW)
        return Wᵣ, Dᵣ
    else
        verbose && println("Green's function is stable: ΔW = ", ΔW, " < δW = ", δW)
        return Wᵤ, D
    end  
end


"""

    fsgn( W::Matrix{Float64} )

Given the equal time Green's function, returns the value of the Fermion sign.

"""
function fsgn(W::Matrix{Float64})
    M_up = W[1:N, :]
    M_dn = W[N+1:2*N, :]

    return sign(det(M_up) * det(M_dn))
end



