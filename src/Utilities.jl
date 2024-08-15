"""

    diagonalize( H::Matrix{AbstractFloat} ) 

Returns all eigenenergies and all eigenstates of the mean-field Hamiltonian, 
the latter being stored in the columns of a matrix. Convention: H(diag) = U⁺HU.

"""
function diagonalize(H)
    # check if Hamiltonian is Hermitian
    @assert ishermitian(H) == true
    F = eigen(H_mf)    
    return F.values, F.vectors  
end


"""

    is_invertible( D::Matrix{AbstractFloat} ) 
    
Runs check on the invertibility of a matrix by checking: the value of the determinant, the condition number,
for singular values, and performs LU decomposition with partial pivoting. 

"""
function is_invertible(D)
    # tolerance
    ϵ = eps(Float64)

    # check the determinant is non-zero
    det_D = det(D)
    norm_D = norm(D)
    det_check = abs(det_D) > ϵ * norm_D

    if !det_check
        if verbose
            println("Singular configuration detected! Generating new configuration...")
            println("Non-zero determinant detected:")
            println("|$det_D| < $ϵ × $norm_D")
        end
        return false
    end

    # check the condition number
    cond_D = cond(D)
    inv_eps = 1/ϵ
    cond_check = cond_D < inv_eps

    if !cond_check
        if verbose
            println("Singular configuration detected! Generating new configuration...")
            println("Condition number test failed:")
            println("$cond_D > $inv_eps")
        end
        return false
    end

    # check for singular values
    singular_values = svdvals(D)
    smallest_singular_value = minimum(singular_values)
    sing_check = smallest_singular_value > ϵ * norm_D

    if !sing_check
        if verbose
            println("Singular configuration detected! Generating new configuration...")
            println("Singular values detected:")
            println("$smallest_singular_value < $ϵ × norm_D")
        end
        return false
    end

    # pivot check with LU decomposition
    LU = lu(D)

    # Access the combined LU matrix
    LU_matrix = LU.factors

    # Extract the diagonal elements of the U matrix (which are on the diagonal of LU_matrix)
    U_diag = diag(LU_matrix)

    # Find the minimum absolute value of the diagonal of U
    min_pivot = minimum(abs.(U_diag))
    pivot_check = min_pivot > ϵ * norm_D

    if !pivot_check
        if verbose
            println("Singular configuration detected! Generating new configuration...")
            println("LU partial pivoting check failed:")
            println("$min_pivot < $ϵ × $norm_D")
        end
        return false
    end

    return true
end


"""

    complex_zeros{T<:AbstractFloat}(rows::Int, cols::Int)

Initialzies a zero matrix with complex type

"""
function complex_zeros( rows::Int, cols::Int)
    return zeros(Complex{AbstractFloat}, rows, cols)
end


"""

    is_openshell( Np::Int, ε::Vector{AbstractFloat} ) 

Checks whether configuration is open shell.

"""
function is_openshell(ε, Np)
    return ε[Np + 1] - ε[Np] < 0.0001
end


"""

    get_num_vpars(determinantal_parameters::DeterminantalParameters, 
    density_jastrow::Jastrow, spin_jastrow::Jastrow, eph_jastrow::Jastrow) 

Returns the total number of variational parameters.

"""
function get_num_vpars(determinantal_parameters, jastrow)
    num_detpars = determinantal_parameters.num_detpars

    num_jpars = jastrow.num_jpars

    num_vpars = num_detpars + num_jpars

    return num_vpars, num_detpars, num_jpars

end


"""
    all_vpars(determinantal_parameters, jastrow)

Generates an array combining all values of determinantal and Jastrow parameters.

"""
function all_vpars(determinantal_parameters, jastrow)
    jpar_map = jastrow.jpar_map

    num_jpars = jastrow.num_jpars

    jpars = [value[2] for (i, value) in enumerate(values(jpar_map)) if i < num_jpars]
    detpars = determinantal_parameters.vals
    
    # concatenate variational parameters
    vpar_array = vcat(detpars, jpars)
    
    return vpar_array
end


"""

    get_parameter_indices()

Get indices of variational parameters from its respective matrix.

"""
function get_parameter_indices(par_matrix)
    nonzero_indices = findall(x -> x != 0, par_matrix)

    parameter_indices = sort(nonzero_indices, by=x->(x[1], x[2]))

    return parameter_indices
end 

"""

    x( i::Int, model_geometry::ModelGeometry )

Convenience function for obtaining the x-coordinate of a lattice site given a 
lattice spindex.

"""
function x(i::Int, model_geometry::ModelGeometry)
    L = model_geometry.lattice.L[1]
    return i % L
end

"""

    y( i::Int, model_geometry::ModelGeometry )

Convenience function for obtaining the y-coordinate of a lattice site given a 
lattice spindex.

"""
function y(i::Int, model_geometry::ModelGeometry)
    L = model_geometry.lattice.L[1]
    return div(i, L)
end


"""

    d( p1::Int, p2::Int, model_geometry::ModelGeometry )

Given lattice indices i and j, returns the distances between those 2 points, accounting 
for the latticed edges with different boundary conditions.

"""
function d(p₁::Int, p₂::Int, model_geometry::ModelGeometry)
    L = model_geometry.lattice.L[1]
    dist = p₂ - p₁

    if dist >div(L, 2)
        dist -= L
    end
    if dist < -div(L,2)
        dist+= L
    end

    return dist
end


"""

    reduce_index( i::Int, j::Int, model_geometry::ModelGeometry )

Reduces the indices of 2 lattice sites (i,j) to irreducible indices (0,k), where k is an integer.

"""
function reduce_index(i, j, model_geometry)
    L = model_geometry.lattice.L[1]

    dx = abs(d(x(i, model_geometry), x(j, model_geometry), model_geometry))
    dy = abs(d(y(i, model_geometry), y(j, model_geometry), model_geometry))

    if dy > dx
        dx, dy = dy, dx
    end

    return dx + L * dy
end


"""

    max_dist( N::Int, L::Int )

Obtains the maximum irreducible index given the total number of sites N and extent of the lattice L.

"""
function max_dist(N, L)
    if L % 2 == 0
        return Int(N / 2 + L / 2)
    else
        return Int(N / 2)
    end
end



