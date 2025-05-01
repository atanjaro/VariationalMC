"""

    diagonalize( H::Matrix{ComplexF64} )::Tuple{Vector{Float64}, Matrix{ComplexF64}}

Returns all eigenenergies and all eigenstates of the mean-field Hamiltonian, 
the latter being stored in the columns of a matrix. Convention: H(diag) = U⁺HU.

"""
function diagonalize(H::AbstractMatrix{<:Complex})
    # check if Hamiltonian is Hermitian
    @assert ishermitian(H) == true
    F = eigen(H)    
    return F.values, F.vectors  
end


"""

    is_openshell( Np::Int, ε::Vector{AbstractFloat} ) 

Checks whether a energy configuration is open shell.

"""
function is_openshell(ε, Np)
    return ε[Np + 1] - ε[Np] < 0.0001
end


"""

    readin_parameters()

"""
function readin_parameters(filename::String)
    lines = filter(x -> !isempty(x), readlines(filename))  # skip empty lines
    parameters = [parse.(Float64, split(line)) for line in lines]

    return Dict(
        :chemical_potential => parameters[1][1],
        :pairing        => parameters[2],
        :afm                => parameters[3][1],
        :sds                => parameters[4],
        :cdw                => parameters[5][1],
        :sdc                => parameters[6],
        :density_jastrow    => parameters[7],
        :spin_jastrow       => parameters[8]
    )
end


"""

    get_num_vpars( determinantal_parameters::DeterminantalParameters, 
                    jastrow::Jastrow )::Tuple{Int64, Int64, Int64}

Returns the total number of variational parameters.

"""
function get_num_vpars(determinantal_parameters::DeterminantalParameters, 
                        jastrow::Jastrow)::Tuple{Int64, Int64, Int64}
    # number of determinantal parameters
    num_detpars = determinantal_parameters.num_detpars;

    # number of Jastrow parameters
    num_jpars = jastrow.num_jpars;

    num_vpars = num_detpars + num_jpars;

    return num_vpars, num_detpars, num_jpars;
end


"""
    all_vpars( determinantal_parameters::DeterminantalParameters, 
                jastrow::Jastrow )::Vector{AbstractFloat}

Concatenates all values of determinantal and Jastrow parameters into a single vector.

"""
function all_vpars(determinantal_parameters::DeterminantalParameters, 
                    jastrow::Jastrow)::Vector{AbstractFloat}
    # get map of Jastrow parameters
    jpar_map = jastrow.jpar_map;

    # total number of Jastrow parameters
    num_jpars = jastrow.num_jpars;

    # Extract the Jastrow parameters
    jpars = [value[2] for (i, value) in enumerate(values(jpar_map)) if i < num_jpars];
    
    # Collect all values from the inner vectors of detpars into a single vector
    detpars_all = vcat(determinantal_parameters.vals...);
    
    # Concatenate all detpars with jpars into a single vector
    vpar_array = vcat(detpars_all, jpars);
    
    return vpar_array;
end


# """

#     convert_par_name( parameters_to_optimize::Vector{String} )

# Converts vector of parameter names to string for appending to the datafolder prefix.

# """
# function convert_par_name(optimize::Vector{String})
#     # Map of special characters to their ASCII replacements
#     replacements = Dict('Δ' => "Delta", 'μ' => "mu")
    
#     # Helper function to sanitize each string
#     function sanitize_string(s::String)
#         # Replace special characters using the replacements dictionary
#         for (char, replacement) in replacements
#             s = replace(s, char => replacement)
#         end
#         # Remove any remaining non-alphanumeric characters
#         s = replace(s, r"[^a-zA-Z0-9]" => "")
#         return s
#     end

#     # Apply sanitization to each parameter and join with underscores
#     cleaned_parameters = join(sanitize_string.(optimize), "_")
#     return cleaned_parameters
# end


"""

    reset_measurements!( measurements::Dict{String, Any} )

Resets value of a dictionary (measurement container) to zero.

"""
function reset_measurements!(measurements::Dict{String, Any})
    for key in keys(measurements)
        value = measurements[key]
        if key == "pconfig"
            # Skip resetting pconfig
            continue
        elseif isa(value, Tuple)
            # Reset tuples: preserve the structure and types
            measurements[key] = map(zero, value)
        elseif isa(value, AbstractVector)
            if eltype(value) <: AbstractVector
                # Nested vectors (e.g., ΔkΔkp or Δk)
                for v in value
                    empty!(v)
                end
            else
                # Single-level vectors (e.g., not applicable anymore here)
                empty!(value)
            end
        elseif isa(value, Number)
            # Reset numbers
            measurements[key] = zero(value)
        else
            # Handle other types if needed (fallback: do nothing)
            @warn "Unhandled type for key $key: $(typeof(value))"
        end
    end
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

    reduce_index_2d( i::Int, j::Int, model_geometry::ModelGeometry )

Reduces the indices of 2 lattice sites (i,j) to irreducible indices (0,k), where k is an integer.

"""
function reduce_index_2d(i::Int, j::Int, model_geometry::ModelGeometry)
    L = model_geometry.lattice.L[1]

    dx = abs(d(x(i, model_geometry), x(j, model_geometry), model_geometry))
        dy = abs(d(y(i, model_geometry), y(j, model_geometry), model_geometry))

        if dy > dx
            dx, dy = dy, dx
        end
    
        return dx + L * dy
end


"""

    reduce_index_1d( i::Int, j::Int, model_geometry::ModelGeometry )

Reduces the indices of 2 lattice sites (i,j) to irreducible indices (0,k), where k is an integer.

"""
function reduce_index_1d(i::Int, j::Int, model_geometry::ModelGeometry)
    L = model_geometry.lattice.L[1]

    dx = abs(d(x(i, model_geometry), x(j, model_geometry), model_geometry))

    return dx
end


"""

    max_dist( N::Int, L::Int )

Obtains the maximum irreducible index given the total number of sites N and extent of the lattice L.

"""
function max_dist(N::Int, L::Int)
    if L % 2 == 0
        return Int(N / 2 + L / 2)
    else
        return Int(N / 2)
    end
end





##################################################### DEPRECATED FUNCTIONS #####################################################
# function reset_dictionary_to_zeros!(dict::AbstractDict)
#     for key in keys(dict)
#         dict[key] = reset_to_zeros(dict[key])
#     end
# end

# function reset_to_zeros(value)
#     if isa(value, Tuple)
#         return tuple(reset_to_zeros(v) for v in value)
#     elseif isa(value, AbstractArray)
#         return zeros(eltype(value), size(value))
#     elseif isa(value, Number)
#         return zero(value)
#     else
#         throw(ArgumentError("Unsupported type for reset: $(typeof(value))"))
#     end
# end


# """

#     is_invertible( D::Matrix{AbstractFloat} ) 
    
# Checks the invertibility of a matrix D. 

# """
# function is_invertible(D)
#     # invertible = inverse(D)

#     # return invertible

#     return !isapprox(det(D), 0.0; atol=1e-12)
# end


# """

#     check_overlap( D::Matrix{ComplexF64} )
    
# Ensures that there is finite overlap with the determinantal state.

# """
# function check_overlap(D::Matrix{ComplexF64})
#     # perform LU decomposition
#     lu_decomp = lu(D)

#     # check for finite overlap
#     if !is_invertible(lu_decomp)
#         debug && println("WARNING! State has no overlap with the determinantal wavefunction!")
#         debug && println("D = ")
#         debug && display(D)

#         return false
#     end

#     return true
# end


# # function inverse(D::AbstractMatrix)
# #     try
# #         D_inv = inv(D)  # Try to compute the inverse
# #         # println("Matrix is invertible.")
# #         return true  # Inversion successful, matrix is invertible
# #     catch e
# #         if isa(e, SingularException)
# #             println("Singular configuration detected! Matrix inversion unsuccessful.")
# #             return false  # Matrix is singular, inversion failed
# #         else
# #             rethrow(e)  # For any other exceptions, rethrow the error
# #         end
# #     end
# # end

# """

#     complex_zeros{T<:AbstractFloat}(rows::Int, cols::Int)

# Initialzies a zero matrix with complex type

# """
# function complex_zeros( rows::Int, cols::Int)
#     return zeros(Complex{AbstractFloat}, rows, cols)
# end

# """

#     get_parameter_indices()

# Get indices of variational parameters from its respective matrix.

# """
# function get_parameter_indices(par_matrix)
#     nonzero_indices = findall(x -> x != 0, par_matrix)

#     parameter_indices = sort(nonzero_indices, by=x->(x[1], x[2]))

#     return parameter_indices
# end 


# """

#     apply_tabc!( H::Matrix{ComplexF64}, Lx::Int, Ly::Int, θ::Tuple{Float64, Float64} )

# Given a Hamiltonian matrix H, phase θ, and lattice dimensions Lx and Ly, applies twisted boundary conditions. 

# """
# function apply_tbc!(H::Matrix{ComplexF64}, θ::Tuple{Float64, Float64}, Lx::Int, Ly::Int)
#     θx, θy = θ
#     N = Lx * Ly

#     # Apply the twist in the x-direction for spin-up and spin-down sectors
#     for y in 1:Lx
#         idx1 = Lx * (y - 1) + 1  # First element of row y
#         idx2 = Lx * y            # Last element of row y

#         # Spin-up sector
#         H[idx1, idx2] *= cis(θx) #exp(1im * θx)
#         H[idx2, idx1] *= cis(-θx) #exp(-1im * θx)

#         # Spin-down sector
#         H[idx1 + N, idx2 + N] *= cis(-θx) #exp(1im * -θx)
#         H[idx2 + N, idx1 + N] *= cis(θx) #exp(-1im * -θx)
#     end

#     # Apply the twist in the y-direction for spin-up and spin-down sectors
#     for x in 1:Ly
#         idx1 = x                  # Top element of column x
#         idx2 = Ly * (Ly - 1) + x  # Bottom element of column x

#         # Spin-up sector
#         H[idx1, idx2] *= cis(θy) #exp(1im * θy)
#         H[idx2, idx1] *= cis(-θy) #exp(-1im * θy)

#         # Spin-down sector
#         H[idx1 + N, idx2 + N] *= cis(-θy) #exp(1im * -θy)
#         H[idx2 + N, idx1 + N] *= cis(θy) #exp(-1im * -θy)
#     end
# end


# # Apply twist phases to the Hamiltonian
# θ = (rand(rng) * 2 * π, rand(rng) * 2 * π)  
# apply_tbc!(H, θ, Lx, Ly)