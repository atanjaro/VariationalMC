"""

    Jastrow( jastrow_type::AbstractString, num_jpars::Int, 
             jpar_map::OrderedDict{Any, Any}, Float64}} 
             Tvec_f::Vector{Float64}, Tvec_b::Vector{Float64} )

A type defining quantities related to a Jastrow factor.

"""
mutable struct Jastrow
    # type of Jastrow parameter
    jastrow_type::String

    # number of Jastrow parameters
    num_jpars::Int

    # map of Jastrow parameters
    jpar_map::OrderedDict{Any, Any}

    # fermionic T vector
    Tvec_f::Vector{Float64}

    # bosonic T vector
    Tvec_b::Vector{Float64}
end


"""

    initialize_jpars( model_geometry::ModelGeometry, rng::Xoshiro, readin_jpars::Bool )

Generates a dictionary of irreducible indices k which reference a tuple consisting of a vector of lattice index 
pairs (i,j) which generate k, and Jastrow parameters vᵢⱼ. The parameter corresponding to the 
largest k is automatically initialized to 0. Parameters are randomly initialized.

"""
function initialize_jpars(model_geometry::ModelGeometry, rng::Xoshiro, readin_jpars::Bool)
    # check 
    @assert readin_jpars == false

    # number of lattice sites
    N = model_geometry.lattice.N

    # one side of the lattice
    L = model_geometry.lattice.L[1]
    
    # vector to store reduced indices
    reduced_indices = []

    # initialize map of Jastrow parameters
    jpar_map = OrderedDict()

    for i in 0:N-1
        red_idx = reduce_index_2d(0, i, model_geometry)
        push!(reduced_indices, red_idx)
        if haskey(jpar_map, red_idx)
            indices, init_val = jpar_map[red_idx]
            push!(indices, (0, i))
            jpar_map[red_idx] = (indices, init_val)
        else
            jpar_map[red_idx] = ([(0, i)], 0.01*rand(rng))
        end
    end

    for i in 1:N-1
        for j in 1:N-1
            red_idx = reduce_index_2d(i, j, model_geometry)
            push!(reduced_indices, red_idx)
            if haskey(jpar_map, red_idx)
                indices, init_val = jpar_map[red_idx]
                push!(indices, (i, j))
                jpar_map[red_idx] = (indices, init_val)
            else
                jpar_map[red_idx] = ([(i, j)], 0.01*rand(rng))
            end
        end
    end

    # set the parameter corresponding to the maximum distance to 0
    max_idx = maximum(keys(jpar_map))
    if haskey(jpar_map, max_idx)
        indices, _ = jpar_map[max_idx]
        jpar_map[max_idx] = (indices, 0.0)
    else
        error("Maximum distance index $max_idx not found in jpar_map")
    end

    # DEBUG
    if debug
        # check indices
        irreducible_indices = unique(reduced_indices)
        println(irreducible_indices)
    end

    # Sort the dictionary by irreducible indices
    sorted_jpar_map = OrderedDict(sort(collect(jpar_map)))

    return sorted_jpar_map
end


"""

    initialize_jpars( model_geometry::ModelGeometry, path_to_jpars::String, readin_jpars::Bool )

Generates a dictionary of irreducible indices k which reference a tuple consisting of a vector of lattice index 
pairs (i,j) which generate k, and Jastrow parameters vᵢⱼ. The parameter corresponding to the 
largest k is automatically initialized to 0. Parameters are read in from a .csv file.

"""
function initialize_jpars(model_geometry::ModelGeometry, path_to_jpars::String, readin_jpars::Bool)
    # check
    @assert readin_jpars == true

    # number of lattice sites
    N = model_geometry.lattice.N

    # one side of the lattice
    L = model_geometry.lattice.L[1]
    
    # vector to store reduced indices
    reduced_indices = []

    # initialize map of Jastrow parameters
    jpar_map = OrderedDict()

    # read the CSV file
    df = CSV.read(path_to_jpars, DataFrame)
        
    # create a dictionary from the CSV data
    csv_jpar_map = Dict{Any, Float64}()
    for row in eachrow(df)
        csv_jpar_map[row.IRR_IDX] = row.MEAN_vij
    end

    # for i in 0:N-1
    #     red_idx = reduce_index(0, i, model_geometry)
    #     push!(reduced_indices, red_idx)
    #     if haskey(jpar_map, red_idx)
    #         val = jpar_map[red_idx]
    #         jpar_map[red_idx] = val
    #     else
    #         # Use the value from the CSV if it exists
    #         mean_vij = csv_jpar_map[red_idx]
    #         jpar_map[red_idx] = mean_vij
    #     end
    # end

    for i in 0:N-1
        red_idx = reduce_index_2d(0, i, model_geometry)
        push!(reduced_indices, red_idx)
        if haskey(jpar_map, red_idx)
            indices, mean_vij = jpar_map[red_idx]
            push!(indices, (0, i))
            jpar_map[red_idx] = (indices, mean_vij)
        else
            # Use the value from the CSV if it exists
            mean_vij = csv_jpar_map[red_idx]
            jpar_map[red_idx] = ([(0, i)], mean_vij)
        end
    end

    for i in 1:N-1
        for j in 1:N-1
            red_idx = reduce_index_2d(i, j, model_geometry)
            push!(reduced_indices, red_idx)
            if haskey(jpar_map, red_idx)
                indices, mean_vij = jpar_map[red_idx]
                push!(indices, (i, j))
                jpar_map[red_idx] = (indices, mean_vij)
            else
                # Use the value from the CSV if it exists
                mean_vij = csv_jpar_map[red_idx]
                jpar_map[red_idx] = ([(i, j)], mean_vij)
            end
        end
    end

    # set the parameter corresponding to the maximum distance to 0
    # technically, the read in values will always have the last Jastrow parameter to be 0
    # but this is just here for safety reasons.
    max_idx = maximum(keys(jpar_map))
    if haskey(jpar_map, max_idx)
        indices, _ = jpar_map[max_idx]
        jpar_map[max_idx] = (indices, 0.0)
    else
        error("Maximum distance index $max_idx not found in jpar_map")
    end

    # DEBUG
    if debug
        # check indices
        irreducible_indices = unique(reduced_indices)
        println(irreducible_indices)
    end

    # Sort the dictionary by key
    sorted_jpar_map = OrderedDict(sort(collect(jpar_map)))

    return sorted_jpar_map
end



"""

    get_Tvec( jastrow_type::AbstractString, jpar_map::Dict{Any,Any}, pconfig::Vector{Int64}, model_geometry::ModelGeometry ) 

Returns vector of T with entries of the form Tᵢ = ∑ⱼ vᵢⱼnᵢ(x) where vᵢⱼ are the associated Jastrow peseudopotentials and nᵢ(x)
is the total electron occupation.

"""
function get_Tvec(jastrow_type::String, jpar_map::OrderedDict{Any,Any}, pconfig::Vector{Int64}, pht::Bool, model_geometry::ModelGeometry)
    # extent of the lattice
    N = model_geometry.lattice.N

    # dimensions
    dims = size(model_geometry.lattice.L)[1]

    # initialize T vectors
    Tvec_f = zeros(N) 
    Tvec_b = zeros(N) 


    for i in 1:N 
        for j in 1:N
            # reduce the index
            if dims == 1
                reduced_index = reduce_index_1d(i, j, model_geometry)
            elseif dims == 2
                reduced_index = reduce_index_2d(i, j, model_geometry)
            end

            if haskey(jpar_map, reduced_index)
                # get_vᵢⱼ
                (_, value) = jpar_map[reduced_index]
                vᵢⱼ = value
    
                # get fermion occupations
                num_up = get_onsite_fermion_occupation(j, pconfig)[1]
                num_dn = get_onsite_fermion_occupation(j, pconfig)[2]

                # populate T vectors
                if jastrow_type == "e-den-den"
                    if pht
                        Tvec_f[i] += vᵢⱼ * (num_up - num_dn)
                    else
                        Tvec_f[i] += vᵢⱼ * (num_up + num_dn)    
                    end
                elseif jastrow_type == "e-spn-spn"
                    if pht
                        Tvec_f[i] += 0.5 * vᵢⱼ * (num_up - num_dn)
                    else
                        Tvec_f[i] += 0.5 * vᵢⱼ * (num_up + num_dn)    
                    end
                end
            end
        end
    end
    
    return Tvec_f, Tvec_b
end


# """

#     get_Tvec( jastrow_type::AbstractString, jpar_map::Dict{Any,Any}, pconfig::Vector{Int64}, phconfig::Vector{Int64}, model_geometry::ModelGeometry ) 

# Returns vectors T_f and T_b with entries of the form Tᵢ = ∑ⱼ vᵢⱼnᵢ(x) where vᵢⱼ are the associated Jastrow peseudopotentials and nᵢ(x)
# is the total electron or phonon occupation. 

# """
# function get_Tvec(jastrow_type::String, jpar_map::OrderedDict{Any,Any}, pconfig::Vector{Int64}, phconfig::Vector{Int64}, pht::Bool, model_geometry::ModelGeometry)
#     # extent of the lattice
#     N = model_geometry.lattice.N

#     # initialize T vectors
#     Tvec_f = zeros(N) #Vector{AbstractFloat}(undef, N)
#     Tvec_b = zeros(N) #Vector{AbstractFloat}(undef, N)

#     for i in 1:N
#         # track the Jastrow parameter sum
#         jpar_sum = 0.0
#         for j in 1:N 
#             # Calculate the reduced index for (i, j)
#             red_idx = reduce_index(i-1, j-1, model_geometry)

#             # Add the appropriate value based on the key of the jpar_map
#             if haskey(jpar_map, red_idx)
#                 (_, vᵢⱼ) = jpar_map[red_idx]
#                 jpar_sum += vᵢⱼ
#             end
#         end

#         # get boson occupations
#         num_phonons = get_phonon_occupation(i, phconfig)

#         if jastrow_type == "eph-den-den" # electron-phonon density-density
#             # get fermion occupations
#             num_up = get_onsite_fermion_occupation(i, pconfig)[1]
#             num_dn = get_onsite_fermion_occupation(i, pconfig)[2]

#             if pht 
#                 Tvec_f[i] = jpar_sum * (2 * num_up - 1)
#                 Tvec_b[i] = jpar_sum * num_phonons
#             else
#                 Tvec_f[i] = jpar_sum * (num_up + num_dn)
#                 Tvec_b[i] = jpar_sum * num_phonons
#             end
#         elseif jastrow_type == "ph-den-den"  # phonon density-density
#             Tvec_b[i] = jpar_sum * num_phonons
#         end
#     end
    
#     return Tvec_f, Tvec_b
# end


# """

#     get_Tvec( jastrow_type::AbstractString, jpar_map::Dict{Any,Any}, pconfig::Vector{Int64}, phconfig::Vector{Int64}, model_geometry::ModelGeometry ) 

# Returns vectors T_f and T_b with entries of the form Tᵢ = ∑ⱼ vᵢⱼnᵢ(x) where vᵢⱼ are the associated Jastrow peseudopotentials and nᵢ(x)
# is the total electron or phonon occupation. 

# """
# function get_Tvec(jastrow_type::String, jpar_map::OrderedDict{Any,Any}, pconfig::Vector{Int64}, phconfig::Matrix{AbstractFloat}, pht::Bool, model_geometry::ModelGeometry)
#     # extent of the lattice
#     N = model_geometry.lattice.N

#     # initialize T vectors
#     Tvec_f = zeros(N) #Vector{AbstractFloat}(undef, N)
#     Tvec_b = zeros(N) #Vector{AbstractFloat}(undef, N)


#     for i in 1:N 
#         for j in 1:N
#             # get vᵢⱼ
#             for (_, (indices, value)) in jpar_map
#                 if (i-1, j-1) in indices
#                     vᵢⱼ = value
#                     break  
#                 end
#             end

#             # get fermion occupations
#             num_up = get_onsite_fermion_occupation(j, pconfig)[1]
#             num_dn = get_onsite_fermion_occupation(j, pconfig)[2]

#             # populate T vectors
#             if jastrow_type == "e-den-den"
#                 if pht
#                     Tvec_f[i] += vᵢⱼ * (num_up - num_dn)
#                 else
#                     Tvec_f[i] += vᵢⱼ * (num_up + num_dn)    
#                 end
#             elseif jastrow_type == "e-spn-spn"
#                 if pht
#                     Tvec_f[i] += 0.5 * vᵢⱼ * (num_up - num_dn)
#                 else
#                     Tvec_f[i] += 0.5 * vᵢⱼ * (num_up + num_dn)    
#                 end
#             end
#         end
#     end
    
#     return Tvec_f, Tvec_b


#     # optical ssh model
#     if size(phconfig)[1] == N
#         for i in 1:N
#             # track the Jastrow parameter sum
#             jpar_sum = 0.0
#             for j in 1:N 
#                 # Calculate the reduced index for (i, j)
#                 red_idx = reduce_index(i-1, j-1, model_geometry)

#                 # Add the appropriate value based on the key of the jpar_map
#                 if haskey(jpar_map, red_idx)
#                     (_, vᵢⱼ) = jpar_map[red_idx]
#                     jpar_sum += vᵢⱼ
#                 end
#             end

#             # get boson displacements
#             (Xᵢ, Yᵢ) = get_onsite_phonon_displacement(i, phconfig)

#             if jastrow_type == "ph-dsp-dsp-x"  # phonon-x-displacement-displacement
#                 Tvec_b[i] = jpar_sum * Xᵢ
#             elseif jastrow_type == "ph-dsp-dsp-y" # phonon-y-displacement-displacement
#                 Tvec_b[i] = jpar_sum * Yᵢ
#             elseif jastrow_type == "eph-den-dsp-x"  # electron-phonon density-x-displacement
#                 # get fermion occupations
#                 num_up = get_onsite_fermion_occupation(i, pconfig)[1]
#                 num_dn = get_onsite_fermion_occupation(i, pconfig)[2]

#                 if pht
#                     Tvec_f[i] = jpar_sum * (num_up + num_dn - 1)
#                 else
#                     Tvec_f[i] = jpar_sum * (num_up + num_dn)
#                 end

#                 Tvec_b[i] = jpar_sum * Xᵢ
#             elseif jastrow_type == "eph-den-dsp-y"  # electron-phonon density-y-displacement
#                 # get fermion occupations
#                 num_up = get_onsite_fermion_occupation(i, pconfig)[1]
#                 num_dn = get_onsite_fermion_occupation(i, pconfig)[2]

#                 if pht
#                     Tvec_f[i] = jpar_sum * (num_up + num_dn - 1)
#                 else
#                     Tvec_f[i] = jpar_sum * (num_up + num_dn)
#                 end

#                 Tvec_b[i] = jpar_sum * Yᵢ
#             end
#         end
#     # bond ssh model
#     elseif size(phconfig)[1] == 2*N
#         # TODO; figure out how to do this for the bond model
#         # for i in 1:N
#         #     # track the Jastrow parameter sum
#         #     jpar_sum = 0.0
#         #     for j in 1:N 
#         #         # Calculate the reduced index for (i, j)
#         #         red_idx = reduce_index(i-1, j-1, model_geometry)

#         #         # Add the appropriate value based on the key of the jpar_map
#         #         if haskey(jpar_map, red_idx)
#         #             (_, vᵢⱼ) = jpar_map[red_idx]
#         #             jpar_sum += vᵢⱼ
#         #         end

#         #         # get boson displacements
#         #         X_ij = get_bond_phonon_displacement(b, phconfig)

#         #     end
#         # end
#     end
    
#     return Tvec_f, Tvec_b
# end


"""

    update_Tvec!( local_acceptance::LocalAcceptance, jastrow::Jastrow, model_geometry::ModelGeometry, pht::Bool )

Updates elements Tᵢ of the vector T after a Metropolis update.

"""
function update_Tvec!(local_acceptance, jastrow::Jastrow, model_geometry::ModelGeometry, pht::Bool)
     # extent of the lattice
     N = model_geometry.lattice.N

     # dimensions
     dims = size(model_geometry.lattice.L)[1]

    # current T vectors
    Tvec_f = jastrow.Tvec_f
    Tvec_b = jastrow.Tvec_b

    # different T vectors
    Tvec_f_diff = zeros(N)
    # Tvec_b_diff = zeros(N)

    # jpar map
    jpar_map = jastrow.jpar_map

    # initial and final sites from the particle hop
    l = local_acceptance.isite
    k = local_acceptance.fsite

    for i in 1:N
        if dims == 1
            reduced_l_index = reduce_index_1d(i, l, model_geometry)
            reduced_k_index = reduce_index_1d(i, k, model_geometry)
        elseif dims == 2
            reduced_l_index = reduce_index_2d(i, l, model_geometry)
            reduced_k_index = reduce_index_2d(i, k, model_geometry)
        end

        if haskey(jpar_map, reduced_l_index) || haskey(jpar_map, reduced_k_index)
            # get_vᵢₗ
            (_, value1) = jpar_map[reduced_l_index]
            vᵢₗ = value1

            (_, value2) = jpar_map[reduced_k_index]
            vᵢₖ = value2

            Tvec_f_diff[i] = vᵢₗ - vᵢₖ
        end
    end

    if local_acceptance.spin == 2
        Tvec_f_updated = Tvec_f - Tvec_f_diff
    else
        Tvec_f_updated = Tvec_f + Tvec_f_diff
    end

    return Tvec_f_updated, Tvec_b
end


"""
    get_jastrow_ratio( k::Int, l::Int, jastrow::Jastrow, pht::Bool, spin::Int )

Calculates ratio J(x₂)/J(x₁) = exp[-s(Tₗ - Tₖ) + vₗₗ - vₗₖ ] of Jastrow factors for particle configurations 
which differ by a single particle hopping from site 'k' (configuration 'x₁') to site 'l' (configuration 'x₂')
using the corresponding T vectors Tₖ and Tₗ, rsepctively.  

"""
function get_jastrow_ratio(k::Int, l::Int, jastrow::Jastrow, pht::Bool, spin::Int, model_geometry)
    # dimensions
    dims = size(model_geometry.lattice.L)[1]

    # T vectors
    Tvec_f = jastrow.Tvec_f
    # Tvec_b = jastrow.Tvec_b

    # jpar map
    jpar_map = jastrow.jpar_map

    # obtain elements 
    if dims == 1
        red_idx_ll = reduce_index_1d(l-1, l-1, model_geometry)
    elseif dims == 2
        red_idx_ll = reduce_index_2d(l-1, l-1, model_geometry)
    end

    if haskey(jpar_map, red_idx_ll)
        (_,  vₗₗ) = jpar_map[red_idx_ll]
    end

    if dims == 1
        red_idx_lk = reduce_index_1d(l-1, k-1, model_geometry)
    elseif dims == 2
        red_idx_lk = reduce_index_2d(l-1, k-1, model_geometry)
    end

    if haskey(jpar_map, red_idx_ll)
        (_,  vₗₖ) = jpar_map[red_idx_lk]
    end

    # select element for the initial and final sites
    # fermionic
    Tₗ_f = Tvec_f[l]
    Tₖ_f = Tvec_f[k]

    # # bosonic
    # Tₗ_b = Tvec_b[l]
    # Tₖ_b = Tvec_b[k]

    # compute ratio
    if pht
        if spin == 2
            jas_ratio_f = exp( (Tₗ_f - Tₖ_f) + vₗₗ - vₗₖ )
        else
            jas_ratio_f = exp( -(Tₗ_f - Tₖ_f) + vₗₗ - vₗₖ )

        end
    else
        jas_ratio_f = exp( -(Tₗ_f - Tₖ_f) + vₗₗ - vₗₖ )
    end

    return jas_ratio_f #, jas_ratio_b
end


"""

    get_jastrow_ratio( i::Int, cran::Int, jastrow::Jastrow, phconfig::Vector{Int} )

Calculates ratio J(x₂)/J(x₁) of Jastrow factors for phonon density configurations
which differ by the addition or removal of a boson. 

"""
function get_jastrow_ratio(i::Int, cran::Int, jastrow::Jastrow, phconfig::Vector{Int}, μₚₕ::AbstractFloat)
    # T vectors
    Tvec_f = jastrow.Tvec_f
    Tvec_b = jastrow.Tvec_b

    # get phonon occupation at site i
    num_phonons = get_phonon_occupation(i, phconfig)

    # get fermionic T vector
    Tᵢ_f = Tvec_f[i]

    if cran == -1
        # removal of a particle
        jas_ratio_b = exp(-μₚₕ) * sqrt(num_phonons) * exp(Tᵢ_f)
    elseif cran == 1
        # addition of a particle
        jas_ratio_b = exp(μₚₕ) * exp(-Tᵢ_f) / (num_phonons + 1)
    end

    return jas_ratio_b
end


"""

    get_jastrow_ratio(  )

Calculates ratio J(x₂)/J(x₁) of Jastrow factors for phonon density configurations
which differ by the addition or removal of a boson. 

"""
function get_jastrow_ratio(i::Int, j::Int, Δ::AbstractFloat, jastrow::Jastrow, phconfig::Matrix{AbstractFloat}, z_x::AbstractFloat, z_y::AbstractFloat)
    # sum nᵢnⱼ over all possible i and j 
    # TODO: this will require a slight change in the defintion of the Jastrow parameters for displacement configurations
            # since we will require ALL distances and the displacement parameters are odd under exchange of i and j i.e. wᵢⱼ = - wⱼᵢ
end


"""

    get_jastrow_ratio( i::Int, cran::Int, jastrow::Jastrow, phconfig::Vector{Int} )

Calculates ratio J(x₂)/J(x₁) of Jastrow factors for phonon density configurations
which differ by the addition or removal of a boson. 

"""
function get_jastrow_ratio(i::Int, j::Int, jastrow::Jastrow, phconfig::Matrix{AbstractFloat}, z_x, z_y)
    # T vectors
    Tvec_f = jastrow.Tvec_f
    Tvec_b = jastrow.Tvec_b

    # get phonon occupation displacement between sites i and j 
   

    # get fermionic T vector
    Tᵢ_f = Tvec_f[i]

    if cran == -1
        # # removal of a particle
        # R = exp(-μₚₕ) * sqrt(num_phonons) * exp(Tᵢ_f)
    elseif cran == 1
        # # addition of a particle
        # R = exp(μₚₕ) * exp(-Tᵢ_f) / (num_phonons + 1)
    end

    return R
end


"""
    build_jastrow_factor( jastrow_type::String, model_geometry::ModelGeometry, 
                          pconfig::Vector{Int64}, pht::Bool, rng::Xoshiro, readin_jpars::Bool )

Constructs relevant Jastrow factor and returns intitial T vector, matrix of Jastrow parameters, and
number of Jastrow parameters. 

"""
function build_jastrow_factor(jastrow_type::String, model_geometry::ModelGeometry, pconfig::Vector{Int64}, pht::Bool, rng::Xoshiro, readin_jpars::Bool)
    # map Jastrow parameters
    jpar_map = initialize_jpars(model_geometry, rng, readin_jpars);

    # generate T vector
    (init_Tvec_f, init_Tvec_b) = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry);

    # get number of Jastrow parameters
    num_jpars = length(jpar_map);
   
    if debug
        # report the number of Jastrow parameters initialized
        println(num_jpars," Jastrow parameters initialized")
        println("Type: ", jastrow_type)
    end

    return Jastrow(jastrow_type, num_jpars, jpar_map, init_Tvec_f, init_Tvec_b)
end


"""

    build_gutzwiller_factor(g, pconfig, model_geometry)

Constructs the Gutwiller correlator for an initial Gutzwiller parameter g, and 
particle configuration. [For testing purposes only.]

"""
function build_gutzwiller_factor(g, pconfig, model_geometry)
    N = model_geometry.lattice.N

    sum_dblocc = 0.0

    for i in 1:N

    end

end


# """
#     build_jastrow_factor( jastrow_type::String, model_geometry::ModelGeometry, 
#                           pconfig::Vector{Int64}, phconfig::Vector{Int64}, pht::Bool, rng::Xoshiro, readin_jpars::Bool )

# Constructs relevant Jastrow factor and returns intitial T vector, matrix of Jastrow parameters, and
# number of Jastrow parameters. 

# """
# function build_jastrow_factor(jastrow_type::String, model_geometry::ModelGeometry, pconfig::Vector{Int64}, electron_phonon_model, pht::Bool, rng::Xoshiro, readin_jpars::Bool)
#     # phonon configuration
#     phconfig = electron_phonon_model.phconfig

#     # map Jastrow parameters
#     jpar_map = initialize_jpars(model_geometry, rng, readin_jpars)

#     # generate T vector
#     (init_Tvec_f, init_Tvec_b) = get_Tvec(jastrow_type, jpar_map, pconfig, phconfig, pht, model_geometry)

#     # get number of Jastrow parameters
#     num_jpars = length(jpar_map)
   
#     if debug
#         # report the number of Jastrow parameters initialized
#         println(num_jpars," Jastrow parameters initialized")
#         println("Type: ", jastrow_type)
#     end

#     return Jastrow(jastrow_type, num_jpars, jpar_map, init_Tvec_f, init_Tvec_b)
# end


"""
    build_jastrow_factor( jastrow_type::String, model_geometry::ModelGeometry, 
                          pconfig::Vector{Int64}, pht::Bool, path_to_jpars::String, readin_jpars::Bool )

Constructs relevant Jastrow factor and returns intitial T vector, matrix of Jastrow parameters, and
number of Jastrow parameters. 

"""
function build_jastrow_factor(jastrow_type::String, model_geometry::ModelGeometry, pconfig::Vector{Int64}, pht::Bool, path_to_jpars::String, readin_jpars::Bool)
    # map Jastrow parameters
    jpar_map = initialize_jpars(model_geometry, path_to_jpars, readin_jpars)

    # generate T vector
    (init_Tvec_f, init_Tvec_b) = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry)

    # get number of Jastrow parameters
    num_jpars = length(jpar_map)
   
    if debug
        # report the number of Jastrow parameters initialized
        println(num_jpars," Jastrow parameters initialized")
        println("Type: ", jastrow_type)
    end

    return Jastrow(jastrow_type, num_jpars, jpar_map, init_Tvec_f, init_Tvec_b)
end


"""
    build_jastrow_factor( jastrow_type::String, model_geometry::ModelGeometry, 
                          pconfig::Vector{Int64}, phconfig::Vector{Int64}, pht::Bool, path_to_jpars::String, readin_jpars::Bool )

Constructs relevant Jastrow factor and returns intitial T vector, matrix of Jastrow parameters, and
number of Jastrow parameters. 

"""
function build_jastrow_factor(jastrow_type::String, model_geometry::ModelGeometry, pconfig::Vector{Int64}, electron_phonon_model, pht::Bool, path_to_jpars::String, readin_jpars::Bool)
    # phonon configuration
    phconfig = electron_phonon_model.phconfig

    # map Jastrow parameters
    jpar_map = initialize_jpars(model_geometry, path_to_jpars, readin_jpars)

    # generate T vector
    (init_Tvec_f, init_Tvec_b) = get_Tvec(jastrow_type, jpar_map, pconfig, phconfig::Vector{Int64}, pht, model_geometry)

    # get number of Jastrow parameters
    num_jpars = length(jpar_map)
   
    if debug
        # report the number of Jastrow parameters initialized
        println(num_jpars," Jastrow parameters initialized")
        println("Type: ", jastrow_type)
    end

    return Jastrow(jastrow_type, num_jpars, jpar_map, init_Tvec_f, init_Tvec_b)
end


"""
    update_jastrow!(jastrow::Jastrow, vpars::Vector{AbstractFloat})

Updates Jastrow parameters after Stochastic Reconfiguration.

"""
function update_jastrow!(jastrow::Jastrow, new_vpars::Vector{Float64})
    # number of Jastrow parameters that were modified
    num_jpars = jastrow.num_jpars - 1;

    # map of Jastrow parameters
    jpar_map = jastrow.jpar_map;

    # updated Jastrow parameters 
    new_jpars = new_vpars[end-num_jpars+1:end]

    # get irreducible indices
    irr_indices = collect(keys(jpar_map))

    # update all parameters except for the last one
    for i in 1:num_jpars
        # Extract the current (indices, jpars) tuple
        indices, _ = jpar_map[irr_indices[i]]
        
        # Update the jpars while maintaining the indices
        jpar_map[irr_indices[i]] = (indices, new_jpars[i])
    end
    
    return nothing
end


"""
    recalc_Tvec_f!(Tᵤ::Vector{AbstractFloat}, δT::AbstractFloat)

Checks floating point error accumulation in the fermionic T vector and if ΔT < δT, 
then the recalculated T vector Tᵣ replaces the updated T vector Tᵤ.

"""
function recalc_Tvec_f!(jastrow::Jastrow, δT::Float64)
    if debug
        println("Checking fermionic T vector...")
    end

    # Jastrow type
    jastrow_type = jastrow.jastrow_type

    # T vector(s) that has been updated during MC cycles
    Tᵤ_f = jastrow.Tvec_f

    # map of Jastrow parameters
    jpar_map = jastrow.jpar_map

    # recomputed T vector (first element is fermionic, second element is bosonic)
    Tᵣ_f = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry)[1]

    # difference in updated T vector and recalculated T vector
    diff_f = Tᵤ_f .- Tᵣ_f

    # sum the absolute differences and the recalculated T vector elements
    diff_sum_f = sum(abs.(diff_f))

    T_sum_f = sum(abs.(Tᵣ_f))

    # rms difference
    ΔT_f = sqrt(diff_sum_f  / T_sum_f)

    if ΔT_f > δT
        debug && println("WARNING! Fermionic T vector has been recalculated: ΔT = ", ΔT_f, " > δT = ", δT)

        # record new T vector(s)
        jastrow.Tvec_f = Tᵣ_f

        return nothing
    else
        debug && println("Fermionic T vector is stable: ΔT = ", ΔT_f, " < δT = ", δT)
        return nothing
    end  
end


"""
    recalc_Tvec_f!(Tᵤ::Vector{AbstractFloat}, δT::AbstractFloat)

Checks floating point error accumulation in the fermionic T vector and if ΔT < δT, 
then the recalculated T vector Tᵣ replaces the updated T vector Tᵤ.

"""
function recalc_Tvec_b!(jastrow::Jastrow, δT::Float64)
    if debug
        println("Checking bosonic T vector...")
    end

    # Jastrow type
    jastrow_type = jastrow.jastrow_type

    # T vector(s) that has been updated during MC cycles
    Tᵤ_b = jastrow.Tvec_b

    # map of Jastrow parameters
    jpar_map = jastrow.jpar_map

    # recomputed T vector (first element is fermionic, second element is bosonic)
    Tᵣ_b = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry)[2]

    # difference in updated T vector and recalculated T vector
    diff_f = Tᵤ_f .- Tᵣ_f
    diff_b = Tᵤ_b .- Tᵣ_b

    # sum the absolute differences and the recalculated T vector elements
    diff_sum_b = sum(abs.(diff_b))

    T_sum_b = sum(abs.(Tᵣ_b))

    # rms difference
    ΔT_b = sqrt(diff_sum_b / T_sum_b)

    if ΔT_f > δT
        debug && println("WARNING! Bosonic T vector has been recalculated: ΔT = ", ΔT_b, " > δT = ", δT)

        # record new T vector(s)
        jastrow.Tvec_b = Tᵣ_b

        return nothing
    else
        debug && println("Bosonic T vector is stable: ΔT = ", ΔT_b, " < δT = ", δT)
        return nothing
    end  
end


