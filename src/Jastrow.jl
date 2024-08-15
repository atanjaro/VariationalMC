"""

    Jastrow( jastrow_type::AbstractString, num_jpars::Int, 
             jpar_map::OrderedDict{Any, Any}, Float64}} 
             Tvec::Vector{Float64} )

A type defining quantities related to a Jastrow factor.

"""
mutable struct Jastrow
    # type of Jastrow parameter
    jastrow_type::String

    # number of Jastrow parameters
    num_jpars::Int

    # map of Jastrow parameters
    jpar_map::OrderedDict{Any, Any}

    # T vector
    Tvec::Vector{Float64}
end


"""

    initialize_jpars( model_geometry::ModelGeometry, rng::Xoshiro, readin_jpars::Bool )

Generates a dictionary of irreducible indices k which reference a tuple consisting of a vector of lattice index 
pairs (i,j) which generate k, and Jastrow parameters vᵢⱼ. The parameter corresponding to the 
largest k is automatically initialized to 0. Parameters are randomly initialized.

"""
function initialize_jpars(model_geometry::ModelGeometry, readin_jpars::Bool)
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
        red_idx = reduce_index(0, i, model_geometry)
        push!(reduced_indices, red_idx)
        if haskey(jpar_map, red_idx)
            indices, init_val = jpar_map[red_idx]
            push!(indices, (0, i))
            jpar_map[red_idx] = (indices, init_val)
        else
            jpar_map[red_idx] = ([(0, i)], 0.0)
        end
    end

    for i in 1:N-1
        for j in 1:N-1
            red_idx = reduce_index(i, j, model_geometry)
            push!(reduced_indices, red_idx)
            if haskey(jpar_map, red_idx)
                indices, init_val = jpar_map[red_idx]
                push!(indices, (i, j))
                jpar_map[red_idx] = (indices, init_val)
            else
                jpar_map[red_idx] = ([(i, j)], 0.0)
            end
        end
    end

    # set the parameter corresponding to the maximum distance to 0
    # max_idx = max_dist(N, L)
    # if haskey(jpar_map, max_idx)
    #     indices, _ = jpar_map[max_idx]
    #     jpar_map[max_idx] = (indices, 0.0)
    # else
    #     error("Maximum distance index $max_idx not found in jpar_map")
    # end

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
        red_idx = reduce_index(0, i, model_geometry)
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
            red_idx = reduce_index(i, j, model_geometry)
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
    max_idx = max_dist(N, L)
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

Returns vector of T with entries Tᵢ = ∑ⱼ vᵢⱼnᵢ(x) if using density Jastrow or 
Tᵢ = ∑ⱼ wᵢⱼSᵢ(x) if using spin Jastrow.

"""
function get_Tvec(jastrow_type::String, jpar_map::OrderedDict{Any,Any}, pconfig::Vector{Int64}, pht::Bool, model_geometry::ModelGeometry)
    # extent of the lattice
    N = model_geometry.lattice.N

    # initialize T vector
    Tvec = Vector{AbstractFloat}(undef, N)

    for i in 1:N
        # track the Jastrow parameter sum
        jpar_sum = 0.0
        for j in 1:N 
            # Calculate the reduced index for (i, j)
            red_idx = reduce_index(i-1, j-1, model_geometry)

            # Add the appropriate value based on the key of the jpar_map
            if haskey(jpar_map, red_idx)
                (_, vᵢⱼ) = jpar_map[red_idx]
                jpar_sum += vᵢⱼ
            end
        end

        # check particle occupations
        num_up = number_operator(i, pconfig)[1]
        num_dn = number_operator(i, pconfig)[2]

        # check Jastrow type
        if jastrow_type == "e-den-den"  # electron density-density
            if pht
                Tvec[i] = jpar_sum * (num_up + num_dn - 1)  
            else
                Tvec[i] = jpar_sum * (num_up + num_dn)  
            end
        elseif jastrow_type == "e-spn-spn"  # electron spin-spin
            if pht 
                Tvec[i] = 0.5 * jpar_sum * (num_up + num_dn - 1)
            else
                Tvec[i] = 0.5 * jpar_sum * (num_up + num_dn)
            end
        elseif jastrow_type == "eph-den-den" # electron-phonon density-density
            # populate electron-phonon T vector
        elseif jastrow_type == "ph-dsp-dsp"  # phonon-displacement-displacement
            # populate
        elseif jastrow_type == "eph-den-dsp"  # electron-phonon density-displacement
            # populate
        end
    end
    
    return Tvec
end


"""
    update_Tvec!( local_acceptance::LocalAcceptance, jastrow::Jastrow, model_geometry::ModelGeometry, pht::Bool )

Updates elements Tᵢ of the vector T after a Metropolis update.

"""
function update_Tvec!(local_acceptance, jastrow::Jastrow, model_geometry::ModelGeometry, pht::Bool)
    # T vector
    Tvec = jastrow.Tvec

    # jpar map
    jpar_map = jastrow.jpar_map

    # extent of the lattice
    N = model_geometry.lattice.N

    # initial and final sites from the particle hop
    l = local_acceptance.isite
    k = local_acceptance.fsite

    # update
    for i in 1:N
        red_idx_il = reduce_index(i-1, l-1, model_geometry)
        if haskey(jpar_map, red_idx_il)
            (_,  vᵢₗ) = jpar_map[red_idx_il]
        end

        red_idx_ik = reduce_index(i-1, k-1, model_geometry)
        if haskey(jpar_map, red_idx_ik)
            (_,  vᵢₖ) = jpar_map[red_idx_ik]
        end

        if pht
            if local_acceptance.spin == 2
               Tvec[i] += - vᵢₗ + vᵢₖ
            else
                Tvec[i] += vᵢₗ - vᵢₖ
            end
        else
            Tvec[i] += vᵢₗ - vᵢₖ
        end
    end

    return nothing
end


"""
    get_jastrow_ratio( local_acceptance, jastrow::Jastrow, pht::Bool )

Calculates ratio J(x₂)/J(x₁) = exp[-s(Tₗ - Tₖ) + vₗₗ - vₗₖ ] of Jastrow factors for particle configurations 
which differ by a single particle hopping from site 'l' (configuration 'x₁') to site 'k' (configuration 'x₂')
using the corresponding T vectors Tₗ and Tₖ, rsepctively.  

"""
function get_jastrow_ratio(l, k, jastrow::Jastrow, pht::Bool, spin::Int)
    # T vector
    Tvec = jastrow.Tvec

    # jpar map
    jpar_map = jastrow.jpar_map

    # obtain elements 
    red_idx_ll = reduce_index(l-1, l-1, model_geometry)
    if haskey(jpar_map, red_idx_ll)
        (_,  vₗₗ) = jpar_map[red_idx_ll]
    end

    red_idx_lk = reduce_index(l-1, k-1, model_geometry)
    if haskey(jpar_map, red_idx_ll)
        (_,  vₗₖ) = jpar_map[red_idx_lk]
    end

    # select element for the initial and final sites
    Tₗ = Tvec[l]
    Tₖ = Tvec[k]

    # compute ratio
    if pht
        if spin == 2
            jas_ratio = exp(-((Tₗ - Tₖ) - vₗₗ + vₗₖ)) 
        else
            jas_ratio = exp(-((Tₗ - Tₖ) + vₗₗ - vₗₖ)) 
        end
    else
        jas_ratio = exp(-((Tₗ - Tₖ) + vₗₗ - vₗₖ)) 
    end

    return jas_ratio
end


"""
    build_jastrow_factor( jastrow_type::String, model_geometry::ModelGeometry, 
                          pconfig::Vector{Int64}, pht::Bool, rng::Xoshiro, readin_jpars::Bool )

Constructs relevant Jastrow factor and returns intitial T vector, matrix of Jastrow parameters, and
number of Jastrow parameters. 

"""
function build_jastrow_factor(jastrow_type::String, model_geometry::ModelGeometry, pconfig::Vector{Int64}, pht::Bool, readin_jpars::Bool)
    # map Jastrow parameters
    jpar_map = initialize_jpars(model_geometry, readin_jpars)

    # generate T vector
    init_Tvec = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry)

    # get number of Jastrow parameters
    num_jpars = length(jpar_map)
   
    if verbose
        # report the number of Jastrow parameters initialized
        println(num_jpars," Jastrow parameters initialized")
        println("Type: ", jastrow_type)
    end

    return Jastrow(jastrow_type, num_jpars, jpar_map, init_Tvec)
end


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
    init_Tvec = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry)

    # get number of Jastrow parameters
    num_jpars = length(jpar_map)
   
    if verbose
        # report the number of Jastrow parameters initialized
        println(num_jpars," Jastrow parameters initialized")
        println("Type: ", jastrow_type)
    end

    return Jastrow(jastrow_type, num_jpars, jpar_map, init_Tvec)
end


"""
    update_jastrow!(jastrow::Jastrow, vpars::Vector{AbstractFloat})

Updates Jastrow parameters after Stochastic Reconfiguration.

"""
function update_jastrow!(jastrow::Jastrow, new_vpars::Vector{Float64})
    # number of Jastrow parameters
    num_jpars = jastrow.num_jpars;

    # map of Jastrow parameters
    jpar_map = jastrow.jpar_map;

    # all new Jastrow parameters except for the last one
    new_jpars = new_vpars[end-num_jpars+1:end-1]

    # get irreducible indices
    irr_indices = collect(keys(jpar_map))

    # update all parameters except for the last one
    for i in 1:(num_jpars-1)
        # Extract the current (indices, jpars) tuple
        indices, _ = jpar_map[irr_indices[i]]
        
        # Update the jpars while maintaining the indices
        jpar_map[irr_indices[i]] = (indices, new_jpars[i])
    end
    
    return nothing
end


"""
    recalc_Tvec(Tᵤ::Vector{AbstractFloat}, δT::AbstractFloat)

Checks floating point error accumulation in the T vector and if ΔT < δT, 
then the recalculated T vector Tᵣ replaces the updated T vector Tᵤ.

"""
function recalc_Tvec!(jastrow::Jastrow, δT::Float64)
    # Jastrow type
    jastrow_type = jastrow.jastrow_type

    # T vector that has been updated during MC cycles
    Tᵤ = jastrow.Tvec

    # get matrix
    jpar_map = jastrow.jpar_map

    # recomputed T vector
    Tᵣ = get_Tvec(jastrow_type, jpar_map, pconfig, pht, model_geometry)

    # difference in updated T vector and recalculated T vector
    diff = Tᵤ .- Tᵣ

    # sum the absolute differences and the recalculated T vector elements
    diff_sum = sum(abs.(diff))
    T_sum = sum(abs.(Tᵣ))

    # rms difference
    ΔT = sqrt(diff_sum / T_sum)

    if ΔT > δT
        verbose && println("WARNING! T vector has been recalculated: ΔT = ", ΔT, " > δT = ", δT)

        # record new T vector
        jastrow.Tvec = Tᵣ

        return nothing
    else
        verbose && println("T vector is stable: ΔT = ", ΔT, " < δT = ", δT)
        return nothing
    end  
end


