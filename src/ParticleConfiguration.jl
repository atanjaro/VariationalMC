"""

    get_particle_numbers( density::Float64 ) 

Given a particle density, returns the total number of particles Np, number of spin-up particles Npu, 
number of spin-down particles Npd, and total number of electrons Ne.

"""
function get_particle_numbers(density::Float64)
    N = model_geometry.lattice.N
    Np = density * N

    # Check if the total number of particles Np is an integer
    @assert isapprox(Np, round(Np), atol=1e-8) "Density does not correspond to a commensurate filling (Np must be an integer)"
    
    # Np must also be even
    @assert Np % 2 == 0 "Np must be even"
    
    if !pht
        # Number of spin-up and spin-down particles for particle-hole transformation off
        Npu = Np ÷ 2
        Npd = Np ÷ 2
        nup = Npu
        ndn = Npd
        Ne = Np
    else
        # Particle-hole transformation on
        nup = Np ÷ 2
        Npu = nup
        ndn = N + nup - Np
        Npd = N - ndn
        Ne = nup + ndn
    end
    
    return Int(Np), Int(Ne), Int(nup), Int(ndn)
end


"""
    get_particle_density( nup::Int64, ndn::Int64 ) 

Given the number of spin-up electrons nup, and number of spin-down electrons ndn, returns 
the particle density, total number of particles Np, and the total number of electrons Ne.

"""
function get_particle_density(nup::Int, ndn::Int)
    N = model_geometry.lattice.N

    if !pht
        # Number of spin-up and spin-down particles for particle-hole transformation off
        Npu = nup  
        Npd = ndn 
        Np = Npu + Npd
        Ne = Np
    else
        # Particle-hole transformation on
        Npu = nup
        Npd = N - ndn 
        Np = Npu + Npd
        Ne = nup + ndn
    end

    # calculate particle density
    density = Np / N

    return density, Int(Np), Int(Ne)
end


"""
    generate_initial_fermion_configuration()::Vector{Int64}

Returns a randomly generated initial configuration of electrons (and holes).

"""
function generate_initial_fermion_configuration()
    init_pconfig = zeros(Int, 2*model_geometry.lattice.N)                                                      
    while sum(init_pconfig) < nup
        init_pconfig[rand(rng, 1:model_geometry.lattice.N)] = 1
    end
    while sum(init_pconfig) < Ne
        init_pconfig[rand(rng, model_geometry.lattice.N+1:2*model_geometry.lattice.N)] = 1
    end
    return init_pconfig
end


"""

    generate_initial_onsite_phonon_configuration( )

Returns initial vector to store onsite phonon configurations.

"""
function generate_initial_onsite_phonon_configuration()
    init_phconfig = zeros(Int, model_geometry.lattice.N)
 
    return init_phconfig
end


"""

    generate_initial_bond_phonon_configuration( )

Returns initial vector to store bond phonon configurations.

"""
function generate_initial_bond_phonon_configuration()
    # create neighbor table
    nbr_table = build_neighbor_table(bonds[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice)

    # maps neighbor table to dictionary of bonds and neighbors                                
    nbrs = map_neighbor_table(nbr_table)

    # Collect all bonds into a Set to ensure uniqueness
    unique_bonds = Set{Int}()

    for (site, bond_info) in nbrs
        # Add bonds to the set (automatically handles duplicates)
        for bond in bond_info.bonds
            push!(unique_bonds, bond)
        end
    end

    # The total number of unique bonds is the length of the set
    total_bonds = length(unique_bonds)

    init_phconfig = zeros(Int, total_bonds)

    return init_phconfig
end






"""
    get_spindex_type( spindex::Int, model_geometry::ModelGeometry )::Int

Returns the spin species at a given spindex.

"""
function get_spindex_type(spindex::Int, model_geometry::ModelGeometry)::Int
    @assert spindex < 2 * model_geometry.lattice.N + 1
    return spindex < model_geometry.lattice.N + 1 ? 1 : 2
end


"""
    get_index_from_spindex( spindex::Int, model_geometry::ModelGeometry )::Int 

Returns the lattice site i for a given spindex.

"""
function get_index_from_spindex(spindex::Int, model_geometry::ModelGeometry)::Int
    L = model_geometry.lattice.N
    @assert spindex < 2 * L + 1
    return spindex <= L ? spindex : spindex - L
end


"""
    get_spindices_from_index( index::Int, model_geometry::ModelGeometry ) 

Returns spin-up and spin-down indices from a given site index.

"""
function get_spindices_from_index(index::Int, model_geometry::ModelGeometry)
    L = model_geometry.lattice.N
    @assert index <= L
    return index, index + L
end


"""
    number_operator( site::Int, pconfig::Vector{Int} )

Returns the number of spin-up and spin-down electrons occupying a real lattice site i.  

"""
function number_operator(site::Int, pconfig::Vector{Int})
    nup = pconfig[site]
    ndn = pconfig[site+model_geometry.lattice.N]
    Ne = pconfig[site] + pconfig[site+model_geometry.lattice.N]
    return nup, ndn, Ne
end


"""
    do_particle_hop!( met_step, pconfig::Vector{Int}, model_geometry::ModelGeometry )

If proposed particle hop is accepted, perform the particle hop, and update the particle 
configuration and positions.

"""
function do_particle_hop!(met_step, pconfig::Vector{Int}, κ::Vector{Int64}, model_geometry::ModelGeometry)
    # particle number
    β = met_step.particle

    # spin of the particle
    spin = met_step.spin

    # initial site
    k = met_step.isite

    # final site
    l = met_step.fsite

    # account for spin-up and spin-down sectors
    if spin == 2
        k_dn = get_spindices_from_index(k, model_geometry)[2]
        l_dn = get_spindices_from_index(l, model_geometry)[2]

        if debug
            @info "Hopping particle from site $k to site $l"
        end

        # update pconfig
        pconfig[k_dn] = 0
        pconfig[l_dn] = 1

        # update κ
        κ[k_dn] = 0 
        κ[l_dn] = β
    else
        # update pconfig
        pconfig[k] = 0
        pconfig[l] = 1 

        # update κ
        κ[k] = 0 
        κ[l] = β
    end
    
    return nothing
end



# """
#     get_particle_positions( pconfig::Vector{Int} )

# Returns a dictionary of particle positions with keys and values,
# "spindex" -> "lattice site".

# """
# function get_particle_positions(pconfig::Vector{Int}, model_geometry::ModelGeometry)
#     particle_positions = Dict()
#     for i in eachindex(pconfig)
#         if pconfig[i] == 1
#             particle_positions[i] = get_index_from_spindex(i, model_geometry)
#         else
#         end
#     end

#     sorted_positions = sort(collect(particle_positions), by = x->x[1])

#     if length(sorted_positions) != Ne
#         @error "Mismatch in particle_positions length: expected $Ne, got $(length(sorted_positions))"
#     end
    
#     return sorted_positions
# end


"""

    get_particle_positions( pconfig::Vector{Int}, model_geometry::ModelGeometry, Ne::Int )

Given vector of spindex occupations, returns a vector κ of particle positions according to particle number.
This vector is gives the order of creation operator which create the configuration. Initial configurations
are automatically normal ordered.

"""
function get_particle_positions(pconfig::Vector{Int}, model_geometry::ModelGeometry, Ne::Int)
    N = model_geometry.lattice.N

    # get current spindex occupations
    config_indices = findall(x -> x == 1, pconfig)

    κ = zeros(Int, 2 * N)

    # fill κ according to position of particles
    for i in 1:Ne
        idx = config_indices[i]
        κ[idx] = i 
    end

    return κ
end

# """
#     update_particle_position!( met_step::LocalAcceptance, paritcle_positions)

# If a particle 'β' at site 'k' successfully hops to a neighboring site 'l', update its
# position in 'particle_positions' as well as 'pconfig.

# """
# function update_particle_position!(met_step, particle_positions)
#     if met_step.acceptance == 1
#         # Update the pair within the vector
#         particle_positions[met_step.particle] = Pair(get_spindices_from_index(met_step.fsite, model_geometry)[met_step.spin], met_step.fsite)
#         return nothing
#     else
#         # DO NOTHING
#         return nothing
#     end
# end
















