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

    generate_initial_phonon_density_configuration()

Returns initial vector to store phonon occupations.

"""
function generate_initial_phonon_density_configuration(model_geometry::ModelGeometry)
    N = model_geometry.lattice.N
    init_phconfig = zeros(Int, N)
 
    return init_phconfig
end



"""

    generate_initial_phonon_displacement_configuration( )

Returns initial matrix to store phonon displacements. Each column represents displacements in each dimension. 

"""
function generate_initial_phonon_displacement_configuration(loc::AbstractString, model_geometry::ModelGeometry)
    # dimensions
    dims = size(model_geometry.lattice.L)[1]
    N = model_geometry.lattice.N

    if loc == "bond"
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

        # initialize phonon configuration
        init_phconfig = zeros(AbstractFloat, total_bonds, dims)        
    elseif loc == "onsite"
        # initialize phonon configuration
        init_phconfig = zeros(AbstractFloat, N, dims)
    else
        println("ERROR: Not a valid location in the lattice!")
    end

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
    get_linked_spindex( i, N ) 

Returns an index in the spin-down sector, given an index in the spin-up sector..

"""
function get_linked_spindex(i, N)
    @assert i < 2 * N
    return i + (1 - 2 * (i ÷ N)) * N
end


"""
    get_onsite_fermion_occupation( site::Int, pconfig::Vector{Int} )

Returns the number of spin-up and spin-down electrons occupying a real lattice site i.  

"""
function get_onsite_fermion_occupation(site::Int, pconfig::Vector{Int})
    nup = pconfig[site]
    ndn = pconfig[site+model_geometry.lattice.N]
    Ne = pconfig[site] + pconfig[site+model_geometry.lattice.N]
    return nup, ndn, Ne
end


"""
    get_phonon_occupation( site::Int, phconfig::Vector{Int} )

Returns the number of phonons occupying a real lattice site i or a lattice bond i.  

"""
function get_phonon_occupation(i::Int, phconfig::Vector{Int})
    nph = phconfig[i]

    return nph
end


"""
    get_onsite_phonon_displacement( site::Int, phconfig::Vector{Int} )

Returns the X and Y displacements of a phonon at site i.

"""
function get_onsite_phonon_displacement(i::Int, phconfig::Matrix{AbstractFloat})
    Xᵢ, Yᵢ = phconfig[i, :]

    return (Xᵢ, Yᵢ)
end


"""
    get_bond_phonon_displacement( b::Int, phconfig::Vector{Int} )

Given a bond in the lattice b, returns the displacement of phonon located on that bond. 

"""
function get_bond_phonon_displacement(b::Int, phconfig::Matrix{AbstractFloat})
    Xᵢ, Yᵢ = phconfig[i, :]

    return [Xᵢ, Yᵢ]
end


"""
    do_particle_hop!( met_step, pconfig::Vector{Int}, model_geometry::ModelGeometry )

If proposed particle hop is accepted, perform the particle hop, and update the particle 
configuration and positions.

"""
function do_particle_hop!(markov_move, pconfig::Vector{Int}, κ::Vector{Int64}, model_geometry::ModelGeometry)
    # particle number
    β = markov_move.particle

    # spin of the particle
    spin = markov_move.spin

    # initial site
    k = markov_move.isite

    # final site
    l = markov_move.fsite

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


"""
    change_particle_number!( met_step, phconfig::Vector{Int}, model_geometry::ModelGeometry )

If proposed particle addition or removal is accepted, add or remove a particle, and update the particle 
configuration..

"""
function change_particle_number!(met_step, phconfig, model_geometry)
    # lattice site
    N = model_geometry.lattice.N
    i = met_step[3]

    # addition or removal
    cran = met_step[2]

    if cran == 1
        phconfig[i] += 1
    elseif cran == -1
        phconfig -= 1
    end

    return nothing
end


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













