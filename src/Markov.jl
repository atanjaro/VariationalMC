"""
    LocalAcceptance( acceptance::Bool, particle::Int, spin::Int, isite::Int, fsite::Int )

A type defining quantities related to local MC update acceptance

"""
struct LocalAcceptance
    # whether move is possible
    acceptance::Int
    # selected particle to be moved
    particle::Int
    # the selected particle's spin
    spin::Int
    # initial site
    isite::Int
    # final site
    fsite::Int
end


"""
    local_acceptance()

Constructor for the local acceptance type.

"""
function local_acceptance()
   acceptance, particle, spin, isite, fsite =  metropolis(W, jastrow, particle_positions, rng)

   return LocalAcceptance(acceptance, particle, spin, isite, fsite)
end




"""
    metropolis( W, jastrow, particle_positions, rng )

Perform accept/reject step of proposed hop using the Metropolis algorithm. If move is
accepted, returns acceptance, particle β and it's spindex, initial position, and final
position.

"""
function metropolis(W, e_jastrow, κ, rng)    
    # create neighbor table
    nbr_table = build_neighbor_table(bonds[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice)

    # maps neighbor table to dictionary of bonds and neighbors                                
    nbrs = map_neighbor_table(nbr_table)

    # randomly select some particle in the lattice
    β = rand(rng, 1:trunc(Int,Ne))                   

    # spindex occupation number of particle β
    β_spindex = findfirst(x -> x == β, κ)

    # real position 'k' of particle 'β' 
    k = get_index_from_spindex(β_spindex, model_geometry)  
    
    # randomly selected neighboring site 'l'
    k_nbrs = nbrs[k][2]
    nbr_rand = rand(rng, 1:length(k_nbrs))
    l = k_nbrs[nbr_rand]          

    # spin of particle particle 'β' 
    β_spin = get_spindex_type(β_spindex, model_geometry)
    
    # checks occupation against spin species of particle 'β'
    # if site is unoccupied by same spin species, hop is possible
    if get_onsite_fermion_occupation(l,pconfig)[β_spin] == 1
        if debug == true
            println("Hop impossible! Rejected!")  
        end
        return LocalAcceptance(0, β, β_spin, k, l)
    else
        # begin Metropolis algorithm

        # get Jastrow ratio (element of T vector)
        Rⱼ = get_jastrow_ratio(k, l, e_jastrow, pht, β_spin)[1]    

        # get wavefunction ratio (correpsonding element of Green's function)
        Rₛ = W[l, β]  
                          
        acceptance_prob = Rⱼ^2 * Rₛ^2    

        if debug == true
            println("Hop possible! =>")
            println("Rⱼ = $Rⱼ")
            println("Rₛ = $Rₛ")
            println("accept prob. = $acceptance_prob")
        end

        if rand(rng) < acceptance_prob
            if debug 
                println("Hop accepted!")
            end
            
            return LocalAcceptance(1, β, β_spin, k, l)  # acceptance, particle number, particle spin, initial site, final site
        else
            if debug 
               println("Hop rejected!")
            end

            return LocalAcceptance(0, β, β_spin, k, l)
        end
    end
end


"""
    metropolis( jastrow, phconf, rng )

Perform accept/reject step of proposed addition/removal of a particle using the Metropolis algorithm. If move is
accepted, returns acceptance, site, and whether a particle was added or removed.

"""
function metropolis(phconfig::Vector{Int}, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
    # randomly select a site on the lattice
    N = model_geometry.lattice.N
    rnd_site = rand(rng, 1:N)

    # deciede whether a particle will be added or removed
    tryrand = rand(rng)
    cran = tryrand > 0.5 ? -1 : 1

    # if tryrand > 0.5
    #     sgn = 0
    # else
    #     sgn = 1
    # end

    # if sgn == 0
    #     cran = -1
    # else
    #     cran = 1
    # end

    if cran == -1 || phconfig[rnd_site] == 0
        # do nothing, since this is the floor of the ladder operator
        println("Removal impossible! Rejected!") 
    else
        # get phonon-phonon and electon-phonon Jastrow ratios
        Rⱼₚₕ = get_jastrow_ratio(k, l, ph_jastrow, pht, β_spin)[2]   
        Rⱼₑₚₕ = get_jastrow_ratio(k, l, eph_jastrow, pht, β_spin)[1]

        # get phononic amplitudes
        if cran == -1
            phnorm = sqrt(phconfig[rand_site])
        else
            phnorm = 1.0 / sqrt(phconfig[rand_site] + 1)
        end

        acceptance_prob = Rⱼₚₕ^2 * Rⱼₑₚₕ^2 * exp(2 * μₚₕ * cran) * phnorm^2

        if debug == true
            println("Change possible! =>")
            println("Rⱼₚₕ = $Rⱼₚₕ")
            println("Rⱼₑₚₕ = $Rⱼₑₚₕ")
            println("norm = $phnorm")
            println("accept prob. = $acceptance_prob")
        end

        if acceptance_prob > rand(rng)
            if debug 
                println("Change accepted!")
            end

            return 1, cran, rand_site
        else
            if debug 
                println("Change rejected!")
             end
 
             return 0, cran, rand_site
        end
    end
end


"""
    local_fermion_update!()

Performs a local MC update. Proposes moves and accept/rejects via Metropolis algorithm,
if accepted, updates particle positions, T vector, and Green's function (W matrix).

"""
function local_fermion_update!(W, D, model_geometry, e_jastrow, pconfig, κ, rng, n_iter, n_stab, mc_meas_freq)
    if debug
        println("Starting new Monte Carlo cycle...")
    end

    # counts number of proposed hops
    proposed_hops = 0
    # counts number of accepted hops
    accepted_hops = 0

    # perform number of metropolis steps equal to the number of electrons
    # for help in decorrelation
    for s in 1:mc_meas_freq
        if debug
            println("Metropolis step = $s")
        end
    
        # increment number of proposed hops
        proposed_hops += 1
    
        # Metropolis step
        met_step = metropolis(W, e_jastrow, κ, rng)    
    
        # whether hop was accepted
        acceptance = met_step.acceptance
    
        # DEBUG
        if debug
            prop_particle = met_step.particle
            prop_spin = met_step.spin
            prop_isite = met_step.isite
            prop_fsite = met_step.fsite
    
            println("Particle: $prop_particle")
            println("Spin: $prop_spin")
            println("initial site: $prop_isite")
            println("final site: $prop_fsite")
    
            @info "Before update:"
            # @info "particle_positions: $particle_positions"
            @info "pconfig: $pconfig"
        end
    
        # if hop is accepted 
        if acceptance == 1
            accepted_hops += 1
    
            # perform hop   
            do_particle_hop!(met_step, pconfig, κ, model_geometry)              
    
            # update Green's function
            update_equal_greens!(met_step, W)   
    
            # update T vector
            update_Tvec!(met_step, e_jastrow, model_geometry, pht)  
        end
        # DEBUG
        if debug
            @info "After update:"
            @info "pconfig: $pconfig"
        end
    end

    
    # # check for numerical stability after a certain number of iterations
    # if n_iter % n_stab == 0
    #     # check stability of Green's function 
    #     (W, D) = recalc_equal_greens(W, δW, D, pconfig)

    #     # check stability of T vector
    #     recalc_Tvec!(jastrow::Jastrow, δT::Float64)
    # end

    # # compute local acceptance rate
    local_acceptance_rate = accepted_hops / proposed_hops     

    return pconfig, κ, e_jastrow, W, D #local_acceptance_rate, 
end


"""
    local_boson_update!( phconfig::Vector{Int64}, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro )

Performs a local MC update. Proposes moves and accept/rejects via Metropolis algorithm,
if accepted, updates phonon configurations and phonon number.

"""
function local_boson_update!(phconfig::Vector{Int64}, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
    # counts number of proposed additions/removals
    proposed_addrms = 0
    # counts number of accepted additions/removals
    accepted_addrms = 0

    for s in 1:mc_meas_freq
        if debug
            println("Metropolis step = $s")
        end

        # increment number of proposed additions/removals
        proposed_addrms += 1

        # Metropolis step
        met_step = metropolis(phconfig, ph_jastrow, eph_jastrow, model_geometry, rng)    

        # whether change was accepted
        acceptance = met_step.acceptance

        if acceptance == 1
            accepted_addrms += 1
    
            # perform change
            change_particle_number!(met_step, phconfig, model_geometry)                   
    
            # update T vector
            update_Tvec!(met_step, ph_jastrow, model_geometry, pht)  
            update_Tvec!(met_step, eph_jastrow, model_geometry, pht)  
        end
    end

    return nothing
end


"""
    local_boson_update!( )

Performs a local MC update. Proposes displacements and accept/rejects via Metropolis algorithm,
if accepted, updates phonon configurations and phonon displacements.

"""
function local_boson_update!(phconfig::Matrix{AbstractFloat}, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
    # counts number of proposed additions/removals
    proposed_disps = 0
    # counts number of accepted additions/removals
    accepted_disps = 0

    for s in 1:mc_meas_freq
        if debug
            println("Metropolis step = $s")
        end

        # increment number of proposed additions/removals
        proposed_disps += 1

        # Metropolis step
        met_step = metropolis(phconfig, ph_jastrow, eph_jastrow, model_geometry, rng)    

        # whether change was accepted
        acceptance = met_step.acceptance

        if acceptance == 1
            accepted_disps += 1
    
            # perform change
            update_displacements!(met_step, phconfig, model_geometry)                   
    
            # update T vector
            update_Tvec!(met_step, ph_jastrow, model_geometry, pht)  
            update_Tvec!(met_step, eph_jastrow, model_geometry, pht)  
        end
    end

    return nothing
end