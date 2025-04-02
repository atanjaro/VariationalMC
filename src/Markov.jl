"""

    local_fermion_update!( detwf::DeterminantalWavefunction, jastrow::Jastrow, 
                                Ne::Int, model_geometry::ModelGeometry, 
                                pht::Bool, δW::Float64, δT::Float64, rng::Xoshiro )

Performs a local update to the electron sector, with a certain number of equilibration steps.

"""
function local_fermion_update!(detwf::DeterminantalWavefunction, jastrow::Jastrow, 
                                Ne::Int, model_geometry::ModelGeometry, 
                                pht::Bool, δW::Float64, δT::Float64, rng::Xoshiro)
    # track acceptances and rejections
    acceptances = 0.0
    rejections = 0.0

    debug && println("Markov:local_fermion_update!() : Starting new Monte Carlo step")
    debug && println("Markov:local_fermion_update!() : Metropolis step = ", mc)

    # TODO: perform number of Metropolis steps equal to number of electrons?
    #       This would give each electron a chance to move.
    # for s in 1:Ne
    #     acceptance = metropolis_step(detwf, jastrow, Ne, n_stab_W, n_stab_T, 
    #                                     δW, δT, model_geometry, pht, rng)

    #     if acceptance == "accepted"
    #         acceptances += 1.0
    #     elseif acceptance == "rejected"
    #         rejections += 1.0
    #     end
    # end

    acceptance = metropolis_step(detwf, jastrow, Ne, n_stab_W, n_stab_T, 
                                        δW, δT, model_geometry, pht, rng)

    if acceptance == "accepted"
        acceptances += 1.0
    elseif acceptance == "rejected"
        rejections += 1.0
    end

    # calculate local acceptance rate
    acceptance_rate = acceptances / (acceptances + rejections)

    return acceptance_rate, detwf, jastrow
end


"""

    metropolis_step( detwf::DeterminantalWavefunction, jastrow::Jastrow, Ne::Int, 
                        model_geometry::ModelGeometry, bonds::Vector{Vector{Bond{1}}}, pht::Bool, rng::Xoshiro )::String

Proposes a Markov move and then accepts or rejects using the Metropolis algorithm. 

"""
function metropolis_step(detwf::DeterminantalWavefunction, jastrow::Jastrow, Ne::Int, 
                        n_stab_W::Int64, n_stab_T::Int64, δW::Float64, δT::Float64, 
                        model_geometry::ModelGeometry, pht::Bool, rng::Xoshiro)::String
    # propose a Markov move
    markov_move = propose_random_move(Ne, detwf.pconfig, model_geometry, rng)

    if markov_move.possible == false
        debug && println("Markov::metropolis_step() : hop impossible!")

        return "impossible"
    else
        # check particle spin
        spin = get_spindex_type(markov_move.k, model_geometry)

        # get Jastrow ratio (element of T vector)
        Rⱼ = get_fermionic_jastrow_ratio(markov_move, jastrow, pht, spin, model_geometry)

        # get wavefunction ratio (element of the equal-time Green's function)
        Rₛ = real(detwf.W[markov_move.l, markov_move.particle])

        # acceptance probability
        acceptance_prob = Rⱼ^2 * Rₛ^2       # TODO: include phase

        debug && println("Markov::metropolis_step() : hop possible =>")
        debug && println("R_j = ", Rⱼ)
        debug && println("R_s = ", Rₛ)
        debug && println("acceptance_prob = ", acceptance_prob)

        if acceptance_prob > rand(rng)
            debug && println("Markov::metropolis_step() : hop accepted!")

            # hop the particle
            hop!(markov_move, detwf.pconfig)

            # perform rank-1 update to W matrix
            update_equal_time_greens!(markov_move, detwf, model_geometry, Ne, n_stab_W, δW) 

            # update T vector
            update_fermionic_Tvec!(markov_move, spin, jastrow, model_geometry, n_stab_T, δT, pht)

            return "accepted"
        else
            debug && println("Markov::metropolis_step() : hop rejected!")

            return "rejected"
        end
    end
end


##################################################### DEPRECATED FUNCTIONS #####################################################
# """
#     local_fermion_update!( W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, model_geometry::ModelGeometry, jastrow::Jastrow,
#                                 pconfig::Vector{Int64}, rng::Xoshiro, n_stab::Int64, mc_meas_freq::Int64 )

# Performs a local MC update. Proposes moves and accept/rejects via Metropolis algorithm,
# if accepted, updates particle positions, T vector, and Green's function (W matrix).

# """
# function local_fermion_update!(W::Matrix{ComplexF64}, D::Matrix{ComplexF64}, model_geometry::ModelGeometry, jastrow::Jastrow,
#                                 pconfig::Vector{Int64}, rng::Xoshiro, n_stab::Int64, mc_meas_freq::Int64, Ne)
#     if debug
#         println("Starting new Monte Carlo cycle...")
#     end

#     # track acceptance and rejections
#     acceptances = 0.0
#     rejections = 0.0

#     # perform thermalization updates
#     for s in 1:mc_meas_freq
#         if debug
#             println("Metropolis step = $s")
#         end
    
#         # Metropolis step
#         markov_move = metropolis(W, jastrow, pconfig, rng)    
    
#         # whether hop was accepted
#         acceptance = markov_move.acceptance

#         # if hop is accepted 
#         if acceptance == 1
#             acceptances += 1.0

#             # perform hop   
#             do_particle_hop!(markov_move, pconfig, model_geometry)              
    
#             # update Green's function
#             W = update_equal_greens(markov_move, W)  

#             # update T vector
#             update_fermionic_Tvec!(markov_move.isite, markov_move.fsite, markov_move.spin, jastrow, model_geometry)
#         end

#         if acceptance == 0
#             rejections += 1.0
#         end

#         # check for numerical stability after a certain number of iterations
#         if s % n_stab == 0
#             # check stability of Green's function 
#             (W, D) = recalc_equal_greens(W, δW, D, pconfig, Ne)
            
#             # check stability of the T vector
#             recalc_fermionic_Tvec!(jastrow, δT)
#         end
#     end

#     # compute local acceptance rate
#     local_acceptance_rate = acceptances / (acceptances + rejections)    

#     return local_acceptance_rate, W, D, pconfig, jastrow
# end



# """
#     local_boson_update!( phconfig::Vector{Int64}, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro )

# Performs a local MC update. Proposes moves and accept/rejects via Metropolis algorithm,
# if accepted, updates phonon configurations and phonon number.

# """
# function local_boson_update!(holstein_model::HolsteinModel, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
#     # counts number of proposed additions/removals
#     proposed_addrms = 0
#     # counts number of accepted additions/removals
#     accepted_addrms = 0

#     # Holstein model fugacity
#     μₚₕ = holstein_model.μₚₕ

#     for s in 1:mc_meas_freq
#         # current phonon density configuration
#         phconfig = holstein_model.phconfig

#         if debug
#             println("Metropolis step = $s")
#         end

#         # increment number of proposed additions/removals
#         proposed_addrms += 1

#         # Metropolis step
#         met_step = metropolis(phconfig, μₚₕ, eph_jastrow, model_geometry, rng)    

#         # whether change was accepted
#         acceptance = met_step[1]

#         if acceptance == 1
#             accepted_addrms += 1
    
#             # perform change
#             change_particle_number!(met_step, phconfig, model_geometry)                
            
#             # update electron-phonon model
#             update_electron_phonon_model!(holstein_model, phconfig, model_geometry)
    
#             # update T vector
#             update_Tvec!(met_step, eph_jastrow, model_geometry, pht)  
#         end
#     end

#     return nothing
# end


# """
#     local_boson_update!( phconfig::Vector{Int64}, eph_jastrow::Jastrow, ph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro )

# Performs a local MC update. Proposes addition or removal of particle and accept/rejects via Metropolis algorithm,
# if accepted, updates phonon configurations and phonon number.

# """
# function local_boson_update!(holstein_model::HolsteinModel, eph_jastrow::Jastrow, ph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
#     # counts number of proposed additions/removals
#     proposed_addrms = 0
#     # counts number of accepted additions/removals
#     accepted_addrms = 0

#     # Holstien model fugacity
#     μₚₕ = holstein_model.μₚₕ

#     for s in 1:mc_meas_freq
#         # current phonon density configuration
#         phconfig = holstein_model.phconfig

#         if debug
#             println("Metropolis step = $s")
#         end

#         # increment number of proposed additions/removals
#         proposed_addrms += 1

#         # Metropolis step
#         met_step = metropolis(phconfig, μₚₕ, ph_jastrow, eph_jastrow, model_geometry, rng)    

#         # whether change was accepted
#         acceptance = met_step[1]

#         if acceptance == 1
#             accepted_addrms += 1
    
#             # perform change
#             change_particle_number!(met_step, phconfig, model_geometry)      
            
#             # update electron-phonon model
#             update_electron_phonon_model!(holstein_model, phconfig, model_geometry)
    
#             # update T vectors
#             update_Tvec!(met_step, ph_jastrow, model_geometry, pht)  
#             update_Tvec!(met_step, eph_jastrow, model_geometry, pht)  
#         end
#     end

#     return nothing
# end


# """
#     local_boson_update!( ssh_model::SSHModel, eph_jastrow::Jastrow, ph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro )

# Performs a local MC update. Proposes displacements and accept/rejects via Metropolis algorithm,
# if accepted, updates phonon configurations and phonon displacements.

# """
# function local_boson_update!(ssh_model::SSHModel, eph_jastrow::Jastrow, ph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
#     # counts number of proposed additions/removals
#     proposed_disps = 0
#     # counts number of accepted additions/removals
#     accepted_disps = 0

#     # SSH fugacities
#     z_x = ssh_model.z_x
#     z_y = ssh_model.z_y

#     for s in 1:mc_meas_freq
#         # current phonon displacement configuration
#         phconfig = ssh_model.phconfig

#         if debug
#             println("Metropolis step = $s")
#         end

#         # increment number of proposed additions/removals
#         proposed_disps += 1

#         # Metropolis step
#         met_step = metropolis(phconfig, z_x, z_y, ph_jastrow, eph_jastrow, model_geometry, rng)    

#         # whether change was accepted
#         acceptance = met_step[1]

#         if acceptance == 1
#             accepted_disps += 1
    
#             # perform change
#             chnage_displacements!(met_step, phconfig, model_geometry)                   
    
#             # update T vector
#             update_Tvec!(met_step, ph_jastrow, model_geometry, pht)  
#             update_Tvec!(met_step, eph_jastrow, model_geometry, pht)  
#         end
#     end

#     return nothing
# end

# """
#     metropolis( W::Matrix{ComplexF64}, jastrow::Jastrow, pconfig, rng::Xoshiro )

# Perform accept/reject step of proposed hop using the Metropolis algorithm. If move is
# accepted, returns acceptance, particle β and it's spindex, initial position, and final
# position.

# """
# function metropolis(W::Matrix{ComplexF64}, jastrow::Jastrow, pconfig::Vector{Int64}, rng::Xoshiro)    
#     # create neighbor table
#     nbr_table = build_neighbor_table(bonds[1],
#                                     model_geometry.unit_cell,
#                                     model_geometry.lattice)

#     # maps neighbor table to dictionary of bonds and neighbors                                
#     nbrs = map_neighbor_table(nbr_table)

#     # randomly select some particle in the lattice
#     β = rand(rng, 1:trunc(Int,Ne))                   

#     # spindex occupation number of particle β
#     β_spindex = findfirst(x -> x == β, pconfig)

#     # real position 'k' of particle 'β' 
#     k = get_index_from_spindex(β_spindex, model_geometry)  
    
#     # randomly selected neighboring site 'l'
#     k_nbrs = nbrs[k][2]
#     nbr_rand = rand(rng, 1:length(k_nbrs))
#     l = k_nbrs[nbr_rand]          

#     # spin of particle particle 'β' 
#     β_spin = get_spindex_type(β_spindex, model_geometry)
    
#     # checks occupation against spin species of particle 'β'
#     # if site is unoccupied by same spin species, hop is possible
#     if get_onsite_fermion_occupation(l,pconfig)[β_spin] == 1
#         if debug == true
#             println("Hop impossible! Rejected!")  
#         end
#         return LocalAcceptance(0, β, β_spin, k, l)
#     else
#         # begin Metropolis algorithm

#         # get Jastrow ratio (element of T vector)
#         Rⱼ = get_fermionic_jastrow_ratio(k,l, jastrow, pht, β_spin, model_geometry)   

#         # get wavefunction ratio (correpsonding element of Green's function)
#         Rₛ = real(W[l, β])  
                          
#         acceptance_prob = Rⱼ^2 * Rₛ^2    

#         if debug == true
#             println("Hop possible! =>")
#             println("Rⱼ = $Rⱼ")
#             println("Rₛ = $Rₛ")
#             println("accept prob. = $acceptance_prob")
#         end

#         # if rand(rng) < acceptance_prob
#         if acceptance_prob > rand(rng)
#             if debug 
#                 println("Hop accepted!")
#             end
            
#             return LocalAcceptance(1, β, β_spin, k, l)  # acceptance, particle number, particle spin, initial site, final site
#         else
#             if debug 
#                println("Hop rejected!")
#             end

#             return LocalAcceptance(0, β, β_spin, k, l)
#         end
#     end
# end


# """
#     metropolis( phconfig::Vector{Int}, μₚₕ::AbstractFloat, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro )

# Perform accept/reject step of proposed addition/removal of a particle using the Metropolis algorithm. If move is
# accepted, returns acceptance, site, and whether a particle was added or removed.

# """
# function metropolis(phconfig::Vector{Int}, μₚₕ::AbstractFloat, jastrow_eph::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
#     # randomly select a site on the lattice
#     N = model_geometry.lattice.N
#     i = rand(rng, 1:N)

#     # decide whether a particle will be added or removed
#     tryrand = rand(rng)
#     cran = tryrand > 0.5 ? -1 : 1

#     # if tryrand > 0.5
#     #     sgn = 0
#     # else
#     #     sgn = 1
#     # end

#     # if sgn == 0
#     #     cran = -1
#     # else
#     #     cran = 1
#     # end

#     if cran == -1 && phconfig[i] == 0
#         # do nothing, since this is the floor of the ladder operator
#         println("Removal impossible! Rejected!") 
#     else
#         # get electon-phonon Jastrow ratio  # TODO: need to redo the Jastrow ratios here
#         Rⱼₑₚₕ = get_jastrow_ratio(i, cran, jastrow_eph, phconfig, μₚₕ)

#         # get phononic amplitudes
#         if cran == -1
#             phnorm = sqrt(phconfig[i])
#         else
#             phnorm = 1.0 / sqrt(phconfig[i] + 1)
#         end

#         acceptance_prob = Rⱼₑₚₕ^2 * exp(2 * μₚₕ * cran) * phnorm^2

#         if debug == true
#             println("Change possible! =>")
#             println("Rⱼₑₚₕ = $Rⱼₑₚₕ")
#             println("norm = $phnorm")
#             println("accept prob. = $acceptance_prob")
#         end

#         if acceptance_prob >= 1.0 || rand(rng) < acceptance_prob
#             if debug 
#                 println("Change accepted!")
#             end

#             return 1, cran, i
#         else
#             if debug 
#                 println("Change rejected!")
#              end
 
#              return 0, cran, i
#         end
#     end
# end


# """
#     metropolis( phconfig::Vector{Int}, μₚₕ::AbstractFloat, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro )

# Perform accept/reject step of proposed addition/removal of a particle using the Metropolis algorithm. If move is
# accepted, returns acceptance, site, and whether a particle was added or removed.

# """
# function metropolis(phconfig::Vector{Int}, μₚₕ::AbstractFloat, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
#     # randomly select a site on the lattice
#     N = model_geometry.lattice.N
#     i = rand(rng, 1:N)

#     # decide whether a particle will be added or removed
#     tryrand = rand(rng)
#     cran = tryrand > 0.5 ? -1 : 1

#     # if tryrand > 0.5
#     #     sgn = 0
#     # else
#     #     sgn = 1
#     # end

#     # if sgn == 0
#     #     cran = -1
#     # else
#     #     cran = 1
#     # end

#     if cran == -1 && phconfig[i] == 0
#         # do nothing, since this is the floor of the ladder operator
#         println("Removal impossible! Rejected!") 
#     else
#         # get phonon-phonon and electon-phonon Jastrow ratios  # TODO: need to redo the Jastrow ratios here
#         Rⱼₚₕ = get_jastrow_ratio(k, l, ph_jastrow, pht, β_spin)[2]   
#         Rⱼₑₚₕ = get_jastrow_ratio(k, l, eph_jastrow, pht, β_spin)[1]

#         # get phononic amplitudes
#         if cran == -1
#             phnorm = sqrt(phconfig[i])
#         else
#             phnorm = 1.0 / sqrt(phconfig[i] + 1)
#         end

#         acceptance_prob = Rⱼₚₕ^2 * Rⱼₑₚₕ^2 * exp(2 * μₚₕ * cran) * phnorm^2

#         if debug == true
#             println("Change possible! =>")
#             println("Rⱼₚₕ = $Rⱼₚₕ")
#             println("Rⱼₑₚₕ = $Rⱼₑₚₕ")
#             println("norm = $phnorm")
#             println("accept prob. = $acceptance_prob")
#         end

#         if acceptance_prob > rand(rng)
#             if debug 
#                 println("Change accepted!")
#             end

#             return 1, cran, i
#         else
#             if debug 
#                 println("Change rejected!")
#              end
 
#              return 0, cran, i
#         end
#     end
# end



# function metropolis(phconfig::Matrix{AbstractFloat}, z_x::AbstractFloat, z_y::AbstractFloat, ph_jastrow::Jastrow, eph_jastrow::Jastrow, model_geometry::ModelGeometry, rng::Xoshiro)
#     # randomly select a site on the lattice
#     N = model_geometry.lattice.N
#     i = rand(rng, 1:N)

#     # # or, if bond SSH
#     # # randomly select a bond in the lattice
#     # rnd_bond = rand(rng, 2*N)

#     # maximum allowed displacement
#     Δ_max = 1.0

#     # number of displacements #TODO: what would be a good number be here?
#     n_disps = 1_000_000

#     # uniform range of displacements
#     Δ_range = range(-Δ_max, Δ_max, length=n_disps)

#     # randomly select a displacement from the range
#     Δ = rand(rng, Δ_range)

#     # get phonon-phonon and electon-phonon Jastrow ratios
#     Rⱼₚₕ = get_jastrow_ratio(k, l, ph_jastrow, pht, β_spin)[2]   
#     Rⱼₑₚₕ = get_jastrow_ratio(k, l, eph_jastrow, pht, β_spin)[1]

#     # get real position of random site
#     r_i = loc_to_pos(i, unit_cell)

#     # get X and and Y displacements of the random site
#     disp_rand_site = get_onsite_phonon_displacement(i,phconfig)

#     # phonon amplitude functions
#     ϕ_x(k, r, z_x, X_i) = im(z_x * sin(k * r)) - 0.25 * (X_i - 2 * z_x * cos(k * r))^2
#     ϕ_y(k, r, z_y, Y_i) = im(z_y * sin(k * r)) - 0.25 * (Y_i - 2 * z_y * cos(k * r))^2

#     # TODO: need to get r from site_to_loc
#     # TODO: do this for all BZ k-points or just a k-point of interest?

#     # calculate all k-points 
#     k_points = calc_k_points(unit_cell, lattice)

#     # phonon Gaussian functions
#     gauss_x = ϕ_x(k, r_i, z_x, disp_rand_site[1])
#     gauss_y = ϕ_x(k, r_i, z_y, disp_rand_site[2])

#     # phonon coherent states
#     coherent_x = exp(ϕ_x)
#     coherent_y = exp(ϕ_y) 

#     acceptance_prob = Rⱼₚₕ^2 * Rⱼₑₚₕ^2 * gaussian_x * gaussian_y

#     if debug == true
#         println("Displacement possible! =>")
#         println("Rⱼₚₕ = $Rⱼₚₕ")
#         println("Rⱼₑₚₕ = $Rⱼₑₚₕ")
#         println("coherent_x = $coherent_x")
#         println("coherent_y = $coherent_y")
#         println("accept prob. = $acceptance_prob")
#     end

#     if acceptance_prob > rand(rng)
#         if debug 
#             println("Displacement accepted!")
#         end

#         return 1, Δ, rand_site
#     else
#         if debug 
#             println("Displacement rejected!")
#          end

#          return 0, Δ, rand_site
#     end
# end