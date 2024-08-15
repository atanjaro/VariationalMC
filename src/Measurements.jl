"""

    initialize_measurement_container( model_geometry::ModelGeometry0 )

Creates dictionaries of generic arrays for storing measurements. Each dictionary in the container
has (keys => values): observable_name => local_values (i.e. ∑ O_loc(x)), binned_values (i.e. ⟨O⟩≈(N)⁻¹local_values)

"""
function initialize_measurement_container(model_geometry::ModelGeometry, variational_parameters, Np, N_equil, N_bins, bin_size)
    # one side of the lattice
    L = model_geometry.lattice.L

    # total number of lattice sites
    N = model_geometry.lattice.N

    # number of variational parameters to be optimized
    num_vpars = length(variational_parameters)

    # number of initial visited configurations
    N_configs = N_equil * Np

    # dictionary to store scalar measurements
    scalar_measurements = Dict{String, Any}([
        ("density", 0.0),        # average density per orbital species
        ("double_occ", 0.0),     # average double occupancy per orbital species
        ("parameters", (variational_parameters, [[zeros(num_vpars) for _ in 1:bin_size] for _ in 1:N_bins])),       # variational parameters
        ("energy", (0.0,  [zeros(bin_size) for _ in 1:N_bins])),          # energy
        ("pconfig", (0.0, [zeros(bin_size) for _ in 1:N_bins]))           # particle configurations
    ])                     

    # dictionary to store derivative measurmements
    derivative_measurements = Dict{String, Any}([
        ("Δk", (zeros(num_vpars), [[zeros(num_vpars) for _ in 1:bin_size] for _ in 1:N_bins])),               
        ("ΔkΔkp", (zeros(num_vpars,num_vpars), [[zeros(num_vpars,num_vpars) for _ in 1:bin_size] for _ in 1:N_bins])),            
        ("ΔkE", (zeros(num_vpars), [[zeros(num_vpars) for _ in 1:bin_size] for _ in 1:N_bins]))      
    ])      

    # dictionary to store correlation measurements
    correlation_measurements = Dict{String, Any}()

    # create container
    measurement_container = (
        scalar_measurements       = scalar_measurements,
        derivative_measurements   = derivative_measurements,         
        correlation_measurements  = correlation_measurements,                       
        L                         = L,
        N                         = N,
        N_configs                 = N_configs,
        N_bins                    = N_bins,
        bin_size                  = bin_size,
        num_vpars                 = num_vpars                      
    )

    return measurement_container
end


"""

    initialize_measurements!( measurement_container, observable::String )

For a certain type of measurment (scalar or correlation), initializes the arrays
necessary to store measurements in respective bins.

"""
function initialize_measurements!(measurement_container, observable)
    (; scalar_measurements, N) = measurement_container

    if observable == "energy"
        # initialize with the initial energy per site
        init_energy = sum(ε₀) / N

        # update the energy container
        init_econt = measurement_container.scalar_measurements["energy"]
        new_econt = (init_energy, init_econt[2])
        measurement_container.scalar_measurements["energy"] = new_econt
    elseif observable == "site_dependent_density"
        scalar_measurements["site_dependent_density"] = (zeros(AbstractFloat, norbs*N),Vector{Vector{AbstractFloat}}(undef, N_iter))    # charge stripe 
    elseif observable == "site_dependent_spin"
        scalar_measurements["site_dependent_spin"] = (zeros(AbstractFloat, norbs*N),Vector{Vector{AbstractFloat}}(undef, N_iter))       # spin stripe
    elseif observable == "phonon_number"
        scalar_measurements["phonon_number"] = (0.0,  Vector{AbstractFloat}(undef, N_iter))                     # phonon number nₚₕ
    elseif observable == "displacement"
        scalar_measurements["displacement"] = (0.0,  Vector{AbstractFloat}(undef, N_iter))                      # phonon displacement X
    end

    return nothing
end


# TODO: implement correlation measurmenets
function initialize_correlation_measurements!(measurement_container,  correlation)
    (; correlation_measurements) = measurement_container

    if correlation == "density-density"
        correlation_measurements["density-density"] = Vector{Vector{AbstractFloat}}(undef, N_iter)
    elseif correlation == "spin-spin"
        correlation_measurements["spin-spin"] = Vector{Vector{AbstractFloat}}(undef, N_iter)
    elseif correlation == "pair"
        correlation_measurements["pair"] = Vector{Vector{AbstractFloat}}(undef, N_iter)
    end

    return nothing
end


"""

    measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, Np, W, A)

Measures logarithmic derivatives for all variational parameters. The first 'p' are derivatives of 
determinantal parameters and the rest are derivatives of Jastrow parameters. Measurments are then written
to the measurement container.

"""
function measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions, Np, W, A, n, bin, N_configs)
    # perform parameter derivatives
    detpar_derivatives = get_local_detpar_derivative(determinantal_parameters, model_geometry, particle_positions, Np, W, A)
    jpar_derivatives = get_local_jpar_derivative(jastrow, pconfig, pht)
    Δk = vcat(detpar_derivatives,jpar_derivatives)

    # current values
    current_container = measurement_container.derivative_measurements["Δk"]

    # update the local value 
    new_local_value = current_container[1] .+ Δk

    # update the current expectation values 
    new_expectation_value = new_local_value / N_configs
    current_expectation_value = current_container[2]
    current_expectation_value[bin][n] .= new_expectation_value

    # combine the updated values 
    updated_values = (new_local_value, current_expectation_value)

    # write the new values to the container
    measurement_container.derivative_measurements["Δk"] = updated_values

    return nothing
end


"""
    measure_ΔkE( determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, Np, W, A )

Measures the product of variational derivatives with the local energy. Measurments are then written
to the measurement container.

"""
# PASSED
function measure_ΔkE!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, particle_positions,  Np, W, A, n, bin, N_configs)
    # perform derivatives
    detpar_derivatives = get_local_detpar_derivative(determinantal_parameters, model_geometry, particle_positions, Np, W, A)
    jpar_derivatives = get_local_jpar_derivative(jastrow,pconfig,pht)
    Δk = vcat(detpar_derivatives,jpar_derivatives)

    # compute local energy
    E = get_local_energy(model_geometry, tight_binding_model, jastrow, pconfig, particle_positions) 

    # compute product of local derivatives with the local energy
    ΔkE = Δk * E

    # current values
    current_container = measurement_container.derivative_measurements["ΔkE"]

    # update the local value 
    new_local_value = current_container[1] .+ ΔkE

    # update the current expectation values 
    N_configs = n + N_equil * Np
    new_expectation_value = new_local_value / N_configs
    current_expectation_value = current_container[2]
    current_expectation_value[bin][n] .= new_expectation_value

    # combine the updated values 
    updated_values = (new_local_value, current_expectation_value)

    # write the new values to the container
    measurement_container.derivative_measurements["ΔkE"] = updated_values

    return nothing
end


"""
    measure_ΔkΔkp(  )

Measures the product of variational derivatives with other variational derivatives. Measurments are then written
to the measurement container.

"""
function measure_ΔkΔkp!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions, Np, W, A, n, bin, N_configs)
    # perform derivatives
    detpar_derivatives = get_local_detpar_derivative(determinantal_parameters, model_geometry, particle_positions, Np, W, A)
    jpar_derivatives = get_local_jpar_derivative(jastrow,pconfig, pht)
    Δk = vcat(detpar_derivatives,jpar_derivatives)

    # inner product of Δk and Δk′
    ΔkΔkp = Δk .* Δk'

    # current values
    current_container = measurement_container.derivative_measurements["ΔkΔkp"]

    # update the local value 
    new_local_value = current_container[1] .+ ΔkΔkp

    # update the current expectation values 
    new_expectation_value = new_local_value / N_configs
    current_expectation_value = current_container[2]
    current_expectation_value[bin][n] = new_expectation_value

    # combine the updated values 
    updated_values = (new_local_value, current_expectation_value)

    # write the new values to the container
    measurement_container.derivative_measurements["ΔkΔkp"] = updated_values

    return nothing
end 


"""
    get_local_energy(model_geometry::ModelGeometry, tight_binding_model::TightBindingModel, jastrow::Jastrow, particle_positions:: )

Calculates the local variational energy. Returns the total local energy and writes to the measurement container.

"""
function get_local_energy(model_geometry, tight_binding_model, jastrow, pconfig, particle_positions)

    N = model_geometry.lattice.N

    E_k = get_local_kinetic_energy(model_geometry, tight_binding_model, jastrow, pconfig, particle_positions)

    E_hubb = get_local_hubbard_energy(U, model_geometry, pconfig)
    
    # calculate total local energy
    E_loc = E_k + E_hubb

    return E_loc / N
end



function get_local_kinetic_energy(model_geometry, tight_binding_model, jastrow, pconfig, particle_positions)
    # number of sites
    N = model_geometry.lattice.N

    # generate neighbor table
    nbr_table = build_neighbor_table(bonds[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice)

    # generate neighbor map
    nbr_map = map_neighbor_table(nbr_table)

    # track kinetic energy
    E_loc_kinetic = 0.0

    # calculate electron kinetic energy
    for β in 1:Np
        # position of particle β
        k = particle_positions[β][2]

        # spin of particle particle 'β' 
        spindex = particle_positions[β][1]
        β_spin = get_spindex_type(spindex, model_geometry)
      
        # loop over nearest neighbors. TODO: add next-nearest neighbors
        sum_nn = 0.0
        for l in nbr_map[k][2]
            # check that neighboring sites are unoccupied
            if number_operator(l, pconfig)[β_spin] == 0
                Rⱼ = get_jastrow_ratio(l, k, jastrow, pht, β_spin)
                sum_nn += Rⱼ * W[l, β]
            end
        end

        # calculate kinetic energy
        if pht 
            if β_spin == 1
                E_loc_kinetic += -tight_binding_model.t[1] * sum_nn        
            else
                E_loc_kinetic += tight_binding_model.t[1] * sum_nn 
            end
        else
            E_loc_kinetic += -tight_binding_model.t[1] * sum_nn
        end
    end

    return E_loc_kinetic
end


function get_local_hubbard_energy(U, model_geometry, pconfig)
    # number of sites
    N = model_geometry.lattice.N

    # track hubbard energy
    dblocc_sum = 0.0

    # calculate Hubbard energy
    for i in 1:N
        if pht
            dblocc_sum += number_operator(i, pconfig)[1] * (1 - number_operator(i, pconfig)[2])
        else
            dblocc_sum += number_operator(i, pconfig)[1] * number_operator(i, pconfig)[2]
        end
    end

    E_loc_hubbard = U * dblocc_sum

    return E_loc_hubbard
end


"""
    measure_local_energy!( measurement_container, model_geometry::ModelGeometry, tight_binding_model::TightBindingModel, 
                jastrow::Jastrow, particle_positions:: )

Measures the total local energy and writes to the measurement container.

"""
# PASSED
function measure_local_energy!(measurement_container, model_geometry, tight_binding_model, jastrow, pconfig, particle_positions, n, bin, N_configs)
    # calculate the current local energy
    E_loc = get_local_energy(model_geometry, tight_binding_model, jastrow, pconfig, particle_positions)

    # current values
    current_container = measurement_container.scalar_measurements["energy"]

    # update the local value 
    new_local_value = current_container[1] + E_loc

    # update the current expectation values
    new_expectation_value = new_local_value / N_configs
    current_expectation_value = current_container[2]
    current_expectation_value[bin][n] = new_expectation_value

    # combine the updated values 
    updated_values = (new_local_value, current_expectation_value)

    # write the new values to the container
    measurement_container.scalar_measurements["energy"] = updated_values


    return nothing
end




"""
    measure_global_energy( model_geometry::ModelGeometry )

Measure the global variational energy ⟨E⟩. This is intended to used at the end of the simulation.

"""
function measure_global_energy(model_geometry, N_bins, bin_size)

    # binned energies
    # E_binned

    # average over all sites
    E_global = E_binned / model_geometry.lattice.N

    return E_global
end


"""
    measure_double_occ( model_geometry::ModelGeometry, pconfig::Vector{Int} )

Measure the average double occupancy ⟨D⟩ = N⁻¹ ∑ᵢ ⟨nᵢ↑nᵢ↓⟩.

"""
function measure_double_occ!(measurement_container, pconfig, model_geometry)
    nup_ndn = 0.0

    for i in 1:model_geometry.lattice.N
        nup_ndn += number_operator(i, pconfig)[1] * (1 - number_operator(i, pconfig)[2])
    end

    dblocc = nup_ndn / model_geometry.lattice.N

    # write to measurment container
    
    
    return nothing
end


"""
    measure_n( site::Int, pconfig::Vector{Int} )

Measure the local particle density ⟨n⟩.

"""
function measure_n(site)
    loc_den = number_operator(site, pconfig)[1] + 1 - number_operator(i, pconfig)[2]

    return loc_den
end


"""
    measure_ρ( site::int )

Measure the local excess hole density ⟨ρ⟩.

"""
function measure_ρ(site)
    ρ = 1 - measure_n(site)

    return ρ
end


"""
    measure_s( site::Int, pconfig::Vector{Int} )

Measure the local spin.

"""
function measure_s()
    loc_spn = number_operator(site,pconfig)[1] - 1 + number_operator(i,pconfig)[2]

    return loc_spn
end




"""
    measure_density_corr( )

Measure the density-density correlation function.

"""
function measure_density_corr()
    return nothing
end


"""
    measure_spin_corr( )

Measure the spin-spin correlation function.

"""
function measure_spin_corr()
    return nothing
end


# """
#     measure_average_X( )

# Measure average phonon displacement ⟨X⟩, in the SSH model (TBD).

# """
# function measure_average_X()
#     return nothing
# end


function write_measurements!()
    return nothing
end
