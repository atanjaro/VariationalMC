"""

    initialize_measurement_container( model_geometry::ModelGeometry0 )

Creates dictionaries of generic arrays for storing measurements. Each dictionary in the container
has (keys => values): observable_name => local_values (i.e. ∑ O_loc(x)), binned_values (i.e. ⟨O⟩≈(N)⁻¹local_values)

observable_name => (sum_of_local_vals, [opt_bin_size],...,[opt_bin_size])

observable_name => (sum_of_local_vals, [opt_bin_size],...(N_opts)...,[opt_bin_size], [bin_size],...(N_bins)...,[bin_size])

"""
function initialize_measurement_container(model_geometry::ModelGeometry, variational_parameters, N_opts, opt_bin_size, N_bins, bin_size)
    # one side of the lattice
    L = model_geometry.lattice.L

    # total number of lattice sites
    N = model_geometry.lattice.N

    # number of determinantal_parameters
    num_detpars = variational_parameters.num_detpars

    # number of Jastrow parameters
    num_jpars = variational_parameters.num_jpars

    # number of variational parameters to be optimized
    num_vpars = num_detpars + num_jpars

    # container to store optimization measurements
    optimization_measurements = Dict{String, Any}([
        ("parameters", (zeros(num_vpars), zeros(num_vpars))),                     # variational parameters
        ("Δk", (zeros(num_vpars), zeros(num_vpars),[])),                                                          # log derivative of variational parameters
        ("ΔkΔkp", (zeros(num_vpars,num_vpars), zeros(num_vpars,num_vpars),[])),                                   # product between log derivatives        
        ("ΔkE", (zeros(num_vpars), zeros(num_vpars),[])),                                                         # product of log derivatives and energy
    ])      

    # dictionary to store simulation measurements
    simulation_measurements = Dict{String, Any}([
        ("density", (0.0,  0.0)),          # average density
        ("double_occ", (0.0,  0.0)),       # average double occupancy 
        ("energy", (0.0,  0.0)),   # local energy
        ("pconfig", zeros(N))      # particle configurations
    ])                     

    # # dictionary to store derivative measurmements
    # derivative_measurements = Dict{String, Any}([
    #     ("Δk", (zeros(num_vpars), [[zeros(num_vpars) for _ in 1:bin_size] for _ in 1:N_bins])),               
    #     ("ΔkΔkp", (zeros(num_vpars,num_vpars), [[zeros(num_vpars,num_vpars) for _ in 1:bin_size] for _ in 1:N_bins])),            
    #     ("ΔkE", (zeros(num_vpars), [[zeros(num_vpars) for _ in 1:bin_size] for _ in 1:N_bins]))      
    # ])      

    # dictionary to store correlation measurements
    correlation_measurements = Dict{String, Any}()

    # create container
    measurement_container = (
        simulation_measurements   = simulation_measurements,
        optimization_measurements = optimization_measurements,         
        correlation_measurements  = correlation_measurements,                       
        L                         = L,
        N                         = N,
        N_opts                    = N_opts,
        opt_bin_size              = opt_bin_size,
        N_updates                 = N_updates,
        N_bins                    = N_bins,
        bin_size                  = bin_size,
        num_vpars                 = num_vpars,
        num_detpars               = num_detpars,
        num_jpars                 = num_jpars                      
    )

    return measurement_container
end


"""

    initialize_measurements!( measurement_container, observable::String )

For a certain type of measurment (simulation or correlation), initializes the arrays
necessary to store measurements in respective bins.

"""
function initialize_measurements!(measurement_container, observable)
    (; simulation_measurements, N) = measurement_container

    if observable == "energy"
        # initialize with the initial energy per site
        init_energy = sum(ε₀) / N

        # update the energy container
        init_econt = measurement_container.simulation_measurements["energy"]
        new_econt = (init_energy, init_econt[2])
        measurement_container.simulation_measurements["energy"] = new_econt
    elseif observable == "site_dependent_density"
        simulation_measurements["site_dependent_density"] = (zeros(AbstractFloat, norbs*N),Vector{Vector{AbstractFloat}}(undef, N_iter))    # charge stripe 
    elseif observable == "site_dependent_spin"
        simulation_measurements["site_dependent_spin"] = (zeros(AbstractFloat, norbs*N),Vector{Vector{AbstractFloat}}(undef, N_iter))       # spin stripe
    elseif observable == "phonon_number"
        simulation_measurements["phonon_number"] = (0.0,  Vector{AbstractFloat}(undef, N_iter))                     # phonon number nₚₕ
    elseif observable == "displacement"
        simulation_measurements["displacement"] = (0.0,  Vector{AbstractFloat}(undef, N_iter))                      # phonon displacement X
    end

    return nothing
end


# TODO: implement correlation measurements
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

    make_measurements!(measurement_container,determinantal_parameters, jastrow, model_geometry,
                        tight_binding_model, pconfig, particle_positions,  Np, W, A, n, bin)

Measure the local energy and logarithmic derivatives for a particular bin.

"""
function make_measurements!(measurement_container,determinantal_parameters, jastrow, model_geometry, 
                            tight_binding_model, pconfig, κ, Np, W, A)

    # record current variational parameters
    parameters_current = all_vpars(determinantal_parameters, jastrow)

    # get current values from the container
    parameters_container = measurement_container.optimization_measurements["parameters"]

    # update value for the current bin
    current_parameters_bin = parameters_container[2]
    current_parameters_bin = parameters_current

    # update accumuator for this bin
    thisbin_parameters_sum = parameters_container[1]
    thisbin_parameters_sum += parameters_current

    # combine the updated values 
    updated_values = (thisbin_parameters_sum, current_parameters_bin)

    # write the new values to the container
    measurement_container.optimization_measurements["parameters"] = updated_values
    
    # measure the energy
    measure_local_energy!(measurement_container, model_geometry, tight_binding_model, jastrow, pconfig, κ)

    # measure the lograithmic derivatives
    measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry,pconfig, κ, Np, W, A)
    measure_ΔkΔkp!(measurement_container, determinantal_parameters,jastrow, model_geometry, pconfig, κ, Np,W,A)
    measure_ΔkE!(measurement_container,determinantal_parameters,jastrow,model_geometry,tight_binding_model,pconfig, κ,Np,W,A)

    # measure double occupancy
    measure_double_occ!(measurement_container, pconfig, model_geometry)

    # measure average density
    measure_n!(measurement_container, pconfig, model_geometry)

    # record the current configuration
    measurement_container.simulation_measurements["pconfig"] = pconfig

    return nothing
end


"""

    write_measurements!( measurement_container, simulation_info, bin )

Writes current measurements in the current bin to a JLD2 file 

"""
function write_measurements!(measurement_container, simulation_info)

    if debug

        push!(energy_bin, simulation_measurements["energy"][2])
        push!(dblocc_bin, simulation_measurements["double_occ"][2])


        # reset all bin values to zeroes



    else

        (; datafolder, pID) = simulation_info

        (; optimization_measurements, simulation_measurements, num_detpars) = measurement_container

        # construct filename
        fn = @sprintf "bin-%d_pID-%d.jld2" bin pID  
        file_path_detpar = joinpath(datafolder, "optimization", "determinantal", fn)        
        file_path_jpar = joinpath(datafolder, "optimization", "Jastrow", fn)
        file_path_energy = joinpath(datafolder, "simulation", "energy", fn)
        file_path_pconfig = joinpath(datafolder, "simulation", "configurations", fn)
        file_path_dblocc = joinpath(datafolder, "simulation", "double_occ", fn)
        file_path_density = joinpath(datafolder, "simulation", "density", fn)

        # Append determinantal parameter measurements to file
        detpar_measurements = optimization_measurements["parameters"][2][1:num_detpars]
        JLD2.@save file_path_detpar detpar_measurements append=true

        # Append Jastrow parameter measurements to file
        jpar_measurements = optimization_measurements["parameters"][2][num_detpars:end]
        JLD2.@save file_path_jpar jpar_measurements append=true

        # Append energy measurements to file
        energy_measurements = simulation_measurements["energy"][2]
        JLD2.@save file_path_energy energy_measurements append=true

        # Append particle configurations to file
        pconfig_measurements = simulation_measurements["pconfig"]
        JLD2.@save file_path_pconfig pconfig_measurements append=true

        # Append double occupancy to file
        dblocc_measurements = simulation_measurements["double_occ"][2]
        JLD2.@save file_path_dblocc dblocc_measurements append=true

        # Append average density to file
        density_measurements = simulation_measurements["density"][2]
        JLD2.@save file_path_density density_measurements append=true

        # reset current bin measurement to 0

    end

    return nothing
end

function initialize_measurement_directories(simulation_info::SimulationInfo, measurement_container::NamedTuple)

    (; datafolder, resuming, pID) = simulation_info
    (; optimization_measurements, simulation_measurements, correlation_measurements) = measurement_container

    # only initialize folders if pID = 0
    if iszero(pID) && !resuming

        # make optimization measurements directory
        optimization_directory = joinpath(datafolder, "optimization")
        mkdir(optimization_directory)

        # make simulation measurements directory
        simulation_directory = joinpath(datafolder, "simulation")
        mkdir(simulation_directory)

        # make global measurements directory
        global_directory = joinpath(datafolder, "global")
        mkdir(global_directory)

        # make directories for each parameter measurement
        detpars_directory = joinpath(optimization_directory, "determinantal")
        mkdir(detpars_directory)
        jpars_directory = joinpath(optimization_directory, "Jastrow")
        mkdir(jpars_directory)

        # make energy measurement directory
        energy_directory = joinpath(simulation_directory, "energy")
        mkdir(energy_directory)

        # make configuration measurement directory
        config_directory = joinpath(simulation_directory, "configurations")
        mkdir(config_directory)

        # make double occupancy measurement directory
        dblocc_directory = joinpath(global_directory, "double_occ")
        mkdir(dblocc_directory)

        # make density measurement directory
        density_directory = joinpath(global_directory, "density")
        mkdir(density_directory)

        # TODO: add correlation measurement directory initialization
    end

    return nothing
end


"""

    measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, Np, W, A)

Measures logarithmic derivatives for all variational parameters. The first 'p' are derivatives of 
determinantal parameters and the rest are derivatives of Jastrow parameters. Measurments are then written
to the measurement container.

"""
function measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, κ, Np, W, A)
    # perform parameter derivatives
    detpar_derivatives = get_local_detpar_derivative(determinantal_parameters, model_geometry, κ, Np, W, A)
    jpar_derivatives = get_local_jpar_derivative(jastrow, pconfig, pht)
    Δk_current = vcat(detpar_derivatives,jpar_derivatives)

    # get current values from the container
    Δk_container = measurement_container.optimization_measurements["Δk"]

    # update value for the current bin
    current_Δk_bin = Δk_container[2]
    current_Δk_bin = Δk_current

    # add to bin history
    bin_Δk_history = Δk_container[3]
    push!(bin_Δk_history, current_Δk_bin)

    # update accumuator for this bin
    thisbin_Δk_sum = Δk_container[1]
    thisbin_Δk_sum += Δk_current

    # combine the updated values 
    updated_values = (thisbin_Δk_sum, current_Δk_bin, bin_Δk_history)

    # write the new values to the container
    measurement_container.optimization_measurements["Δk"] = updated_values

    return nothing
end


"""
    measure_ΔkE!( determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, Np, W, A )

Measures the product of variational derivatives with the local energy. Measurments are then written
to the measurement container.

"""
# PASSED
function measure_ΔkE!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, κ,  Np, W, A)
    # perform derivatives
    detpar_derivatives = get_local_detpar_derivative(determinantal_parameters, model_geometry, κ, Np, W, A)
    jpar_derivatives = get_local_jpar_derivative(jastrow,pconfig,pht)
    Δk = vcat(detpar_derivatives,jpar_derivatives)

    # compute local energy
    E_loc = get_local_energy(model_geometry, tight_binding_model, jastrow, pconfig, κ) 

    # compute product of local derivatives with the local energy
    ΔkE_current = Δk * E_loc

    # get current values from the container
    ΔkE_container = measurement_container.optimization_measurements["ΔkE"]

    # update value for the current bin
    current_ΔkE_bin = ΔkE_container[2]
    current_ΔkE_bin = ΔkE_current

    # add to bin history
    bin_ΔkE_history = ΔkE_container[3]
    push!(bin_ΔkE_history, current_ΔkE_bin)

    # update accumuator for this bin
    thisbin_ΔkE_sum = ΔkE_container[1]
    thisbin_ΔkE_sum += ΔkE_current

    # combine the updated values 
    updated_values = (thisbin_ΔkE_sum, current_ΔkE_bin, bin_ΔkE_history)

    # write the new values to the container
    measurement_container.optimization_measurements["ΔkE"] = updated_values

    return nothing
end



"""
    measure_ΔkΔkp(  )

Measures the product of variational derivatives with other variational derivatives. Measurments are then written
to the measurement container.

"""
function measure_ΔkΔkp!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, κ, Np, W, A)
    # perform derivatives
    detpar_derivatives = get_local_detpar_derivative(determinantal_parameters, model_geometry, κ, Np, W, A)
    jpar_derivatives = get_local_jpar_derivative(jastrow,pconfig, pht)
    Δk = vcat(detpar_derivatives,jpar_derivatives)

    # inner product of Δk and Δk′
    ΔkΔkp_current = Δk .* Δk'

    # get current values from the container
    ΔkΔkp_container = measurement_container.optimization_measurements["ΔkΔkp"]

    # update value for the current bin
    current_ΔkΔkp_bin = ΔkΔkp_container[2]
    current_ΔkΔkp_bin = ΔkΔkp_current

    # add to bin history
    bin_ΔkΔkp_history = ΔkΔkp_container[3]
    push!(bin_ΔkΔkp_history, current_ΔkΔkp_bin)

    # update accumuator for this bin
    thisbin_ΔkΔkp_sum = ΔkΔkp_container[1]
    thisbin_ΔkΔkp_sum += ΔkΔkp_current

    # combine the updated values 
    updated_values = (thisbin_ΔkΔkp_sum, current_ΔkΔkp_bin, bin_ΔkΔkp_history)

    # write the new values to the container
    measurement_container.optimization_measurements["ΔkΔkp"] = updated_values

    return nothing
end 


"""
    measure_local_energy!( measurement_container, model_geometry::ModelGeometry, tight_binding_model::TightBindingModel, 
                jastrow::Jastrow, particle_positions:: )

Measures the total local energy and writes to the measurement container.

"""
function measure_local_energy!(measurement_container, model_geometry, tight_binding_model, jastrow, pconfig, κ)

   # calculate the current local energy
    E_loc_current = get_local_energy(model_geometry, tight_binding_model, jastrow, pconfig, κ)

    # get current values from the container
    energy_container = measurement_container.simulation_measurements["energy"]

    # update value for the current bin
    current_E_loc_bin = energy_container[2]
    current_E_loc_bin = E_loc_current

    # update accumuator for this bin
    thisbin_E_loc_sum = energy_container[1]
    thisbin_E_loc_sum += E_loc_current

    # combine the updated values 
    updated_values = (thisbin_E_loc_sum, current_E_loc_bin)

    # write the new values to the container
    measurement_container.simulation_measurements["energy"] = updated_values

    return nothing
end


"""
    measure_double_occ( )

Measure the average double occupancy ⟨D⟩ = N⁻¹ ∑ᵢ ⟨nᵢ↑nᵢ↓⟩.

"""
function measure_double_occ!(measurement_container, pconfig, model_geometry)
    nup_ndn = 0.0

    for i in 1:model_geometry.lattice.N
        nup_ndn += get_onsite_fermion_occupation(i, pconfig)[1] * (1 - get_onsite_fermion_occupation(i, pconfig)[2])
    end

    # calculate the current double occupancy
    dblocc_current = nup_ndn / model_geometry.lattice.N

    # get current values from the container
    dblocc_container = measurement_container.simulation_measurements["double_occ"]

    # update value for the current bin
    current_dblocc_bin = dblocc_container[2]
    current_dblocc_bin = dblocc_current

    # update accumuator for this bin
    thisbin_dblocc_sum = dblocc_container[1]
    thisbin_dblocc_sum += dblocc_current

    # combine the updated values 
    updated_values = (thisbin_dblocc_sum, current_dblocc_bin)

    # write the new values to the container
    measurement_container.simulation_measurements["double_occ"] = updated_values

    return nothing
end


"""
    measure_n( site::Int, pconfig::Vector{Int} )

Measure the local particle density ⟨n⟩.

"""
function measure_n!(measurement_container, pconfig, model_geometry)
    total_occ = 0.0
    N = model_geometry.lattice.N

    for i in 1:N
        local_occ = get_onsite_fermion_occupation(i, pconfig)[1] + 1 - get_onsite_fermion_occupation(i, pconfig)[2]
        total_occ += local_occ
    end

    density_current = total_occ / N

    # get current values from the container
    density_container = measurement_container.simulation_measurements["density"]

    # update value for the current bin
    current_density_bin = density_container[2]
    current_density_bin = density_current

    # update accumuator for this bin
    thisbin_density_sum = density_container[1]
    thisbin_density_sum += density_current

    # combine the updated values 
    updated_values = (thisbin_density_sum, current_density_bin)

    # write the new values to the container
    measurement_container.simulation_measurements["density"] = updated_values


    return nothing
end




"""
    get_local_energy(model_geometry::ModelGeometry, tight_binding_model::TightBindingModel, jastrow::Jastrow, particle_positions:: )

Calculates the local variational energy per site E/N, where N is the number fo sites.

"""
function get_local_energy(model_geometry, tight_binding_model, jastrow, pconfig, κ)
    # number of lattice sites
    N = model_geometry.lattice.N

    # calculate kinetic energy
    E_k = get_local_kinetic_energy(model_geometry, tight_binding_model, jastrow, pconfig, κ)

    # calculate Hubbard energy
    E_hubb = get_local_hubbard_energy(U, model_geometry, pconfig)
    
    # calculate total local energy
    E_loc = E_k + E_hubb

    return E_loc
end



function get_local_kinetic_energy(model_geometry, tight_binding_model, jastrow, pconfig, κ)
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
    for β in 1:Ne
        # spindex occupation number of particle β
        β_spindex = findfirst(x -> x == β, κ)

        # real position 'k' of particle 'β' 
        k = get_index_from_spindex(β_spindex, model_geometry) 

        # spin of particle particle 'β' 
        β_spin = get_spindex_type(β_spindex, model_geometry)
      
        # loop over nearest neighbors. TODO: add next-nearest neighbors
        sum_nn = 0.0
        for l in nbr_map[k][2]
            # check that neighboring sites are unoccupied
            if get_onsite_fermion_occupation(l, pconfig)[β_spin] == 0
                Rⱼ = get_jastrow_ratio(k, l, jastrow, pht, β_spin)[1]
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

    return real(E_loc_kinetic)
end


function get_local_hubbard_energy(U, model_geometry, pconfig)
    # number of sites
    N = model_geometry.lattice.N

    # track hubbard energy
    dblocc_sum = 0.0

    # calculate Hubbard energy
    for i in 1:N
        if pht
            dblocc_sum += get_onsite_fermion_occupation(i, pconfig)[1] * (1 - get_onsite_fermion_occupation(i, pconfig)[2])
        else
            dblocc_sum += get_onsite_fermion_occupation(i, pconfig)[1] * get_onsite_fermion_occupation(i, pconfig)[2]
        end
    end

    E_loc_hubbard = U * dblocc_sum

    return E_loc_hubbard
end


"""
    measure_global_energy( model_geometry::ModelGeometry )

Measure the global variational energy ⟨E⟩. This is intended to used at the end of the simulation.

"""
# function measure_global_energy(model_geometry, N_bins, bin_size)

#     # number of lattice sites
#     N = model_geometry.lattice.N
#     # binned energies
#     # E_binned

#     # average over all sites
#     E_global = E_binned / N

#     return E_global
# end







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
    loc_spn = get_onsite_fermion_occupation(site,pconfig)[1] - 1 + get_onsite_fermion_occupation(i,pconfig)[2]

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

