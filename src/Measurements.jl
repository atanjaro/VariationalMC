"""

    initialize_measurement_container( N_opts::Int, opt_bin_size::Int, N_bins::Int, bin_size::Int,
                                        model_geometry::ModelGeometry, determinantal_parameters::DeterminantalParameters, 
                                        jastrow::Jastrow )::NamedTuple

Creates dictionaries of generic arrays for storing measurements. Each dictionary in the container
has (keys => values): observable_name => local_values (i.e. ∑ O_loc(x)), binned_values (i.e. ⟨O⟩≈(N)⁻¹local_values)

"""
function initialize_measurement_container(N_opts::Int, opt_bin_size::Int, N_bins::Int, bin_size::Int,
                                         determinantal_parameters::DeterminantalParameters, jastrow::Jastrow, 
                                         model_geometry::ModelGeometry)::NamedTuple
    # total number of lattice sites
    N = model_geometry.lattice.N
    
    # one side of the lattice
    L = model_geometry.lattice.L

    # number of determinantal_parameters
    num_detpars = determinantal_parameters.num_detpars

    # number of Jastrow parameters
    num_jpars = jastrow.num_jpars - 1

    # number of variational parameters to be optimized
    num_vpars = num_detpars + num_jpars

    # container to store optimization measurements
    optimization_measurements = Dict{String, Any}([
        ("parameters", (zeros(num_vpars), zeros(num_vpars))),                     # variational parameters
        ("Δk", (zeros(num_vpars), zeros(num_vpars),[])),                          # log derivative of variational parameters
        ("ΔkΔkp", (zeros(num_vpars,num_vpars), zeros(num_vpars,num_vpars),[])),   # product between log derivatives        
        ("ΔkE", (zeros(num_vpars), zeros(num_vpars),[])),                         # product of log derivatives and energy
    ])      

    # dictionary to store simulation measurements
    simulation_measurements = Dict{String, Any}([
        ("density", (0.0,  0.0)),          # average density
        ("double_occ", (0.0,  0.0)),       # average double occupancy 
        ("energy", (0.0,  0.0)),           # local energy
        ("pconfig", zeros(N))              # particle configurations
    ])                     

    # TODO: dictionary to store correlation measurements
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

    make_measurements!( measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                        tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                        jastrow::Jastrow, model_geometry::ModelGeometry, Ne::Int64, pht::Bool )

Measure the local energy and logarithmic derivatives for a particular bin.

"""
function make_measurements!(measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                            tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                            jastrow::Jastrow, model_geometry::ModelGeometry, Ne::Int64, pht::Bool)

    # measure the variational parameters
    measure_parameters!(measurement_container, determinantal_parameters, jastrow)
    
    # measure the energy
    measure_local_energy!(measurement_container, detwf, tight_binding_model, jastrow, model_geometry, Ne, pht)

    # measure the lograithmic derivatives
    measure_Δk!(measurement_container, detwf, determinantal_parameters, jastrow, model_geometry, Ne, pht)
    measure_ΔkΔkp!(measurement_container, detwf, determinantal_parameters, jastrow, model_geometry, Ne, pht)
    measure_ΔkE!(measurement_container, detwf, tight_binding_model, determinantal_parameters, jastrow, model_geometry, Ne, pht)

    # measure double occupancy
    measure_double_occ!(measurement_container, detwf, model_geometry)

    # measure average density
    measure_n!(measurement_container, detwf, model_geometry)

    # record the current particle configuration
    measurement_container.simulation_measurements["pconfig"] = detwf.pconfig

    return nothing
end


"""

    write_measurements!( measurement_container::NamedTuple, simulation_info::SimulationInfo )::Nothing

Writes current measurements in the current bin to a JLD2 file. 

""" 
function write_measurements!(measurement_container::NamedTuple, simulation_info::SimulationInfo)::Nothing
    # extract container info
    simulation_measurements = measurement_container.simulation_measurements
    optimization_measurements = measurement_container.optimization_measurements

    (; datafolder, pID) = simulation_info

    # extract other container components
    (; optimization_measurements, simulation_measurements) = measurement_container

    # construct filenames
    fn = @sprintf "bin-%d_pID-%d.jld2" bin pID  
    file_path_energy = joinpath(datafolder, "simulation", "energy", fn)
    file_path_dblocc = joinpath(datafolder, "simulation", "double_occ", fn)

    # append energy measurements to file
    energy_measurements = simulation_measurements["energy"][1]
    JLD2.@save file_path_energy energy_measurements append=true

    # append double occupancy to file
    dblocc_measurements = simulation_measurements["double_occ"][1]
    JLD2.@save file_path_dblocc dblocc_measurements append=true

    # reset all measurements
    reset_measurements!(simulation_measurements)
    reset_measurements!(optimization_measurements)    

    return nothing
end


"""

    write_measurements!( measurement_container::NamedTuple, energy_bin::Vector{Any}, 
                        dblocc_bin::Vector{Any}, param_bin::Vector{Any} )::Nothing

DEBUG version of the write_measurements!() method. Will write binned energies, double occupancy, and parameters
to specified vectors.  

""" 
function write_measurements!(measurement_container::NamedTuple, energy_bin::Vector{Float64}, 
                                dblocc_bin::Vector{Float64}, param_bin::Vector{Any})::Nothing
    # extract container info
    simulation_measurements = measurement_container.simulation_measurements
    optimization_measurements = measurement_container.optimization_measurements

    # append accumulated values to the storage vectors
    push!(energy_bin, simulation_measurements["energy"][1])
    push!(dblocc_bin, simulation_measurements["double_occ"][1])
    push!(param_bin, optimization_measurements["parameters"][1])
    
    # reset all measurements
    reset_measurements!(simulation_measurements)
    reset_measurements!(optimization_measurements)

    return nothing
end


"""

    initialize_measurement_directories(simulation_info::SimulationInfo, measurement_container::NamedTuple)

Creates file directories for storing measurements. 

"""
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

    measure_parameters!( measurement_container::NamedTuple, 
                        determinantal_parameters::DeterminantalParameters, jastrow::Jastrow )::Nothing

Measures all variational parameters. The first 'p' are determinantal parameters and the rest are of Jastrow parameters. 
Measurements are then written to the measurement container.

"""
function measure_parameters!(measurement_container::NamedTuple, 
                            determinantal_parameters::DeterminantalParameters, jastrow::Jastrow)::Nothing
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

    return nothing
end


"""

    measure_Δk!( measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                    determinantal_parameters::DeterminantalParameters, jastrow::Jastrow, 
                    model_geometry::ModelGeometry, Ne::Int64, pht::Bool )::Nothing

Measures logarithmic derivatives for all variational parameters. The first 'p' are derivatives of 
determinantal parameters and the rest are derivatives of Jastrow parameters. Measurements are then written
to the measurement container.

"""
function measure_Δk!(measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                    determinantal_parameters::DeterminantalParameters, jastrow::Jastrow, 
                    model_geometry::ModelGeometry, Ne::Int64, pht::Bool)::Nothing
    # calculate variational parameter derivatives
    detpar_derivatives = get_detpar_derivatives(detwf, determinantal_parameters, model_geometry, Ne)
    jpar_derivatives = get_jpar_derivatives(detwf, jastrow, pht)
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

    measure_ΔkΔkp!( measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                        determinantal_parameters::DeterminantalParameters, jastrow::Jastrow, 
                        model_geometry::ModelGeometry, Ne::Int64, pht::Bool )::Nothing

Measures the product of variational derivatives with other variational derivatives. Measurements are then written
to the measurement container.

"""
function measure_ΔkΔkp!(measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                        determinantal_parameters::DeterminantalParameters, jastrow::Jastrow, 
                        model_geometry::ModelGeometry, Ne::Int64, pht::Bool)::Nothing
    # calculate variational parameter derivatives
    detpar_derivatives = get_detpar_derivatives(detwf, determinantal_parameters, model_geometry, Ne)
    jpar_derivatives = get_jpar_derivatives(detwf, jastrow, pht)
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

    measure_ΔkE!( measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                    tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                    jastrow::Jastrow, model_geometry::ModelGeometry, Ne::Int64, pht::Bool )::Nothing

Measures the product of variational derivatives with the local energy. Measurements are then written
to the measurement container.

"""
function measure_ΔkE!(measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                    tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                    jastrow::Jastrow, model_geometry::ModelGeometry, Ne::Int64, pht::Bool)::Nothing
    # calculate variational parameter derivatives
    detpar_derivatives = get_detpar_derivatives(detwf, determinantal_parameters, model_geometry, Ne)
    jpar_derivatives = get_jpar_derivatives(detwf, jastrow, pht)
    Δk = vcat(detpar_derivatives, jpar_derivatives)

    # compute local energy
    E_loc = get_local_energy(detwf, tight_binding_model, jastrow, model_geometry, Ne, pht) 

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

    measure_local_energy!( measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                            tight_binding_model::TightBindingModel, jastrow::Jastrow, model_geometry::ModelGeometry )

Measures the total local energy for a Hubbard model and writes to the measurement container.

"""
function measure_local_energy!(measurement_container::NamedTuple, detwf::DeterminantalWavefunction, 
                                tight_binding_model::TightBindingModel, jastrow::Jastrow, 
                                model_geometry::ModelGeometry, Ne::Int64, pht::Bool)
   # calculate the current local energy
    E_loc_current = get_local_energy(detwf, tight_binding_model, jastrow, model_geometry, Ne, pht)

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

    measure_double_occ!( measurement_container::NamedTuple, 
                        detwf::DeterminantalWavefunction, model_geometry::ModelGeometry )

Measure the average double occupancy ⟨D⟩ = N⁻¹ ∑ᵢ ⟨nᵢ↑nᵢ↓⟩.

"""
function measure_double_occ!(measurement_container::NamedTuple, 
                            detwf::DeterminantalWavefunction, model_geometry::ModelGeometry)
    # calculate the current double occupancy
    dblocc_current = get_double_occ(detwf, model_geometry)

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

    measure_n!( measurement_container::NamedTuple, 
                detwf::DeterminantalWavefunction, model_geometry::ModelGeometry )

Measure the local particle density ⟨n⟩.

"""
function measure_n!(measurement_container::NamedTuple, 
                    detwf::DeterminantalWavefunction, model_geometry::ModelGeometry)
    # calculate current density
    density_current = get_n(detwf, model_geometry)

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
    get_local_energy( detwf::DeterminantalWavefunction, tight_binding_model::TightBindingModel, 
                        jastrow::Jastrow, model_geometry::ModelGeometry )

Calculates the local variational energy per site for a Hubbard model.

"""
function get_local_energy(detwf::DeterminantalWavefunction, tight_binding_model::TightBindingModel, 
                        jastrow::Jastrow, model_geometry::ModelGeometry, Ne::Int64, pht::Bool)
    # number of lattice sites
    N = model_geometry.lattice.N

    # calculate kinetic energy
    E_k = get_local_kinetic_energy(detwf, tight_binding_model, jastrow, model_geometry, Ne, pht)

    # calculate Hubbard energy
    E_hubb = get_local_hubbard_energy(U, detwf, model_geometry, pht)

    # calculate total local energy
    E_loc = E_k + E_hubb
    
    return E_loc/N
end


"""
    get_local_kinetic_energy( detwf::DeterminantalWavefunction, tight_binding_model::TightBindingModel, 
                                jastrow::Jastrow, model_geometry::ModelGeometry, pht::Bool )

Calculates the local electronic kinetic energy. 

"""
function get_local_kinetic_energy(detwf::DeterminantalWavefunction, tight_binding_model::TightBindingModel, 
                                    jastrow::Jastrow, model_geometry::ModelGeometry, Ne::Int64, pht::Bool)
    # number of sites
    N = model_geometry.lattice.N

    # generate neighbor table
    nbr_table = build_neighbor_table(model_geometry.bond[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice)

    # generate neighbor map
    nbr_map = map_neighbor_table(nbr_table)

    # # checks for next nearest neighbors
    # if length(model_geometry.bond) == 2
    #     nbr_table_p = build_neighbor_table(bonds[2],
    #                                       model_geometry.unit_cell,
    #                                       model_geometry.lattice)
    #     nbr_map_p = map_neighbor_table(nbr_table_p)
    # end

    E_loc_kinetic = 0.0

    for β in 1:Ne
        # spindex of particle
        k = findfirst(x -> x == β, detwf.pconfig)

        # real position position 
        ksite = get_index_from_spindex(k, model_geometry) 

        # check spin of particle  
        spin = get_spindex_type(k, model_geometry)
      
        # loop over nearest neighbors. TODO: add next-nearest neighbors
        sum_nn = 0.0
        for lsite in nbr_map[ksite][2]
            if spin == 1
                l = get_spindices_from_index(lsite, model_geometry)[1]
            else
                l = get_spindices_from_index(lsite, model_geometry)[2]
            end

            # check that neighboring site is unoccupied
            if detwf.pconfig[l] == 0
                Rⱼ = get_fermionic_jastrow_ratio(k, l, jastrow, pht, spin, model_geometry)
                sum_nn += Rⱼ * detwf.W[l, β]
            end
        end

        # hopping amplitudes       
        t = tight_binding_model.t

        if pht 
            if spin == 1
                E_loc_kinetic += -t[1] * sum_nn        
            else
                E_loc_kinetic += t[1] * sum_nn 
            end
        else
            E_loc_kinetic += -t[1] * sum_nn
        end
    end

    return real(E_loc_kinetic)
end


"""
    get_local_hubbard_energy( U::Float64, detwf::DeterminantalWavefunction, 
                            model_geometry::ModelGeometry, pht::Bool )::Float64

Calculates the energy due to onsite Hubbard repulsion.  

"""
function get_local_hubbard_energy(U::Float64, detwf::DeterminantalWavefunction, 
                                    model_geometry::ModelGeometry, pht::Bool)::Float64
    # number of sites
    N = model_geometry.lattice.N
        
    hubbard_sum = 0.0
    for i in 1:N
        occ_up, occ_dn, occ_e = get_onsite_fermion_occupation(i, detwf.pconfig)
        if occ_e == 2
            if pht
                hubbard_sum += occ_up .* (1 .- occ_dn)
            else
                hubbard_sum+= occ_up .* occ_dn
            end
        end
    end

    E_loc_hubbard = U * hubbard_sum

    return E_loc_hubbard
end


"""

    get_double_occ( detwf::DeterminantalParameters, model_geometry::ModelGeometry )

Calculates the double occupancy. 

"""
function get_double_occ(detwf::DeterminantalWavefunction, model_geometry::ModelGeometry)
    N = model_geometry.lattice.N

    spin_up = detwf.pconfig[1:N]    
    spin_down = detwf.pconfig[N+1:end]  

    nup_ndn = 0.0
    for i in 1:N
        if spin_up[i] != 0 && spin_down[i] != 0
            nup_ndn += 1.0
        end
    end

    return nup_ndn / N
end

"""

    get_n( detwf::DeterminantalWavefunction, model_geometry::ModelGeometry )

Calculate the local density.

"""
function get_n(detwf::DeterminantalWavefunction, model_geometry::ModelGeometry)
    total_occ = 0.0
    N = model_geometry.lattice.N

    for i in 1:N
        local_occ = get_onsite_fermion_occupation(i, detwf.pconfig)[1] + 1 - get_onsite_fermion_occupation(i, detwf.pconfig)[2]
        total_occ += local_occ
    end

    return total_occ / N
end


##################################################### DEPRECATED FUNCTIONS #####################################################
# """
#     measure_global_energy( model_geometry::ModelGeometry )

# Measure the global variational energy ⟨E⟩. This is intended to used at the end of the simulation.

# """
# # function measure_global_energy(model_geometry, N_bins, bin_size)

# #     # number of lattice sites
# #     N = model_geometry.lattice.N
# #     # binned energies
# #     # E_binned

# #     # average over all sites
# #     E_global = E_binned / N

# #     return E_global
# # end

# """
#     measure_ρ( site::int )

# Measure the local excess hole density ⟨ρ⟩.

# """
# function measure_ρ(site)
#     ρ = 1 - measure_n(site)

#     return ρ
# end


# """
#     measure_s( site::Int, pconfig::Vector{Int} )

# Measure the local spin.

# """
# function measure_s()
#     loc_spn = get_onsite_fermion_occupation(site,pconfig)[1] - 1 + get_onsite_fermion_occupation(i,pconfig)[2]

#     return loc_spn
# end




# """
#     measure_density_corr( )

# Measure the density-density correlation function.

# """
# function measure_density_corr()
#     return nothing
# end


# """
#     measure_spin_corr( )

# Measure the spin-spin correlation function.

# """
# function measure_spin_corr()
#     return nothing
# end


# # """
# #     measure_average_X( )

# # Measure average phonon displacement ⟨X⟩, in the SSH model (TBD).

# # """
# # function measure_average_X()
# #     return nothing
# # end

# TODO: implement correlation measurements
# function initialize_correlation_measurements!(measurement_container,  correlation)
#     (; correlation_measurements) = measurement_container

#     if correlation == "density-density"
#         correlation_measurements["density-density"] = Vector{Vector{AbstractFloat}}(undef, N_iter)
#     elseif correlation == "spin-spin"
#         correlation_measurements["spin-spin"] = Vector{Vector{AbstractFloat}}(undef, N_iter)
#     elseif correlation == "pair"
#         correlation_measurements["pair"] = Vector{Vector{AbstractFloat}}(undef, N_iter)
#     end

#     return nothing
# end


# """

#     initialize_measurements!( measurement_container, observable::String )

# For other types of measurements, initializes the arrays necessary to store 
# measurements in respective bins.

# """
# function initialize_measurements!(measurement_container, observable::String)
#     (; simulation_measurements, N) = measurement_container

#     if observable == "site_dependent_density"
#         simulation_measurements["site_dependent_density"] = (zeros(AbstractFloat, norbs*N),Vector{Vector{AbstractFloat}}(undef, N_iter))  
#     elseif observable == "site_dependent_spin"
#         simulation_measurements["site_dependent_spin"] = (zeros(AbstractFloat, norbs*N),Vector{Vector{AbstractFloat}}(undef, N_iter))   
#     elseif observable == "phonon_number"
#         simulation_measurements["phonon_number"] = (0.0,  Vector{AbstractFloat}(undef, N_iter))                  
#     elseif observable == "displacement"
#         simulation_measurements["displacement"] = (0.0,  Vector{AbstractFloat}(undef, N_iter))                      
#     end

#     return nothing
# end