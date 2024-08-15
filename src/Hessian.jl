"""

    get_local_jpar_derivative( jastrow::Jastrow, pconfig::Vector{Int}, pht::Bool )

Calculates the local logarithmic derivative Δₖ(x) = ∂lnΨ(x)/∂vₗₘ, with respect to the kth Jastrow parameter vₗₘ. Returns 
a vector of derivatives.

"""
function get_local_jpar_derivative(jastrow::Jastrow, pconfig::Vector{Int}, pht::Bool)
    # jastrow type
    jastrow_type = jastrow.jastrow_type;

    # number of Jastrow parameters, except for the last one
    num_jpars = jastrow.num_jpars - 1;

    # map of Jastrow parameters
    jpar_map = jastrow.jpar_map;

    # vector to store derivatives
    derivatives = zeros(AbstractFloat, num_jpars)

    # get irreducible indices
    irr_indices = collect(keys(jpar_map))
                
    if jastrow_type == "e-den-den"
        for num in 1:num_jpars
            # Extract the current (indices, jpars) tuple
            indices, _ = jpar_map[irr_indices[num]]

            for idx in indices
                i = idx[1]
                j = idx[2]
                nup_i = number_operator(i+1, pconfig)[1]
                ndn_i = number_operator(i+1, pconfig)[2]
                nup_j = number_operator(j+1, pconfig)[1]
                ndn_j = number_operator(j+1, pconfig)[2]
                if pht
                    derivatives[num] += -0.5 * (nup_i + ndn_i - 1) * (nup_j + ndn_j - 1)
                else
                    derivatives[num] += -0.5 * (nup_i + ndn_i) * (nup_j + ndn_j)
                end
            end
        end

        return derivatives
    elseif jastrow_type == "e-spn-spn"
        for num in 1:num_jpars
            # Extract the current (indices, jpars) tuple
            indices, _ = jpar_map[irr_indices[num]]

            for idx in indices
                i = idx[1]
                j = idx[2]
                nup_i = number_operator(i+1, pconfig)[1]
                ndn_i = number_operator(i+1, pconfig)[2]
                nup_j = number_operator(j+1, pconfig)[1]
                ndn_j = number_operator(j+1, pconfig)[2]
                if pht
                    derivatives[num] += -0.4 * (nup_i + ndn_i - 1) * (nup_j + ndn_j - 1)
                else
                    derivatives[num] += -0.4 * (nup_i + ndn_i) * (nup_j + ndn_j)
                end
            end
        end

        return derivatives
    end
end


"""

    get_local_detpar_derivative( determinantal_parameters::DeterminantalParameters, model_geometry::ModelGeometry
                                     pconfig::Vector{Int}, W::Matrix{AbstractFloat}, A::Matrix{AbstractFloat}  )

Calculates the local logarithmic derivative Δₖ(x) = ∂lnΨ(x)/∂αₖ, with respect to the kth variational parameter αₖ,
in the determinantal part of the wavefunction. Returns a vector of derivatives.

"""
function get_local_detpar_derivative(determinantal_parameters, model_geometry, particle_positions, Np, W, A)  

    # dimensions
    dims = model_geometry.unit_cell.n * model_geometry.lattice.N

    # number of determinantal parameters
    num_detpars = determinantal_parameters.num_detpars

    # vector to store derivatives
    derivatives = zeros(AbstractFloat, num_detpars)
    

    # loop over Nₚ particles 
    G = zeros(AbstractFloat, 2*dims, 2*dims)
    for β in 1:Np
        k = particle_positions[β][2]  
        G[k,:] .= W[:,β]
        # for j in 1:2*dims
            # for (spindex, iᵦ) in particle_positions
            #     G[iᵦ,j] = W[j,β]
            # # G[iᵦ,:] = W[:,β]
        #     end
        # end
    end

    # loop over the number of determinantal parameters
    for num in 1:num_detpars
        derivatives[num] += sum(A[num] .* G)
    end

    return derivatives
end


"""

    get_hessian_matrix(measurement_container, determinantal_parameters, 
                    jastrow, model_geometry, tight_binding_model, pconfig, 
                    Np, W, A )

Generates the covariance (Hessian) matrix S, for Stochastic Reconfiguration

The matrix S has elements S_kk' = <Δ_kΔk'> - <Δ_k><Δ_k'>

"""
function get_hessian_matrix(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions,Np, W, A, n, bin, N_configs)

    # measure local parameters derivatives ⟨Δₖ⟩ (also ⟨Δₖ'⟩), for this configuration
    measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions, Np, W, A, n, bin, N_configs)
    Δk = measurement_container.derivative_measurements["Δk"][2][bin][n]       

    measure_ΔkΔkp!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions, Np, W, A, n, bin, N_configs)
    ΔkΔkp = measurement_container.derivative_measurements["ΔkΔkp"][2][bin][n] 
    
    # calculate the product of local derivatives ⟨Δk⟩⟨Δkp⟩
    ΔkΔk = Δk * Δk'  

    # generate covariance matrix
    S = ΔkΔkp - ΔkΔk
    
    return S
end


"""

    get_force_vector(measurement_container, determinantal_parameters, 
                jastrow, model_geometry, tight_binding_model, pconfig,
                 Np, W, A )

Generates the force vector f, for Stochastic Reconfiguration.

The vector f has elements f_k = <Δ_k><H> - <Δ_kH>

"""
function get_force_vector(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, particle_positions, Np, W, A, n, bin, N_configs)
    
    # initialize force vector
    f = [] 

    # measure local parameters derivatives ⟨Δₖ⟩, for this configuration
    # measure_Δk!(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions,Np, W, A,iter)
    Δk = measurement_container.derivative_measurements["Δk"][2][bin][n]         

    # measure local energy E = ⟨H⟩, for this configuration
    measure_local_energy!(measurement_container, model_geometry, tight_binding_model, jastrow, pconfig, particle_positions,n, bin, N_configs)
    E = measurement_container.scalar_measurements["energy"][2][bin][n]      

    # measure product of local derivatives with energy ⟨ΔkE⟩, for this configuration
    measure_ΔkE!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, particle_positions, Np, W, A,n, bin, N_configs)
    ΔkE = measurement_container.derivative_measurements["ΔkE"][2][bin][n]       

    # product of local derivative with the local energy ⟨Δk⟩⟨H⟩
    ΔktE = Δk * E
    
    for (i,j) in zip(ΔktE,ΔkE)
        fk = i - j
        push!(f, fk)
    end

    return f  # the length of f == number of vpars where the first p are the determinantal parameters and the rest are Jastrow parameters
end


"""

    parameter_gradient( S::Matrix{AbstractFloat}, η::Float64 )

Perform gradient descent on variational parameters for Stochastic 
Reconfiguration.

"""
function parameter_gradient(S, f, η)

    # Convert f to a vector of Float64
    f = convert(Vector{Float64}, f)

    # add small variation to diagonal of S for numerical stabilization
    S += η * I

    # solve for δα 
    δvpars = S \ f
    
    return δvpars   # the length of δvpars == number of vpars where the first p are the determinantal parameters and the rest are Jastrow parameters
end


"""

    sr_update!(measurement_container, determinantal_parameters, 
                jastrow, model_geometry, tight_binding_model, 
                pconfig, Np, W, A, η, dt)

Update variational parameters through stochastic optimization.

"""
function sr_update!(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, Np, W, A, η, dt, n, bin)
    # get particle positions
    particle_positions = get_particle_positions(pconfig, model_geometry)

    # current number of visited configurations
    N_configs = measurement_container.N_configs
    N_configs += n
    # update number of visited configurations
    measurement_container = merge(measurement_container, (N_configs = N_configs,))

    # get covariance (Hessian) matrix
    S = get_hessian_matrix(measurement_container, determinantal_parameters, jastrow, model_geometry, pconfig, particle_positions, Np, W, A, n, bin, N_configs)
    # get force vector
    f = get_force_vector(measurement_container, determinantal_parameters, jastrow, model_geometry, tight_binding_model, pconfig, particle_positions, Np, W, A, n, bin, N_configs)

    # perform gradient descent
    δvpars = parameter_gradient(S,f,η)     

    # # update parameters
    # # get parameters from measurement container
    # vpars = measurement_container.scalar_measurements["parameters"][1]
    # vpars += dt * δvpars    # TODO: start with a large dt and reduce as energy is minimized

    # push back all parameters to container
    current_container = measurement_container.scalar_measurements["parameters"]
    current_local_parameters = current_container[1]
    current_local_parameters += dt * δvpars
    current_parameter_values = current_container[2]
    current_parameter_values[bin][n] .= current_local_parameters

    # combine the updated values 
    updated_values = (current_local_parameters, current_parameter_values)

    # write the new values to the container
    measurement_container.scalar_measurements["parameters"] = updated_values

    # # push back Jastrow parameters
    # update_jastrow!(jastrow, vpars)

    # # push back determinantal_parameters
    # update_detpars!(determinantal_parameters, vpars)

    # update measurement container
    # current_values = measurement_container.scalar_measurements["parameters"][1]
    # current_values[iter] = vpars

    return measurement_container
end

