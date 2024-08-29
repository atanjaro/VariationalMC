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

    get_hessian_matrix( measurement_container, bin )

Generates the covariance (Hessian) matrix S, for Stochastic Reconfiguration

The matrix S has elements S_kk' = <Δ_kΔk'> - <Δ_k><Δ_k'>

"""
function get_hessian_matrix(measurement_container)
    # get size of bin
    bin_size = measurement_container.opt_bin_size

    # measure local parameters derivatives ⟨Δₖ⟩ for the current bin
    Δk = measurement_container.optimization_measurements["Δk"][2]/bin_size
    
    # measure the product of local derivatives ⟨ΔₖΔₖ'⟩ for the current bin
    ΔkΔkp = measurement_container.optimization_measurements["ΔkΔkp"][2]/bin_size
    
    # calculate the product of local derivatives ⟨Δₖ⟩⟨Δₖ'⟩
    ΔkΔk = Δk * Δk'  

    # generate covariance matrix
    S = ΔkΔkp - ΔkΔk
    
    return S
end


"""

    get_force_vector( measurement_container, bin )

Generates the force vector f, for Stochastic Reconfiguration.

The vector f has elements f_k = <Δ_k><H> - <Δ_kH>

"""
function get_force_vector(measurement_container)
    # get size of bin
    bin_size = measurement_container.opt_bin_size
    
    # initialize force vector
    f = [] 

    # measure local parameters derivatives ⟨Δₖ⟩ for the current bin
    Δk = measurement_container.optimization_measurements["Δk"][2]/bin_size

    # measure local energy E = ⟨H⟩ for the current bin
    E = measurement_container.simulation_measurements["energy"][2]/bin_size

    # measure product of local derivatives with energy ⟨ΔkE⟩ for the current bin
    ΔkE = measurement_container.optimization_measurements["ΔkE"][2]/bin_size         

    # calculate product of local derivative with the local energy ⟨Δk⟩⟨H⟩
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
function sr_update!(measurement_container, determinantal_parameters, jastrow, η, dt)
    if verbose
        println("Begin optimization step...")
    end

    # get covariance (Hessian) matrix
    S = get_hessian_matrix(measurement_container)

    # get force vector
    f = get_force_vector(measurement_container)

    # perform gradient descent
    δvpars = parameter_gradient(S,f,η)     

    # new varitaional parameters
    vpars = all_vpars(determinantal_parameters, jastrow)
    vpars += dt * δvpars

    # push back Jastrow parameters
    update_jastrow!(jastrow, vpars)

    # push back determinantal_parameters
    update_detpars!(determinantal_parameters, vpars)

    # measure parameters
    # get current values from the container
    current_container = measurement_container.optimization_measurements["parameters"]

    # update value for the current bin
    current_bin_values = current_container[2]
    current_bin_values .= vpars 

    # update accumuator for average measurements
    new_avg_value = current_container[1] .+ vpars

    # combine the updated values 
    updated_values = (new_avg_value, current_bin_values)

    # write the new values to the container
    measurement_container.optimization_measurements["parameters"] = updated_values

    if verbose
        println("End optimization step")
    end

    return nothing
end

