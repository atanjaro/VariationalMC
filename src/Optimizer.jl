"""

    get_local_detpar_derivative( determinantal_parameters::DeterminantalParameters, model_geometry::ModelGeometry
                                     pconfig::Vector{Int}, W::Matrix{AbstractFloat}, A::Matrix{AbstractFloat}  )

Calculates the local logarithmic derivative Δₖ(x) = ∂lnΨ(x)/∂αₖ, with respect to the kth variational parameter αₖ,
in the determinantal part of the wavefunction. Returns a vector of derivatives.

"""
function get_local_detpar_derivative(determinantal_parameters::DeterminantalParameters, model_geometry::ModelGeometry, 
                                        κ::Vector{Int64}, Ne::Int64, W::Matrix{ComplexF64}, A::Vector{Any})  

    # number of lattice sites
    N = model_geometry.unit_cell.n * model_geometry.lattice.N

    # number of determinantal parameters
    num_detpars = determinantal_parameters.num_detpars

    # vector to store derivatives
    derivatives = zeros(AbstractFloat, num_detpars)
    

    # loop over Nₚ particles 
    G = zeros(Complex, 2*N, 2*N)
    for β in 1:Ne
        # spindex occupation number of particle β
        β_spindex = findfirst(x -> x == β, κ)

        # # real position 'k' of particle 'β' 
        # k = get_index_from_spindex(β_spindex, model_geometry) 

        G[β_spindex,:] .= W[:,β]
    end

    # loop over the number of determinantal parameters
    for num in 1:num_detpars
        derivatives[num] += sum(A[num] .* G)
    end

    return derivatives
end


"""

    get_local_jpar_derivative( jastrow::Jastrow, pconfig::Vector{Int}, pht::Bool )

Calculates the local logarithmic derivative Δₖ(x) = ∂lnΨ(x)/∂vₗₘ, with respect to the kth Jastrow parameter vₗₘ. Returns 
a vector of derivatives.

"""
function get_local_jpar_derivative(jastrow::Jastrow, pconfig::Vector{Int64}, pht::Bool)
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

                # check for double counting
                dblcount_correction = (j==i) ? 0.5 : 1.0

                # get occupations
                nup_i = get_onsite_fermion_occupation(i+1, pconfig)[1]
                ndn_i = get_onsite_fermion_occupation(i+1, pconfig)[2]
                nup_j = get_onsite_fermion_occupation(j+1, pconfig)[1]
                ndn_j = get_onsite_fermion_occupation(j+1, pconfig)[2]
                if pht
                    derivatives[num] += -dblcount_correction * (nup_i - ndn_i) * (nup_j - ndn_j)
                else
                    derivatives[num] += -dblcount_correction * (nup_i + ndn_i) * (nup_j + ndn_j)
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

                # check for double counting
                dblcount_correction = (j==i) ? 0.5 : 1.0

                # get occupations
                nup_i = get_onsite_fermion_occupation(i+1, pconfig)[1]
                ndn_i = get_onsite_fermion_occupation(i+1, pconfig)[2]
                nup_j = get_onsite_fermion_occupation(j+1, pconfig)[1]
                ndn_j = get_onsite_fermion_occupation(j+1, pconfig)[2]
                if pht
                    derivatives[num] += -0.5 * (nup_i - ndn_i) * (nup_j - ndn_j)
                else
                    derivatives[num] += -0.5 * (nup_i + ndn_i) * (nup_j + ndn_j)
                end
            end
        end

        return derivatives
    end
end


"""

    get_hessian_matrix( measurement_container, bin )

Generates the covariance (Hessian) matrix S, for Stochastic Reconfiguration

The matrix S has elements S_kk' = <Δ_kΔk'> - <Δ_k><Δ_k'>

"""
function get_hessian_matrix(measurement_container, opt_bin_size)

    # measure local parameters derivatives ⟨Δₖ⟩ for the current bin
    Δk = measurement_container.optimization_measurements["Δk"][1]/opt_bin_size
    
    # measure the product of local derivatives ⟨ΔₖΔₖ'⟩ for the current bin
    ΔkΔkp = measurement_container.optimization_measurements["ΔkΔkp"][1]/opt_bin_size
    
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
function get_force_vector(measurement_container, opt_bin_size)
    
    # initialize force vector
    f = [] 

    # measure local parameters derivatives ⟨Δₖ⟩ for the current bin
    Δk = measurement_container.optimization_measurements["Δk"][1]/opt_bin_size

    # measure local energy E = ⟨H⟩ for the current bin
    E = measurement_container.simulation_measurements["energy"][1]/opt_bin_size

    # measure product of local derivatives with energy ⟨ΔkE⟩ for the current bin
    ΔkE = measurement_container.optimization_measurements["ΔkE"][1]/opt_bin_size

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
    S̃ = S + (η * I)

    # solve for δα 
    δvpars = S̃ \ f

    return δvpars
end


"""

    sr_update!(measurement_container, determinantal_parameters, 
                model_geometry, tight_binding_model, 
                pconfig, Np, W, A, η, dt)

Update variational parameters through stochastic optimization.

"""
function sr_update!(measurement_container, determinantal_parameters, η, dt, opt_bin_size)
    if debug
        println("Begin optimization...")
    end

    # get covariance (Hessian) matrix
    S = get_hessian_matrix(measurement_container, opt_bin_size)

    # get force vector
    f = get_force_vector(measurement_container, opt_bin_size)

    # perform gradient descent
    δvpars = parameter_gradient(S,f,η)     

    if debug
        println("Parameters updated!")
    end

    # new varitaional parameters
    vpars = reduce(vcat, determinantal_parameters.vals)
    vpars += dt * δvpars

    if debug
        println("New parameters: ", vpars)
    end

    # push back determinantal_parameters
    update_detpars!(determinantal_parameters, vpars)

    if debug
        println("End of optimization...")
    end

    return nothing
end


"""

    sr_update!(measurement_container, determinantal_parameters, 
                jastrow, model_geometry, tight_binding_model, 
                pconfig, Np, W, A, η, dt)

Update variational parameters through stochastic optimization.

"""
function sr_update!(measurement_container, determinantal_parameters, jastrow, η, dt, bin_size)
    if debug
        println("Begin optimization...")
    end

    # get covariance (Hessian) matrix
    S = get_hessian_matrix(measurement_container, bin_size)

    # get force vector
    f = get_force_vector(measurement_container, bin_size)

    # perform gradient descent
    δvpars = parameter_gradient(S,f,η)     

    if debug
        println("Parameters updated!")
    end

    # new varitaional parameters
    vpars = all_vpars(determinantal_parameters, jastrow)
    vpars += dt * δvpars

    if debug
        println("New parameters: ", vpars)
    end

    # push back Jastrow parameters
    update_jastrow!(jastrow, vpars)

    # push back determinantal_parameters
    update_detpars!(determinantal_parameters, vpars)

    if debug
        println("End of optimization...")
    end

    return nothing
end

