"""

    stochastic_reconfiguration!( measurement_container::NamedTuple, determinantal_parameters::DeterminantalParameters, 
                                    jastrow::Jastrow, η::Float64, dt::Float64, bin_size::Int64 )::Nothing

Updates variational parameters through stochastic optimization.

"""
function stochastic_reconfiguration!(measurement_container::NamedTuple, determinantal_parameters::DeterminantalParameters, 
                                    jastrow::Jastrow, η::Float64, dt::Float64, bin_size::Int64)::Nothing
    debug && println("Optimizer::stochastic_reconfiguration!() : ")
    debug && println("Start of optimization")

    # get S matrix
    S = get_covariance_matrix(measurement_container, bin_size)

    # get f vector
    f = get_force_vector(measurement_container, bin_size)

    # solve for variation in the parameters
    δvpars = variations(S, f, η)     

    # new varitaional parameters
    vpars = all_vpars(determinantal_parameters, jastrow)
    vpars += dt * δvpars

    debug && println("Optimizer::stochastic_reconfiguration!() : ")
    debug && println("Parameters have been updated")
    debug && println("new parameters = ", vpars)

    # push back Jastrow parameters
    update_jastrow!(jastrow, vpars)

    # push back determinantal_parameters
    update_detpars!(determinantal_parameters, vpars)

    debug && println("Optimizer::stochastic_reconfiguration!() : ")
    debug && println("End of optimization")

    return nothing
end


"""

    get_detpar_derivatives( detwf::DeterminantalWavefunction, determinantal_parameters::DeterminantalParameters, 
                            model_geometry::ModelGeometry, Ne::Int64 )::Vector{Float64}

Calculates the local logarithmic derivative Δₖ(x) = ∂lnΨ(x)/∂αₖ, with respect to the kth variational parameter αₖ,
in the determinantal part of the wavefunction. Returns a vector of derivatives.

"""
function get_detpar_derivatives(detwf::DeterminantalWavefunction, determinantal_parameters::DeterminantalParameters, 
                                model_geometry::ModelGeometry, Ne::Int64)::Vector{Float64}
    # number of lattice sites
    N = model_geometry.unit_cell.n * model_geometry.lattice.N

    # number of determinantal parameters
    num_detpars = determinantal_parameters.num_detpars

    # vector to store derivatives
    det_par_derivatives = zeros(Float64, num_detpars)
    
    # loop over Nₑ particles 
    G = zeros(Complex, 2*N, 2*N)
    for β in 1:Ne
        k = findfirst(x -> x == β, detwf.pconfig)

        G[k, :] .= detwf.W[:, β]
    end

    # loop over the number of determinantal parameters
    for num in 1:num_detpars
        det_par_derivatives[num] += sum(detwf.A[num] .* G)
    end

    return det_par_derivatives
end


"""

    get_jpar_derivatives( detwf::DeterminantalWavefunction, jastrow::Jastrow, pht::Bool )::Vector{Float64}

Calculates the local logarithmic derivative Δₖ(x) = ∂lnΨ(x)/∂vₗₘ, with respect to the kth 
Jastrow parameter vₗₘ. Returns a vector of derivatives.

"""
function get_jpar_derivatives(detwf::DeterminantalWavefunction, jastrow::Jastrow, pht::Bool)::Vector{Float64}
    # jastrow type
    jastrow_type = jastrow.jastrow_type;

    # number of Jastrow parameters, except for the last one
    num_jpars = jastrow.num_jpars - 1;

    # map of Jastrow parameters
    jpar_map = jastrow.jpar_map;

    # vector to store derivatives
    jpar_derivatives = zeros(Float64, num_jpars)

    # get irreducible indices
    irr_indices = collect(keys(jpar_map))
                
    if jastrow_type == "e-den-den"
        for num in 1:num_jpars
            # Extract the current (indices, jpars) tuple
            indices, _ = jpar_map[irr_indices[num]]

            for idx in indices
                i = idx[1]
                j = idx[2]

                # double counting
                dblcount_correction = (j==i) ? 0.5 : 1.0

                # get occupations
                nup_i = get_onsite_fermion_occupation(i+1, detwf.pconfig)[1]
                ndn_i = get_onsite_fermion_occupation(i+1, detwf.pconfig)[2]
                nup_j = get_onsite_fermion_occupation(j+1, detwf.pconfig)[1]
                ndn_j = get_onsite_fermion_occupation(j+1, detwf.pconfig)[2]
                if pht
                    jpar_derivatives[num] += -dblcount_correction * (nup_i - ndn_i) * (nup_j - ndn_j)
                else
                    jpar_derivatives[num] += -dblcount_correction * (nup_i + ndn_i) * (nup_j + ndn_j)
                end
            end
        end
    elseif jastrow_type == "e-spn-spn"
        for num in 1:num_jpars
            # Extract the current (indices, jpars) tuple
            indices, _ = jpar_map[irr_indices[num]]

            for idx in indices
                i = idx[1]
                j = idx[2]

                # double counting
                dblcount_correction = (j==i) ? 0.5 : 1.0

                # get occupations
                nup_i = get_onsite_fermion_occupation(i+1, detwf.pconfig)[1]
                ndn_i = get_onsite_fermion_occupation(i+1, detwf.pconfig)[2]
                nup_j = get_onsite_fermion_occupation(j+1, detwf.pconfig)[1]
                ndn_j = get_onsite_fermion_occupation(j+1, detwf.pconfig)[2]
                if pht
                    jpar_derivatives[num] += -0.5 * dblcount_correction * (nup_i - ndn_i) * (nup_j - ndn_j)
                else
                    jpar_derivatives[num] += -0.5 * dblcount_correction * (nup_i + ndn_i) * (nup_j + ndn_j)
                end
            end
        end
    end
    return jpar_derivatives
end


"""

    get_covariance_matrix( measurement_container::NamedTuple, opt_bin_size::Int64 )

Calculates the covariance matrix S, for Stochastic Reconfiguration, with elements
S_kk' = <Δ_kΔk'> - <Δ_k><Δ_k'>.

"""
function get_covariance_matrix(measurement_container::NamedTuple, opt_bin_size::Int64)

    # measure local parameters derivatives ⟨Δₖ⟩ for the current bin
    Δk = measurement_container.optimization_measurements["Δk"][1]/opt_bin_size
    
    # measure the product of local derivatives ⟨ΔₖΔₖ'⟩ for the current bin
    ΔkΔkp = measurement_container.optimization_measurements["ΔkΔkp"][1]/opt_bin_size
    
    # calculate the product of local derivatives ⟨Δₖ⟩⟨Δₖ'⟩
    ΔkΔk = Δk * Δk'  

    # get S
    S = ΔkΔkp - ΔkΔk
    
    return S
end


"""

    get_force_vector( measurement_container::NamedTuple, opt_bin_size::Int64 )

Generates the force vector f, for Stochastic Reconfiguration with elements 
f_k = <Δ_k><H> - <Δ_kH>.

"""
function get_force_vector(measurement_container::NamedTuple, opt_bin_size::Int64)
    
    # initialize force vector
    f = [] # TODO: change this initialize a vector with Float64 elements

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

    return f 
end


"""

    variations( S::Matrix, f::Vector, η::Float64 )

Solves for the variation in parameters for Stochastic Reconfiguration.

"""
function variations(S, f, η::Float64)
    # Convert f to a vector of Float64
    f = convert(Vector{Float64}, f)

    # solve for variation in the parameters
    δvpars = (S + (η * I)) \ f

    return δvpars
end




