"""

    ModelGeometry( unit_cell::UnitCell, lattice::Lattice )

A type defining model geometry.

"""
struct ModelGeometry
    # unit cell
    unit_cell::UnitCell
    # extent of the lattice
    lattice::Lattice
    # lattice bonds
    bond::Vector{Vector{Any}}
end



"""
    TightBindingModel( t::Vector{AbstractFloat}, μ::AbstractFloat, 
                    model_geometry::ModelGeometry, nbr_table::Matrix{Int64} )

A type defining a non-interacting tight binding model

"""
struct TightBindingModel    # TODO: add onsite energy?
    # hopping amplitudes
    t::Vector{AbstractFloat}   # [t, t']    # TODO: change to [[t₁, t₂, t₃], t'] for SSH models
    # chemical potentials
    μ::AbstractFloat           # TODO: change to vector based on number of orbitals [μ₁,μ₂,μ₃,...]
end


"""

    DeterminantalParameters( pars::Vector{AbstractString}, 
                        vals::Vector{AbstractFloat}, num_detpars::Int )

A type defining a set of variational parameters obtained from the fermionic determinant.

"""
struct DeterminantalParameters
    # name of order parameter
    pars::Vector{AbstractString}
    # variational parameter values
    vals::Vector{Vector{AbstractFloat}}
    # number of determinantal parameters
    num_detpars::Int
end


"""

    initialize_determinantal_parameters(pars:;Vector{AbstractString}, vals::Vector{AbstractFloat} ) 

Constructor for the variational parameters type.

"""
function initialize_determinantal_parameters(pars, vals)
    @assert length(pars) == length(vals) "Input vectors must have the same length"
    
    # Calculate num_detpars as the total number of elements in all inner vectors of vals
    num_detpars = sum(length, vals)

    return DeterminantalParameters(pars, vals, num_detpars)
end


"""

    update_detpars!( determinantal_parameters::DeterminantalParameters, new_vpars::Vector{AbstractFloat} )

Updates determinantal_parameters.

"""
function update_detpars!(determinantal_parameters, new_vpars)
    num_detpars = determinantal_parameters.num_detpars
    current_vals = determinantal_parameters.vals

    new_detpars = new_vpars[1:num_detpars]

    for i in 1:num_detpars
        current_vals[i][1] = new_detpars[i]
    end

    return nothing
end



"""
    map_determinantal_parameters( determinantal_parameters::DeterminantalParameters ) 

For a given set of variational parameters, returns a dictionary of 
that reports the value and optimization flag for a given parameter.

"""
function map_determinantal_parameters(determinantal_parameters)
    vparam_map = Dict()
    for i in 1:length(determinantal_parameters.vals)
       vparam_map[determinantal_parameters.pars[i]] = determinantal_parameters.vals[i]
    end
    return vparam_map
end


"""

    build_tight_binding_model( tight_binding_model::TightBindingModel ) 

Constructs a 2 × n × N by 2 × n × N Hamiltonian matrix, where n is the
number of orbitals per unit cell and N is the number of lattice sites,
given tight binding parameters t, t', and μ. TODO: change tight binding
parameters to tx,ty,t' for future SSH model functionality.

"""
function build_tight_binding_model(tight_binding_model)
    N = model_geometry.unit_cell.n*model_geometry.lattice.N 
    nbr_table = build_neighbor_table(bonds[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice);
    H_t = zeros(Complex, 2*N, 2*N);
    H_tp = zeros(Complex, 2*N, 2*N);
    μ_vec = Vector{Complex}(undef, 2*N);

    if debug
        println("Building tight binding model...")
        println("Hopping parameters:")
        println("t0 = ", tight_binding_model.t[1])
        println("t1 = ", tight_binding_model.t[2])
    end

    if pht == true
        # particle-hole transformed chemical potential
        if !("μ_BCS" in parameters_to_optimize)     
            for i in 1:N
                for j in N+1:2*N
                    μ_vec[i] = -tight_binding_model.μ;
                    μ_vec[j] = tight_binding_model.μ;
                end 
            end
        end
        # particle-hole transformed nearest neighbor hopping
        if Lx == 2 && Ly == 2 
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -tight_binding_model.t[1];
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += tight_binding_model.t[1];
            end
        # special case for 1D
        elseif  Lx == 1 && Ly > Lx || Ly == 1 && Lx > Ly
            for (i,j) in eachcol(nbr_table[:,1:N])
                H_t[i,j] += -tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -tight_binding_model.t[1];
                end
            end
            for (i,j) in eachcol(nbr_table[:,1:N] .+ N)    
                H_t[i,j] += tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += tight_binding_model.t[1];
                end
            end
        # special case for Lx = 2 
        elseif Lx == 2 && Ly > Lx
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)])
                H_t[i,j] += -tight_binding_model.t[1];
                H_t[j,i] += -tight_binding_model.t[1];
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)] .+ N)
                H_t[i,j] += tight_binding_model.t[1];
                H_t[j,i] += tight_binding_model.t[1];
            end 
        # special case for Ly = 2 
        elseif Ly == 2 && Lx > Ly
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)])
                H_t[i,j] += -tight_binding_model.t[1];
                H_t[j,i] += -tight_binding_model.t[1];
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)] .+ N)
                H_t[i,j] += tight_binding_model.t[1];
                H_t[j,i] += tight_binding_model.t[1];
            end 
        else
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -tight_binding_model.t[1];
                else
                end
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += tight_binding_model.t[1];
                else
                end
            end
        end
        # particle-hole transformed next nearest neighbor hopping
        if tight_binding_model.t[2] != 0.0
            nbr_table_p = build_neighbor_table(bonds[2],
                                            model_geometry.unit_cell,
                                            model_geometry.lattice);
            if Lx == 2 && Ly == 2
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += tight_binding_model.t[2]/2;
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += -tight_binding_model.t[2]/2;
                end
            else
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += tight_binding_model.t[2];
                    H_tp[j,i] += tight_binding_model.t[2];
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += -tight_binding_model.t[2];
                    H_tp[j,i] += -tight_binding_model.t[2];
                end
            end
        else
        end
    else
        # chemical potential
        if !("μ_BCS" in parameters_to_optimize)
            for i in 1:N
                for j in N+1:2*N
                    μ_vec[i] = -tight_binding_model.μ;
                    μ_vec[j] = -tight_binding_model.μ;
                end 
            end
        end
        # nearest neighbor hopping
        if Lx == 2 && Ly == 2 
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -tight_binding_model.t[1];
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += -tight_binding_model.t[1];
            end
        # special case for 1D  
        elseif  Lx == 1 && Ly > Lx || Ly == 1 && Lx > Ly
            for (i,j) in eachcol(nbr_table[:,1:N])
                H_t[i,j] += -tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -tight_binding_model.t[1];
                end
            end
            for (i,j) in eachcol(nbr_table[:,1:model_geometry.lattice.N] .+ N)    
                H_t[i,j] += -tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -tight_binding_model.t[1];
                end
            end
        # special case for Lx = 2 
        elseif Lx == 2 && Ly > Lx
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)])
                H_t[i,j] += -tight_binding_model.t[1];
                H_t[j,i] += -tight_binding_model.t[1];
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)] .+ N)
                H_t[i,j] += -tight_binding_model.t[1];
                H_t[j,i] += -tight_binding_model.t[1];
            end 
        # special case for Ly = 2 
        elseif Ly == 2 && Lx > Ly
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)])
                H_t[i,j] += -tight_binding_model.t[1];
                H_t[j,i] += -tight_binding_model.t[1];
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)] .+ N)
                H_t[i,j] += -tight_binding_model.t[1];
                H_t[j,i] += -tight_binding_model.t[1];
            end  
        else
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -tight_binding_model.t[1];
                else
                end
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += -tight_binding_model.t[1];
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -tight_binding_model.t[1];
                else
                end
            end
        end
        # next nearest neighbor hopping
        if tight_binding_model.t[2] != 0.0
            nbr_table_p = build_neighbor_table(bonds[2],
                                            model_geometry.unit_cell,
                                            model_geometry.lattice);
            if Lx == 2 && Ly ==2 
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += tight_binding_model.t[2]/2;
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += tight_binding_model.t[2]/2;
                end
            else
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += tight_binding_model.t[2];
                    H_tp[j,i] += tight_binding_model.t[2];
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += tight_binding_model.t[2];
                    H_tp[j,i] += tight_binding_model.t[2];
                end
            end
        end
    end

    if !("μ_BCS" in parameters_to_optimize)
        if debug
            println("Adding starting chemical potential...")
        end

        return H_t + H_tp + LinearAlgebra.Diagonal(μ_vec);
    else
        return H_t + H_tp;
    end
end


"""

    build_variational_terms( determinantal_parameters::DeterminantalParameters ) 

Constructs a 2 × n × N by 2 × n × N matrices to be added to the non-interacting tight binding
Hamiltonian for each variational parameter. Returns a vector of the sum of
matrices and a vector of individual matrix terms.

"""
function build_variational_terms(determinantal_parameters)
    # model parameters
    N = model_geometry.unit_cell.n*model_geometry.lattice.N
    L = model_geometry.lattice.L
    vparam_map = map_determinantal_parameters(determinantal_parameters) 

    # initial matrices
    Hs = zeros(Complex, 2*N, 2*N) 
    Ha = zeros(Complex, 2*N, 2*N)    
    Hc = zeros(Complex, 2*N, 2*N) 
    Hμ = zeros(Complex, 2*N, 2*N)
    Hd = zeros(Complex, 2*N, 2*N)       
    
    # store vpar matrices
    H_vpars = []
    V = []

    # s-wave pairing order
    if haskey(vparam_map, "Δs") == true
        # ensure that particle-hole transformation is on
        @assert pht == true

        if debug
            println("Adding Δs term...")
            println("Initial Δs = ", vparam_map["Δs"][1])
        end

        bA = zeros(AbstractFloat, N, N);
        bD = zeros(AbstractFloat, N, N);
        Δ_vec_bB = Vector{AbstractFloat}(undef, N);
        Δ_vec_bC = Vector{AbstractFloat}(undef, N);
        for i in 1:N
            Δ_vec_bB[i] = 1;
            Δ_vec_bC[i] = 1;
        end
        bB = LinearAlgebra.Diagonal(Δ_vec_bB);
        bC = LinearAlgebra.Diagonal(Δ_vec_bC);
        Vs = Matrix([bA bB; bC bD]);
        Hs += vparam_map["Δs"][1]*Vs;
        push!(H_vpars,Hs);
        push!(V, Vs);
    end

    # d-wave pairing order
    if haskey(vparam_map, "Δd") == true
        # ensure that particle-hole transformation is on
        @assert pht == true

        if debug
            println("Adding Δd term...")
            println("Initial Δd = ", vparam_map["Δd"][1])
        end

        # create neighbor table
        nbr_table = build_neighbor_table(bonds[1],
        model_geometry.unit_cell,
        model_geometry.lattice)

        # maps neighbor table to dictionary of bonds and neighbors                                
        nbrs = map_neighbor_table(nbr_table)

        # Predefine the displacement-to-sign map
        disp_sign_map = Dict([1,0] => 1, [0,1] => -1, [-1,0] => 1, [0,-1] => -1)

        # initial variational operator matrix
        Vdwave = zeros(AbstractFloat, 2*N, 2*N)

        for i in 1:N
            # get all neighbors of site i
            nn = nbrs[i][2]

            # loop over neighbors
            for j in nn
                # Find the displacement between site i and one of its neighbors j
                disp = sites_to_displacement(i, j, unit_cell, lattice)

                # Lookup sign of Δd
                dsgn = get(disp_sign_map, disp, 0)  # Default to 0 if no match

                if dsgn != 0
                    println(dsgn > 0 ? "+1" : "-1")

                    # Store spin-down indices
                    idn_idx = get_spindices_from_index(i, model_geometry)[2]
                    jdn_idx = get_spindices_from_index(j, model_geometry)[2]

                    # Add elements to variational operator
                    Vdwave[i, jdn_idx] = dsgn 
                    Vdwave[j, idn_idx] = dsgn 
                    Vdwave[jdn_idx, i] = dsgn 
                    Vdwave[idn_idx, j] = dsgn 
                end
            end
        end

        Hd += vparam_map["Δd"][1]*Vdwave
        push!(H_vpars,Hd)
        push!(V, Vdwave)
    end

    # antiferromagnetic (Neél) order
    if haskey(vparam_map, "Δa") == true
        if debug
            println("Adding Δa term...")
            println("Initial Δa = ", vparam_map["Δa"][1])
        end
        # diagonal vector
        afm_vec = fill(1,2*N)

        # account for particle-hole transformation
        if pht
            # stagger
            for s in 1:2*N
                idx = get_index_from_spindex(s, model_geometry)
                loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)
                if length(L) == 1
                    afm_vec[s] *= (-1)^(loc[1][1])
                else
                    afm_vec[s] *= (-1)^(loc[1][1]+loc[1][2])
                end
            end

            # store variational operator
            Vafm = LinearAlgebra.Diagonal(afm_vec)
            
            # create Hamiltonian term
            Ha += vparam_map["Δa"][1]*Vafm
            push!(H_vpars,Ha)
            push!(V, Vafm)
        else
            # account for minus sign 
            afm_vec_neg = copy(afm_vec)
            afm_vec_neg[N+1:2*N] .= -afm_vec_neg[N+1:2*N]

            # stagger
            for s in 1:2*N
                idx = get_index_from_spindex(s, model_geometry)
                loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)
                if length(L) == 1
                    afm_vec_neg[s] *= (-1)^(loc[1][1])
                else
                    afm_vec_neg[s] *= (-1)^(loc[1][1]+loc[1][2])
                end
            end

            # store variational operator
            Vafm_neg = LinearAlgebra.Diagonal(afm_vec_neg)
            
            # create Hamiltonian term
            Ha += vparam_map["Δa"][1]*Vafm_neg
            push!(H_vpars,Ha)
            push!(V, Vafm_neg)
        end
    end

    # uniform charge density wave order
    if haskey(vparam_map, "Δc") == true
        if debug
            println("Adding Δc term...")
            println("Initial Δc = ", vparam_map["Δc"][1])
        end
        # diagonal vector
        cdw_vec = fill(1,2*N)

        # account for particle-hole transformation
        if pht
            # account for minus sign 
            cdw_vec_neg = copy(cdw_vec)
            cdw_vec_neg[N+1:2*N] .= -cdw_vec_neg[N+1:2*N]

            # stagger
            for s in 1:2*N
                idx = get_index_from_spindex(s, model_geometry)
                loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)
                cdw_vec_neg[s] *= (-1)^(loc[1][1]+loc[1][2])
            end

            # store variational operator
            Vcdw = LinearAlgebra.Diagonal(cdw_vec)

            # account for particle-hole transformation
            Hc += vparam_map["Δc"][1]*Vcdw
            push!(H_vpars,Hc)
            push!(V, Vcdw)
        else
            # stagger
            for s in 1:2*N
                idx = get_index_from_spindex(s, model_geometry)
                loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)
                cdw_vec_neg[s] *= (-1)^(loc[1][1]+loc[1][2])
            end

            # store variational operator
            Vcdw_neg = LinearAlgebra.Diagonal(cdw_vec_neg)

            # create Hamiltonian term
            Hc += vparam_map["Δc"][1]*Vcdw_neg
            push!(H_vpars,Hc)
            push!(V, Vcdw_neg)
        end
    end

    # BCS chemical potential
    if haskey(vparam_map, "μ_BCS") == true
        if debug
            println("Adding μ_BCS term...")
            println("Initial μ_BCS = ", vparam_map["μ_BCS"][1])
        end
        # diagonal vector
        μ_vec = fill(-1,2*N);

        if pht
            # account for minus sign 
            μ_vec_neg = copy(μ_vec);
            μ_vec_neg[N+1:2*N] .= -μ_vec_neg[N+1:2*N];

            # store variational operator
            Vμ_neg = LinearAlgebra.Diagonal(μ_vec_neg);

            #
            Hμ += vparam_map["μ_BCS"][1]*Vμ_neg;
            push!(H_vpars,Hμ);
            push!(V, Vμ_neg);
        else
            Vμ = LinearAlgebra.Diagonal(μ_vec);

            Hμ += vparam_map["μ_BCS"][1]*Vμ;
            push!(H_vpars,Hμ);
            push!(V, Vμ);
        end
    end

    # charge stripe order
    if haskey(vparam_map, "Δcs") == true
        # ensure that particle-hole transformation is off
        @assert pht == false

        if debug
            println("Adding Δcs term...")
        end

        # store diagonal vectors
        cs_vectors = []

        # populate vectors
        for shift in 0:(L[1]-1)
            vec = zeros(Int, 2 * N)  # Initialize a vector of zeros with length 2*N
            for i in 1:2*L[1]
                idx = (i-1)*L[1] + 1 + shift  # Compute the index
                if idx <= 2 * N  # Ensure the index does not exceed the bounds of vec
                    vec[idx] = 1  # Place a "1" if the index is valid
                end
            end
            # for i in 1:2*L[1]
            #     vec[(i-1)*L[1] + 1 + shift] = 1  # Place a "1" every Lth element, shifted by `shift`
            # end
            push!(cs_vectors, vec)  # Store the vector in the list of vectors
        end

        iter = 0

        for cs_vec in cs_vectors
            Hcs = zeros(AbstractFloat, 2*N, 2*N)
            iter += 1
            Vcs = LinearAlgebra.Diagonal(cs_vec)

            Hcs += vparam_map["Δcs"][iter]*Vcs
            push!(H_vpars,Hcs)
            push!(V, Vcs)
        end
    end

    # spin stripe order
    if haskey(vparam_map, "Δss") == true
        # ensure that particle-hole transformation is off
        @assert pht == false

        if debug
            println("Adding Δss term...")
        end


        # store diagonal vectors
        ss_vectors = []

        # populate vectors
        for shift in 0:(L[1]-1)
            vec = zeros(Int, 2 * N)  # Initialize a vector of zeros with length 2*N
            for i in 1:2*L[1]
                idx = (i-1)*L[1] + 1 + shift  # Compute the index
                if idx <= 2 * N  # Ensure the index does not exceed the bounds of vec
                    vec[idx] = 1  # Place a "1" if the index is valid
                end
            end
            # for i in 1:(2*L[1])
            #     vec[(i-1)*L[1] + 1 + shift] = 1  # Place a "1" every Lth element, shifted by `shift`
            # end
            push!(ss_vectors, vec)  # Store the vector in the list of vectors
        end

        iter = 0

        for ss_vec in ss_vectors
            iter += 1

            Hss = zeros(AbstractFloat, 2*N, 2*N)
            ss_vec_neg = copy(ss_vec)
            ss_vec_neg[N+1:2*N] .= -ss_vec_neg[N+1:2*N]

            for s in 1:2*N
                idx = get_index_from_spindex(s, model_geometry)
                loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)
                ss_vec_neg[s] *= (-1)^(loc[1][1]+loc[1][2])
            end

            # Vss = LinearAlgebra.Diagonal(ss_vec)
            Vss_neg = LinearAlgebra.Diagonal(ss_vec_neg)

            Hss += vparam_map["Δss"][iter]*Vss_neg
            push!(H_vpars,Hss)
            push!(V, Vss_neg)
        end
    end

    return [sum(H_vpars),V];
end


"""

    build_mean_field_hamiltonian( tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters ) 

Constructs a matrix by combining the non-interacting Hamiltonian with
matrix of variational terms.

"""
function build_mean_field_hamiltonian(tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters)
    if debug
        println("Building mean-field Hamiltonian...")
    end
    return build_tight_binding_model(tight_binding_model) + build_variational_terms(determinantal_parameters)[1], build_variational_terms(determinantal_parameters)[2];
end


"""

    get_Ak_matrices( V::Vector{Matrix{AbstractFloat}}, U::Matrix{AbstractFloat}, ε::Vector{AbstractFloat}, model_geometry::ModelGeometry ) 
    
Returns variational parameter matrices Aₖ from the corresponding Vₖ. Computes Qₖ = (U⁺VₖU)_(ην) / (ε_η - ε_ν), for η > Nₚ and ν ≤ Nₚ and is 0 otherwise
(η and ν run from 1 to 2L)

"""
function get_Ak_matrices(V, Uₑ, ε, model_geometry)
    if debug
        println("Building A matrices...")
    end

    N = model_geometry.unit_cell.n * model_geometry.lattice.N;

    # define perturbation mask
    ptmask = zeros(Float64, 2*N, 2*N);
    # for η in 1:2*N
    #     for ν in 1:2*N
    #         if η >= Ne + 1 && ν < Ne + 1
    #             ptmask[η, ν] = 1.0 / (ε[ν] - ε[η]);
    #         end
    #     end
    # end

    # # calculate A matrices of variational parameters
    # int_A = [];
    # for it in V
    #     A = Uₑ * (Uₑ' * it * Uₑ) .* ptmask * Uₑ';
    #     A = A[end:-1:1,:];
    #     push!(int_A, A);
    # end
        
    for η in 1:2*N
        for ν in 1:2*N
            if η > Ne && ν <= Ne
                ptmask[η, ν] = 1.0 / ε[ν] - ε[η]
            end
        end
    end

    int_A = [];
    for v in V
        push!(int_A, Uₑ * ((Uₑ' * v * Uₑ) .* ptmask) * Uₑ')
    end
    
    return int_A;
end

















