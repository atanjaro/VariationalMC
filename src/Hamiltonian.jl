"""

    ModelGeometry( unit_cell::UnitCell, lattice::Lattice, bond::Vector{Vector{Any}} )

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

    TightBindingModel( t::Vector{AbstractFloat}, μ::AbstractFloat )

A type defining a non-interacting tight binding model.

"""
struct TightBindingModel    
    # hopping amplitudes
    t::Vector{AbstractFloat}   # [t, t']  

    # chemical potentials
    μ::AbstractFloat           
end


"""

    SpinModel

A type defining a spin model.

"""
struct SpinModel
    # nearest neighbor spin exchange coupling 
    J₁::Float64

    # next nearest neighbor spin exchange coupling
    J₂::Float64

    # next next nearest neighbor spin exchange coupling
    J₃::Float64
end


"""

    DeterminantalParameters( pars::Vector{AbstractString}, 
                            vals::Vector{AbstractFloat}, num_detpars::Int )

A type defining a set of variational parameters for the determinantal wavefunction.

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

    initialize_determinantal_parameters( pars:;Vector{AbstractString}, 
                                        vals::Vector{AbstractFloat} )::DeterminantalParameters

Creates an instances of the DeterminantalParameters type.

"""
function initialize_determinantal_parameters(pars, vals)::DeterminantalParameters
    @assert length(pars) == length(vals) "Input vectors must have the same length"
    
    # Calculate num_detpars as the total number of elements in all inner vectors of vals
    num_detpars = sum(length, vals)

    return DeterminantalParameters(pars, vals, num_detpars)
end


"""

    build_mean_field_hamiltonian( tight_binding_model::TightBindingModel, 
                                    determinantal_parameters::DeterminantalParameters ) 

Constructs a matrix by combining the non-interacting Hamiltonian with
matrix of variational terms.

"""
function build_mean_field_hamiltonian(tight_binding_model::TightBindingModel, 
                                        determinantal_parameters::DeterminantalParameters)
    return build_tight_binding_model(tight_binding_model) + build_variational_terms(determinantal_parameters)[1], 
            build_variational_terms(determinantal_parameters)[2];
end


"""

    update_detpars!( determinantal_parameters::DeterminantalParameters, Vector{Float64} )

Updates determinantal_parameters.

"""
function update_detpars!(determinantal_parameters::DeterminantalParameters, new_vpars::Vector{Float64})
    num_detpars = determinantal_parameters.num_detpars
    current_vals = determinantal_parameters.vals

    new_detpars = new_vpars[1:num_detpars]

    for i in 1:num_detpars
        current_vals[i][1] = new_detpars[i]
    end

    return nothing
end


"""

    build_tight_binding_model( tight_binding_model::TightBindingModel ) 

Constructs a 2 × n × N by 2 × n × N Hamiltonian matrix, where n is the number of orbitals 
per unit cell and N is the number of lattice sites, given tight binding parameters t, t', and μ. 

"""
function build_tight_binding_model(tight_binding_model::TightBindingModel)
    # number of sites
    N = model_geometry.unit_cell.n*model_geometry.lattice.N 

    # generate neighbor table
    nbr_table = build_neighbor_table(bonds[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice);

    # initialize matrices
    H_t = zeros(Complex, 2*N, 2*N);
    H_tp = zeros(Complex, 2*N, 2*N);
    μ_vec = Vector{Complex}(undef, 2*N);

    # hopping parameters
    t0 = tight_binding_model.t[1]
    t1 = tight_binding_model.t[2]

    # initial chemical potential
    μ = tight_binding_model.μ

    debug && println("Hamiltonian::build_tight_binding_model() : ")
    debug && println("building tight binding model")
    debug && println("hopping : t0 = ", t0)
    debug && println("hopping : t1 = ", t1)
    debug && println("particle-hole transformation : ", pht)
    debug && println("optimize μ_BCS : ", "μ_BCS" in parameters_to_optimize)

    if pht == true
        # particle-hole transformed chemical potential
        if !("μ_BCS" in parameters_to_optimize)     
            for i in 1:N
                for j in N+1:2*N
                    μ_vec[i] = -μ;
                    μ_vec[j] = μ;
                end 
            end
        end
        # particle-hole transformed nearest neighbor hopping
        if Lx == 2 && Ly == 2 
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -t0;
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += t0;
            end
        # special case for Lx = 2 
        elseif Lx == 2 && Ly > Lx
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)])
                H_t[i,j] += -t0;
                H_t[j,i] += -t0;
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)] .+ N)
                H_t[i,j] += t0;
                H_t[j,i] += t0;
            end 
        # special case for Ly = 2 
        elseif Ly == 2 && Lx > Ly
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)])
                H_t[i,j] += -t0;
                H_t[j,i] += -t0;
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)] .+ N)
                H_t[i,j] += t0;
                H_t[j,i] += t0;
            end 
        else
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -t0;
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -t0;
                else
                end
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += t0;
                if model_geometry.lattice.N > 2
                    H_t[j,i] += t0;
                else
                end
            end
        end
        # particle-hole transformed next nearest neighbor hopping
        if t1 != 0.0
            nbr_table_p = build_neighbor_table(bonds[2],
                                            model_geometry.unit_cell,
                                            model_geometry.lattice);
            if Lx == 2 && Ly == 2
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += t1/2;
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += -t1/2;
                end
            else
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += t1;
                    H_tp[j,i] += t1;
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += -t1;
                    H_tp[j,i] += -t1;
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
                H_t[i,j] += -t0
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += -t0;
            end
        # special case for Lx = 2 
        elseif Lx == 2 && Ly > Lx
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)])
                H_t[i,j] += -t0;
                H_t[j,i] += -t0;
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)] .+ N)
                H_t[i,j] += -t0;
                H_t[j,i] += -t0;
            end 
        # special case for Ly = 2 
        elseif Ly == 2 && Lx > Ly
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)])
                H_t[i,j] += -t0;
                H_t[j,i] += -t0;
            end
            for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)] .+ N)
                H_t[i,j] += -t0;
                H_t[j,i] += -t0;
            end  
        else
            for (i,j) in eachcol(nbr_table)
                H_t[i,j] += -t0;
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -t0;
                else
                end
            end
            for (i,j) in eachcol(nbr_table .+ N)    
                H_t[i,j] += -t0;
                if model_geometry.lattice.N > 2
                    H_t[j,i] += -t0;
                else
                end
            end
        end
        # next nearest neighbor hopping
        if t1 != 0.0
            nbr_table_p = build_neighbor_table(bonds[2],
                                            model_geometry.unit_cell,
                                            model_geometry.lattice);
            if Lx == 2 && Ly ==2 
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += t1/2;
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += t1/2;
                end
            else
                for (i,j) in eachcol(nbr_table_p)
                    H_tp[i,j] += t1;
                    H_tp[j,i] += t1;
                end
                for (i,j) in eachcol(nbr_table_p .+ N)    
                    H_tp[i,j] += t1;
                    H_tp[j,i] += t1;
                end
            end
        end
    end

    if !("μ_BCS" in parameters_to_optimize)
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
    # lattice sites
    N = model_geometry.unit_cell.n*model_geometry.lattice.N

    # one side of the lattice
    L = model_geometry.lattice.L

    # map of available variational parameters
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

    # s-wave pairing 
    if haskey(vparam_map, "Δs") == true
        # ensure that particle-hole transformation is on
        @assert pht == true

        Vs = copy(Hs)

        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding s-wave term")
        debug && println("initial Δs = ", vparam_map["Δs"][1])
        
        # populate variational operator
        for i in 0:(2 * N - 1)
            Vs[i + 1, get_linked_spindex(i, N) + 1] = 1.0  
        end

        # add variational term
        Hs += vparam_map["Δs"][1]*Vs;
        push!(H_vpars,Hs);
        push!(V, Vs);
    end

    # d-wave pairing 
    if haskey(vparam_map, "Δd") == true
        # ensure that particle-hole transformation is on
        @assert pht == true

        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding d-wave term")
        debug && println("initial Δd = ", vparam_map["Δd"][1])

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
        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding spin-z term")
        debug && println("initial Δa = ", vparam_map["Δa"][1])

        # diagonal vector
        afm_vec = fill(1,2*N)

        # account for particle-hole transformation
        if pht
            # stagger
            for s in 1:2*N
                # get proper site index
                idx = get_index_from_spindex(s, model_geometry)

                # 1D
                if length(L) == 1
                    # get site coordinates
                    ix = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1][1]

                    # apply phase
                    afm_vec[s] *= (-1)^(ix)
                # 2D
                elseif length(L) == 2
                    # get site coordinates
                    (ix, iy) = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1]

                    # apply phase
                    afm_vec[s] *= (-1)^(ix + iy)
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

            # flip the spin down sector in 2D
            if length(L) == 2
                afm_vec_neg[N+1:2*N] .= -afm_vec_neg[N+1:2*N]
            end

            # stagger
            for s in 1:2*N
                # get proper site index
                idx = get_index_from_spindex(s, model_geometry)

                # 1D
                if length(L) == 1
                    # get site coordinates
                    ix = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1][1]

                    # apply phase
                    afm_vec_neg[s] *= (-1)^(ix)
                # 2D
                elseif length(L) == 2
                    # get site coordinates
                    (ix, iy) = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1]

                    # apply phase
                    afm_vec_neg[s] *= (-1)^(ix + iy)
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

    # uniform charge density wave 
    if haskey(vparam_map, "Δc") == true
        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding charge term")
        debug && println("initial Δc = ", vparam_map["Δc"][1])
      
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

    # (BCS) chemical potential
    if haskey(vparam_map, "μ_BCS") == true
        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding chemical potential term")
        debug && println("initial μ_BCS = ", vparam_map["μ_BCS"][1])
     
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

    # charge stripe 
    if haskey(vparam_map, "Δcs") == true
        # ensure that particle-hole transformation is off
        @assert pht == false

        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding stripe term")
        debug && println("initial Δcs = ", vparam_map["Δcs"][1])

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

    # spin stripe 
    if haskey(vparam_map, "Δss") == true
        # ensure that particle-hole transformation is off
        @assert pht == false

        debug && println("Hamiltonian::build_variational_terms() : ")
        debug && println("adding stripe term")
        debug && println("initial Δss = ", vparam_map["Δss"][1])

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

    get_variational_matrices( V::Vector{Any}, U_int::Matrix{ComplexF64}, 
                            ε::Vector{Float64}, model_geometry::ModelGeometry )::Vector{Any} 
    
Returns variational parameter matrices Aₖ from the corresponding Vₖ. Computes Qₖ = (U⁺VₖU)_(ην) / (ε_η - ε_ν), 
for η > Nₚ and ν ≤ Nₚ and is 0 otherwise (η and ν run from 1 to 2L).

"""
function get_variational_matrices(V::Vector{Any}, U_int::Matrix{ComplexF64}, 
                                    ε::Vector{Float64}, model_geometry::ModelGeometry)::Vector{Any}
    # number of lattice sites
    N = model_geometry.unit_cell.n * model_geometry.lattice.N;

    # define perturbation mask
    ptmask = zeros(Float64, 2*N, 2*N);
        
    for η in 1:2*N
        for ν in 1:2*N
            if η > Ne && ν <= Ne
                ptmask[η, ν] = 1.0 / (ε[ν] - ε[η])
            end
        end
    end

    int_A = [];
    for v in V
        push!(int_A, U_int * ((U_int' * v * U_int) .* ptmask) * U_int')
    end

    return int_A;
end


"""
    map_determinantal_parameters( determinantal_parameters::DeterminantalParameters ) 

For a given set of variational parameters, returns a dictionary of that reports the value 
and optimization flag for a given parameter.

"""
function map_determinantal_parameters(determinantal_parameters::DeterminantalParameters)
    vparam_map = Dict()

    for i in 1:length(determinantal_parameters.vals)
       vparam_map[determinantal_parameters.pars[i]] = determinantal_parameters.vals[i]
    end

    return vparam_map
end


"""

    get_tb_chem_pot( Ne::Int64, tight_binding_model::TightBindingModel, model_geometry::ModelGeometry )::Float64

For a tight-binding model that has not been particle-hole transformed, returns the  
chemical potential.

"""
function get_tb_chem_pot(Ne::Int64, tight_binding_model::TightBindingModel, model_geometry::ModelGeometry)::Float64
    @assert pht == false

    # number of lattice sites
    N = model_geometry.lattice.N
    
    # preallocate matrices
    H_t = zeros(Complex, 2*N, 2*N);
    H_tp = zeros(Complex, 2*N, 2*N);

    # hopping amplitudes
    t0 = tight_binding_model.t[1];
    t1 = tight_binding_model.t[2];

    # nearest neighbor table
    nbr_table = build_neighbor_table(bonds[1],
                                        model_geometry.unit_cell,
                                        model_geometry.lattice);


    # nearest neighbor hopping
    # special case for Lx, Ly = 2
    if Lx == 2 && Ly == 2 
        for (i,j) in eachcol(nbr_table)
            H_t[i,j] += -t0;
        end
        for (i,j) in eachcol(nbr_table .+ N)    
            H_t[i,j] += -t0;
        end
    # special case for Lx = 2 
    elseif Lx == 2 && Ly > Lx
        for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)])
            H_t[i,j] += -t0;
            H_t[j,i] += -t0;
        end
        for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Ly)] .+ N)
            H_t[i,j] += -t0;
            H_t[j,i] += -t0;
        end 
    # special case for Ly = 2 
    elseif Ly == 2 && Lx > Ly
        for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)])
            H_t[i,j] += -t0;
            H_t[j,i] += -t0;
        end
        for (i,j) in eachcol(nbr_table[:,1:(size(nbr_table,2) - Lx)] .+ N)
            H_t[i,j] += -t0;
            H_t[j,i] += -t0;
        end  
    else
        for (i,j) in eachcol(nbr_table)
            H_t[i,j] += -t0;
            if model_geometry.lattice.N > 2
                H_t[j,i] += -t0;
            else
            end
        end
        for (i,j) in eachcol(nbr_table .+ N)    
            H_t[i,j] += -t0;
            if model_geometry.lattice.N > 2
                H_t[j,i] += -t0;
            else
            end
        end
    end
    # next nearest neighbor hopping
    if t1 != 0.0
        # next nearest neighbor table
        nbr_table_p = build_neighbor_table(bonds[2],
                                        model_geometry.unit_cell,
                                        model_geometry.lattice);
        if Lx == 2 && Ly ==2 
            for (i,j) in eachcol(nbr_table_p)
                H_tp[i,j] += 0.5 * t1;
            end
            for (i,j) in eachcol(nbr_table_p .+ N)    
                H_tp[i,j] += 0.5 * t1;
            end
        else
            for (i,j) in eachcol(nbr_table_p)
                H_tp[i,j] += t1;
                H_tp[j,i] += t1;
            end
            for (i,j) in eachcol(nbr_table_p .+ N)    
                H_tp[i,j] += t1;
                H_tp[j,i] += t1;
            end
        end
    end

    # full tight-binding Hamiltonian
    H_tb = H_t + H_tp

    # solve for eigenvalues
    ε_F, Uₑ = diagonalize(H_tb)

    # tight-binding chemical potential
    μ = 0.5 * (ε_F[Ne + 1] + ε_F[Ne])

    debug && println("Hamiltonian::get_tb_chem_pot() : ")
    debug && println("tight-binding chemical potential")
    debug && println("μ = ", μ)

    return μ
end

















