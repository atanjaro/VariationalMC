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

    TightBindingModel( t::Vector{AbstractFloat})

A type defining a non-interacting tight binding model.

"""
struct TightBindingModel    
    # nearest neighbor hopping amplitude
    t₀::Float64

    # next nearest neighbor hopping amplitude
    t₁::Float64
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
mutable struct DeterminantalParameters
    # determinantal parameters and their values
    det_pars::NamedTuple

    # total number of determinantal parameters
    num_det_pars::Int

    # total number of determinantal parameters being optimized
    num_det_opts::Int
end


"""

    DeterminantalParameters( optimize::NamedTuple, tight_binding_model::TightBindingModel, 
                                model_geometry::ModelGeometry, minabs_vpar::Float64, Ne::Int, pht::Bool )

Given an intial set of parameters and set of optimization flags, generates a set of variational parameters.
"""
function DeterminantalParameters(optimize::NamedTuple, tight_binding_model::TightBindingModel, 
                                model_geometry::ModelGeometry, minabs_vpar, Ne::Int, pht::Bool)
    # dimensions
    dims = size(model_geometry.lattice.L)[1]

    # x-dimension
    Lx = model_geometry.lattice.L[1]

    # TODO: change starting values of these parameters
    if dims > 1
        if pht
            det_pars = (
                μ = 0.0,
                Δ_0 = minabs_vpar,
                Δ_d = 0.0,
                Δ_afm = minabs_vpar,
                Δ_cdw = 0.0,
                Δ_sdc = fill(0.0, Lx),
                Δ_sds = fill(0.0, Lx)
            )
        else
            det_pars = (
                μ = get_tb_chem_pot(Ne, tight_binding_model, model_geometry),
                Δ_afm = minabs_vpar,
                Δ_cdw = 0.0,
                Δ_sdc = fill(0.0, Lx),
                Δ_sds = fill(0.0, Lx)
            )
        end
    else
        if pht
            det_pars = (
                μ = 0.0,
                Δ_0 = minabs_vpar,
                Δ_afm = minabs_vpar,
                Δ_cdw = 0.0,
            )
        else
            det_pars = (
                μ = get_tb_chem_pot(Ne, tight_binding_model, model_geometry),
                Δ_afm = minabs_vpar,
                Δ_cdw = 0.0,
            )
        end
    end

    # determine total number of determinantal parameters being added to the model
    num_det_pars = sum(x -> isa(x, AbstractArray) ? length(x) : 1, values(det_pars))

    # determine the number of determinantal parameters being optimized
    opt_keys = intersect(keys(optimize), keys(det_pars))
    num_det_opts = sum(opt_keys) do key
        opt = getfield(optimize, key)
        val = getfield(det_pars, key)
        opt ? (isa(val, AbstractArray) ? length(val) : 1) : 0
    end


    debug && println("Hamiltonian::DeterminantalParameters() : ")
    debug && println("Number of determinantal parameters = $num_det_pars")
    debug && println("Number of determinantal parameters to be optimized = $num_det_opts")

    return DeterminantalParameters(det_pars, num_det_pars, num_det_opts)
end


"""

    DeterminantalParameters( optimize::NamedTuple, model_geometry::ModelGeometry, pht::Bool, path_to_parameter_file::String )

Given an intial set of parameters and set of optimization flags, generates a set of variational parameters.
"""
function DeterminantalParameters(optimize::NamedTuple, model_geometry::ModelGeometry, pht::Bool, path_to_parameter_file::String)
    # dimensions
    dims = size(model_geometry.lattice.L)[1]

    # get parameters from file
    vpar_dict = readin_parameters(path_to_parameter_file)

    if dims > 1
        if pht
            det_pars = (
                μ = vpar_dict[:chemical_potential],
                Δ_0 = vpar_dict[:pairing][1],
                Δ_d = vpar_dict[:pairing][2],
                Δ_afm = vpar_dict[:afm],
                Δ_cdw = vpar_dict[:cdw],
                Δ_sdc = vpar_dict[:sdc],
                Δ_sds = vpar_dict[:sds]
            )
        else
            det_pars = (
                μ = vpar_dict[:chemical_potential],
                Δ_afm = vpar_dict[:afm],
                Δ_cdw = vpar_dict[:cdw],
                Δ_sdc = vpar_dict[:sdc],
                Δ_sds = vpar_dict[:sds]
            )
        end
    else
        if pht
            det_pars = (
                μ = vpar_dict[:chemical_potential],
                Δ_0 = Δ_0 = vpar_dict[:pairing][1],
                Δ_afm = vpar_dict[:afm],
                Δ_cdw = vpar_dict[:cdw],
            )
        else
            det_pars = (
                μ = vpar_dict[:chemical_potential],
                Δ_afm = vpar_dict[:afm],
                Δ_cdw = vpar_dict[:cdw],
            )
        end
    end
    
    # determine total number of determinantal parameters being added to the model
    num_det_pars = sum(x -> isa(x, AbstractArray) ? length(x) : 1, values(det_pars))

    # determine the number of determinantal parameters being optimized
    opt_keys = intersect(keys(optimize), keys(det_pars))
    num_det_opts = sum(opt_keys) do key
        opt = getfield(optimize, key)
        val = getfield(det_pars, key)
        opt ? (isa(val, AbstractArray) ? length(val) : 1) : 0
    end

    debug && println("Hamiltonian::DeterminantalParameters() : ")
    debug && println("Number of determinantal parameters = $num_det_pars")
    debug && println("Number of determinantal parameters to be optimized = $num_det_opts")

    return DeterminantalParameters(det_pars, num_det_pars, num_det_opts)
end


"""

    build_auxiliary_hamiltonian( tight_binding_model::TightBindingModel, 
                                    determinantal_parameters::DeterminantalParameters, pht::Bool ) 

Constructs a matrix by combining the non-interacting Hamiltonian with
matrix of variational terms.

"""
function build_auxiliary_hamiltonian(tight_binding_model::TightBindingModel, determinantal_parameters::DeterminantalParameters, 
                                    optimize::NamedTuple, model_geometry::ModelGeometry, pht::Bool)
    # hopping matrix
    H_tb = build_tight_binding_hamiltonian(tight_binding_model, model_geometry, pht);

    # variational matrices and operators
    H_var, V = build_variational_hamiltonian(determinantal_parameters, optimize, pht);

    return H_tb + H_var, V
end


"""

    update_parameters!( new_vpars::AbstractVector, determinantal_parameters::DeterminantalParameters, optimize::NamedTuple )

Updates variational parameters.

"""
function update_parameters!(new_vpars::AbstractVector, determinantal_parameters::DeterminantalParameters)
    # extract current parameters
    current_pars = determinantal_parameters.det_pars

    # check that the new values match the number of existing parameters
    if length(new_vpars) != length(current_pars)
        error("Mismatch: got $(length(new_vpars)) new values but $(length(current_pars)) parameters to update.")
    end

    # preserve parameter names from current_pars
    param_names = keys(current_pars)

    # build updated NamedTuple with same keys and new values
    new_det_pars = NamedTuple{Tuple(param_names)}(Tuple(new_vpars))

    # update the struct
    determinantal_parameters.det_pars = new_det_pars

    return nothing
end


# function update_parameters!(new_vpars::AbstractVector, determinantal_parameters::DeterminantalParameters, jastrow::Jastrow, optimize::NamedTuple)

# end


"""

    build_tight_binding_hamiltonian( tight_binding_model::TightBindingModel ) 

Constructs a 2 × n × N by 2 × n × N Hamiltonian matrix, where n is the number of orbitals 
per unit cell and N is the number of lattice sites, given tight binding parameters t, t', and μ. 

"""
function build_tight_binding_hamiltonian(tight_binding_model::TightBindingModel, model_geometry::ModelGeometry, pht::Bool)
    # number of sites
    N = model_geometry.unit_cell.n*model_geometry.lattice.N 

    # generate neighbor table
    nbr0 = build_neighbor_table(bonds[1],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice);

    # initialize matrices
    H_t₀ = zeros(Complex, 2*N, 2*N);
    H_t₁ = zeros(Complex, 2*N, 2*N);

    # hopping parameters
    t₀ = tight_binding_model.t₀; 
    t₁ = tight_binding_model.t₁; 

    debug && println("Hamiltonian::build_tight_binding_hamiltonian() : ")
    debug && println("building tight-binding hopping matrix")
    debug && println("hopping : t₀ = ", t₀)
    debug && println("hopping : t₁ = ", t₁)
    debug && println("particle-hole transformation : ", pht)

    if pht == true
        # add nearest-neighbor hopping
        if Lx == 2 && Ly == 2 
            for (i,j) in eachcol(nbr0)
                H_t₀[i,j] += -t₀;
            end
            for (i,j) in eachcol(nbr0 .+ N)    
                H_t₀[i,j] += t₀;
            end
        # special case for Lx = 2 
        elseif Lx == 2 && Ly > Lx
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Ly)])
                H_t₀[i,j] += -t₀;
                H_t₀[j,i] += -t₀;
            end
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Ly)] .+ N)
                H_t₀[i,j] += t₀;
                H_t₀[j,i] += t₀;
            end 
        # special case for Ly = 2 
        elseif Ly == 2 && Lx > Ly
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Lx)])
                H_t₀[i,j] += -t₀;
                H_t₀[j,i] += -t₀;
            end
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Lx)] .+ N)
                H_t₀[i,j] += t₀;
                H_t₀[j,i] += t₀;
            end 
        else
            for (i,j) in eachcol(nbr0)
                H_t₀[i,j] += -t₀;
                if model_geometry.lattice.N > 2
                    H_t₀[j,i] += -t₀;
                else
                end
            end
            for (i,j) in eachcol(nbr0 .+ N)    
                H_t₀[i,j] += t₀;
                if model_geometry.lattice.N > 2
                    H_t₀[j,i] += t₀;
                else
                end
            end
        end

        # add next-nearest neighbor hopping
        nbr1 = build_neighbor_table(bonds[2],
                                            model_geometry.unit_cell,
                                            model_geometry.lattice);
        if Lx == 2 && Ly == 2
            for (i,j) in eachcol(nbr1)
                H_t₁[i,j] += t₁/2;
            end
            for (i,j) in eachcol(nbr1 .+ N)    
                H_t₁[i,j] += -t₁/2;
            end
        else
            for (i,j) in eachcol(nbr1)
                H_t₁[i,j] += t₁;
                H_t₁[j,i] += t₁;
            end
            for (i,j) in eachcol(nbr1 .+ N)    
                H_t₁[i,j] += -t₁;
                H_t₁[j,i] += -t₁;
            end
        end
    else
        # nearest neighbor hopping
        if Lx == 2 && Ly == 2 
            for (i,j) in eachcol(nbr0)
                H_t₀[i,j] += -t₀;
            end
            for (i,j) in eachcol(nbr0 .+ N)    
                H_t₀[i,j] += -t₀;
            end
        # special case for Lx = 2 
        elseif Lx == 2 && Ly > Lx
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Ly)])
                H_t₀[i,j] += -t₀;
                H_t₀[j,i] += -t₀;
            end
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Ly)] .+ N)
                H_t₀[i,j] += -t₀;
                H_t₀[j,i] += -t₀;
            end 
        # special case for Ly = 2 
        elseif Ly == 2 && Lx > Ly
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Lx)])
                H_t₀[i,j] += -t₀;
                H_t₀[j,i] += -t₀;
            end
            for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Lx)] .+ N)
                H_t₀[i,j] += -t₀;
                H_t₀[j,i] += -t₀;
            end  
        else
            for (i,j) in eachcol(nbr0)
                H_t₀[i,j] += -t₀;
                if model_geometry.lattice.N > 2
                    H_t₀[j,i] += -t₀;
                else
                end
            end
            for (i,j) in eachcol(nbr0 .+ N)    
                H_t₀[i,j] += -t₀;
                if model_geometry.lattice.N > 2
                    H_t₀[j,i] += -t₀;
                else
                end
            end
        end

        # add next-nearest neighbor hopping 
        nbr1 = build_neighbor_table(bonds[2],
                                            model_geometry.unit_cell,
                                            model_geometry.lattice);
        if Lx == 2 && Ly ==2 
            for (i,j) in eachcol(nbr1)
                H_t₁[i,j] += t₁/2;
            end
            for (i,j) in eachcol(nbr1 .+ N)    
                H_t₁[i,j] += t₁/2;
            end
        else
            for (i,j) in eachcol(nbr1)
                H_t₁[i,j] += t₁;
                H_t₁[j,i] += t₁;
            end
            for (i,j) in eachcol(nbr1 .+ N)    
                H_t₁[i,j] += t₁;
                H_t₁[j,i] += t₁;
            end
        end
    end

    return H_t₀ + H_t₁;
end


"""

    build_variational_hamiltonian( determinantal_parameters::DeterminantalParameters ) 

Constructs a 2 × n × N by 2 × n × N matrices to be added to the non-interacting tight binding
Hamiltonian for each variational parameter. Returns a vector of the sum of
matrices and a vector of individual matrix terms.

"""
function build_variational_hamiltonian(determinantal_parameters::DeterminantalParameters, optimize::NamedTuple, pht::Bool)
    # dimensions
    dims = size(model_geometry.lattice.L)[1];
   
    # initialize Hamiltonian and operator matrices
    H_vpars = [];
    V = [];
    
    # add chemical potential term
    add_chemical_potential!(determinantal_parameters, optimize, H_vpars, V, model_geometry, pht);

    debug && println("Hamiltonian::build_variational_hamiltonian() : ")
    debug && println("adding chemical potential matrix")
    debug && println("initial μ = ", determinantal_parameters.det_pars.μ)
    if optimize.μ
        debug && println("optimize = true")
    else
        debug && println("optimize = false")
    end

    # add s-wave term
    if pht == true
        add_pairing_symmetry!("s", determinantal_parameters, optimize, H_vpars, V, model_geometry, pht)

        debug && println("Hamiltonian::build_variational_hamiltonian() : ")
        debug && println("adding s-wave pairing matrix")
        debug && println("initial Δ_0 = ", determinantal_parameters.det_pars.Δ_0)
        if optimize.Δ_0
            debug && println("optimize = true")
        else
            debug && println("optimize = false")
        end

        # add d-wave pairing 
        if dims > 1
            add_pairing_symmetry!("d", determinantal_parameters, optimize, H_vpars, V, model_geometry, pht);

            debug && println("Hamiltonian::build_variational_hamiltonian() : ")
            debug && println("adding d-wave pairing matrix")
            debug && println("initial Δ_d = ", determinantal_parameters.det_pars.Δ_d)
            if optimize.Δ_d
                debug && println("optimize = true")
            else
                debug && println("optimize = false")
            end
        end
    end

    # add antiferromagnetic (Neél) term
    add_spin_order!("spin-z", determinantal_parameters, optimize, H_vpars, V, model_geometry, pht);

    debug && println("Hamiltonian::build_variational_hamiltonian() : ")
    debug && println("adding spin-z matrix")
    debug && println("initial Δ_afm = ", determinantal_parameters.det_pars.Δ_afm)
    if optimize.Δ_afm
        debug && println("optimize = true")
    else
        debug && println("optimize = false")
    end

    # add charge-density-wave term
    add_charge_order!("density wave", determinantal_parameters, optimize, H_vpars, V, model_geometry, pht);

    debug && println("Hamiltonian::build_variational_hamiltonian() : ")
    debug && println("adding charge density wave matrix")
    debug && println("initial Δ_cdw = ", determinantal_parameters.det_pars.Δ_cdw)
    if optimize.Δ_cdw
        debug && println("optimize = true")
    else
        debug && println("optimize = false")
    end

    # add site-dependent charge term
    if dims > 1
        add_charge_order!("site-dependent", determinantal_parameters, optimize, H_vpars, V, model_geometry, pht);

        debug && println("Hamiltonian::build_variational_hamiltonian() : ")
        debug && println("adding site-dependent charge matrix")
        debug && println("initial Δ_sdc = ", determinantal_parameters.det_pars.Δ_sdc)
        if optimize.Δ_sdc
            debug && println("optimize = true")
        else
            debug && println("optimize = false")
        end
    end

    # add site-dependent spin term
    if dims > 1
        add_spin_order!("site-dependent", determinantal_parameters, optimize, H_vpars, V, model_geometry, pht);

        debug && println("Hamiltonian::build_variational_hamiltonian() : ")
        debug && println("adding site-dependent spin matrix")
        debug && println("initial Δ_sds = ", determinantal_parameters.det_pars.Δ_sds)
        if optimize.Δ_sds
            debug && println("optimize = true")
        else
            debug && println("optimize = false")
        end
    end

    @assert length(H_vpars) == determinantal_parameters.num_det_pars
    @assert length(V) == determinantal_parameters.num_det_opts

    return sum(H_vpars), V;
end


"""

    add_pairing_symmetry!( symmetry::String, determinantal_parameters::DeterminantalParameters, 
                                Hs::::AbstractMatrix{<:Complex}, Hd::::AbstractMatrix{<:Complex}, V, N::Int )

Adds specified pairing symmetry to the auxiliary Hamiltonian. 

"""
function add_pairing_symmetry!(symmetry::String, determinantal_parameters::DeterminantalParameters, optimize, H_vpars, V, model_geometry, pht)
    # lattice sites
    N = model_geometry.lattice.N;

    # add s-wave pairing 
    if symmetry == "s"
        @assert pht == true

        # s-wave parameter
        Δ_0 = determinantal_parameters.det_pars.Δ_0;

        # add s-wave symmetry
        H_s = zeros(Complex, 2*N, 2*N) 
        V_s = copy(H_s);
        for i in 0:(2 * N - 1)
            V_s[i + 1, get_linked_spindex(i, N) + 1] = 1.0  
        end

        # add s-wave matrix
        H_s += Δ_0 * V_s;
        push!(H_vpars,H_s);

        if optimize.Δ_0
            push!(V, V_s);
        end
    elseif symmetry == "d"
        @assert pht == true
        @assert dims > 1

        # d-wave parameter
        Δ_d = determinantal_parameters.det_pars.Δ_d;

        # create neighbor table
        nbr_table = build_neighbor_table(bonds[1],
                                        model_geometry.unit_cell,
                                        model_geometry.lattice);

        # maps neighbor table to dictionary of bonds and neighbors                                
        nbrs = map_neighbor_table(nbr_table);

        # Predefine the displacement-to-sign map
        disp_sign_map = Dict([1,0] => 1, [0,1] => -1, [-1,0] => 1, [0,-1] => -1);

        # initial variational operator matrix
        H_d = zeros(Complex, 2*N, 2*N);
        V_dwave = zeros(AbstractFloat, 2*N, 2*N);

        # add d-wave symmetry
        for i in 1:N
            # get all neighbors of site i
            nn = nbrs[i][2];

            # loop over neighbors
            for j in nn
                # Find the displacement between site i and one of its neighbors j
                disp = sites_to_displacement(i, j, unit_cell, lattice);

                # Lookup sign of Δd
                dsgn = get(disp_sign_map, disp, 0);  # Default to 0 if no match

                if dsgn != 0
                    println(dsgn > 0 ? "+1" : "-1");

                    # Store spin-down indices
                    idn_idx = get_spindices_from_index(i, model_geometry)[2];
                    jdn_idx = get_spindices_from_index(j, model_geometry)[2];

                    # Add elements to variational operator
                    Vdwave[i, jdn_idx] = dsgn; 
                    Vdwave[j, idn_idx] = dsgn; 
                    Vdwave[jdn_idx, i] = dsgn; 
                    Vdwave[idn_idx, j] = dsgn; 
                end
            end
        end

        # add d-wave matrix
        H_d += Δ_d * V_dwave;
        push!(H_vpars,H_d);

        # if Δ_d is being optimized, store Vdwave matrix
        if optimize.Δ_d
            push!(V, V_dwave);
        end
    end

    return nothing
end


"""

    add_spin_order!()

Add spin order to the auxiliary Hamiltonian. 

"""
function add_spin_order!(order, determinantal_parameters, optimize, H_vpars, V, model_geometry, pht)
    # lattice sites
    N = model_geometry.lattice.N;

    # dimensions
    dims = size(model_geometry.lattice.L)[1];

    if order == "spin-z"
        afm_vec = fill(1,2*N);

        # antiferromagnetic parameter
        Δ_afm = determinantal_parameters.det_pars.Δ_afm; 

        if pht
            # stagger
            for s in 1:2*N
                # get proper site index
                idx = get_index_from_spindex(s, model_geometry);

                # 1D
                if dims == 1
                    # get site coordinates
                    ix = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1][1];

                    # apply phase
                    afm_vec[s] *= (-1)^(ix);
                # 2D
                elseif dims == 2
                    # get site coordinates
                    (ix, iy) = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1];

                    # apply phase
                    afm_vec[s] *= (-1)^(ix + iy);
                end
            end

            # add afm matrix
            H_afm = zeros(Complex, 2*N, 2*N); 
            V_afm = LinearAlgebra.Diagonal(afm_vec);
            H_afm += Δ_afm * V_afm;
            push!(H_vpars, H_afm);

            # if Δ_afm is being optimized, store Vafm matrix
            if optimize.Δ_afm
                push!(V, V_afm);
            end
        else
            # additional sign flip in spin down sector
            afm_vec_neg = copy(afm_vec);
            afm_vec_neg[N+1:2*N] .= -afm_vec_neg[N+1:2*N];

            # stagger
            for s in 1:2*N
                # get proper site index
                idx = get_index_from_spindex(s, model_geometry);

                # 1D
                if dims == 1
                    # get site coordinates
                    ix = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1][1];

                    # apply phase
                    afm_vec_neg[s] *= (-1)^(ix);
                # 2D
                elseif dims == 2
                    # get site coordinates
                    (ix, iy) = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1];

                    # apply phase
                    afm_vec_neg[s] *= (-1)^(ix + iy);
                end
            end

            # add afm matrix
            H_afm = zeros(Complex, 2*N, 2*N); 
            V_afm_neg = LinearAlgebra.Diagonal(afm_vec_neg);
            H_afm += Δ_afm * V_afm_neg;
            push!(H_vpars, H_afm);

            # if Δ_afm is being optimized, store Vafm matrix
            if optimize.Δ_afm
                push!(V, V_afm_neg);
            end
        end
    elseif order == "site-dependent"
        @assert pht == false        # TODO: add PHT version

        # lattice dimensions
        L = model_geometry.lattice.L

        Δ_sds = determinantal_parameters.det_pars.Δ_sds;

        sds_vectors = [];
        for shift in 0:(L[1]-1)
            vec = zeros(Int, 2 * N);  
            for i in 1:2*L[1]
                idx = (i-1)*L[1] + 1 + shift;  
                if idx <= 2 * N  
                    vec[idx] = 1;  
                end
            end
            # for i in 1:(2*L[1])
            #     vec[(i-1)*L[1] + 1 + shift] = 1  
            # end
            push!(sds_vectors, vec);  
        end

        for (i, sds_vec) in enumerate(sds_vectors)
            H_sds = zeros(AbstractFloat, 2*N, 2*N)
            sds_vec_neg = copy(sds_vec)
            sds_vec_neg[N+1:2*N] .= -sds_vec_neg[N+1:2*N]
        
            for s in 1:2*N
                idx = get_index_from_spindex(s, model_geometry)
                loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)
                sds_vec_neg[s] *= (-1)^(loc[1][1] + loc[1][2])
            end
        
            V_sds_neg = LinearAlgebra.Diagonal(sds_vec_neg)
            H_sds += Δ_sds[i] * V_sds_neg
            push!(H_vpars, H_sds)
        
            if optimize.Δ_sds
                push!(V, V_sds_neg)
            end
        end

        # for sds_vec in sds_vectors

        #     H_sds = zeros(AbstractFloat, 2*N, 2*N);
        #     sds_vec_neg = copy(sds_vec);
        #     sds_vec_neg[N+1:2*N] .= -sds_vec_neg[N+1:2*N];

        #     for s in 1:2*N
        #         idx = get_index_from_spindex(s, model_geometry);
        #         loc = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice);
        #         sds_vec_neg[s] *= (-1)^(loc[1][1]+loc[1][2]);
        #     end

        #     V_sds_neg = LinearAlgebra.Diagonal(sds_vec_neg);
        #     H_sds += Δ_sds * V_sds_neg;
        #     push!(H_vpars, H_sds);

        #     # if Δ_sds is being optimized, store Vsds matrix
        #     if optimize.Δ_sds
        #         push!(V, V_sds_neg);
        #     end
        # end
    end

    return nothing
end


"""

    add_charge_order!()

Add charge order to the auxiliary Hamiltonian.

"""
function add_charge_order!(order, determinantal_parameters, optimize, H_vpars, V, model_geometry, pht)
    # lattice sites
    N = model_geometry.lattice.N;

    # dimensions
    dims = size(model_geometry.lattice.L)[1];

    if order == "density wave"
        # charge density wave parameter
        Δ_cdw = determinantal_parameters.det_pars.Δ_cdw;

        # diagonal vector
        cdw_vec = fill(1,2*N);

        # account for particle-hole transformation
        if pht
            # sign flip in the spin-down sector
            cdw_vec_neg = copy(cdw_vec);
            cdw_vec_neg[N+1:2*N] .= -cdw_vec_neg[N+1:2*N];

            # stagger
            for s in 1:2*N
                # get proper site index
                idx = get_index_from_spindex(s, model_geometry);

                # 1D
                if dims == 1
                    # get site coordinates
                    ix = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1][1];

                    # apply phase
                    cdw_vec_neg[s] *= (-1)^(ix);
                # 2D
                elseif dims == 2
                    # get site coordinates
                    (ix, iy) = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1];

                    # apply phase
                    cdw_vec_neg[s] *= (-1)^(ix + iy);
                end
            end

            # add cdw matrix
            H_cdw = zeros(Complex, 2*N, 2*N); 
            V_cdw = LinearAlgebra.Diagonal(cdw_vec_neg);
            H_cdw += Δ_cdw * V_cdw;
            push!(H_vpars, H_cdw);

            # if Δ_cdw is being optimized, save Vcdw matrix
            if optimize.Δ_cdw
                push!(V, V_cdw);
            end
        else
            # stagger
            for s in 1:2*N
                # get proper site index
                idx = get_index_from_spindex(s, model_geometry);

                # 1D
                if dims == 1
                    # get site coordinates
                    ix = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1][1];

                    # apply phase
                    cdw_vec[s] *= (-1)^(ix);
                # 2D
                elseif dims == 2
                    # get site coordinates
                    (ix, iy) = site_to_loc(idx, model_geometry.unit_cell, model_geometry.lattice)[1];

                    # apply phase
                    cdw_vec[s] *= (-1)^(ix + iy);
                end
            end

            # add cdw matrix
            H_cdw = zeros(Complex, 2*N, 2*N); 
            V_cdw = LinearAlgebra.Diagonal(cdw_vec);
            H_cdw += Δ_cdw * V_cdw;
            push!(H_vpars, H_cdw);

            # if Δ_cdw is being optimized, save Vcdw matrix
            if optimize.Δ_cdw
                push!(V, V_cdw);
            end
        end
    elseif order == "site-dependent"
        @assert pht == false    # TODO: add PHT version
        # lattice dimensions
        L = model_geometry.lattice.L

        # site-dependent charge parameters
        Δ_sdc = determinantal_parameters.det_pars.Δ_sdc;

        # store diagonal vectors
        sdc_vectors = [];

        # populate vectors
        for shift in 0:(L[1]-1)
            vec = zeros(Int, 2 * N);  
            for i in 1:2*L[1]
                idx = (i-1)*L[1] + 1 + shift;  
                if idx <= 2 * N  
                    vec[idx] = 1;
                end
            end
            # for i in 1:2*L[1]
            #     vec[(i-1)*L[1] + 1 + shift] = 1  
            # end
            push!(sdc_vectors, vec);  
        end

        for (i, sdc_vec) in enumerate(sdc_vectors)
            H_sdc = zeros(AbstractFloat, 2*N, 2*N);
            V_sdc = LinearAlgebra.Diagonal(sdc_vec);
            H_sdc += Δ_sdc[i] .* V_sdc;
            push!(H_vpars, H_sdc);

            # if Δ_sdc is being optimized, store Vsdc matrix
            if optimize.Δ_sdc
                push!(V, V_sdc);
            end
        end
    end

    return nothing
end


"""

    add_chemical_potential!()

Adds chemical potential term to the auxiliary Hamiltonian.

"""
function add_chemical_potential!(determinantal_parameters, optimize, H_vpars, V, model_geometry, pht)
    # lattice sites
    N = model_geometry.lattice.N;

    # initial chemical potential value
    μ = determinantal_parameters.det_pars.μ

    μ_vec = fill(-1,2*N);
    if pht
        # account for minus sign 
        μ_vec_neg = copy(μ_vec);
        μ_vec_neg[N+1:2*N] .= -μ_vec_neg[N+1:2*N];

        # add chemical potential matrix
        H_μ = zeros(Complex, 2*N, 2*N)
        V_μ_neg = LinearAlgebra.Diagonal(μ_vec_neg);
        H_μ += μ * V_μ_neg;
        push!(H_vpars, H_μ);

        # if μ is being optimized, save Vμ matrix
        if optimize.μ
            push!(V, V_μ_neg);
        end
    else
        # add chemical potential matrix
        H_μ = zeros(Complex, 2*N, 2*N)
        V_μ = LinearAlgebra.Diagonal(μ_vec);
        H_μ += μ * V_μ;
        push!(H_vpars,H_μ);

        # if μ is being optimized, save Vμ matrix
        if optimize.μ
            push!(V, V_μ);
        end
    end

    return nothing
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

    get_tb_chem_pot( Ne::Int64, tight_binding_model::TightBindingModel, model_geometry::ModelGeometry )::Float64

For a tight-binding model that has not been particle-hole transformed, returns the  
chemical potential.

"""
function get_tb_chem_pot(Ne::Int64, tight_binding_model::TightBindingModel, model_geometry::ModelGeometry)::Float64
    @assert pht == false

    # number of lattice sites
    N = model_geometry.lattice.N;
    
    # preallocate matrices
    H_t₀ = zeros(Complex, 2*N, 2*N);
    H_t₁ = zeros(Complex, 2*N, 2*N);

    # hopping amplitudes
    t₀ = tight_binding_model.t₀;
    t₁ = tight_binding_model.t₁;

    # add nearest neighbor hopping
    nbr0 = build_neighbor_table(bonds[1],
                                        model_geometry.unit_cell,
                                        model_geometry.lattice);
    # special case for Lx, Ly = 2
    if Lx == 2 && Ly == 2 
        for (i,j) in eachcol(nbr0)
            H_t₀[i,j] += -t₀;
        end
        for (i,j) in eachcol(nbr0 .+ N)    
            H_t₀[i,j] += -t₀;
        end
    # special case for Lx = 2 
    elseif Lx == 2 && Ly > Lx
        for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Ly)])
            H_t₀[i,j] += -t₀;
            H_t₀[j,i] += -t₀;
        end
        for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Ly)] .+ N)
            H_t₀[i,j] += -t₀;
            H_t₀[j,i] += -t₀;
        end 
    # special case for Ly = 2 
    elseif Ly == 2 && Lx > Ly
        for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Lx)])
            H_t₀[i,j] += -t₀;
            H_t₀[j,i] += -t₀;
        end
        for (i,j) in eachcol(nbr0[:,1:(size(nbr0,2) - Lx)] .+ N)
            H_t₀[i,j] += -t₀;
            H_t₀[j,i] += -t₀;
        end  
    else
        for (i,j) in eachcol(nbr0)
            H_t₀[i,j] += -t₀;
            if model_geometry.lattice.N > 2
                H_t₀[j,i] += -t₀;
            else
            end
        end
        for (i,j) in eachcol(nbr0 .+ N)    
            H_t₀[i,j] += -t₀;
            if model_geometry.lattice.N > 2
                H_t₀[j,i] += -t₀;
            else
            end
        end
    end
    # add next nearest neighbor hopping
    nbr1 = build_neighbor_table(bonds[2],
                                    model_geometry.unit_cell,
                                    model_geometry.lattice);
    if Lx == 2 && Ly ==2 
        for (i,j) in eachcol(nbr1)
            H_t₁[i,j] += 0.5 * t₁;
        end
        for (i,j) in eachcol(nbr1 .+ N)    
            H_t₁[i,j] += 0.5 * t₁;
        end
    else
        for (i,j) in eachcol(nbr1)
            H_t₁[i,j] += t₁;
            H_t₁[j,i] += t₁;
        end
        for (i,j) in eachcol(nbr1 .+ N)    
            H_t₁[i,j] += t₁;
            H_t₁[j,i] += t₁;
        end
    end

    # full tight-binding Hamiltonian
    H_tb = H_t₀ + H_t₁;

    # solve for eigenvalues
    ε_F, Uₑ = diagonalize(H_tb);

    # tight-binding chemical potential
    μ = 0.5 * (ε_F[Ne + 1] + ε_F[Ne]);

    debug && println("Hamiltonian::get_tb_chem_pot() : ")
    debug && println("tight-binding chemical potential")
    debug && println("μ = ", μ)

    return μ
end


#####################################       DEPRECATED FUNCTIONS        #################################################
# """

#     initialize_determinantal_parameters( pars:;Vector{AbstractString}, 
#                                         vals::Vector{AbstractFloat} )::DeterminantalParameters

# Creates an instances of the DeterminantalParameters type.

# """
# function initialize_determinantal_parameters(pars, vals)::DeterminantalParameters
#     @assert length(pars) == length(vals) "Input vectors must have the same length"
    
#     # Calculate num_detpars as the total number of elements in all inner vectors of vals
#     num_detpars = sum(length, vals)

#     return DeterminantalParameters(pars, vals, num_detpars)
# end


# """
# #     map_determinantal_parameters( determinantal_parameters::DeterminantalParameters ) 

# # For a given set of variational parameters, returns a dictionary of that reports the value 
# # and optimization flag for a given parameter.

# # """
# function map_determinantal_parameters(determinantal_parameters::DeterminantalParameters)
#     vparam_map = Dict()

#     for i in 1:length(determinantal_parameters.vals)
#        vparam_map[determinantal_parameters.pars[i]] = determinantal_parameters.vals[i]
#     end

#     return vparam_map
# end















