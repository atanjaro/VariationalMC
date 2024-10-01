"""

    PhononParameters

A type defining quantities realted to phonons.

"""
struct PhononParameters
    # phonon frequency
    Ω::AbstractFloat

    # microscopic coupling constant
    α::AbstractFloat
end


struct HolsteinModel
    # phonon parameters
    phonon_parameters::PhononParameters

    # fugacity
    μₚₕ::AbstractFloat

    # phonon number
    Nₚₕ::Int

    # phonon density configuration
    phconfig::Vector{Int}
end



struct SSHModel
    # phonon parameters
    phonon_parameters::PhononParameters

    # coherent state parameters
    z::Matrix{AbstractFloat}

    # phonon displacement configuration 
    phconfig::Matrix{AbstractFloat}
end



"""

    initialize_phonon_parameters( Ω::AbstractFloat, α::AbstractFloat )

Initializes an instance of the PhononParameters type

"""
function initialize_phonon_parameters(Ω::AbstractFloat, α::AbstractFloat)
    return PhononParameters(Ω, α)
end


function initialize_electron_phonon_model(model::AbstractString)

end















# Given a quantum harmonic oscillator with frequency Ω and mass M at an
# inverse temperature of β, return the standard deviation of the equilibrium
# distribution for the phonon position.
function std_x_qho(Ω::AbstractFloat, M::AbstractFloat) 
    # (sufficiently) low temperature
    β = 1_000_000

    ΔX = inv(sqrt(2 * M * Ω * tanh(β*Ω/2)))
    return ΔX
end
