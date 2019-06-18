module MPCsgan

using AutomotiveDrivingModels
using Random
using JuMP

export
        MpcSganMonteDriver,
        set_desired_speed!

include("behaviors/mpc_sgan_monte_driver.jl")

end
