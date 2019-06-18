"""
	AccelSteeringAngle
Allows driving the car in a circle based on the steering angle
If steering angle less than threshold 0.01 radian, just drives straight

# Fields
- `a::Float64` longitudinal acceleration [m/s^2]
- `δ::Float64` Steering angle [rad]
"""
struct AccelSteeringAngle
    a::Float64 # accel [m/s²]
    δ::Float64 # steering angle [rad]
end
Base.show(io::IO, a::AccelSteeringAngle) = @printf(io, "AccelSteeringAngle(%6.3f,%6.3f)", a.a, a.δ)
Base.length(::Type{AccelSteeringAngle}) = 2

"""
Take a vector containing acceleration and desired steering angle and
convert to AccelSteeringAngle
"""
Base.convert(::Type{AccelSteeringAngle}, v::Vector{Float64}) = AccelSteeringAngle(v[1], v[2])

"""
Extract acceleration and steering angle components from AccelSteeringAngle
and return them into a vector
"""
function Base.copyto!(v::Vector{Float64}, a::AccelSteeringAngle)
    v[1] = a.a
    v[2] = a.δ
    v
end

"""
propagate vehicle forward in time given a desired acceleration and
steering angle. If steering angle higher than 0.1 radian, the vehicle
drives in a circle
"""
function propagate(veh::Entity{VehicleState, D, I}, action::AccelSteeringAngle, roadway::Roadway, Δt::Float64) where {D, I}
    a = 1.5; b = 1.5
    # L = veh.def.a + veh.def.b
    # l = -veh.def.b
    L = a+b
    l = -b

    a = action.a # accel [m/s²]
    δ = action.δ # steering wheel angle [rad]

    x = veh.state.posG.x
    y = veh.state.posG.y
    θ = veh.state.posG.θ
    v = veh.state.v

    s = v*Δt + a*Δt*Δt/2 # distance covered
    v′ = v + a*Δt

    if abs(δ) < 0.01 # just drive straight
        posG = veh.state.posG + polar(s, θ)
    else # drive in circle

        R = L/tan(δ) # turn radius

        β = s/R
        xc = x - R*sin(θ) + l*cos(θ)
        yc = y + R*cos(θ) + l*sin(θ)

        θ′ = mod(θ+β, 2π)
        x′ = xc + R*sin(θ+β) - l*cos(θ′)
        y′ = yc - R*cos(θ+β) - l*sin(θ′)

        posG = VecSE2(x′, y′, θ′)
    end

    VehicleState(posG, roadway, v′)
end
