"""
Speed limit: 15m/s​
 (The average speed in dense traffic will be less than 10m/s)​

Acceleration limits: -4m/s^2 to 3.5 m/s^2​
 (Same Range might be used for other cars on the roamodel. The consequence is that IDM might become a little bit more aggressive. For other cars, a lower max acceleration (~2m/s^2) will result in smoother actions)​

Jerk Limits:​
·       Acc >0 (while accelerating):  -20m/s^3 to 2m/s^3 (It means you can release the accelerator pedal almost immediately, but you have a limit on increasing the acc value)​
·       Acc<0 (while braking): -4m/s^3 to +20m/s^2 (it means you can release the brake almost immediately, but there is a limit in applying the brake)​

​Steering Limits:​
·       Wheel maximum steering: 0.6 Radian (34.3°)​
·       Maximum steering rate: 0.4 Radian/sec (22.9°/sec)
"""

mutable struct MpcSganMonteDriver <: DriverModel{AccelSteeringAngle}
    rec::SceneRecord
    mlane::LaneChangeModel
    δ_max::Float64
    δ_min::Float64
    a_max::Float64
    a_min::Float64
    N_sim::Float64
    T::Float64
    ΔT::Float64
    T_obs::Float64
    T_pred::Float64
    Δa_max::Float64
    Δδ_max::Float64
    λ_v::Float64
    λ_div::Float64
    λ_δ::Float64
    λ_a::Float64
    λ_Δδ::Float64
    λ_Δa::Float64
    v_des::Float64
    a::Float64
    δ::Float64

    function MpcSganMonteDriver(
        timestep::Float64;
        mlane::LaneChangeModel= BafflingLaneChanger(timestep),
        rec::SceneRecord = SceneRecord(1, timestep),
        δ_max::Float64 = 0.6,
        δ_min::Float64 = -0.4,
        a_max::Float64 = 3.0,
        a_min::Float64 = -3.0,
        N_sim::Float64 = 500.0,
        T::Float64 = 0.2,
        ΔT::Float64 = timestep,
        T_obs::Float64 = 2.0,
        T_pred::Float64 = 2.0,
        Δa_max::Float64 = 5.0,
        Δδ_max::Float64 = 0.6,
        λ_v::Float64 = 10.0,
        λ_div::Float64 = 10e2+0.0,
        λ_δ::Float64 = 1.0,
        λ_a::Float64 = 1.0,
        λ_Δδ::Float64 = 3.0,
        λ_Δa::Float64 = 1.0,
        v_des::Float64 = 30.0)

        retval = new()

        retval.rec = rec
        retval.mlane = mlane
        retval.δ_max = δ_max
        retval.δ_min = δ_min
        retval.a_max = a_max
        retval.a_min = a_min
        retval.N_sim = trunc(Int,N_sim)
        retval.T = T
        retval.ΔT = ΔT
        retval.T_obs = T_obs
        retval.T_pred = T_pred
        retval.Δa_max = Δa_max
        retval.Δδ_max = Δδ_max
        retval.λ_v = λ_v
        retval.λ_div = λ_div
        retval.λ_δ = λ_δ
        retval.λ_a = λ_a
        retval.λ_Δδ = λ_Δδ
        retval.λ_Δa = λ_Δa
        retval.v_des = v_des
        retval.a = NaN
        retval.δ = NaN

        retval
    end
end
get_name(::MpcSganMonteDriver) = "MpcSganMonteDriver"
function set_desired_speed!(model::MpcSganMonteDriver, v_des::Float64)
    model.v_des = v_des
    model
end
function observe!(model::MpcSganMonteDriver, scene::Scene, roadway::Roadway, egoid::Int)
    # δ_max,δ_min,a_max,a_min,N_sim,T,ΔT,T_obs,T_pred,Δa_max,Δδ_max=0.6,-0.4,3.0,-3.0,1000,4,0.5,2,2,5,0.6

    update!(model.rec, scene)
    observe!(model.mlane, scene, roadway, egoid) # receive lane change action from higher level decision maker

    vehicle_index = findfirst(egoid, scene)
    lane_change_action = rand(model.mlane)

    a_seq, δ_seq = gen_rand_inputs(model)

    cost_min = Inf
    global y_ref = -DEFAULT_LANE_WIDTH # TEMP middle lane
    for n = 1 : model.N_sim
        # initialize the vehicle object
        ego_veh = scene[vehicle_index] # Entity{VehicleStateBuffer,BicycleModel,Int}

        # initialize the vehicle object for the other vehicles
        indices_near_vehs = get_indices_near_vehs(vehicle_index, scene)

        # vehs = Array{Entity{VehicleState, BicycleModel, Int}}
        # v
        # push!(vehs,ego_veh) # push the vehicle object of thewatch -n 1 nvidia-smi ego vehicle.
        # indices_other_vehs = [] # index of the other vehicles
        # for i in indices_other_vehs
        #     vehᵢ = scene[i]
        #     push!(vehs,vehᵢ)
        # end

        # randomly sample the sequence of control
        a_ = a_seq[1:end,rand(1:size(a_seq,2))]
        δ_ = δ_seq[1:end,rand(1:size(δ_seq,2))]

        # evaluate the control sequences
        cost_n = 0
        for ℓ = 1 : size(a_seq,1)
            # global cost_n
            # (1) predict other drivers' motion (SGAN) (for the next step)
            # if length(vehs) > 1
            #     predict!(vehs)
            # end

            # (2) update the vehicle states with a,δ
            a,δ = a_[ℓ],δ_[ℓ]
            state′ = propagate(ego_veh, AccelSteeringAngle(a,δ), roadway, model.ΔT)
            ego_veh = Entity(state′, ego_veh.def, ego_veh.id)

            # (3) check the feasibility (break if violated)
            # - input the current positions of the vehicles
            # - compute the minimum distance between vehicles: skip if distance < 0, otherwise continue
            # if length(vehs) > 1
            #     isSafe(vehs) ? true : break
            # end

            # (4) compute the cumulative costs

            # TODO: update the cost function with lane angle (currently assumed there's no angle)
            # TODO: get the reference y coordinate
            a_prev = ℓ == 1 ? 0 : a_[ℓ-1]
            δ_prev = ℓ == 1 ? 0 : δ_[ℓ-1]

            cost_n += ( model.λ_v*(ego_veh.state.v-model.v_des)^2
                    + model.λ_div*(ego_veh.state.posG.y-y_ref)^2
                    + model.λ_δ*(δ)^2
                    + model.λ_a*(a)^2
                    + model.λ_Δδ*(δ-δ_prev)^2
                    + model.λ_Δa*(a-a_prev)^2)
        end
        if cost_n < cost_min
            # update with the optimal control sequence
            cost_min = cost_n
            model.a = a_[1]
            model.δ = δ_[1]
        end
    end

    if abs(scene[vehicle_index].state.posG.y-y_ref) <= 0.05
        model.δ = 0.0
    end
    model
end


"""
    get_indices_near_vehs(model::MpcSganMonteDriver, scene::Scene; range::Float64 = 5 [m])

find indices of near vehicles around the ego vehicle
"""
function get_indices_near_vehs(vehicle_index::Integer, scene::Scene; range::Float64 = 5)
    ego_veh = scene[vehicle_index]
    x = ego_veh.state.posG.x
    y = ego_veh.state.posG.y
    indices_near_vehs = []
    for i in 1 : length(scene)
        if i == vehicle_index
            continue
        end
        xᵢ = scene[i].state.posG.x
        yᵢ = scene[i].state.posG.y
        if sqrt((x-xᵢ)^2+(y-yᵢ)^2) <= range
            append!(indices_near_vehs, i)
        end
    end

    return indices_near_vehs
end


"""
    gen_rand_inputs(model::MpcSganMonteDriver)

returns randomly generated sequence of a and δ, both of which satisfies the
maximum rate constraints (jerk, steering rate)
"""
function gen_rand_inputs(model::MpcSganMonteDriver)
    n_row = trunc(Int,model.T/model.ΔT)
    n_col = trunc(Int,model.N_sim)

    # generate random sequences of action pair (a, δ)
    a_seq_raw = rand(Random.MersenneTwister(),model.a_min:0.01:model.a_max, (n_row,n_col)) #0.002 sec
    δ_seq_raw = rand(Random.MersenneTwister(),model.δ_min:0.01:model.δ_max, (n_row,n_col))

    # filter out the infeasible jerks
    a_seq_check = [zeros(1,n_col);a_seq_raw]
    a_diff = abs.(reshape([diff(reshape(a_seq_check, length(a_seq_check)));0],n_row+1,n_col)[1:end-1,1:end])
    bitarr = sum(a_diff .< model.Δa_max,dims=1) .== n_row
    a_seq = []
    for i in 1:length(bitarr)
        if bitarr[i] == true
            append!(a_seq, a_seq_raw[1:end,i])
        end
    end
    a_seq = reshape(a_seq, n_row, trunc(Int,length(a_seq)/n_row))

    # filter out the infeasible steering rate
    δ_seq_check = [zeros(1,n_col);δ_seq_raw]
    δ_diff = abs.(reshape([diff(reshape(δ_seq_check, length(δ_seq_check)));0],n_row+1,n_col)[1:end-1,1:end])
    bitarr = sum(δ_diff .< model.Δδ_max,dims=1) .== n_row
    δ_seq = []
    for i in 1:length(bitarr)
        if bitarr[i] == true
            append!(δ_seq, δ_seq_raw[1:end,i])
        end
    end
    δ_seq = reshape(δ_seq, n_row, trunc(Int,length(δ_seq)/n_row))

    return a_seq, δ_seq
end


"""
    predict!(vehs::Array{Entity{VehicleState, D, Int}}) where {D}
predicts motions of all vehicles within a range, using Social GAN.
"""
function predict!(vehs::Array{Entity{VehicleState, D, Int}}) where {D}
    # TODO: call SGAN function to predict

    # arrange input for SGAN
    inp = []
    for vehᵢ in vehs # other vehicles
        pos = [(vehᵢ.state.buffer[k].posG.x, vehᵢ.state.buffer[k].posG.y) for k in 1:length(vehᵢ.state.buffer)]
        inp = [inp reshape(pos,length(vehᵢ.state.buffer),1)]
    end

    # call SGAN and return the predicted positions
    # out = CALL_SGAN_FUNCTION(inp)

    # TODO: update vehicle object with the predicted positions
end


"""
    isSafe(ind_ego::Integer, ind_others::Array{Integer}, scene::Scene)
check collision with any of adjacent vehicles
"""
function isSafe(ind_ego::Integer, ind_others::Array{Integer}, scene::Scene)
    veh = scene[ind_ego]
    ϵ = 10e-5
    for i in ind_others
        if min_distance(veh, scene[i]) > ϵ
            continue
        else
            return false
        end
    end
    return true
end


"""
    min_distance(ego::Entity{VehicleState, D, Int},
                other::Entity{VehicleState, D, Int};
                type::AbstractString="ellipsoid",
                height::Float64=3.0,
                width::Float64=1.0) where {D}

returns distance between ego vehicle and an adjacent vehicle
"""
function min_distance(ego::Entity{VehicleState, D, Int},
                    other::Entity{VehicleState, D, Int};
                    type::AbstractString="ellipsoid",
                    height::Float64=3.0,
                    width::Float64=1.0) where {D}
    # ego: ego vehicle object
    # other: other vehicle object
    x,y,θ = ego.state.posG.x, ego.state.posG.y, ego.state.posG.θ
    xᵢ,yᵢ,θᵢ = other.state.posG.x, other.state.posG.y, other.state.posG.θᵢ
    # x,y,θ = 0,0,0.5
    # xᵢ,yᵢ,θᵢ = 0,5,0

    if type == "ellipsoid"
        model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
        @variable(model, x′); @variable(model, y′)
        @variable(model, xᵢ′); @variable(model, yᵢ′)
        @objective(model, MOI.MIN_SENSE, (x′-xᵢ′)^2+(y′-yᵢ′)^2)
        @constraint(model, ((x′-x)*cos(θ) + (y′-y)*sin(θ))^2/(height^2) + ((x′-x)*sin(θ) + (y′-y)*cos(θ))^2/(width^2) <= 1.0)
        @constraint(model, ((xᵢ′-xᵢ)*cos(θᵢ) + (yᵢ′-yᵢ)*sin(θᵢ))^2/(height^2) + ((xᵢ′-xᵢ)*sin(θᵢ) + (yᵢ′-yᵢ)*cos(θᵢ))^2/(width^2) <= 1.0)
        optimize!(model)
        return sqrt(objective_value(model))
    elseif type == "circle"
        return max(sqrt((x-xᵢ)^2+(y-yᵢ)^2)-2*height, 0)
    else
        error("ERROR: vehicle shape type error. It must be either circle or ellipsoid")
    end
end

Base.rand(model::MpcSganMonteDriver) = AccelSteeringAngle(model.a,model.δ)
# Distributions.pdf(driver::MpcSganMonteDriver, a::AccelSteeringAngle) = pdf(driver.mlat, a.a_lat) * pdf(driver.mlon, a.a_lon)
# Distributions.logpdf(driver::MpcSganMonteDriver, a::AccelSteeringAngle) = logpdf(driver.mlat, a.a_lat) * logpdf(driver.mlon, a.a_lon)
