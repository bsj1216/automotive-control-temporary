"""
MpcSganMonteDriver
Driver that randomly changes lanes and speeds.

# Constructors
BafflingDriver(timestep::Float64;mlon::LaneFollowingDriver=IntelligentDriverModel(), mlat::LateralDriverModel=ProportionalLaneTracker(), mlane::LaneChangeModel=RandLaneChanger(timestep),rec::SceneRecord = SceneRecord(1, timestep))

# Fields
- `rec::SceneRecord` A record that will hold the resulting simulation results
- `mlon::LaneFollowingDriver = IntelligentDriverModel()` Longitudinal driving model
- `mlat::LateralDriverModel = ProportionalLaneTracker()` Lateral driving model
- `mlane::LaneChangeModel =RandLaneChanger` Lane change model (randomly)
"""
# using PyCall

mutable struct MpcSganMonteDriver <: DriverModel{AccelSteeringAngle}
    rec::SceneRecord
    n_ticks::Integer
    mlane::LaneChangeModel
    δ_max::Float64
    δ_min::Float64
    a_max::Float64
    a_min::Float64
    N_sim::Float64
    T::Float64
    ΔT::Float64
    T_obs_given::Float64
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
    height::Float64
    width::Float64
    models::Dict{Int, DriverModel}
    thred_safety::Float64
    isDebugMode::Bool
    pred_model_type::AbstractString  # {"sgan", "perfect"}
    predictor::Any
    sgan_path::AbstractString
    sgan_model_path::AbstractString

    function MpcSganMonteDriver(
        timestep::Float64;
        n_ticks::Integer = 5,
        mlane::LaneChangeModel= BafflingLaneChanger(timestep),
        rec::SceneRecord = SceneRecord(n_ticks, timestep),
        δ_max::Float64 = 0.6,
        δ_min::Float64 = -0.4,
        a_max::Float64 = 3.5,
        a_min::Float64 = -4.0,
        N_sim::Float64 = 200.0,
        T::Float64 = 0.2,
        ΔT::Float64 = timestep,
        T_obs::Float64 = 1.0,
        T_obs_given::Float64 = T_obs,
        T_pred::Float64 = 1.0,
        Δa_max::Float64 = 2.0,
        Δδ_max::Float64 = 0.4,
        λ_v::Float64 = 1.0,
        λ_div::Float64 = 10e2+0.0,
        λ_δ::Float64 = 1.0,
        λ_a::Float64 = 1.0,
        λ_Δδ::Float64 = 3.0,
        λ_Δa::Float64 = 1.0,
        height::Float64 = 4.0,
        width::Float64 = 1.8,
        v_des::Float64 = 30.0,
        thred_safety::Float64 = 0.5,
        isDebugMode::Bool = false,
        pred_model_type::AbstractString = "sgan",
        sgan_path::AbstractString = "/home/sbae/automotive-control-temporary/python/nnmpc/sgan",
        sgan_model_path::AbstractString = "/home/sbae/automotive-control-temporary/python/nnmpc/sgan/models/sgan-models/eth_8_model.pt")

        retval = new()

        pushfirst!(PyVector(pyimport("sys")."path"),sgan_path)
        sganPredictor = pyimport("sgan.predictor")
        predictor = sganPredictor.Predictor(sgan_model_path)

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
        retval.T_obs_given = T_obs_given
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
        retval.a = a_min
        retval.δ = 0
        retval.height = height
        retval.width = width
        retval.models = Dict{Int64,DriverModel}()
        retval.thred_safety = thred_safety
        retval.isDebugMode = isDebugMode
        retval.predictor = predictor
        retval.pred_model_type = pred_model_type

        retval
    end
end

get_name(::MpcSganMonteDriver) = "MpcSganMonteDriver"
function set_desired_speed!(model::MpcSganMonteDriver, v_des::Float64)
    model.v_des = v_des
    model
end

function set_other_models!(model::MpcSganMonteDriver, models::Dict{Int, DriverModel})
    model.models = models
    model
end


"""
    compute_longitudinal_acceleration(model::MpcSganMonteDriver, scene::Scene, roadway::Roadway, egoid::Int)

"""
function compute_longitudinal_acceleration(model::MpcSganMonteDriver, scene::Scene, roadway::Roadway, egoid::Int)
    # local variables -- from IDM
    k_spd = 1.0 # proportional constant for speed tracking when in freeflow [s⁻¹]
    δ = 4.0 # acceleration exponent [-]
    T = 1.5 # desired time headway [s]
    # v_des = 29.0 # desired speed [m/s]
    s_min = 5.0 # minimum acceptable gap [m]
    a_max = 3.0 # maximum acceleration ability [m/s²]
    d_cmf = 2.0 # comfortable deceleration [m/s²] (positive)
    d_max = 9.0 # maximum deceleration [m/s²] (positive)
    ΔT = 0.2 # timestep to simulate [s]
    v_offset = 5.0 # offset from the desired speed [m/s]

    fore = get_neighbor_fore_along_lane(scene, ind_ego, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    v_ego = scene[ind_ego].state.v
    headway, v_oth = fore.Δs, scene[fore.ind].state.v
    if !isnan(v_oth)
        Δv = v_oth - v_ego
        s_des = s_min + v_ego*T - v_ego*Δv / (2*sqrt(model.a_max*abs(model.a_min))
        v_ratio = model.v_des > 0.0 ? (v_ego/model.v_des) : 1.0
        a = model.a_max * (1.0 - v_ratio^δ - (s_des/headway)^2)
    else
        # no lead vehicle, just drive to match desired speed
        Δv = model.v_des - v_ego
        model.a = Δv*k_spd # predicted accel to match target speed
    end

    if v_ego + model.ΔT * a < 0
        a = max(model.a_min, -v_ego/model.ΔT)
    end

    return a
end

"""
    observe!(model::MpcSganMonteDriver, scene::Scene, roadway::Roadway, egoid::Int)

(1) randomly generates a sequence of control inputs (acceleration and steering angle)
(2) evaluates Social GAN to predict motions of neighboring vehicles within a range
(3) checks the safety constraint at each time step
(4) evalautes the cumulative cost over the receding time horizon T
(5) finds the optimal pair of control inputs
"""
function observe!(model::MpcSganMonteDriver, scene::Scene, roadway::Roadway, egoid::Int)
    # δ_max,δ_min,a_max,a_min,N_sim,T,ΔT,T_obs,T_pred,Δa_max,Δδ_max=0.6,-0.4,3.0,-3.0,1000,4,0.5,2,2,5,0.6
    ind_ego = findfirst(egoid, scene)
    lane_change_action = rand(model.mlane)

    update!(model.rec, scene)
    observe!(model.mlane, scene, roadway, egoid) # receive lane change action from higher level decision maker
    # global y_ref = DEFAULT_LANE_WIDTH * lane_change_action.dir
    global y_ref = -1*DEFAULT_LANE_WIDTH

    # check if we have enough dataset to use SGAN
    isSganDataReady = model.rec.nframes*model.ΔT < model.T_obs_given
    model.isDebugMode ? println("isSganDataReady: ", isSganDataReady)

    # if sgan data is not ready, then it just keeps the lane
    if !isSganDataReady
        model.a = compute_longitudinal_acceleration(model, scene, roadway, ind_ego)
        model.δ = 0
        return model
    end

    model.T_obs = min(model.rec.nframes*model.ΔT, model.T_obs_given)
    model.isDebugMode ? println("T_obs: ", model.T_obs) : nothing

    # local variables
    N_receding = trunc(Int, model.T/model.ΔT)
    N_sgan_obs = trunc(Int, model.T_obs/model.ΔT)
    N_sgan_pred = trunc(Int, model.T_pred/model.ΔT)

    # generate sequences of control inputs
    a_seq, δ_seq = gen_rand_inputs(model, y_ref)

    cost_min = Inf
    n_opt = -1            # index of the optimal sequence of the controls
    model.a = model.a_min # maximum braking by default
    model.δ = 0           # zero steering by default
    for n = 1 : model.N_sim
        model.isDebugMode ? println("n: ",n) : nothing

        # initialize the vehicle object
        ego_veh = scene[ind_ego] # Entity{VehicleStateBuffer,BicycleModel,Int}

        # initialize the vehicle object for the other vehicles
        ind_near_vehs = get_indices_near_vehs(ind_ego, scene, 15.0)

        # initialize the record scene queue
        rec′ = SceneRecord(N_receding, model.ΔT)
        for k = 1 : N_sgan_obs
            update!(rec′,model.rec[-N_sgan_obs + k])
        end

        # randomly sample the sequence of control
        a_ = a_seq[1:end,rand(1:size(a_seq,2))]
        δ_ = δ_seq[1:end,rand(1:size(δ_seq,2))]

        # initialize the scene with the recent one
        scene′= Scene()
        for i = 1 : length(scene)
            push!(scene′,scene[i])
        end

        # evaluate the control sequences
        cost_n = 0
        for ℓ = 1 : N_receding
            model.isDebugMode ? println("ℓ: ", ℓ) : nothing
            # (1) predict other drivers' motion (SGAN) (for the next step)
            if length(ind_near_vehs) >= 1
                if model.pred_model_type == "perfect"
                    scene′ = propagate_other_vehs(Any, model, ind_ego, scene′, roadway, rec′)
                elseif model.pred_model_type == "sgan"
                    if isSganDataReady
                        scene′ = predict(model, ind_ego, ind_near_vehs, scene′, roadway, rec′)
                    else

                    end
                else
                    error("ERROR: prediction model type error. It must be either {perfect} (benchmark) or {sgan}")
                end
            end

            # (2) update the vehicle states with a,δ
            a,δ = a_[ℓ],δ_[ℓ]
            model.isDebugMode ? println("try a = ",a, ", δ = ", δ) : nothing
            state′ = propagate(ego_veh, AccelSteeringAngle(a,δ), roadway, model.ΔT)
            ego_veh = Entity(state′, ego_veh.def, ego_veh.id)
            scene′[ind_ego] = ego_veh
            update!(rec′,scene′)

            # (3) check the feasibility (break if violated)
            if length(ind_near_vehs) >= 1
                if !isSafe(model, ind_ego, ind_near_vehs, scene′)
                    cost_n = Inf
                    break
                end
            end

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
            n_opt = n
        end
    end

    model.isDebugMode ? println("optimal controls (n = ", n_opt,") : a = ",model.a, ", δ = ", model.δ) : nothing
    if n_opt == -1
        @warn "No feasible solution found. Maximum braking applied."
    end
    # prevent the vehicle going backward
    if scene[ind_ego].state.v + model.ΔT*model.a < 0
        model.a = max(-scene[ind_ego].state.v/model.ΔT, model.a_min)
    end
    model
end


"""
    get_indices_near_vehs(model::MpcSganMonteDriver, scene::Scene; range::Float64 = 5 [m])

find indices of near vehicles around the ego vehicle
"""
function get_indices_near_vehs(ind_ego::Int, scene::Scene, range::Float64)
    ego_veh = scene[ind_ego]
    x = ego_veh.state.posG.x
    y = ego_veh.state.posG.y
    ind_near_vehs = []
    for i in 1 : length(scene)
        if i == ind_ego
            continue
        end
        xᵢ = scene[i].state.posG.x
        yᵢ = scene[i].state.posG.y
        if sqrt((x-xᵢ)^2+(y-yᵢ)^2) <= range
            append!(ind_near_vehs, i)
        end
    end
    return ind_near_vehs
end


"""
    gen_rand_inputs(model::MpcSganMonteDriver)

returns randomly generated sequence of a and δ, both of which satisfies the
maximum rate constraints (jerk, steering rate)
"""
function gen_rand_inputs(model::MpcSganMonteDriver, y_ref::Float64)
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
    function propagate(model::MpcSganMonteDriver, ind_near_vehs::Vector{Any}, rec::SceneRecord)
returns positions of the neighboring vehicles, based on their Driver Model. This is equivalent to the SGAN that is perfectly trained.
"""
function propagate_other_vehs(::Type{A}, model::MpcSganMonteDriver, ind_ego::Int, scene::Scene, roadway::Roadway, rec::SceneRecord) where {A}
    models = Dict{Int, DriverModel}()
    for i in 1 : length(model.models)
        if i == ind_ego
            models[i] = BafflingDriver(rec.timestep)
        else
            models[i] = model.models[i]
        end
    end
    actions = Array{A}(undef, length(scene))
    get_actions!(actions, scene, roadway, models)
    tick!(scene, roadway, actions, rec.timestep)

    return scene
end


"""
    predict(model::MpcSganMonteDriver, ind_ego::Integer, ind_near_vehs::Vector{Any}, scene::Scene, roadway::Roadway, rec::SceneRecord)
predicts motions of all vehicles within a range, using Social GAN.
"""
function predict(model::MpcSganMonteDriver, ind_ego::Integer, ind_near_vehs::Vector{Any}, scene::Scene, roadway::Roadway, rec::SceneRecord)
    # arrange input for SGAN
    # example:
    # obs_traj = [  [ 1,  1.000e+00,  8.460e+00,  3.590e+00],
    #           [ 1,  2.000e+00,  1.364e+01,  5.800e+00],
    #           [ 2,  1.000e+00,  9.570e+00,  3.790e+00],
    #           [ 2,  2.000e+00,  1.364e+01,  5.800e+00],
    #           [ 3,  1.000e+00,  1.067e+01,  3.990e+00],
    #           [ 3,  2.000e+00,  1.364e+01,  5.800e+00],
    #           [ 4,  1.000e+00,  1.173e+01,  4.320e+00],
    #           [ 4,  2.000e+00,  1.209e+01,  5.750e+00],
    #           [ 5,  1.000e+00,  1.281e+01,  4.610e+00],
    #           [ 5,  2.000e+00,  1.137e+01,  5.800e+00],
    #           [ 6,  1.000e+00,  1.281e+01,  4.610e+00],
    #           [ 6,  2.000e+00,  1.031e+01,  5.970e+00],
    #           [ 7,  1.000e+00,  1.194e+01,  6.770e+00],
    #           [ 7,  2.000e+00,  9.570e+00,  6.240e+00],
    #           [ 8,  1.000e+00,  1.103e+01,  6.840e+00],
    #           [ 8,  2.000e+00,  8.730e+00,  6.340e+00]]

    N_obs = trunc(Int, round(model.T_obs/model.ΔT))
    N_vehs = length(ind_near_vehs)
    model.isDebugMode ? println("N_vehs: ",N_vehs) : nothing

    # bulid the input matrix for sgan
    obs_traj = []
    for seq in 1 : N_obs # sequence
        for i in 1 : N_vehs + 1 # vehicles
            ind = i == 1 ? ind_ego : ind_near_vehs[i-1]
            veh = rec[0-(N_obs-1)+(seq-1)][ind]
            l = [seq, i, veh.state.posG.x, veh.state.posG.y]
            obs_traj = [obs_traj;[l]]
        end
    end

    if model.isDebugMode
        for i in 1:length(obs_traj)
              println(obs_traj[i])
        end
    end

    # call SGAN and return the predicted positions
    next_pred_traj = model.predictor.predict(obs_traj)
    if model.isDebugMode
        for i in 1:size(next_pred_traj,1)
              println(next_pred_traj[i,:])
        end
    end

    # update the scene with the predicted positions
    for i in 1 : N_vehs + 1
        ind = i == 1 ? ind_ego : ind_near_vehs[i-1]
        veh = scene[ind]

        x′ = next_pred_traj[i,1]; x_diff = x′-veh.state.posG.x
        y′ = next_pred_traj[i,2]; y_diff = y′-veh.state.posG.y
        θ′ = atan(y_diff/x_diff)
        v′ = sqrt(x_diff^2+y_diff^2)/model.ΔT

        posG = VecSE2(x′, y′, θ′)
        state′  = VehicleState(posG, roadway, v′)

        scene[ind] = Entity(state′, veh.def, veh.id)
    end

    return scene
end


"""
    isSafe(ind_ego::Int, ind_others::Array{Int}, scene::Scene)
check collision with any of adjacent vehicles
"""
function isSafe(model::MpcSganMonteDriver, ind_ego::Int, ind_others::Vector{Any}, scene::Scene)
    ego_veh = scene[ind_ego]
    ϵ = model.thred_safety
    for i in ind_others
        model.isDebugMode ? println("distance to vehicle ", i, ": ", min_distance(model, ego_veh, scene[i])) : nothing
        if min_distance(model, ego_veh, scene[i]) > ϵ
            continue
        else
            model.isDebugMode ? println("   COLISION DETECTED. actions discarded") : nothing
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
function min_distance(model::MpcSganMonteDriver,
                    ego::Entity{VehicleState, D, Int},
                    other::Entity{VehicleState, D, Int};
                    type::AbstractString="ellipsoid") where {D}
    # ego: ego vehicle object
    # other: other vehicle object
    x,y,θ = ego.state.posG.x, ego.state.posG.y, ego.state.posG.θ
    xᵢ,yᵢ,θᵢ = other.state.posG.x, other.state.posG.y, other.state.posG.θ
    h = model.height/2
    w = model.width/2

    if type == "ellipsoid"
        prob = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
        @variable(prob, x′); @variable(prob, y′)
        @variable(prob, xᵢ′); @variable(prob, yᵢ′)
        @objective(prob, MOI.MIN_SENSE, (x′-xᵢ′)^2+(y′-yᵢ′)^2)
        @constraint(prob, ((x′-x)*cos(θ) + (y′-y)*sin(θ))^2/(h^2) + ((x′-x)*sin(θ) + (y′-y)*cos(θ))^2/(w^2) <= 1.0)
        @constraint(prob, ((xᵢ′-xᵢ)*cos(θᵢ) + (yᵢ′-yᵢ)*sin(θᵢ))^2/(h^2) + ((xᵢ′-xᵢ)*sin(θᵢ) + (yᵢ′-yᵢ)*cos(θᵢ))^2/(w^2) <= 1.0)
        optimize!(prob)
        return sqrt(max(objective_value(prob), 0))
    elseif type == "circle"
        return max(sqrt((x-xᵢ)^2+(y-yᵢ)^2)-2*h, 0)
    else
        error("ERROR: vehicle shape type error. It must be either circle or ellipsoid")
    end
end

Base.rand(model::MpcSganMonteDriver) = AccelSteeringAngle(model.a,model.δ)
