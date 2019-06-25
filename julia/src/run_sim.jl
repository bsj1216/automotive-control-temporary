using AutoViz
using Distributions
using Interact
using AutomotiveDrivingModels
using Dates

roadway = gen_stadium_roadway(3)
timestep = 0.2
scene = Scene()
nticks = 5

# ==========================
# Generate Vehicles
# ==========================
n_vehs_top = 2; origin_top = 3.0;
n_vehs_middle = 3; origin_middle = 3.0;
n_vehs_bottom = 2; origin_bottom = 3.0;
n_vehs_tot = n_vehs_top + n_vehs_middle + n_vehs_bottom
dist_betw_cars = 6.0
global num_vehs = 0
# top lane
for i in 1: n_vehs_top
    global num_vehs
    num_vehs += 1
    if i == 1
        curr_speed = 5.0
    else
        curr_speed = 10.0
    end
    push!(scene, Vehicle(VehicleState(VecSE2(origin_top+dist_betw_cars*(i-1),0.0,0.0),roadway,curr_speed),
                         VehicleDef(),
                         num_vehs));
end
# middle lane
for i in 1: n_vehs_middle
    global num_vehs
    num_vehs += 1
    if i == 1
        curr_speed = 5.0
    elseif i == 2
        curr_speed = 10.0
    else
        curr_speed = 10.0
    end
    push!(scene, Vehicle(VehicleState(VecSE2(origin_middle+dist_betw_cars*(i-1),-DEFAULT_LANE_WIDTH,0.0),roadway,curr_speed),
                 VehicleDef(),
                 num_vehs));
end
# bottom lane
for i in 1: n_vehs_bottom
    global num_vehs
    num_vehs += 1
    if i == 1
        curr_speed = 5.0
    else
        curr_speed = 10.0
    end
    push!(scene, Vehicle(VehicleState(VecSE2(origin_bottom+dist_betw_cars*(i-1),-2*DEFAULT_LANE_WIDTH,0.0),roadway,curr_speed),
                 VehicleDef(),
                 num_vehs));
end

# ==========================
# Assign Driver models
# ==========================
ind_ego = 2
models = Dict{Int, DriverModel}()
for j in 1:num_vehs
    if j == ind_ego
        models[j] = MpcSganMonteDriver(timestep,
                                        n_ticks = nticks+1,
                                        N_sim = 5.0,
                                        T=0.4,
                                        λ_div=5000.0,
                                        λ_v=1.0,
                                        λ_δ=1000.0,
                                        λ_Δδ=10.0,
                                        λ_a = 100.0,
                                        δ_max= 0.15,
                                        δ_min = -0.15,
                                        a_min = -9.0,
                                        Δa_max = 10.0,
                                        Δδ_max = 0.2,
                                        width = 2.0,
                                        height = 4.2,
                                        thred_safety = 1.3,
                                        isDebugMode = true)
    else
        models[j] = IntelligentDriverModel()
    end
    set_desired_speed!(models[j], 10.0)
    # if j == 4 || j == 8 || j == 12
    #     set_desired_speed!(models[j], 3.0)
    # end
end
set_other_models!(models[ind_ego],models)
set_desired_speed!(models[ind_ego], 10.0)


# ==========================
# Set vehicle colors
# ==========================
# Set colors
saturation=0.85
value=0.85
car_colors = Dict{Int,Colorant}()
n = length(scene)
for (i,veh) in enumerate(scene)
    car_colors[veh.id] = convert(RGB, HSV(0, saturation, value))
end
car_colors[ind_ego] = convert(RGB, HSV(100, saturation, value))
# car_colors = get_pastel_car_colors(scene)
cam = FitToContentCamera()


# ==========================
# Run simulation
# ==========================
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene, roadway, models, nticks)


# ===========================
# Store the variables
# ===========================
using JLD
JLD.save("../sim_records/$(Dates.DateTime(Dates.now()))_N$(trunc(Int,models[ind_ego].N_sim))_tick$(nticks)_vehs$(n_vehs_tot)_Nhorz$(trunc(Int,round(models[ind_ego].T/timestep))).jld",
    "rec", rec, "roadway", roadway, "cam", cam,
    "colors", car_colors, "ind", ind_ego)

# ===========================
# Rendering
# ===========================
render(rec[0], roadway, cam=cam, car_colors=car_colors)
@manipulate for frame_index in 1:nframes(rec)
    render(rec[frame_index-nframes(rec)],roadway,cam=cam,car_colors=car_colors)
end
