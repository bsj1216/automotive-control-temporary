using AutoViz
using Distributions
using Interact
using AutomotiveDrivingModels

# using LaneChangeRL

# include("./mpc_sgan_monte_driver.jl")
# using .MPCsganMonteDriver

roadway = gen_stadium_roadway(3)
# roadway = gen_straight_roadway(4,100.0)

num_vehs = 6
timestep = 0.2

scene = Scene()
global i=1
for j in 1:num_vehs/3
    i = (j-1)*3+1
    push!(scene, Vehicle(VehicleState(VecSE2(0.0+2.0*(i-1),0.0,0.0),roadway,10.0), VehicleDef(),i)); i+=1 # top lane
    push!(scene, Vehicle(VehicleState(VecSE2(0.0+2.0*(i-2),-DEFAULT_LANE_WIDTH,0.0),roadway,10.0), VehicleDef(),i)); i+=1 # middle lane
    push!(scene, Vehicle(VehicleState(VecSE2(0.0+2.0*(i-3),-2*DEFAULT_LANE_WIDTH,0.0),roadway,10.0), VehicleDef(),i)); # bottm lane
end

# Driver models
ind_ego = 1
models = Dict{Int, DriverModel}()
for j in 1:num_vehs
    if j == ind_ego
        models[j] = MpcSganMonteDriver(timestep,
                                        N_sim = 50.0,
                                        T=0.4,
                                        λ_div=5000.0,
                                        λ_v=1.0,
                                        λ_δ=1000.0,
                                        λ_Δδ=10.0,
                                        λ_a = 100.0,
                                        δ_max= 0.15,
                                        δ_min = -0.15,
                                        Δδ_max = 0.2,
                                        height = 4.0,
                                        width = 1.8)
    else
        models[j] = Tim2DDriver(timestep,
                        mlane = MOBIL(timestep),
                    )
    end
    set_desired_speed!(models[j], 10.0)
end
set_other_models!(models[ind_ego],models)
set_desired_speed!(models[ind_ego], 20.0)

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


nticks = 20
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene, roadway, models, nticks)
render(rec[0], roadway, cam=cam, car_colors=car_colors)

@manipulate for frame_index in 1:nframes(rec)
    render(rec[frame_index-nframes(rec)],roadway,cam=cam,car_colors=car_colors)
end
