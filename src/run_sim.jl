using AutomotiveDrivingModels
using LaneChangeRL
using AutoViz
using Distributions
using Interact

# include("MPCsgan.jl")

roadway = gen_stadium_roadway(3)
# roadway = gen_straight_roadway(4,100.0)

num_vehs = 1
timestep = 0.2

scene = Scene()
global i=1
for j in 1:num_vehs
    i = (j-1)*3+1
    push!(scene, Vehicle(VehicleState(VecSE2(0.0+3.0*(i-1),0.0,0.0),roadway,15.0), VehicleDef(),i)); i+=1 # top lane
    # push!(scene, Vehicle(VehicleState(VecSE2(0.0+3.0*(i-1),-DEFAULT_LANE_WIDTH-0.5,0.0),roadway,10.0), VehicleDef(),i)); i+=1 # middle lane
    # push!(scene, Vehicle(VehicleState(VecSE2(0.0+3.0*(i-1),-2*DEFAULT_LANE_WIDTH,0.0),roadway,20.0), VehicleDef(),i)); # bottm lane
end
car_colors = get_pastel_car_colors(scene)
cam = FitToContentCamera()

models = Dict{Int, DriverModel}()
models[1] = MpcSganMonteDriver(timestep, λ_div=5000.0, λ_δ=1000.0, λ_Δδ=100.0, N_sim = 5000.0, δ_max= 0.2, δ_min = -0.2, Δδ_max = 0.3, T=0.4)
set_desired_speed!(models[1], 40.0)
# for i in 1:num_vehs
#     # models[i] = BafflingDriver(timestep)
#     models[i] = MpcSganMonteDriver(timestep)
#     set_desired_speed!(models[i], 40.0)
# end

nticks = 15
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene, roadway, models, nticks)
render(rec[0], roadway, cam=cam, car_colors=car_colors)

@manipulate for frame_index in 1:nframes(rec)
    render(rec[frame_index-nframes(rec)],roadway,cam=cam,car_colors=car_colors)
end
