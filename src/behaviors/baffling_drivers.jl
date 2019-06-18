"""
	BafflingDriver
Driver that randomly changes lanes and speeds.

# Constructors
	BafflingDriver(timestep::Float64;mlon::LaneFollowingDriver=IntelligentDriverModel(), mlat::LateralDriverModel=ProportionalLaneTracker(), mlane::LaneChangeModel=RandLaneChanger(timestep),rec::SceneRecord = SceneRecord(1, timestep))

# Fields
- `rec::SceneRecord` A record that will hold the resulting simulation results
- `mlon::LaneFollowingDriver = IntelligentDriverModel()` Longitudinal driving model
- `mlat::LateralDriverModel = ProportionalLaneTracker()` Lateral driving model
- `mlane::LaneChangeModel =RandLaneChanger` Lane change model (randomly)
"""
mutable struct BafflingDriver <: DriverModel{LatLonAccel}
    rec::SceneRecord
    mlon::LaneFollowingDriver
    mlat::LateralDriverModel
    mlane::LaneChangeModel

    function BafflingDriver(
        timestep::Float64;
        mlon::LaneFollowingDriver=BafflingLongitudinalTracker(),
        mlat::LateralDriverModel=ProportionalLaneTracker(),
        mlane::LaneChangeModel=BafflingLaneChanger(timestep),
        rec::SceneRecord = SceneRecord(1, timestep)
        )

        retval = new()

        retval.rec = rec
        retval.mlon = mlon
        retval.mlat = mlat
        retval.mlane = mlane

        retval
    end
end
get_name(::BafflingDriver) = "BafflingDriver"
function set_desired_speed!(model::BafflingDriver, v_des::Float64)
    set_desired_speed!(model.mlon, v_des)
    set_desired_speed!(model.mlane, v_des)
    model
end
# function track_longitudinal!(driver::LaneFollowingDriver, scene::Scene, roadway::Roadway, vehicle_index::Int, fore::NeighborLongitudinalResult)
#     v_ego = scene[vehicle_index].state.v
#     if fore.ind != nothing
#         headway, v_oth = fore.Δs, scene[fore.ind].state.v
#     else
#         headway, v_oth = NaN, NaN
#     end
#     return track_longitudinal!(driver, v_ego, v_oth, headway)
# end
function observe!(driver::BafflingDriver, scene::Scene, roadway::Roadway, egoid::Int)

    update!(driver.rec, scene)
    observe!(driver.mlane, scene, roadway, egoid) # receive action from the lane change controller

    vehicle_index = findfirst(egoid, scene)
    lane_change_action = rand(driver.mlane)
    laneoffset = get_lane_offset(lane_change_action, driver.rec, roadway, vehicle_index)
    lateral_speed = convert(Float64, get(VELFT, driver.rec, roadway, vehicle_index))

    if lane_change_action.dir == DIR_MIDDLE
        fore = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    elseif lane_change_action.dir == DIR_LEFT
        fore = get_neighbor_fore_along_left_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    else
        @assert(lane_change_action.dir == DIR_RIGHT)
        fore = get_neighbor_fore_along_right_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    end

    track_lateral!(driver.mlat, laneoffset, lateral_speed) # receive acceleration from the lateral controller
    track_longitudinal!(driver.mlon, scene, roadway, vehicle_index, fore) # receive acceleration from the longitudinal controller

    driver
end
Base.rand(driver::BafflingDriver) = LatLonAccel(rand(driver.mlat), rand(driver.mlon).a)
Distributions.pdf(driver::BafflingDriver, a::LatLonAccel) = pdf(driver.mlat, a.a_lat) * pdf(driver.mlon, a.a_lon)
Distributions.logpdf(driver::BafflingDriver, a::LatLonAccel) = logpdf(driver.mlat, a.a_lat) * logpdf(driver.mlon, a.a_lon)
