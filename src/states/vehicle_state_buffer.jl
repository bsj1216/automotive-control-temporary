"""
    VehicleStateBuffer
A default type to represent an agent physical state (position, velocity).
It contains the position in the global frame, Frenet frame and the longitudinal velocity

# fields
- `buffer::CircularBuffer{VehicleState}` buffer of VehicleState
"""

struct VehicleStateBuffer
    buffer::CircularBuffer{VehicleState}
end

VehicleStateBuffer() = VehicleStateBuffer(5)
VehicleStateBuffer(capacity::Int) = VehicleStateBuffer(CircularBuffer{VehicleState}(capacity))
