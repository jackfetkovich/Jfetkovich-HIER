
#include "Trajectory.hpp"
#include "Waypoint.hpp"

Trajectory::Trajectory(){}

Waypoint Trajectory::sample(double time){
    return Waypoint{State(0.0, 0.0, 0.0, 0.0, 0.0), 0.0};
} 