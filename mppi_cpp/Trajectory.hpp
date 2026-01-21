#pragma once
#include <vector>
#include <functional>
#include "Waypoint.hpp"
#include "MotionParams.hpp"

std::function<Waypoint(double)> generate_n_bezier(std::vector<Waypoint> ctrl_points);

class Trajectory {
    public:
        Trajectory(std::vector<Waypoint> waypoints, MotionParams mp);
        Waypoint sample(double time);
    private:
        std::vector<Waypoint> points;
        MotionParams params;
};