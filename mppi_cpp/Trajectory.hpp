#pragma once
#include <vector>
#include "Waypoint.hpp"


class Trajectory {
    public:
        Trajectory(std::vector<Waypoint> waypoints);
        Waypoint sample(double time);
    private:
        std::vector<Waypoint> points;
};