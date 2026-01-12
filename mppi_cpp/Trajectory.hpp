#pragma once
#include "Waypoint.hpp"

class Trajectory {
    public:
        Trajectory();
        Waypoint sample(double time);
};