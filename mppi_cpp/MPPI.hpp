#pragma once

#include "State.hpp"
#include "Waypoint.hpp"
#include <vector>
#include "Control.hpp"

/* 
Inputs:
    - Starting state
    - Series of waypoints
    - Timestep
    - Number of paths (K)
    - Horizon (T)
Output:
    - Single control output
*/

class MPPI {
    public:
        MPPI(int K, int T);
        Control get_control(State state, std::vector<Waypoint> waypoints, double dt);
        
    private:
        int K;
        int T;
};