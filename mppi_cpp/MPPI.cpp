#include<iostream>
#include "MPPI.hpp"

MPPI::MPPI(int K, int T) : K(K), T(T){};
Control MPPI::get_control(State state, std::vector<Waypoint> waypoints, double dt){
    std::cout << "x: " << state.x << ", dt: " << dt << std::endl;
    return Control{1.0, 2.0};
};
