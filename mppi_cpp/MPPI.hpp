#pragma once

#include "State.hpp"
#include "Waypoint.hpp"
#include <vector>
#include "Control.hpp"
#include <Eigen/Dense>

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
        Eigen::MatrixXd gen_rand_ctrl_seq(double mu_v, double sigma_v, double mu_omega, double sigma_omega);
        
        
};