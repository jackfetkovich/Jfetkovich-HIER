#pragma once

#include "State.hpp"
#include "Waypoint.hpp"
#include "Trajectory.hpp"
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


// dynamics, cost functions
class MPPI {
    public:
        MPPI(int K, int T, double lambda);
        Control get_control(State state, Trajectory traj, double t, double dt);

    private:
        int K;
        int T;
        double lambda;
        Eigen::MatrixXd gen_rand_ctrl_seq(double mu_v, double sigma_v, double mu_omega, double sigma_omega);
        double cost_func(State state, Control ctrl, Waypoint target);
        double terminal_cost_func(State state, Waypoint target);
};