#pragma once

#include "State.hpp"
#include "Waypoint.hpp"
#include "Trajectory.hpp"
#include "MotionParams.hpp"
#include <vector>
#include <rerun.hpp>
#include "Control.hpp"
#include <Eigen/Dense>


class MPPI {
    public:
        MPPI(int K, int T, double lambda,  MotionParams mp, const rerun::RecordingStream& rec);
        Eigen::Matrix<double, 2, 25> get_control(State state, Trajectory traj, double t, double dt, Eigen::Matrix<double, 2, 25>& u_nom);

    private:
        int K;
        int T;
        double lambda;
        MotionParams mp;
        const rerun::RecordingStream& rec;
        Eigen::MatrixXd gen_rand_ctrl_seq(double mu_v, double sigma_v, double mu_omega, double sigma_omega);
        double cost_func(State state, Control ctrl, Waypoint target, int step);
        double terminal_cost_func(State state, Waypoint target);
};