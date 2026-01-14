#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include "MPPI.hpp"
#include "Trajectory.hpp"
#include "Dynamics.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

MPPI::MPPI(int K, int T, double lambda) : K(K), T(T), lambda(lambda){};

Control MPPI::get_control(State state, Trajectory traj, double t, double dt){
    MatrixXd ctrls = gen_rand_ctrl_seq(0.3, 1.0, 0.0, 2.0); // Generate random control inputs
    VectorXd costs = VectorXd(K); // Store cost of each sampled path

    std::vector<Waypoint> discretized_waypoints = std::vector<Waypoint>();
    
    double time = t;
    while(time < t + T*dt){
        discretized_waypoints.push_back(traj.sample(time));
        time += dt;
    }

    State temp_state = state;
    double temp_cost = 0;

    for (int k = 0; k < K; k++){
        for (int t = 0; t < T; t++){
            double v = ctrls(0, k*T + t);
            double w = ctrls(1, k*T + t);
            Control ctrl = Control{v, w};
            temp_state = unicyle_dynamics(temp_state, ctrl, dt);
            if(t < T-1){
                temp_cost += cost_func(temp_state, ctrl, discretized_waypoints.at(t));
            } else {
                temp_cost += terminal_cost_func(temp_state, discretized_waypoints.at(t));
            }
        }
        costs(k) = temp_cost;
        temp_state = state;
        temp_cost = 0.0;
    }
    
    // Calculate smallest cost
    double min_cost = INFINITY;
    for (int i = 0; i < K; i++){
        if(costs(i) < min_cost) min_cost = costs(i);
    }


    // Calculate weights
    VectorXd weights = VectorXd(K);
    for (int k = 0; k < K; k++){
        weights(k) = exp(-(costs(k)-min_cost)/lambda);
    }

    double sum_w = weights.sum();
    if (sum_w > 1e-12) {
        weights /= sum_w;
    } else {
        weights.setConstant(1.0 / K);
    }

    /////////////
    for(int i = 0; i < weights.size(); i++){
        std::cout << weights(i) << std::endl;
    }
    /////////////

    Vector2d ctrl_out = Vector2d(0.0, 0.0);
    for (int k = 0; k < K; k++){
        ctrl_out += weights(k) * ctrls.col(k*T);
    }

    return Control{ctrl_out};
};

MatrixXd MPPI::gen_rand_ctrl_seq(double mu_v, double sigma_v, double mu_omega, double sigma_omega){
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution v_rand{mu_v, sigma_v};
    std::normal_distribution omega_rand{mu_omega, sigma_omega};

    MatrixXd controls = MatrixXd(2, K*T); // Access i-th control of k-th sample as K*T + i

    for(int k = 0; k < K; k++){
        for(int t = 0; t < T; t++){
            controls(0, t+k*T) = v_rand(gen);
            controls(1, t+k*T) = omega_rand(gen);
        }
    }

    return controls;
}

double MPPI::cost_func(State state, Control ctrl, Waypoint target){
    Eigen::DiagonalMatrix Q = Eigen::DiagonalMatrix<double, 5>(16.0, 16.0, 3.0, 0.0, 0.0);
    Eigen::DiagonalMatrix R = Eigen::DiagonalMatrix<double, 2>(0.0005, 0.0001);
    
    VectorXd state_diff = target.state.val - state.val;
    state_diff(2) = std::fmod((state_diff(2) + M_PI), (M_PI * 2)) - M_PI;

    double cost = state_diff.transpose() * Q * state_diff; // State cost
    cost += ctrl.val.transpose() * R * ctrl.val; // Control cost
    
    return cost;
}

double MPPI::terminal_cost_func(State state, Waypoint target){
    Eigen::DiagonalMatrix Q = Eigen::DiagonalMatrix<double, 5>(16.0, 16.0, 3.0, 0.0, 0.0);
    VectorXd state_diff = target.state.val - state.val;
    state_diff(2) = std::fmod((state_diff(2) + M_PI), (M_PI * 2)) - M_PI;

    double cost = state_diff.transpose() * Q * state_diff; // State cost
    
    return cost;
}



