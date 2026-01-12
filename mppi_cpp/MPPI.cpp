#include<iostream>
#include <random>
#include "MPPI.hpp"
#include <Eigen/Dense>
#include <cmath>


using Eigen::MatrixXd;

MPPI::MPPI(int K, int T) : K(K), T(T){};

Control MPPI::get_control(State state, std::vector<Waypoint> waypoints, double dt){
    MatrixXd ctrls = gen_rand_ctrl_seq(0.0, 10.0, 0.0, 2.0);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return Control{1.0, 2.0};
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



