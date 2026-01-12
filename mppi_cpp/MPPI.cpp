#include<iostream>
#include <random>
#include "MPPI.hpp"
#include <Eigen/Dense>


using Eigen::MatrixXd;

MPPI::MPPI(int K, int T) : K(K), T(T){};

Control MPPI::get_control(State state, std::vector<Waypoint> waypoints, double dt){
    

    MatrixXd ctrls = gen_rand_ctrl_seq(0.0, 10.0, 0.0, 2.0);
    std::cout << ctrls << std::endl;
    
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
            std::cout <<  t+k*T << std::endl;
            controls(0, t+k*T) = v_rand(gen);
            controls(1, t+k*T) = omega_rand(gen);
        }
    }

    return controls;

}



