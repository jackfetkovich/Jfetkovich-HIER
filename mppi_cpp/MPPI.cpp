#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <rerun.hpp>
#include <rerun/demo_utils.hpp>
#include "MPPI.hpp"
#include "Trajectory.hpp"
#include "Dynamics.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace rerun::demo;

struct Rollout {
    std::vector<State> logging_states;
};

MPPI::MPPI(int K, int T, double lambda, MotionParams mp, const rerun::RecordingStream& rec) : K(K), T(T), lambda(lambda), mp(mp), rec(rec){};

Control MPPI::get_control(State state, Trajectory traj, double t, double dt){
    rec.set_time_duration_secs("sim_time", t); // New time on timeline
    
    MatrixXd ctrls = gen_rand_ctrl_seq(0.05, 0.2, 0.0, 0.4); // Generate random control inputs
    VectorXd costs = VectorXd(K); // Store cost of each sampled path

    std::vector<Waypoint> discretized_waypoints = std::vector<Waypoint>();
    
    double time = t;
    while(time < t + T*dt){
        discretized_waypoints.push_back(traj.sample(time));
        time += dt;
    }

    State temp_state = state;
    double temp_cost = 0;

    std::vector<Rollout> rollouts = std::vector<Rollout>();
    std::vector<State> logging_states = std::vector<State>();



    for (int k = 0; k < K; k++){
        for (int t_ = 0; t_ < T; t_++){
            double v = ctrls(0, k*T + t_);
            double w = ctrls(1, k*T + t_);
            Control ctrl = Control{v, w};
            temp_state = unicyle_dynamics(temp_state, ctrl, mp, dt);
            logging_states.push_back(temp_state);

            if(t_ < T-1){
                temp_cost += cost_func(temp_state, ctrl, discretized_waypoints.at(t_), t_) * pow(0.95, t_);
            } else {
                // temp_cost += terminal_cost_func(temp_state, discretized_waypoints.at(t_));
            }
        }
        costs(k) = temp_cost;
        rollouts.push_back(Rollout{logging_states});

        // Reset per-rollout variables
        temp_state = state;
        temp_cost = 0.0;
        logging_states.clear();
    }

    double max_cost = costs.maxCoeff();
    double min_cost = costs.minCoeff();
    std::cout << "Cost diff: " << max_cost - min_cost << std::endl;
    

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

    rec.log(
        "mppi/highest_weight",
        rerun::Scalars(weights.maxCoeff())
    );

    double max_weight = weights.maxCoeff();

    for (int i = 0; i < rollouts.size(); i++) {

        if(weights(i) > max_weight/2){

            double alpha = weights(i)/max_weight * 255.0;

            std::vector<rerun::Position2D> pts;
            pts.reserve(rollouts[i].logging_states.size());

            for (const auto& s : rollouts[i].logging_states) {
                pts.emplace_back(
                    static_cast<float>(s.val(0)),
                    static_cast<float>(s.val(1))
                );
            }

            std::vector<std::vector<rerun::Position2D>> strips;
            strips.push_back(pts);

            rec.log(
                "mppi/sample/" + std::to_string(i),
                rerun::LineStrips2D(strips)
                    .with_colors(rerun::Color(0, 255, 0, alpha))
            );
        }
    }

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

double MPPI::cost_func(State state, Control ctrl, Waypoint target, int step){
    Eigen::DiagonalMatrix Q = Eigen::DiagonalMatrix<double, 5>(4.0, 4.0, 0.0, 0.0, 0.0);
    Eigen::DiagonalMatrix R = Eigen::DiagonalMatrix<double, 2>(0.05, 0.01);
    
    VectorXd state_diff = target.state.val - state.val;
    state_diff(2) = std::fmod((state_diff(2) + M_PI), (M_PI * 2)) - M_PI;

    double cost = state_diff.transpose() * Q * state_diff; // State cost
    cost += ctrl.val.transpose() * R * ctrl.val; // Control cost
    
    return cost;
}

double MPPI::terminal_cost_func(State state, Waypoint target){
    Eigen::DiagonalMatrix Q = Eigen::DiagonalMatrix<double, 5>(10.0, 10.0, 3.0, 0.0, 0.0);
    VectorXd state_diff = target.state.val - state.val;
    state_diff(2) = std::fmod((state_diff(2) + M_PI), (M_PI * 2)) - M_PI;

    double cost = state_diff.transpose() * Q * state_diff; // State cost
    
    return cost;
}



