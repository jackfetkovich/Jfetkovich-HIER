#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <rerun.hpp>
#include <rerun/demo_utils.hpp>
#include "State.hpp"
#include "Control.hpp"
#include "Waypoint.hpp"
#include "MPPI.hpp"
#include "Trajectory.hpp"
#include "Timer.hpp"
#include "Dynamics.hpp"
#include "MotionParams.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace rerun::demo;

int main(){    
    State init_state = State(
        0.0, 
        0.0,
        0.0, 
        0.0,
        0.0
    );

    MotionParams mp = MotionParams{0.35, 1.0, 1.9, 1.5};

    // Note: Thetas are ignored and recomputed by geometry
    std::vector<Waypoint> waypoints{
        Waypoint{init_state, 0.0},
        Waypoint{State(0.25, 0.0, 0.0, 0.0, 0.0), 1.0},
        Waypoint{State(0.4, 0.15, 0.0, 0.0, 0.0), 2.0},
        Waypoint{State(0.65, 0.25, 0.0, 0.0, 0.0), 3.0},

    };

    Trajectory traj = Trajectory(waypoints);

    // Create a new `RecordingStream` which sends data over gRPC to the viewer process.
    const auto rec = rerun::RecordingStream("rerun_example_cpp");
    // Try to spawn a new viewer instance.
    rec.spawn().exit_on_failure();

    std::vector<rerun::Position2D> rr_traj_points = std::vector<rerun::Position2D>();
    for (int i = 0; i < waypoints.size(); i++){
        rerun::Position2D point = rerun::Position2D(waypoints.at(i).state.val(0), waypoints.at(i).state.val(1));
        rr_traj_points.push_back(point);
    }

    rec.set_time_duration_secs("sim_time", 0.0); // New time on timeline


    rec.log(
        "mppi/trajectory/points",
        rerun::Points2D(rr_traj_points).with_radii({0.08f})
    );

    Timer time = Timer();

    time.reset();
    double elapsed_time = time.elapsed();

    MPPI mppi = MPPI(2000, 30, 0.05, mp, rec);
    State robot_state = init_state;
    
    while(elapsed_time < 3.0){
        Control ctrl = mppi.get_control(robot_state, traj, elapsed_time, 0.05);
        rec.set_time_duration_secs("sim_time", elapsed_time);

        rec.log(
            "/mppi/control/velocity",
            rerun::Scalars(ctrl.val(0))
        );

        rec.log(
            "/mppi/control/omega",
            rerun::Scalars(ctrl.val(1))
        );

        robot_state = unicyle_dynamics(robot_state, ctrl, mp, 0.05);
        rerun::Position2D loc = rerun::Position2D(robot_state.val(0), robot_state.val(1));

        rec.log(
            "mppi/telemetry",
            rerun::Points2D(loc).with_radii({0.01f}).with_colors(rerun::Color(0, 0, 255))
        );

        Waypoint goal = traj.sample(elapsed_time);
        rerun::Position2D goal_pos = rerun::Position2D(goal.state.val(0), goal.state.val(1));

        rec.log(
            "mppi/goal",
            rerun::Points2D(goal_pos).with_radii({0.01f}).with_colors(rerun::Color(255, 255, 128))
        );

        // std::cout << elapsed_time << std::endl;
        
        elapsed_time = time.elapsed();
    }

    std::cout << robot_state.val << std::endl;

    return 0;
}