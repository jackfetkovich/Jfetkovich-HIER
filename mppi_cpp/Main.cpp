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

    // Note: Thetas are ignored and recomputed by geometry
    std::vector<Waypoint> waypoints{
        Waypoint{init_state, 0.0},
        Waypoint{State(1.0, 0.0, 0.0, 0.0, 0.0), 0.5},
        Waypoint{State(1.0, 1.5, 0.0, 0.0, 0.0), 1.0}
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


    rec.log(
        "mppi/trajectory/points",
        rerun::Points2D(rr_traj_points).with_radii({0.08f})
    );

    MPPI mppi = MPPI(200, 10, 5, rec);



    Control ctrl = mppi.get_control(init_state, traj, 0.0, 0.05);
    std::cout << ctrl.val << std::endl;

    return 0;
}