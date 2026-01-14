#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "State.hpp"
#include "Control.hpp"
#include "Waypoint.hpp"
#include "MPPI.hpp"
#include "Trajectory.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main(){
    MPPI mppi = MPPI(1000, 20, 5);
    
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

    Control ctrl = mppi.get_control(init_state, traj, 0.0, 0.05);
    std::cout << ctrl.val << std::endl;

    return 0;
}