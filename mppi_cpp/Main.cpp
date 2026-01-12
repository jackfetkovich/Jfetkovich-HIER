#include <vector>
#include <Eigen/Dense>
#include "State.hpp"
#include "Control.hpp"
#include "Waypoint.hpp"
#include "MPPI.hpp"
#include "Trajectory.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main(){
    MPPI mppi = MPPI(3, 5, 0.5);
    
    State init_state = State(
        0.0, 
        0.0,
        0.0, 
        0.0,
        0.0
    );

    Trajectory traj = Trajectory();

    Control ctrl = mppi.get_control(init_state, traj, 3.0, 0.05);

    return 0;
}