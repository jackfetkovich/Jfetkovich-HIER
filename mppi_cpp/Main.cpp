#include <vector>
#include <Eigen/Dense>
#include "State.hpp"
#include "Control.hpp"
#include "Waypoint.hpp"
#include "MPPI.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main(){
    MPPI mppi = MPPI(3, 5);
    
    std::vector<Waypoint> wp = {
        Waypoint{0.0, 0.0, 0.0},
        Waypoint{1.0, 0.0, 0.5}
    };

    State init_state = State(
        0.0, 
        0.0,
        0.0, 
        0.0,
        0.0
    );

    Control ctrl = mppi.get_control(init_state, wp, 0.5);

    return 0;
}