
#include <cmath>
#include <Eigen/Dense>
#include "Trajectory.hpp"
#include "Waypoint.hpp"
#include "State.hpp"

using Eigen::VectorXd;

Trajectory::Trajectory(std::vector<Waypoint> waypoints)
    :points(waypoints){
        // Populate headings
        for(int i = 0; i < points.size(); i++){
            if (i < points.size() - 1) {
                int dx = points.at(i+1).state.val(0) -  points.at(i).state.val(0);
                int dy = points.at(i+1).state.val(1) -  points.at(i).state.val(1);
                points.at(i).state.val(2) = atan2(dy, dx);
            } else {
                points.at(i).state.val(2) = points.at(i-1).state.val(2);
            }
        }
    }

Waypoint Trajectory::sample(double time){
    
    // Bound time to trajectory start and end
    if (time >= points.at(points.size()-1).t){
        return points.at(points.size()-1);
    } else if (time <= 0.0){
        return points.at(0);
    }
    
    int idx_above = -1;

    for (int i = 0; i < points.size(); i++){
        if (points.at(i).t > time) {
            idx_above = i;
            break;
        }
    }

    Waypoint above = points.at(idx_above);
    Waypoint below = points.at(idx_above - 1);

    VectorXd interp_vec = below.state.val + (above.state.val - below.state.val) * (time / (above.t - below.t));
    Waypoint interp_point = Waypoint{State(interp_vec), time};

    return interp_point;
} 