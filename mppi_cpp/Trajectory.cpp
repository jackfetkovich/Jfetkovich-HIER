
// Standard Library
#include <cmath>
#include <functional>

// Imported Library
#include <Eigen/Dense>

// My code
#include "Trajectory.hpp"
#include "Waypoint.hpp"
#include "State.hpp"
#include "MotionParams.hpp"
#include "MathFuncs.hpp"

using Eigen::VectorXd;

Trajectory::Trajectory(std::vector<Waypoint> waypoints, MotionParams mp)
    :points(waypoints), params(mp){
        // Populate headings
        for(int i = 0; i < points.size(); i++){
            if (i < points.size() - 1) {
                double dx = points.at(i+1).state.val(0) -  points.at(i).state.val(0);
                double dy = points.at(i+1).state.val(1) -  points.at(i).state.val(1);
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

    VectorXd interp_vec = below.state.val + (above.state.val - below.state.val) * ((time - below.t) / (above.t - below.t));
    Waypoint interp_point = Waypoint{State(interp_vec), time};

    return interp_point;
} 

std::function<Waypoint(double)> generate_n_bezier(std::vector<Waypoint> ctrl_points){
    
    return [ctrl_points](double t){ 
        double x {0.0};
        double y {0.0};
        double weight {0.0};
        int n = ctrl_points.size() - 1;
        
        for(int i = 0; i < ctrl_points.size(); i++){
            weight = choose(n, i) * pow(t, i) * pow((1-t), (n-i));
            x += weight * ctrl_points.at(i).state.val(0);
            y += weight * ctrl_points.at(i).state.val(1);
        }

        return Waypoint{State(x, y, 0, 0, 0), 0.0};
    };
}


