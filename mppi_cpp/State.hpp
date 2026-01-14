#pragma once
#include <Eigen/Dense>

using Eigen::VectorXd;

struct State {
    VectorXd val;

    State(double x, double y, double theta, double v, double w)
        : val(5)
        {
            val << x, y, theta, v, w;
        };

    State(VectorXd val):val(val){};
};