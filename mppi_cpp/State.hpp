#pragma once
#include <Eigen/Dense>

using Eigen::VectorXd;

struct State {
    VectorXd value;

    State(double x, double y, double theta, double v, double w)
        : value(5)
        {
            value << x, y, theta, v, w;
        };
};