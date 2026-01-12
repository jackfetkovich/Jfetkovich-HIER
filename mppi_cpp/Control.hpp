#pragma once
#include <Eigen/Dense>

using Eigen::Vector2d;

struct Control {
    Vector2d val;
    Control(double v, double w): val(Vector2d(v, w)){};
};