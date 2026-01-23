#pragma once

#include <vector>
#include "gurobi_c++.h"
#include "RobotParams.hpp"
#include "Control.hpp"
#include "State.hpp"
#include "Obstacle.hpp"

class SafetyFilter {
    public:
        SafetyFilter(RobotParams params, GRBEnv *env, double alpha, Eigen::Matrix2d Q);
        Control filter(Control u_nom, State state, Obstacle ob);

    private:
        RobotParams params;
        GRBEnv *env;
        GRBModel model;
        GRBConstr inequality;
        std::vector<GRBVar> u;
        Eigen::Matrix2d Q;
        double h;
        double lgh1;
        double lgh2;
        double dhdt;
        double alpha;
};