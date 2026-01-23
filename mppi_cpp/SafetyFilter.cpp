
#include <cmath>
#include "SafetyFilter.hpp"
#include <Eigen/Dense>
#include "Control.hpp"
#include "State.hpp"
#include "Obstacle.hpp"
#include "gurobi_c++.h"


/* 
    argmin(u) {u - u_nom}
    s.t.
    v_min <= v <= vmax
    w_min <= w <= wmax
    Lg_h @ u + dh_dt[i] + alpha * h[i] >= 0
    For now, just one obstacle (maybe the closest?)
*/

SafetyFilter::SafetyFilter(RobotParams params, GRBEnv *env, double alpha, Eigen::Matrix2d Q) :
    params(params),
    env(env),
    model(GRBModel(*env)), // Pointer?
    h(0.0),
    lgh1(0.0),
    lgh2(0.0),
    dhdt(0.0),
    alpha(alpha),
    Q(Q)
    {
        u = std::vector<GRBVar>();
        u.push_back(model.addVar(-params.mp.max_v, params.mp.max_v, 0.0, GRB_CONTINUOUS));
        u.push_back(model.addVar(-params.mp.max_omega, params.mp.max_omega, 0.0, GRB_CONTINUOUS));

        inequality = model.addConstr(lgh1 * u.at(0) + lgh2 * u.at(1) >= -(dhdt + alpha * h));

        double Q0 = Q(0,0); // Weighting coefficients
        double Q1 = Q(1,1);
        GRBQuadExpr obj = 0.0;

        obj += Q0 * u.at(0) * u.at(0);
        obj += Q1 * u.at(1) * u.at(1);

        model.setObjective(obj, GRB_MINIMIZE);
        model.update();
    }

Control SafetyFilter::filter(Control u_nom, State state, Obstacle ob){
    double dx = state.val(0) - ob.x + params.length * cos(state.val(1));
    double dy = state.val(1) - ob.y + params.length * sin(state.val(1));
    h = pow(dx, 2) + pow(dy, 2) - pow(ob.radius, 2);
    lgh1 = 2*dx*cos(state.val(2)) + 2*dy*sin(state.val(2));
    lgh2 = -2 * dx * params.length * sin(state.val(2)) + 2 * dy * params.length * cos(state.val(2));
    dhdt = -2 * dx * ob.vx - 2 * dy * ob.vy;

    model.chgCoeff(inequality, u.at(0), lgh1);
    model.chgCoeff(inequality, u.at(1), lgh2);

    double rhs = -(dhdt + alpha * h);
    inequality.set(GRB_DoubleAttr_RHS, rhs);

    model.update();
    model.optimize();

    Control u_safe = Control(
        u.at(0).get(GRB_DoubleAttr_X),
        u.at(1).get(GRB_DoubleAttr_X)
    );

    return u_safe;
}
