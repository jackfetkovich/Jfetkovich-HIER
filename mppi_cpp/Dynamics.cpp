#include <cmath>
#include <algorithm>
#include "Dynamics.hpp"
#include "MotionParams.hpp"
#include "State.hpp"
#include "Control.hpp"

template <typename T>
int sign(T x) {
    return (T(0) < x) - (x < T(0));
}

State unicyle_dynamics(State state, Control ctrl, MotionParams mp, double dt){
    
    // Limiting linear and angular velocity
    ctrl.val(0) = std::max(std::min(ctrl.val(0), mp.max_v), -mp.max_v);
    ctrl.val(1) = std::max(std::min(ctrl.val(1), mp.max_omega), -mp.max_omega);

    // Limiting linear and angular acceleration
    if (abs(ctrl.val(0) - state.val(3)) > mp.max_v_dot * dt){
        ctrl.val(0) = state.val(3) + mp.max_v_dot * sign(ctrl.val(0) - state.val(3)) * dt;
    }
    if (abs(ctrl.val(1) - state.val(4)) > mp.max_omega_dot * dt){
        ctrl.val(1) = state.val(4) + mp.max_omega_dot * sign(ctrl.val(1) - state.val(4)) * dt;
    }

    
    State x_star = State(
        state.val(0) + ctrl.val(0) * cos(state.val(2)) * dt,
        state.val(1) + ctrl.val(1) * sin(state.val(2)) * dt,
        state.val(2) + ctrl.val(1),
        ctrl.val(0),
        ctrl.val(1)
    );
    return x_star;

}

