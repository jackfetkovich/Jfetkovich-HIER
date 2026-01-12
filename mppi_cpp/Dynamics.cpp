#include <math.h>
#include "Dynamics.hpp"
#include "State.hpp"
#include "Control.hpp"

State unicyle_dynamics(State state, Control ctrl, double dt){
    State x_star = State(
        state.val(0) + ctrl.val(0) * cos(state.val(2)) * dt,
        state.val(1) + ctrl.val(1) * sin(state.val(2)) * dt,
        state.val(2) + ctrl.val(1),
        ctrl.val(0),
        ctrl.val(1)
    );
    return x_star;
}

