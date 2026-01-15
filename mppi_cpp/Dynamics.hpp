#pragma once

#include "State.hpp"
#include "Control.hpp"
#include "MotionParams.hpp"

State unicyle_dynamics(State state, Control ctrl, MotionParams mp, double dt);