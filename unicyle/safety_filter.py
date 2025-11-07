import cvxpy as cp
import numpy as np
from numba import int32, float32
from numba.experimental import jitclass
import time
from parameters import Parameters

class SafetyFilter():
    def __init__(self, params, alpha, q, dt, output=False):
        self.num_obstacles = len(params.obstacles)
        self.offsets = [ (params.l, 0.0),          # center front
                        (params.l,  0.3),         # right side
                        (params.l, -0.3)  ]
        self.u_nom = cp.Parameter(2) # MPPI Output (what's changing)
        self.last_u = cp.Parameter(2)
        self.x = cp.Parameter(5)
        self.lgh1 = [[cp.Parameter() for _ in range(len(self.offsets))] for _ in range(self.num_obstacles)]
        self.lgh2 = [[cp.Parameter() for _ in range(len(self.offsets))] for _ in range(self.num_obstacles)]
        self.h = [[cp.Parameter() for _ in range(len(self.offsets))] for _ in range(self.num_obstacles)]
        self.dh_dt = [[cp.Parameter() for _ in range(len(self.offsets))] for _ in range(self.num_obstacles)]
        self.u = cp.Variable(2) # Control output (decision variable)
        self.q = q
        self.cost = cp.quad_form(self.u - self.u_nom, q)
        self.filter_outputs = np.zeros((3,2), dtype=np.float32)
        self.alpha = alpha
        self.last_u_var = np.zeros(2)
        self.dt = dt
        self.last_obstacle_pos = np.copy(params.last_obstacle_pos)
        self.output = output

        self.constraints = [
            self.u[0] <= params.max_v, # Box constraints
            self.u[0] >= -params.max_v,
            self.u[1] <= params.max_w,
            self.u[1] >= -params.max_w,
            self.u[0] - self.last_u[0] <= params.max_v_dot * dt, # Slew-rate limiting
            self.u[0] - self.last_u[0] >= -params.max_v_dot * dt,
            self.u[1] - self.last_u[1] <= params.max_w_dot * dt,
            self.u[1] - self.last_u[1] >= -params.max_w_dot * dt,
        ]

        for i in range(self.num_obstacles):
            for j in range(len(self.offsets)):
                Lg_h = cp.hstack([
                    self.lgh1[i][j], 
                    self.lgh2[i][j]
                ])
                self.constraints.append(Lg_h @ self.u + self.dh_dt[i][j] + alpha * self.h[i][j] >= 0)

        self.objective = cp.Minimize(self.cost)
        self.prob = cp.Problem(self.objective, self.constraints)

    def filter(self, u_in, x, params, last_u):
        self.u_nom.value = u_in
        # print(u_in)
        self.x.value = x
        self.last_u.value = last_u

        # print(f"u_nom: {self.u_nom.value}")
        # print(f"x: {self.x.value}")
        # print(f"last_U: {self.last_u.value}")

        # Loop through all obstacles
        for i in range(self.num_obstacles):
            c = params.obstacles[i][0:2]   # obstacle center (x, y)
            r = params.obstacles[i][2]     # obstacle radius

            vx_obs = (c[0] - self.last_obstacle_pos[i][0]) / self.dt
            # print(f"vx_obs:{vx_obs}")
            vy_obs = (c[1] - self.last_obstacle_pos[i][1]) / self.dt
            # print(f"vy_obs:{vy_obs}")
            self.last_obstacle_pos[i] = np.array([c[0], c[1]])

            ct = np.cos(x[2])
            st = np.sin(x[2])
            for j, (l, b) in enumerate(self.offsets):
                px = x[0] + l*ct - b*st
                py = x[1] + l*st + b*ct
                dx = px - c[0]
                dy = py - c[1]

                self.h[i][j].value = dx**2 + dy**2 - (r + 0.1)**2

                # Control influence
                self.lgh1[i][j].value = 2*dx*ct + 2*dy*st
                self.lgh2[i][j].value = 2*dx*(-l*st - b*ct) + 2*dy*(l*ct - b*st)

                self.dh_dt[i][j].value = -2*dx*vx_obs - 2*dy*vy_obs

        if np.isnan(u_in[0]) or np.isnan(u_in[1]):
            print("NAN")
            print("x", x)
            print("Ob 1 pos:", params.obstacles[0, :])
            # print("Ob 2 pos:", params.obstacles[1, :])

        try:
            self.prob.solve(solver=cp.OSQP,verbose=False)
            if self.prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError("Infeasible or failed solve")
            u_out = self.u.value

        except cp.error.SolverError:
        # Fallback strategy
            u_out = np.array([0, 0])
        # Solve
        # print(f"u_out: {u_out}")
        # if self.output:
        #     for i in range(self.num_obstacles):
        #         print(f"Obstacle: {i}")
        #         #Lg_h @ self.u + self.dh_dt[i] + alpha * self.h[i]
        #         print(f"h[{i}]: {self.h[i].value}")
        #         print(f"[{self.lgh1[i].value}, {self.lgh2[i].value}] * {u_out} + {self.dh_dt[i].value} + {self.alpha} * {self.h[i].value}")
        #         term = np.array([self.lgh1[i].value, self.lgh2[i].value]) @ u_out
        #         print(f"{term} + {self.dh_dt[i].value} + {self.alpha* self.h[i].value}")
        #         print(f"{term + self.dh_dt[i].value + self.alpha * self.h[i].value}")
        if self.output:
            print(f"status:{self.prob.status}")
            print("**************************")
        return u_out