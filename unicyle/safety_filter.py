import cvxpy as cp
import numpy as np
import time

class SafetyFilter:
    def __init__(self, params, alpha, q, dt):
        self.num_obstacles = len(params.obstacles)
        self.u_nom = cp.Parameter(2) # MPPI Output (what's changing)
        self.last_u = cp.Parameter(2)
        self.x = cp.Parameter(5)
        self.lgh1 = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.lgh2 = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.h = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.dh_dt = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.u = cp.Variable(2) # Control output (decision variable)
        self.q = q
        self.cost = cp.quad_form(self.u - self.u_nom, q)
        self.filter_outputs = np.zeros((3,2), dtype=np.float32)
        self.alpha = alpha
        self.last_u_var = np.zeros(2)
        self.dt = dt

        self.constraints = [
            self.u[0] <= params.max_v, # Box constraints
            self.u[0] >= -params.max_v,
            self.u[1] <= params.max_w,
            self.u[1] >= -params.max_w,
            self.u[0] - self.last_u[0] <= params.max_v_dot, # Slew-rate limiting
            self.u[0] - self.last_u[0] >= -params.max_v_dot,
            self.u[1] - self.last_u[1] <= params.max_w_dot,
            self.u[1] - self.last_u[1] >= -params.max_w_dot,
        ]

        for i in range(self.num_obstacles):
            Lg_h = cp.hstack([
                self.lgh1[i], 
                self.lgh2[i]
            ])
            self.constraints.append(Lg_h @ self.u + self.dh_dt[i] + alpha * self.h[i] >= 0)

        self.objective = cp.Minimize(self.cost)
        self.prob = cp.Problem(self.objective, self.constraints)

    def filter(self, u_in, x, params, last_u):
        self.u_nom.value = u_in
        self.x.value = x
        self.last_u.value = last_u

        # Loop through all obstacles
        for i in range(self.num_obstacles):
            c = params.obstacles[i][0:2]   # obstacle center (x, y)
            r = params.obstacles[i][2]     # obstacle radius

            dx = x[0] - c[0] + params.l * np.cos(x[2])
            dy = x[1] - c[1] + params.l * np.sin(x[2])

            vx_obs = (c[0] - params.last_obstacle_pos[i][0]) / self.dt
            vy_obs = (c[1] - params.last_obstacle_pos[i][1]) / self.dt
            params.last_obstacle_pos[i] = np.array([c[0], c[1]])

            # Barrier function
            self.h[i].value = (dx)**2 + (dy)**2 - (r+0.1)**2
            # Lie derivative term
            self.lgh1[i].value = 2*dx*np.cos(x[2]) + 2*dy*np.sin(x[2])
            self.lgh2[i].value = -2*dx*params.l*np.sin(x[2]) + 2*dy*params.l*np.cos(x[2])
            self.dh_dt[i].value = -2*(dx)*vx_obs - 2*(dy)*vy_obs # Obstacle time_varying


        if np.isnan(u_in[0]) or np.isnan(u_in[1]):
            print("NAN")
            print("x", x)
            print("Ob 1 pos:", params.obstacles[0, :])
            # print("Ob 2 pos:", params.obstacles[1, :])

        try:
            start_time = time.perf_counter()
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            end_time = time.perf_counter()
            print(f"solve time: {end_time - start_time}")
            if self.prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.error.SolverError("Infeasible or failed solve")
            u_out = self.u.value
        except cp.error.SolverError:
        # Fallback strategy
            u_out = np.array([0, 0])
        # Solve
        return u_out