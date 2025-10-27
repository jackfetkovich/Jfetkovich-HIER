import cvxpy as cp
import numpy as np
import time

class SafetyFilter:
    def __init__(self, params, alpha):
        self.num_obstacles = len(params.obstacles)
        self.u_nom = cp.Parameter(2) # MPPI Output (what's changing)
        self.last_u = cp.Parameter(2)
        self.x = cp.Parameter(5)
        self.vx_obs = cp.Parameter()
        self.vy_obs = cp.Parameter()
        self.lgh1 = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.lgh2 = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.h = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.dh_dt = [cp.Parameter() for _ in range(self.num_obstacles)]
        self.D = cp.Parameter((2, 2))  # will hold diag(self.sqrt_q)
        self.u = cp.Variable(2) # Control output (decision variable)
        self.cost = cp.sum_squares(self.D @ (self.u - self.u_nom))
        self.filter_outputs = np.zeros((3,2), dtype=np.float32)
        self.alpha = alpha

        self.obstacle_center_params = np.empty((self.num_obstacles,2), dtype=object)
        for i in range(self.num_obstacles):
            self.obstacle_center_params[i, 0] = cp.Parameter() # x coordinate
            self.obstacle_center_params[i, 1] = cp.Parameter() # y coordinate

        self.obstacle_radius_params = np.empty(self.num_obstacles, dtype=object)
        for i in range(self.num_obstacles):
            self.obstacle_radius_params[i] = cp.Parameter(1, nonneg=True) # obstacle radius (probably static, but who knows?)

        self.last_obstacle_center_params = np.empty((self.num_obstacles,2), dtype="object")
        for i in range(self.num_obstacles):
            self.last_obstacle_center_params[i, 0] = cp.Parameter() # last x coordinate
            self.last_obstacle_center_params[i, 1] = cp.Parameter() # last y coordinate
        
        self.obstacle_velocity_params = np.empty((self.num_obstacles,2), dtype=object)
        for i in range(self.num_obstacles):
            self.obstacle_velocity_params[i, 0] = cp.Parameter() # x velocity
            self.obstacle_velocity_params[i, 1] = cp.Parameter() # y velocity


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
        # Loop through all obstacles
        for i in range(self.num_obstacles):
            c = params.obstacles[i][0:2]   # obstacle center (x, y)
            r = params.obstacles[i][2]     # obstacle radius

            dx = x[0] - c[0] + params.l * np.cos(x[2])
            dy = x[1] - c[1] + params.l * np.sin(x[2])

            vx_obs = (c[0] - params.last_obstacle_pos[i][0]) / params.safety_dt
            vy_obs = (c[1] - params.last_obstacle_pos[i][1]) / params.safety_dt
            params.last_obstacle_pos[i] = np.array([c[0], c[1]])

            v_obs = np.array([vx_obs, vy_obs])

            # Barrier function
            h = (dx)**2 + (dy)**2 - (r+0.1)**2
            # Lie derivative term
            Lg_h = np.array([
                2*dx*np.cos(x[2]) + 2*dy*np.sin(x[2]),
                -2*dx*params.l*np.sin(x[2]) + 2*dy*params.l*np.cos(x[2])
            ])

            dh_dt = -2*(dx)*vx_obs - 2*(dy)*vy_obs # Obstacle time_varying

        if np.isnan(u_in[0]) or np.isnan(u_in[1]):
            print("NAN")
            print("x", x)
            print("Ob 1 pos:", params.obstacles[0, :])
            # print("Ob 2 pos:", params.obstacles[1, :])

        compute_times = np.zeros(3)
        for j in range(len(self.filter_outputs)):
                
            # Define QP
            self.Q = np.diag([40.0 * ((j+1)/3), 1.0])   # heavier cost on v
            cost = cp.quad_form(self.u - self.u_nom, self.Q)
            prob = cp.Problem(cp.Minimize(cost), self.constraints)
            
            try:
                start_time = time.perf_counter()
                prob.solve(solver=cp.OSQP, warm_start=True)
                end_time = time.perf_counter()
                print(f"solve time: {end_time - start_time}")
                compute_times[j] = end_time-start_time
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    raise cp.error.SolverError("Infeasible or failed solve")
                u_out = self.u.value
                self.filter_outputs[j] = u_out
            except cp.error.SolverError:
            # Fallback strategy
                u_out = np.array([0, 0])
        
        # with open('./data/compute_time.csv', 'a', newline='', encoding='utf-8') as file:
        #     # 'Step',  'v_nom', 'v_q1', 'v_q2', 'v_q3', 'w_nom','w_q1', 'w_q2', 'w_q3' 'x', 'y', 'obs_x']
        #     writer = csv.writer(file)
        #     writer.writerow([compute_times[0]])
        #     writer.writerow([compute_times[1]])
        #     writer.writerow([compute_times[2]])

        params.first_filter = False
        # Solve
        return self.filter_outputs