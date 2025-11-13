import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mppi import unicyle_dynamics

def animate(x_traj, y_traj, output_frames, params):
    """
    Animates the movement of an object in 2D space given its state variables over time.
    Also plots a given trajectory as a dotted line.

    Parameters:
    x_vals (list or np.array): X positions over time.
    y_vals (list or np.array): Y positions over time.
    theta_vals (list or np.array): Orientations (in radians) over time.
    x_traj (list or np.array, optional): X values of the reference trajectory.
    y_traj (list or np.array, optional): Y values of the reference trajectory.
    """
    matplotlib.use("TkAgg")
    # Set up the figure
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    # fig, ax = plt.subplots()
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)


    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    circles = []

    for i in range(len(params.obstacles)):
        circles.append(plt.Circle((params.obstacles[i][0], params.obstacles[i][1]), params.obstacles[i][2], color='y'))
        ax.add_patch(circles[i])

    # Plot the trajectory if provided
    if x_traj is not None and y_traj is not None:
        ax.plot(x_traj, y_traj, 'k--', linewidth=1.5, label="Trajectory")  # Dotted reference path

    samples = []
    for i in range(params.K):
        samples.append(ax.plot([], [], color=[0.5, 0.5, 0.5], linewidth=0.5)[0])

    # l1 = ax.plot([], [], color=(0.698, 0.031, 0.988, 0.75), linewidth=3)[0]
    # l2 = ax.plot([], [], color=(0.988, 0.561, 0.031, 0.75), linewidth=3)[0]
    # l3 = ax.plot([], [], color=(0.031, 0.788, 0.988, 0.75), linewidth=3)[0]

    # Initialize plot elements
    line, = ax.plot([], [], 'r-', linewidth=2)  # History line
    point, = ax.plot([], [], 'bo', markersize=8)  # Current position
    ghost,  = ax.plot([], [], 'gx', markersize=6)  # Desired position

    x_vals = []
    y_vals = []
    # Update function
    def update(frame):
        x, y, theta, v, omega = frame["x"], frame["y"], frame["theta"], frame["v"], frame["w"]
        x_ob, y_ob = frame["x_ob"], frame["y_ob"]
        weights = frame["weights"]
        sample_trajs = frame["samples"]
        t = frame["t"]
        safe_outputs = frame["safe_outputs"]

        x_vals.append(x)
        y_vals.append(y)

        # # Obstacles
        # for i in range(len(circles)):
        #     circles[i].center = (x_ob[i], y_ob[i])

        # History and current position
        line.set_data(x_vals, y_vals)
        point.set_data([x], [y])

        # Samples
        max_intensity = np.max(weights)
        if max_intensity > 0:
            norm_weights = weights / max_intensity
        else:
            norm_weights = np.zeros_like(weights)

        for i, s in enumerate(samples):
            w = norm_weights[i]
            s.set_color([0, w, 0, w])
            s.set_data(sample_trajs[i, 0, :params.T], sample_trajs[i, 1, :params.T])

        safe_vs = safe_outputs[:, 0]
        safe_ws = safe_outputs[:, 1]

        steps_forward = 10
        current_state = np.array([x, y, theta, v, omega])
        prop_paths = np.zeros((3, steps_forward, 5))
        prop_paths[0, 0] = current_state
        prop_paths[1, 0] = current_state
        prop_paths[2, 0] = current_state

        for i in range(steps_forward-1):
            prop_paths[0, i+1]= unicyle_dynamics(prop_paths[0, i], np.array([safe_vs[0], safe_ws[0]]), params)
            prop_paths[1, i+1]= unicyle_dynamics(prop_paths[1, i], np.array([safe_vs[1], safe_ws[1]]), params)
            prop_paths[2, i+1]= unicyle_dynamics(prop_paths[2, i], np.array([safe_vs[2], safe_ws[2]]), params)

        # l1.set_data(prop_paths[0, :, 0], prop_paths[0, :, 1])
        # l2.set_data(prop_paths[1, :, 0], prop_paths[1, :, 1])
        # l3.set_data(prop_paths[2, :, 0], prop_paths[2, :, 1])


        # Ghost point
        if x_traj is not None and y_traj is not None:
            idx = int(t / int(params.dt / params.safety_dt))
            ghost.set_data([x_traj[idx]], [y_traj[idx]])

        # return [line, point, ghost, l1, l2, l3, *samples] + circles
        return [line, point, ghost, *samples]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=output_frames, interval=1, blit=True)
    plt.title(f"K={params.K}, T={params.T} - Warm Start on Safe Outputs")
    plt.legend()
    # filename=f"./animations/{params.K}-{params.T}-generation.gif"
    # ani.save(filename, writer='pillow', fps=10, )
    # print(f"Animation saved as {filename}")
    plt.show()