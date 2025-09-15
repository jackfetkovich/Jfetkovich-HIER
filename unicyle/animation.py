import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate(x_vals, y_vals, x_traj, y_traj, x_ob, y_ob, sample_trajs, weights, params):
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
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)


    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    circles = np.empty(len(params.obstacles), dtype=plt.Circle)

    for i in range(len(params.obstacles)):
        circles[i] = plt.Circle((params.obstacles[i][0], params.obstacles[i][1]), params.obstacles[i][2], color='y')
        ax.add_patch(circles[i])

    # Plot the trajectory if provided
    if x_traj is not None and y_traj is not None:
        ax.plot(x_traj, y_traj, 'k--', linewidth=1.5, label="Trajectory")  # Dotted reference path

    samples = []
    for i in range(params.K):
        samples.append(ax.plot([], [], color=[0.5, 0.5, 0.5], linewidth=0.5)[0])

    # Initialize plot elements
    line, = ax.plot([], [], 'r-', linewidth=2)  # History line
    point, = ax.plot([], [], 'bo', markersize=8)  # Current position
    ghost,  = ax.plot([], [], 'gx', markersize=6)  # Desired position


    # Update function
    def update(frame):
        for i in range(len(circles)):
            circles[i].set_center((x_ob[i][frame], y_ob[i][frame]))
 
        # circle.set_center((x_ob[frame], y_ob[frame]))
        
        x, y = x_vals[frame], y_vals[frame]

        # Update history path
        line.set_data(x_vals[:frame + 1], y_vals[:frame + 1])
                # Update point position
        point.set_data([x], [y])

        max_intensity = -1
        for i in range(params.K):
            this_intensity = weights[frame][i]
            if this_intensity > max_intensity:
                max_intensity = this_intensity
        # Update generated trajectories
        for i in range(params.K):
            intensity = min(weights[frame][i], 1)
            samples[i].set_data([], [])  # Clears previous data
            samples[i].set_color([0, intensity/max_intensity , 0, intensity/max_intensity])
            samples[i].set_data(sample_trajs[frame, i, 0, 0 : params.T], sample_trajs[frame, i, 1, 0 : params.T])

        # Update ghost point if reference trajectory exists
        if x_traj is not None and y_traj is not None:
            ghost.set_data([x_traj[int(frame/2)]], [y_traj[int(frame/2)]])


        return [line, point, ghost].append(samples)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=60, blit=False)
    plt.title(f"K={params.K}, T={params.T} - Safety Filter, Removal of unsafe paths")
    plt.legend()
    filename=f"./animations/{params.K}-{params.T}-obstacle_back_away.gif"
    ani.save(filename, writer='pillow', fps=20, )
    print(f"Animation saved as {filename}")
    plt.show()