import matplotlib.pyplot as plt
import numpy as np

# Global list to store clicked points
points = []

my_points = np.array([     
        (-0.008, -0.009),
        (0.962, -0.009),
        (1.476, 0.443),
        (1.474, 1.164),
        (1.505, 1.972),
        (1.255, 2.350),
        (0.408, 2.437),
        (-0.233, 2.294),
        (-0.200, 3.050),
        (-0.233, 3.692),
        (-0.297, 4.623),
        (-0.079, 4.818),
        (0.562, 4.839),
        (1.749, 4.774),
        (3.084, 4.968),
        (3.935, 4.774),
        (4.447, 4.559),
        (4.778, 4.276),
        (4.803, 2.850),
        (4.693, 2.374),
        (3.802, 2.449),
        (3.138, 2.556),
        (2.862, 2.386),
        (2.928, 1.839),
        (3.488, 1.283),
        (3.926, 1.137),
        (4.367, 1.004),
        (4.551, 0.578),
        (4.656, 0.056),
        (4.546, -0.126),
        (3.395, -0.115),
        (3.065, -0.137),
        (2.668, -0.653),
        (2.103, -0.842),
        (1.617, -0.714),
        (0.658, -0.692),
        (0.119, -0.284),
        (-0.012, 0.002),
    ])
fig, ax = plt.subplots()
count = 0
def onclick(event):
    # Only record clicks inside the axes
    global count
    if event.inaxes:
        ax.scatter(my_points[count+1, 0],my_points[count+1, 1], color="blue")
        count+=1
        x, y = event.xdata, event.ydata
        points.append((x, y))
        print(f"({x:.3f}, {y:.3f}),")
        # Draw the point on the plot
        plt.plot(x, y, 'ro')
        plt.draw()

def main():
    print("Click on the plot to select points. Close the window when done.")

    print(my_points.shape)
    
    ax.set_title("Click to select waypoints (close window when done)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-1, 5])
    ax.set_xlim([-1, 5])
    ax.scatter(my_points[0, 0],my_points[0, 1], color="blue")
    # ax.scatter(my_points[:, 0], my_points[:, 1])
    ax.grid(True)

    # Connect the click handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    # After window is closed, print results
    print("\nFinal list of points (in order):")
    for p in points:
        print(f"({p[0]:.6f}, {p[1]:.6f})")

if __name__ == "__main__":
    main()