import matplotlib.pyplot as plt
import numpy as np

# Global list to store clicked points
points = []

def onclick(event):
    # Only record clicks inside the axes
    if event.inaxes:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        print(f"({x:.3f}, {y:.3f}),")
        # Draw the point on the plot
        plt.plot(x, y, 'ro')
        plt.draw()

def main():
    print("Click on the plot to select points. Close the window when done.")
    my_points = np.array([        (0.009, -0.022),
        (0.514, 0.035),
        (0.987, 0.436),
        (1.376, 0.743),
        (1.822, 0.855),
        (1.970, 0.855),
        (2.397, 0.812),
        (2.494, 0.714),
        (2.615, 0.422),
        (2.765, 0.178),
        (2.985, 0.007),
        (3.269, 0.017),
        (3.594, 0.008),
        (3.928, 0.008),
        (4.171, 0.052),
        (4.246, 0.117),
        (4.274, 0.207),
        (4.311, 0.307),
        (4.311, 0.460),
        (4.311, 0.570),
        (4.329, 0.843),
        (4.368, 1.124),
        (4.399, 1.544),
        (4.399, 2.492),
        (4.398, 3.271),
        (4.382, 3.508),
        (4.336, 3.607),
        (4.236, 3.623),
        (4.168, 3.623),
        (4.139, 3.633),
        (4.034, 3.674),
        (3.866, 3.696),
        (3.702, 3.696),
        (3.364, 3.690),
        (2.875, 3.711),
        (2.039, 3.752),
        (1.293, 3.752),
        (0.888, 3.795),
        (0.587, 3.795),
        (0.470, 3.766),
        (0.379, 3.724),
        (0.379, 3.649),
        (0.364, 3.577),
        (0.393, 3.554),
        (0.436, 3.491),
        (0.566, 3.399),
        (0.689, 3.316),
        (0.888, 3.217),
        (1.183, 3.032),
        (1.416, 2.899),
        (1.800, 2.732),
        (2.232, 2.560),
        (2.378, 2.511),
        (2.530, 2.385),
        (2.555, 2.304),
        (2.516, 2.262),
        (2.452, 2.225),
        (2.400, 2.225),
        (2.328, 2.205),
        (2.270, 2.205),
        (2.200, 2.184),
        (2.000, 2.184),
        (1.652, 2.184),
        (1.138, 2.184),
        (0.572, 2.205),
        (-0.025, 2.205)])
    print(my_points.shape)
    fig, ax = plt.subplots()
    ax.set_title("Click to select waypoints (close window when done)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-1, 5])
    ax.set_xlim([-1, 5])
    ax.scatter(my_points[:, 0], my_points[:, 1])
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