import matplotlib.pyplot as plt
import numpy as np

# Global list to store clicked points
points = []

my_points = np.array([     
    (-0.000, -0.011),
    (-0.016, 0.502),
    (-0.009, 1.001),
    (-0.012, 1.578),
    (0.010, 1.987),
    (-0.008, 2.506),
    (-0.008, 3.000),
    (0.499, 3.009),
    (0.988, 3.030),
    (1.483, 3.030),
    (1.970, 3.030),
    (2.439, 3.063),
    (2.960, 3.063),
    (3.443, 3.063),
    (3.999, 3.047),
    (4.009, 2.487),
    (4.005, 2.006),
    (3.980, 1.541),
    (4.017, 1.063),
    (4.002, 0.631),
    (4.021, -0.000),
    (3.519, 0.043),
    (2.986, 0.029),
    (2.480, 0.032),
    (2.024, 0.035),
    (1.499, 0.014),
    (1.021, 0.027),
    (0.550, 0.005),
    (0.008, -0.009),
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