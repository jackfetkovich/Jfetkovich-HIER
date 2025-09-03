import numpy as np



def distance_of_path(p):
    distance = 0
    for x in range(len(p)-1):
        distance += np.sqrt((p[x+1,0] - p[x, 0])**2 + (p[x+1,1] - p[x, 1])**2)
    
    return distance

points = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (4.0, 2.0),
        (4.0, 4.0),
        (2.0, 4.0),
        (0.0, 4.0),
        (0.0, 2.0),
        (0.0, 0.0)
    ]
p = np.array(points)
print(p)

k = distance_of_path(np.array(points))
print(k)