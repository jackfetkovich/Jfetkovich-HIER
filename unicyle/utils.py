from numba import njit
import numpy as np

@njit
def gen_normal_control_seq(mu_v, sigma_v, mu_w, sigma_w, K, T):
    return np.dstack((
        np.random.normal(loc=mu_v, scale=sigma_v, size=(K, T)), #v
        np.random.normal(loc=mu_w, scale=sigma_w, size=(K, T)), #omega
        # np.random.uniform(low=-20, high=20, size=(K, T)),
    )) 

def distance_of_path(p):
    distance = 0
    for x in range(len(p)-1):
        distance += np.sqrt((p[x+1,0] - p[x, 0])**2 + (p[x+1,1] - p[x, 1])**2)
    return distance

@njit
def closest_point_on_path(waypoints, point, last_index):
    """
    waypoints: (N,2) array
    point: (2,) array
    Returns: (closest_x, closest_y, distance, seg_idx, t)
    """
    best_d2 = 1e18
    best_point = np.zeros(2)
    best_idx = last_index
    best_t = 0.0

    for i in range(max(best_idx, 0), min(best_idx + 20, waypoints.shape[0]-2)):
        A = waypoints[i]
        B = waypoints[i + 1]
        AB = B - A
        AP = point - A
        denom = AB[0] * AB[0] + AB[1] * AB[1]
        if denom > 1e-12:
            t = (AP[0] * AB[0] + AP[1] * AB[1]) / denom
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        else:
            t = 0.0
        proj = A + t * AB
        dx = point[0] - proj[0]
        dy = point[1] - proj[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_point = proj
            best_idx = i
            best_t = t

    return best_point[0], best_point[1], best_idx


@njit
def point_in_obstacle(point, obstacles):
    pt_x = point[0]
    pt_y = point[1]
    for ob in obstacles:
        if (pt_x - ob[0])**2 + (pt_y - ob[1])**2 - ob[3]**2 <= 0:
            return True
    return False
    