import numpy as np


new_thetas = np.hstack([
        np.zeros(14),
        np.pi/4 * np.ones(13),
        np.pi/2 * np.ones(13),
        3*np.pi /4 * np.ones(13),
        np.pi * np.ones(13),
        5*np.pi /4 * np.ones(13),
        3 * np.pi / 2 *np.ones(13),
        7*np.pi /4 * np.ones(14)
])

print(new_thetas)