"""Simulate autocorrelation for gaussian beams
"""

import numpy as np
import matplotlib.pyplot as plt

WIDTH = 10.0e-3
HEIGHT = 10.0e-3
DEPTH = 1.0e-3
C = 2.99e8


def gaussian_pulse(
    x,
    y,
    z,
    w: float,
    delay: float,
    duration: float,
    angle: float,
    x_offset: float,
    y_offset: float,
):
    """Generate a pulse with gaussian spatial and temporal profile pulse

    Args:
        x (numpy float array): Horizontal axis
        y (numpy float array): Vertical axis
        z (numpy float array): Longitudinal axis
        w (float): waist
        delay (float): Temporal delay
        duration (float): Full Width Half Max duration
        angle (float): Angle of refraction
        x_offset (float): Shift in the horizontal direction
        y_offset (float): Shift in the vertical direction

    Returns:
        numpy float array: Output intensity
    """
    # Change duration from FWHM to stdev
    duration /= (8.0 * np.log(2)) ** 0.5

    x_transformed = np.cos(angle) * x + np.sin(angle) * z
    z_transformed = -np.sin(angle) * x + np.cos(angle) * z

    exponent = -(((z_transformed / C - delay) / duration) ** 2) / 2
    exponent += -(((x_transformed - x_offset) / w) ** 2)
    exponent += -(((y - y_offset) / w) ** 2)
    return np.exp(exponent)


x = np.linspace(-WIDTH / 2, WIDTH / 2, 200)
y = np.linspace(-HEIGHT / 2, HEIGHT / 2, 200)
z = np.linspace(0, DEPTH, 200)

X, Z = np.meshgrid(x, z)

beam_1 = gaussian_pulse(
    X,
    0.0,
    Z,
    w=2.0e-3,
    delay=500.0e-15,
    duration=250.0e-15,
    angle=10 / 180 * np.pi,
    x_offset=0.0,
    y_offset=0.0,
)
beam_2 = gaussian_pulse(
    X,
    0.0,
    Z,
    w=2.0e-3,
    delay=500.0e-15,
    duration=250.0e-15,
    angle=-10 / 180 * np.pi,
    x_offset=0.0,
    y_offset=0.0,
)

plt.pcolormesh(X, Z, beam_1 + beam_2)
plt.gca().set_aspect("equal")
plt.show()
