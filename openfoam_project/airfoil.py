from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NACA4Spec:
    max_camber: int
    camber_position: int = 4
    thickness: int = 12

    @property
    def digits(self) -> str:
        return f"{self.max_camber}{self.camber_position}{self.thickness:02d}"


def generate_naca4_airfoil(spec: NACA4Spec, chord: float = 1.0, points: int = 100) -> np.ndarray:
    """Generate a closed 2D airfoil loop ordered TE -> LE -> TE."""
    if points < 3:
        raise ValueError("points must be at least 3")

    m = spec.max_camber / 100.0
    p = spec.camber_position / 10.0
    t = spec.thickness / 100.0

    beta = np.linspace(0.0, np.pi, points)
    x = (1.0 - np.cos(beta)) / 2.0
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if p > 0:
        front = x <= p
        yc[front] = (m / p**2) * (2.0 * p * x[front] - x[front] ** 2)
        dyc_dx[front] = (2.0 * m / p**2) * (p - x[front])

        back = x > p
        yc[back] = (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x[back] - x[back] ** 2)
        dyc_dx[back] = (2.0 * m / (1.0 - p) ** 2) * (p - x[back])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack((x_coords, y_coords)) * chord
