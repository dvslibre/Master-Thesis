"""Sampling and camera utility functions for pieNeRF (sphere sampling, look-at matrices, vector helpers)."""

import numpy as np


def to_sphere(u, v):
    """NICHT GENUTZT; Definiert geleichverteilte (u,v) ([0,1]^2) in 3D-Punktkoordinaten auf Einheitskugel."""
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz])


def sample_on_sphere(range_u=(0.0, 1.0), range_v=(0.0, 1.0)):
    """NICHT GENUTZT; Zufälliger Punkt auf der Einheitskugel innerhalb der gegebenen Parameterbereiche."""
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    """Erzeuge eine Kamerarotationsmatrix, die von eye nach at blickt."""
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), eps)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), eps)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), eps)

    r_mat = np.concatenate(
        (
            x_axis.reshape(-1, 3, 1),
            y_axis.reshape(-1, 3, 1),
            z_axis.reshape(-1, 3, 1),
        ),
        axis=2,
    )
    return r_mat
