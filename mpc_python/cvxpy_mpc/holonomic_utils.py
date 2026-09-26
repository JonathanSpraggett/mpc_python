from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .utils import get_nn_idx


def get_holonomic_ref_trajectory(
    state: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    target_speed: float,
    T: float,
    DT: float,
) -> npt.NDArray[np.float64]:
    """Build [x, y, vx, vy, theta, omega] reference states.

    The path is expected to be the repository's normal 3xN path:
        path[0] -> x
        path[1] -> y
        path[2] -> heading/yaw

    The reference advances along path arc length at ``target_speed``.  vx/vy
    are generated from the resulting reference positions, so they remain
    consistent with the requested path.  Yaw follows the path tangent by
    default; because the vehicle is holonomic, you can later replace the yaw
    row with an independent camera/mission yaw reference without changing the
    MPC dynamics.
    """
    state = np.asarray(state, dtype=float)
    path = np.asarray(path, dtype=float)

    if state.shape != (6,):
        raise ValueError("state must be [x, y, vx, vy, theta, omega]")
    if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] < 2:
        raise ValueError("path must have shape (3, N), N >= 2")

    K = int(T / DT)
    xref = np.zeros((6, K + 1), dtype=float)

    nearest_idx = get_nn_idx(state, path)

    cumulative_distance = np.zeros(path.shape[1], dtype=float)
    cumulative_distance[1:] = np.cumsum(
        np.hypot(np.diff(path[0]), np.diff(path[1]))
    )
    total_distance = cumulative_distance[-1]
    start_distance = cumulative_distance[nearest_idx]

    requested_distance = (
        start_distance + np.arange(K + 1, dtype=float) * DT * target_speed
    )
    interp_distance = np.clip(requested_distance, 0.0, total_distance)

    xref[0] = np.interp(interp_distance, cumulative_distance, path[0])
    xref[1] = np.interp(interp_distance, cumulative_distance, path[1])
    # xref[0] = path[0, nearest_idx]  # X
    # xref[1] = path[1, nearest_idx]  # Y
    # Interpolate an unwrapped angle to avoid averaging across +/-pi.
    path_theta = np.unwrap(path[2])
    theta_ref = np.interp(interp_distance, cumulative_distance, path_theta)

    # Shift the entire yaw reference onto the 2*pi branch closest to the
    # vehicle's current yaw.  Do not wrap it again: the MPC should see a
    # continuous angle signal.
    theta_ref += 2.0 * np.pi * np.round(
        (state[4] - theta_ref[0]) / (2.0 * np.pi)
    )
    xref[4] = theta_ref

    # Generate dynamically consistent reference velocities from positions.
    xref[2, :-1] = np.diff(xref[0]) / DT
    xref[3, :-1] = np.diff(xref[1]) / DT
    xref[5, :-1] = np.diff(xref[4]) / DT

    if K > 0:
        reached_end = requested_distance[-1] >= total_distance - 1.0e-9
        if reached_end:
            xref[2, -1] = 0.0
            xref[3, -1] = 0.0
            xref[5, -1] = 0.0
        else:
            xref[2, -1] = xref[2, -2]
            xref[3, -1] = xref[3, -2]
            xref[5, -1] = xref[5, -2]

    # Once any reference node has reached the end, all later velocity/yaw-rate
    # references should be zero so the optimizer plans a stop.
    at_end = interp_distance >= total_distance - 1.0e-9
    xref[2, at_end] = 0.0
    xref[3, at_end] = 0.0
    xref[5, at_end] = 0.0

    return xref
