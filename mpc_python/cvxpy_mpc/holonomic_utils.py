from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .utils import get_nn_idx


def get_holonomic_ref_trajectory2(
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


def get_holonomic_ref_trajectory(
    state: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    target_v: float,
    T: float,
    DT: float,
    ego_frame: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Builds a reference trajectory from the path for the MPC horizon.

    Args:
        state: Vehicle state in global frame [x, y, theta, vx, vy, omega].
        path: Array of shape (3, N) with rows [x, y, heading] in global frame.
        target_v: Desired forward speed.
        T: Control horizon duration.
        DT: Control horizon time-step.
        ego_frame: If True, returns trajectory in the vehicle's ego frame.
            If False, returns in the global world frame.

    Returns:
        Array of shape (6, K+1) with rows [x, y, theta, vx, vy, omega] representing
        the reference trajectory over the horizon.
    """
    K = int(T / DT) # length of control horizon
    if K < 1 or DT <= 0 or target_v < 0:
        raise ValueError("Invalid horizon, DT, or speed")
    # Allocate K + 1 elements to map exactly from k=0 (
    # ) to k=K (terminal)
    xref = np.zeros((6, K + 1))
    ind = get_nn_idx(state, path)

    # 1. Calculate cumulative distance along path
    # todo could be in its own path class
    cdist = np.append(
        [0.0], np.cumsum(np.hypot(np.diff(path[0, :]), np.diff(path[1, :])))
    )
    # Assumes path points have strictly increasing
    # cumulative distance (remove duplicates first).
    cdist = np.clip(cdist, cdist[0], cdist[-1])

    total_dist = cdist[-1]

    start_dist = cdist[ind]

    # range is (0, K + 1) to include the t=0 starting node
    # using target velocity to look ahead of the path and find where the vehicle will be along the path
    # for the control horizon
    # the desired path indexes at the predicted path distance
    # todo replace with a speed profile or with upcoming curvature modifiers
    requested_distance = (
            start_dist + np.arange(K + 1, dtype=float) * DT * target_v
    )
    interp_distance = np.clip(requested_distance, 0.0, total_dist)

    # Compute interpolation (automatically maps across all K + 1 points)
    # Based on the prediction of the control horizon, we interpolate to get accurate and evenly spaced
    # x,y, theta predictions along the control horizon
    # ie convert path distance to xy
    xref[0, :] = np.interp(interp_distance, cdist, path[0, :]) # x
    xref[1, :] = np.interp(interp_distance, cdist, path[1, :]) # y

    # Interpolate an unwrapped angle to avoid averaging across +/-pi.
    path_theta = np.unwrap(path[2])
    theta_ref = np.interp(interp_distance, cdist, path_theta)

    # Shift the entire yaw reference onto the 2*pi branch closest to the
    # vehicle's current yaw.  Do not wrap it again: the MPC should see a
    # continuous angle signal.
    theta_ref += 2.0 * np.pi * np.round(
        (state[2] - theta_ref[0]) / (2.0 * np.pi)
    )
    xref[2] = theta_ref # theta

    # Generate dynamically consistent reference velocities from positions.
    xref[3, :-1] = np.diff(xref[0]) / DT # vx
    xref[4, :-1] = np.diff(xref[1]) / DT # vy
    xref[5, :-1] = np.diff(xref[2]) / DT # vtheta

    reached_end = requested_distance[-1] >= total_dist - 1.0e-9
    if reached_end:
        xref[3, -1] = 0.0
        xref[4, -1] = 0.0
        xref[5, -1] = 0.0
    else:
        xref[3, -1] = xref[3, -2]
        xref[4, -1] = xref[4, -2]
        xref[5, -1] = xref[5, -2]



    # Once any reference node has reached the end, all later velocity/yaw-rate
    # references should be zero so the optimizer plans a stop.
    at_end = interp_distance >= total_dist - 1.0e-9
    xref[3, at_end] = 0.0
    xref[4, at_end] = 0.0
    xref[5, at_end] = 0.0

    if ego_frame:
        theta = state[2]
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        xref[0:2, :] = rot.T @ (xref[0:2, :] - state[0:2, None])

        xref[3:5, :] = rot.T @ xref[3:5, :]

        xref[2, :] -=  state[2]  # Local Theta
        # xref[5, :] -= state[5]
        # Angular velocity remains unchanged and vehicle theta is constant through each step meaning todo why

    return xref




def ego_to_global(
    state: npt.NDArray[np.float64], x_mpc: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Transforms MPC trajectory from ego frame to global frame.

    Args:
        state: Vehicle state [x, y, v, heading] in global frame.
        x_mpc: MPC solution trajectory in ego frame (2, N).

    Returns:
        Trajectory in global frame (2, N).
    """
    traj = x_mpc[:2, :].copy()
    ct, st = np.cos(state[2]), np.sin(state[2])
    R = np.array([[ct, -st], [st, ct]])
    traj = R @ traj
    traj[0, :] += state[0]
    traj[1, :] += state[1]
    return traj

def compute_errors(
    current_state: npt.NDArray[np.float64], path: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Compute signed cross-track error and heading error.

    Signed_cte is positive if vehicle is to the left.
    """
    assert path.shape[1] >= 2, "path must have at least 2 points"
    # Find the closest waypoint index
    dx = current_state[0] - path[0, :]
    dy = current_state[1] - path[1, :]
    distances = np.hypot(dx, dy)
    idx = np.argmin(distances)

    # Determine segment direction for true cross-track projection
    # If we are at the very last point, look backward, otherwise look forward
    if idx == path.shape[1] - 1:
        idx_start = idx - 1
        idx_end = idx
    else:
        idx_start = idx
        idx_end = idx + 1
    # Calculate forward-facing tangent vector
    tx = path[0, idx_end] - path[0, idx_start]
    ty = path[1, idx_end] - path[1, idx_start]
    seg_len = np.hypot(tx, ty)

    if seg_len > 1e-5:
        # Normalize tangent vector
        tx /= seg_len
        ty /= seg_len

        # Vector from waypoint to vehicle
        vx = current_state[0] - path[0, idx]
        vy = current_state[1] - path[1, idx]

        # True Cross-Track Error is the perpendicular scalar projection (Using 2D Cross Product)
        cte = (vy * tx) - (vx * ty)
    else:
        cte = distances[idx]

    # Heading Error (Normalized between -pi and pi)
    target_heading = path[2, idx_start]
    heading_err = (current_state[2] - target_heading + np.pi) % (2.0 * np.pi) - np.pi

    return (cte, heading_err)
