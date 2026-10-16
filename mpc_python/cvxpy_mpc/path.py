from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splprep, splev

from mpc_python.cvxpy_mpc.state import StateData


@dataclass
class PathStruct:
    """All tuneable gimbal parameters — sourced from ROS params at startup."""

    sim: bool = False
    rate: float = 50.0
    error_tol: float = 0.1


X_IDX = 0
Y_IDX = 1
THETA_IDX = 2
class Path:
    def __init__(self,
                 state:StateData ,
                 start_xp: list[float] | npt.NDArray[np.float64],
                start_yp: list[float] | npt.NDArray[np.float64],
                step: float = 0.1
                 ):
        self.state = state
        self.path = self.compute_path_from_wp(start_xp, start_yp, step)
        self.cdist = self.compute_path_distance()


    def __getitem__(self, index) -> Any:
        return self.path[index]

    def compute_path_distance(self):
        # Calculate cumulative distance along the path
        # todo could be in its own path class
        cdist = np.append(
            [0.0], np.cumsum(np.hypot(np.diff(self.path[X_IDX, :]), np.diff(self.path[Y_IDX, :])))
        )
        # Assumes path points have strictly increasing
        # cumulative distance (remove duplicates first).
        return np.clip(cdist, cdist[0], cdist[-1])

    def compute_path_from_wp(self,
            start_xp: list[float] | npt.NDArray[np.float64],
            start_yp: list[float] | npt.NDArray[np.float64],
            step: float = 0.1,
    ) -> npt.NDArray[np.float64]:
        """
        Generates a physically drivable, smooth C2 continuous path.

        Args:
            start_xp: X-coordinates of waypoints.
            start_yp: Y-coordinates of waypoints.
            step: Arc-length step between consecutive output points.

        Returns:
            Array of shape (3, N) with rows [x, y, heading] along the path.
        """
        # TODO could be a class, also calc curvature

        # Fit a cubic spline (B-spline) to the waypoints
        # s=0 forces the spline to pass exactly through your waypoints.
        tck, u = splprep([start_xp, start_yp], s=0.0)

        # increase resolution to calculate arc length accurately
        u_fine = np.linspace(0, 1, 2000)
        x_fine, y_fine = splev(u_fine, tck)

        # calculate arc length
        arc_lengths = np.zeros(len(u_fine))
        arc_lengths[1:] = np.cumsum(np.hypot(np.diff(x_fine), np.diff(y_fine)))
        total_len = arc_lengths[-1]

        # interpolate by step size
        num_points = int(total_len / step)
        # Arg1: uniform distance along path length by step size
        # Arg2:  Arc length along the fine resolution.
        # Use interpolation to find corresponding u from the path length
        # That converts your spline from being parameterized arbitrarily by u into something that is approximately
        # parameterized by physical distance along the path
        u_uniform = np.interp(np.linspace(0, total_len, num_points), arc_lengths, u_fine)
        final_xp, final_yp = splev(u_uniform, tck)

        dx = np.gradient(final_xp)
        dy = np.gradient(final_yp)
        theta = np.arctan2(dy, dx)

        return np.vstack((final_xp, final_yp, theta))

    def get_nn_idx(self) -> int:
        """
        Finds the index of the closest path point to the vehicle, with forward projection.

        Args:
            state: Vehicle state [x, y, ...]. Only the first two elements are used.
            path: Array of shape (2, N) of [x, y] path points.

        Returns:
            Index of the closest or next-forward path point.
        """
        dx = self.state.x - self.path[X_IDX, :]
        dy = self.state.y - self.path[Y_IDX, :]
        dist = np.hypot(dx, dy)
        nn_idx = np.argmin(dist)
        try:
            v = np.array(
                [
                    self.path[X_IDX, nn_idx + 1] - self.path[X_IDX, nn_idx],
                    self.path[Y_IDX, nn_idx + 1] - self.path[Y_IDX, nn_idx],
                ]
            )
            assert np.linalg.norm(v) > 0, "zero-length path segment"
            v /= np.linalg.norm(v)
            d = [self.path[X_IDX, nn_idx] - self.state.x, self.path[Y_IDX, nn_idx] - self.state.y]
            if np.dot(d, v) > 0:
                target_idx = nn_idx
            else:
                target_idx = nn_idx + 1
        except IndexError:
            target_idx = nn_idx
        return target_idx

    def get_holonomic_ref_trajectory( self,
            target_v: float,
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
        K = self.state.control_horizon # length of control horizon

        # Allocate K + 1 elements to map exactly from k=0 (
        # ) to k=K (terminal)
        xref = np.zeros((6, K + 1))
        ind = self.get_nn_idx()


        total_dist = self.cdist[-1]

        start_dist = self.cdist[ind]

        # range is (0, K + 1) to include the t=0 starting node
        # using target velocity to look ahead of the path and find where the vehicle will be along the path
        # for the control horizon
        # the desired path indexes at the predicted path distance
        # todo replace with a speed profile or with upcoming curvature modifiers
        requested_distance = (
                start_dist + np.arange(K + 1, dtype=float) * self.state.dt * target_v
        )
        # print(self.state.dt)
        interp_distance = np.clip(requested_distance, 0.0, total_dist)

        # Compute interpolation (automatically maps across all K + 1 points)
        # Based on the prediction of the control horizon, we interpolate to get accurate and evenly spaced
        # x,y, theta predictions along the control horizon
        # ie convert path distance to xy
        xref[self.state.X_IDX, :] = np.interp(interp_distance, self.cdist, self.path[X_IDX, :])  # x
        xref[self.state.Y_IDX, :] = np.interp(interp_distance, self.cdist, self.path[Y_IDX, :])  # y

        # Interpolate an unwrapped angle to avoid averaging across +/-pi.
        path_theta = np.unwrap(self.path[THETA_IDX])
        theta_ref = np.interp(interp_distance, self.cdist, path_theta)

        # Shift the entire yaw reference onto the 2*pi branch closest to the
        # vehicle's current yaw.  Do not wrap it again: the MPC should see a
        # continuous angle signal.
        theta_ref += 2.0 * np.pi * np.round(
            (self.state.theta - theta_ref[0]) / (2.0 * np.pi)
        )
        xref[self.state.THETA_IDX] = theta_ref  # theta

        # Generate dynamically consistent reference velocities from positions.
        xref[self.state.VX_IDX, :-1] = np.diff(xref[self.state.X_IDX]) / self.state.dt  # vx
        xref[self.state.VY_IDX, :-1] = np.diff(xref[self.state.Y_IDX]) / self.state.dt  # vy
        xref[self.state.OMEGA_IDX, :-1] = np.diff(xref[self.state.THETA_IDX]) / self.state.dt  # vtheta

        reached_end = requested_distance[-1] >= total_dist - 1.0e-9
        if reached_end:
            xref[self.state.VX_IDX, -1] = 0.0
            xref[self.state.VY_IDX, -1] = 0.0
            xref[self.state.OMEGA_IDX, -1] = 0.0
        else:
            xref[self.state.VX_IDX, -1] = xref[self.state.VX_IDX, -2]
            xref[self.state.VY_IDX, -1] = xref[self.state.VY_IDX, -2]
            xref[self.state.OMEGA_IDX, -1] = xref[self.state.OMEGA_IDX, -2]

        # Once any reference node has reached the end, all later velocity/yaw-rate
        # references should be zero so the optimizer plans a stop.
        at_end = interp_distance >= total_dist - 1.0e-9
        xref[self.state.VX_IDX, at_end] = 0.0
        xref[self.state.VY_IDX, at_end] = 0.0
        xref[self.state.OMEGA_IDX, at_end] = 0.0

        if ego_frame:
            # print(self.state.theta)
            vec = (xref[self.state.X_IDX:self.state.THETA_IDX, :] - self.state[self.state.X_IDX:self.state.THETA_IDX, None])
            xref[self.state.X_IDX:self.state.THETA_IDX, :] = self.state.world_to_ego(vec)  #rot.T @ (xref[0:2, :] - state[0:2, None])

            xref[self.state.VX_IDX:self.state.OMEGA_IDX, :] =  self.state.world_to_ego(xref[self.state.VX_IDX:self.state.OMEGA_IDX, :]) # rot.T @ xref[3:5, :]

            xref[self.state.THETA_IDX, :] -= self.state.theta  # Local Theta
            # xref[5, :] -= self.state.alpha
            # Angular velocity remains unchanged and vehicle theta is constant through each step meaning todo why

        return xref

    def compute_errors( self,) -> tuple[float, float]:
        """Compute signed cross-track error and heading error.

        Signed_cte is positive if vehicle is to the left.
        """
        # Find the closest waypoint index
        dx = self.state.x - self.path[X_IDX, :]
        dy = self.state.y - self.path[Y_IDX, :]
        distances = np.hypot(dx, dy)
        idx = np.argmin(distances)

        # Determine segment direction for true cross-track projection
        # If we are at the very last point, look backward, otherwise look forward
        if idx == self.path.shape[Y_IDX] - 1:
            idx_start = idx - 1
            idx_end = idx
        else:
            idx_start = idx
            idx_end = idx + 1
        # Calculate forward-facing tangent vector
        tx = self.path[X_IDX, idx_end] - self.path[X_IDX, idx_start]
        ty = self.path[Y_IDX, idx_end] - self.path[Y_IDX, idx_start]
        seg_len = np.hypot(tx, ty)

        if seg_len > 1e-5:
            # Normalize tangent vector
            tx /= seg_len
            ty /= seg_len

            # Vector from waypoint to vehicle
            vx = self.state.x - self.path[X_IDX, idx]
            vy = self.state.y - self.path[Y_IDX, idx]

            # True Cross-Track Error is the perpendicular scalar projection (Using 2D Cross Product)
            cte = (vy * tx) - (vx * ty)
        else:
            cte = distances[idx]

        # Heading Error (Normalized between -pi and pi)
        target_heading = self.path[THETA_IDX, idx_start]
        heading_err = (self.state.theta - target_heading + np.pi) % (2.0 * np.pi) - np.pi

        return (cte, heading_err)

    def goal_dist(self):
        return np.sqrt(
                        (self.state.x - self.path[X_IDX, -1]) ** 2
                        + (self.state.y - self.path[Y_IDX, -1]) ** 2
                    )