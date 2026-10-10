"""Separable 3D holonomic acceleration MPC, with independent altitude control."""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from .holonomic_acceleration_mpc import HolonomicAccelerationMPC
from .holonomic_utils import get_holonomic_ref_trajectory
from .utils import compute_path_from_wp


def build_drone_path(config: dict) -> np.ndarray:
    """Return [x, y, z, yaw] samples, with all waypoints at the configured altitude."""
    route = config["path"]
    planar = compute_path_from_wp(
        route["waypoints_x"], route["waypoints_y"], route["interpolation_step"]
    )
    return np.vstack((planar[:2], np.full(planar.shape[1], route["altitude"]), planar[2]))


class Drone3DAccelerationMPC:
    """State [x,y,z,vx,vy,vz,yaw,omega]; input [ax,ay,az,alpha].

    XY/yaw and altitude costs and constraints are separable, so two small
    convex problems give the same solution as the combined block-diagonal QP.
    Existing obstacle half-planes remain XY constraints (vertical columns).
    """

    PLANAR = [0, 1, 3, 4, 6, 7]

    def __init__(self, config, *, max_vz=1.0, max_az=1.5, max_jerk_z=3.0):
        self.planar = HolonomicAccelerationMPC(config)
        self.dt = self.planar.dt
        self.control_horizon = self.planar.control_horizon
        self.max_vz, self.max_az, self.max_jerk_z = max_vz, max_az, max_jerk_z
        self.Az = np.array([[1., self.dt], [0., 1.]])
        self.Bz = np.array([0.5 * self.dt**2, self.dt])
        n = self.control_horizon
        self._z = cp.Variable((2, n + 1))
        self._az = cp.Variable(n)
        self._z0 = cp.Parameter(2)
        self._ref = cp.Parameter((2, n + 1))
        self._last = cp.Parameter()
        self._previous_az = 0.0
        constraints = [self._z[:, 0] == self._z0,
                       cp.abs(self._z[1, 1:]) <= max_vz,
                       cp.abs(self._az) <= max_az]
        cost = 0
        for k in range(n):
            delta = self._az[k] - (self._last if k == 0 else self._az[k-1])
            constraints += [self._z[:, k+1] == self.Az @ self._z[:, k] + self.Bz * self._az[k],
                            cp.abs(delta) <= max_jerk_z * self.dt]
            cost += cp.quad_form(self._z[:, k] - self._ref[:, k], np.diag([60., 10.]))
            cost += cp.square(self._az[k]) + 5 * cp.square(delta)
        cost += cp.quad_form(self._z[:, -1] - self._ref[:, -1], np.diag([120., 15.]))
        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    def reference(self, state, path, speed):
        planar_ref = get_holonomic_ref_trajectory(
            np.asarray(state)[self.PLANAR], path[[0, 1, 3]], speed,
            self.control_horizon * self.dt, self.dt)
        reference = np.zeros((8, self.control_horizon + 1))
        reference[self.PLANAR] = planar_ref
        reference[2] = path[2, 0]
        return reference

    def solve(self, state, reference, *, obstacle=None):
        state, reference = np.asarray(state, dtype=float), np.asarray(reference, dtype=float)
        if state.shape != (8,) or reference.shape != (8, self.control_horizon + 1):
            raise ValueError("Expected state (8,) and reference (8, horizon+1)")
        xy_states, xy_controls = self.planar.solve(
            state[self.PLANAR], reference[self.PLANAR], obstacle=obstacle)
        self._z0.value = state[[2, 5]]
        self._ref.value = reference[[2, 5]]
        self._last.value = self._previous_az
        self._problem.solve(solver=cp.CLARABEL, warm_start=True)
        if self._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            zs, az = self._z.value, self._az.value
        else:
            # Bounded braking when the altitude problem is infeasible.
            zs = np.zeros((2, self.control_horizon + 1))
            az = np.zeros(self.control_horizon)
            zs[:, 0] = state[[2, 5]]
            for k in range(self.control_horizon):
                az[k] = np.clip(-zs[1, k] / self.dt, -self.max_az, self.max_az)
                zs[:, k+1] = self.Az @ zs[:, k] + self.Bz * az[k]
        self._previous_az = float(az[0])
        states = np.zeros((8, self.control_horizon + 1))
        controls = np.zeros((4, self.control_horizon))
        states[self.PLANAR] = xy_states
        states[[2, 5]] = zs
        controls[[0, 1, 3]] = xy_controls
        controls[2] = az
        return states, controls

    def acceleration_command(self, controls):
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[0] != 4 or controls.shape[1] < 1:
            raise ValueError("controls must have shape (4, N), N >= 1")
        if not np.isfinite(controls[:, 0]).all():
            raise ValueError("Acceleration command must be finite")
        command = np.zeros(4)
        command[[0, 1, 3]] = self.planar.acceleration_command(controls[[0, 1, 3]])
        command[2] = np.clip(controls[2, 0], -self.max_az, self.max_az)
        return command

    @staticmethod
    def predict(state, acceleration, dt):
        result = np.asarray(state, dtype=float).copy()
        acceleration = np.asarray(acceleration, dtype=float)
        result[[0, 1, 2, 6]] += result[[3, 4, 5, 7]] * dt + 0.5 * acceleration * dt**2
        result[[3, 4, 5, 7]] += acceleration * dt
        return result
