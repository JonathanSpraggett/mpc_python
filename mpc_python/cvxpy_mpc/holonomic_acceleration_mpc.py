"""Acceleration-output interface for the existing holonomic MPC."""
from __future__ import annotations

import numpy as np

from .holonomic_mpc import HolonomicMPC


class HolonomicAccelerationMPC(HolonomicMPC):
    """State [x, y, vx, vy, yaw, yaw_rate]; output [ax, ay, alpha].

    Reuses the exact double-integrator dynamics, obstacle constraints, velocity,
    acceleration and jerk bounds of HolonomicMPC. Commands are world-frame
    accelerations and are held constant until the next solve.
    """

    def acceleration_command(self, controls: np.ndarray) -> np.ndarray:
        """Extract and bound the first acceleration from solve()'s control horizon."""
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[0] != 3 or controls.shape[1] < 1:
            raise ValueError("controls must have shape (3, N), N >= 1")
        command = controls[:, 0]
        if not np.all(np.isfinite(command)):
            raise ValueError("acceleration command must be finite")
        limits = np.array([self.max_ax, self.max_ay, self.max_yaw_accel])
        return np.clip(command, -limits, limits)

