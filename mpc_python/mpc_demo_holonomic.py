#! /usr/bin/env python

from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yaml

from cvxpy_mpc.holonomic_mpc import HolonomicMPC
from cvxpy_mpc.holonomic_utils import get_holonomic_ref_trajectory
from cvxpy_mpc.utils import (
    compute_path_from_wp,
    detect_obstacle_camera,
    update_path_obstacles,
)


class HolonomicMPCSim:
    """Small no-physics demo using the repository's existing simulation.yaml."""

    def __init__(self) -> None:
        root = pathlib.Path(__file__).parent
        sim_config = yaml.safe_load((root / "config" / "simulation.yaml").read_text())
        start = sim_config["start"]

        heading = float(start["heading"])
        speed = float(start["velocity"])

        # [x, y, vx, vy, theta, omega]
        self.state: npt.NDArray[np.float64] = np.array(
            [
                float(start["x"]),
                float(start["y"]),
                speed * np.cos(heading),
                speed * np.sin(heading),
                heading,
                0.0,
            ],
            dtype=float,
        )

        self.target_speed = float(sim_config["target_speed"])
        self.sensor_max_range = float(sim_config["sensor"]["max_range"])
        self.sensor_fov_deg = float(sim_config["sensor"]["fov_deg"])
        self.goal_threshold = float(sim_config["goal_threshold"])

        self.mpc = HolonomicMPC("config/mpc_holonomic.yaml")

        self.path = compute_path_from_wp(
            sim_config["path"]["waypoints_x"],
            sim_config["path"]["waypoints_y"],
            sim_config["path"]["interpolation_step"],
        )
        self.path_obstacles = list(sim_config["obstacles"])

        self.sim_time = 0.0
        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]
        self.vx_history = [self.state[2]]
        self.vy_history = [self.state[3]]
        self.yaw_history = [self.state[4]]
        self.omega_history = [self.state[5]]
        self.ax_history = [0.0]
        self.ay_history = [0.0]
        self.alpha_history = [0.0]

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.plot(self.path[0], self.path[1], "--", label="reference path")
        (self.history_line,) = self.ax.plot([], [], label="vehicle")
        (self.preview_line,) = self.ax.plot([], [], ".-", label="MPC preview")
        (self.velocity_line,) = self.ax.plot([], [], label="velocity direction")
        self.ax.legend()
        self.fig.canvas.draw()
        plt.ion()
        plt.show()

    def run(self) -> None:
        while plt.fignum_exists(self.fig.number):
            distance_to_goal = np.hypot(
                self.state[0] - self.path[0, -1],
                self.state[1] - self.path[1, -1],
            )
            speed = np.hypot(self.state[2], self.state[3])
            if distance_to_goal < self.goal_threshold and speed < 0.1:
                print("Success: goal reached and vehicle stopped.")
                plt.ioff()
                plt.show()
                return

            obstacle = self._detect_obstacle()

            target = get_holonomic_ref_trajectory(
                self.state,
                self.path,
                self.target_speed,
                self.mpc.control_horizon * self.mpc.dt,
                self.mpc.dt,
            )

            t0 = time.perf_counter()
            x_mpc, u_mpc = self.mpc.solve(
                self.state,
                target,
                obstacle=obstacle,
            )
            solve_ms = 1000.0 * (time.perf_counter() - t0)

            # ---------------------------------------------------------------
            # THIS IS THE INTERFACE YOU WOULD SEND TO THE REAL DRONE/PX4.
            # The optimizer controls acceleration internally, but the vehicle
            # receives the resulting next-step velocity target.
            # ---------------------------------------------------------------
            velocity_command = self.mpc.velocity_command(x_mpc)
            vx_cmd, vy_cmd, yaw_rate_cmd = velocity_command

            # In this toy constant-acceleration simulation, propagate using
            # the first optimized acceleration.  A real vehicle would instead
            # track the velocity_command with its low-level controller and
            # you would replace self.state on the next loop with measured
            # odometry/VIO state.
            acceleration = u_mpc[:, 0]
            self.state = self.mpc.predict_next_state(self.state, acceleration)

            self.sim_time += self.mpc.dt
            self.x_history.append(self.state[0])
            self.y_history.append(self.state[1])
            self.vx_history.append(self.state[2])
            self.vy_history.append(self.state[3])
            self.yaw_history.append(self.state[4])
            self.omega_history.append(self.state[5])
            self.ax_history.append(acceleration[0])
            self.ay_history.append(acceleration[1])
            self.alpha_history.append(acceleration[2])

            print(
                f"t={self.sim_time:6.2f}s  "
                f"vel_cmd=[{vx_cmd:+.2f}, {vy_cmd:+.2f}, {yaw_rate_cmd:+.2f}]  "
                f"acc=[{acceleration[0]:+.2f}, {acceleration[1]:+.2f}, {acceleration[2]:+.2f}]  "
                f"solve={solve_ms:.1f} ms"
            )

            self._plot(x_mpc)

    def _detect_obstacle(self):
        if not self.path_obstacles:
            return None

        dynamic_obstacles = update_path_obstacles(
            self.path_obstacles, self.path, self.mpc.dt
        )

        # Unlike the car demo, theta is state[4].  Obstacles and MPC both stay
        # in the same world/local navigation frame, so there is no ego-frame
        # obstacle transformation here.
        return detect_obstacle_camera(
            dynamic_obstacles,
            self.state[0],
            self.state[1],
            self.state[4],
            self.sensor_max_range,
            self.sensor_fov_deg,
        )

    def _plot(self, x_mpc: npt.NDArray[np.float64]) -> None:
        self.history_line.set_data(self.x_history, self.y_history)
        self.preview_line.set_data(x_mpc[0], x_mpc[1])

        v_scale = 0.5
        self.velocity_line.set_data(
            [self.state[0], self.state[0] + v_scale * self.state[2]],
            [self.state[1], self.state[1] + v_scale * self.state[3]],
        )

        self.ax.set_title(
            "Holonomic constant-acceleration MPC\n"
            f"t={self.sim_time:.1f}s, yaw={np.degrees(self.state[4]):.1f} deg"
        )
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


if __name__ == "__main__":
    HolonomicMPCSim().run()
