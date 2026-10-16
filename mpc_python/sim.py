import math
import pathlib

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge

from mpc_python.cvxpy_mpc.path import Path
from mpc_python.cvxpy_mpc.utils import update_path_obstacles
import numpy.typing as npt


class Sim:
    def __init__(self) -> None:
        self.config = yaml.safe_load(
            (pathlib.Path(__file__).parent / "config" / "simulation.yaml").read_text()
        )

        self.target_speed = self.config["target_speed"]
        self.sensor_max_range = self.config["sensor"]["max_range"]
        self.sensor_fov_deg = self.config["sensor"]["fov_deg"]
        self.goal_threshold = self.config["goal_threshold"]
        
        self.path_obstacles = list(self.config["obstacles"])
        self.detected_obs: tuple[float, float, float] | None = None
        self.optimized_trajectory: npt.NDArray[np.float64] | None = None

        self.mpc_solve_time: float = 0.0

        
        # Helper variables to keep track of the sim
        self.time: float = 0.0
        

class SimPlot:
    def __init__(self, sim:Sim, path: Path, mpc) -> None:
        self.sim = sim
        self.path = path
        self.mpc = mpc

        # Persistent plot (no clf flickering)
        plt.style.use("ggplot")
        self.fig: plt.Figure = plt.figure()
        gs = plt.GridSpec(3, 3)

        self.ax_main = plt.subplot(gs[0:3, 0:2])
        self.ax_main.set_xlabel("map x")
        self.ax_main.set_ylabel("map y")
        self.ax_main.set_aspect("equal")
        self.ax_main.plot(
            self.path[0, :],
            self.path[1, :],
            c="tab:orange",
            marker=".",
            label="reference track",
        )
        x_pad, y_pad = 1.0, 1.0
        self.ax_main.set_xlim(
            self.path[0, :].min() - x_pad, self.path[0, :].max() + x_pad
        )
        self.ax_main.set_ylim(
            self.path[1, :].min() - y_pad, self.path[1, :].max() + y_pad
        )

        # Obstacle visualization
        self.obs_circles: list[plt.Circle] = []
        
        if self.sim.path_obstacles:
            initial_obs = update_path_obstacles(self.sim.path_obstacles, self.path.path, 0.0)
            for ox, oy, rad, _, _ in initial_obs:
                c = plt.Circle(
                    (ox, oy),
                    rad,
                    color="red",
                    alpha=0.4,
                    label="obstacle",
                )
                self.ax_main.add_patch(c)
                self.obs_circles.append(c)

            # Sensor FOV wedge
            self.fov_patch = Wedge(
                (0, 0),
                self.sim.sensor_max_range,
                0,
                0,
                color="gold",
                alpha=0.15,
            )
            self.ax_main.add_patch(self.fov_patch)

        (self.traj_line,) = self.ax_main.plot(
            [],
            [],
            c="tab:blue",
            marker=".",
            alpha=0.5,
            label="vehicle trajectory",
        )
        (self.mpc_line,) = self.ax_main.plot(
            [],
            [],
            c="tab:green",
            marker="+",
            alpha=0.5,
            label="mpc opt trajectory",
        )
        self.mpc_line.set_visible(False)

        self.car_line: plt.Line2D | None = None

        # HUD overlay
        self.hud = self.ax_main.text(
            0.02,
            0.98,
            "",
            transform=self.ax_main.transAxes,
            va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Acceleration subplot
        self.ax_accel = plt.subplot(gs[0, 2])
        self.ax_accel.set_ylabel("a(t) [m/ss]")
        self.ax_accel.set_xlabel("t [s]")
        self.ax_accel.axhline(y=self.mpc.max_ax, c="gray", ls="--", lw=0.8)
        self.ax_accel.axhline(y=-self.mpc.max_ax, c="gray", ls="--", lw=0.8)
        self.ax_accel.set_ylim(-self.mpc.max_ax * 1.5, self.mpc.max_ax * 1.5)
        (self.accel_line,) = self.ax_accel.plot([], [], c="tab:orange")

        # Steering subplot
        self.ax_steer = plt.subplot(gs[1, 2])
        self.ax_steer.set_ylabel("yaw(t) [deg/s]")
        self.ax_steer.set_xlabel("t [s]")
        max_steer_deg = np.degrees(self.mpc.max_yaw_rate)
        self.ax_steer.axhline(y=max_steer_deg, c="gray", ls="--", lw=0.8)
        self.ax_steer.axhline(y=-max_steer_deg, c="gray", ls="--", lw=0.8)
        self.ax_steer.set_ylim(-max_steer_deg * 1.5, max_steer_deg * 1.5)
        (self.steer_line,) = self.ax_steer.plot([], [], c="tab:orange")

        # Velocity subplot
        self.ax_vel = plt.subplot(gs[2, 2])
        self.ax_vel.set_ylabel("v(t) [m/s]")
        self.ax_vel.set_xlabel("t [s]")
        self.ax_vel.axhline(y=self.sim.target_speed, c="tab:orange", ls="--", label="target speed")
        self.ax_vel.set_ylim(-self.mpc.max_vx * 1.2, self.mpc.max_vx * 1.2)
        (self.vel_line,) = self.ax_vel.plot([], [], c="tab:blue", label="vehicle speed")
        self.ax_vel.legend(loc="lower right")

        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def plot_sim(self, state) -> None:
        # Title
        self.ax_main.set_title(
            f"MPC Simulation\nSimulation elapsed time {self.sim.time:.1f}s"
        )

        # Trajectory history
        self.traj_line.set_data(state.state_history["x"], state.state_history["y"])

        # MPC preview
        if self.sim.optimized_trajectory is not None:
            self.mpc_line.set_data(
                self.sim.optimized_trajectory[0, :], self.sim.optimized_trajectory[1, :]
            )
            self.mpc_line.set_visible(True)
        else:
            self.mpc_line.set_visible(False)

        if self.car_line is not None:
            self.car_line.remove()
        self.car_line = self.plot_car(
            self.ax_main, state.state_history["x"][-1], state.state_history["y"][-1], state.state_history["theta"][-1]
        )

        if self.sim.path_obstacles:
            current_obs = update_path_obstacles(self.sim.path_obstacles, self.path.path, 0.0)
            for i, (ox, oy, _, _, _) in enumerate(current_obs):
                self.obs_circles[i].set_center((ox, oy))

            # Sensor FOV wedge
            half_fov = self.sim.sensor_fov_deg / 2
            theta1 = np.degrees(state.state_history["theta"][-1]) - half_fov
            theta2 = np.degrees(state.state_history["theta"][-1]) + half_fov
            self.fov_patch.set_center((state.state_history["x"][-1], state.state_history["y"][-1]))
            self.fov_patch.set_theta1(theta1)
            self.fov_patch.set_theta2(theta2)

        # HUD
        goal_dist = self.path.goal_dist()
        avoiding = (
            "YES"
            if self.sim.detected_obs is not None
            else "no" if self.sim.path_obstacles else "off"
        )
        self.hud.set_text(
            f"v: {math.sqrt(state.vx**2 + state.vy**2) :.2f} m/s  |  goal: {goal_dist:.2f} m  |  avoid: {avoiding}  |  MPC: {self.sim.mpc_solve_time*1000:.0f} ms"
        )

        # Subplot data: plot against time
        t = np.arange(len(state.control_history["ax"])) * self.mpc.dt
        self.accel_line.set_data(t, state.control_history["ax"])
        self.accel_line.set_data(t, state.control_history["ay"])
        self.ax_accel.relim()
        self.ax_accel.autoscale_view(scalex=True, scaley=False)

        self.steer_line.set_data(t, np.degrees(state.control_history["alpha"]))
        self.ax_steer.relim()
        self.ax_steer.autoscale_view(scalex=True, scaley=False)

        self.vel_line.set_data(t, state.state_history["vx"])
        # self.vel_line.set_data(t, state.state_history["vx"])
        self.ax_vel.relim()
        self.ax_vel.autoscale_view(scalex=True, scaley=False)

        plt.draw()
        plt.pause(0.001)

    def check_completion(self,goal_dist):
        # Check for completion
        if (
                goal_dist
                < self.sim.goal_threshold
        ):
            print(
                "Success! Goal Reached\nClose the plot window or press CTRL-C to exit."
            )
            while plt.fignum_exists(self.fig.number):
                plt.pause(0.1)
            return True
        return False

    def plot_car(self, ax: plt.Axes, x: float, y: float, yaw: float) -> plt.Line2D:
        CAR_LENGTH = 0.5
        CAR_WIDTH = 0.25
        CAR_OFFSET = CAR_LENGTH

        outline = np.array(
            [
                [
                    -CAR_OFFSET,
                    CAR_LENGTH - CAR_OFFSET,
                    CAR_LENGTH - CAR_OFFSET,
                    -CAR_OFFSET,
                    -CAR_OFFSET,
                ],
                [
                    CAR_WIDTH / 2,
                    CAR_WIDTH / 2,
                    -CAR_WIDTH / 2,
                    -CAR_WIDTH / 2,
                    CAR_WIDTH / 2,
                ],
            ]
        )

        Rotm = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        outline = (outline.T @ Rotm).T
        outline[0, :] += x
        outline[1, :] += y

        return ax.plot(outline[0, :].flatten(), outline[1, :].flatten(), "tab:blue")[0]