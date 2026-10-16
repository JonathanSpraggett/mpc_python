#! /usr/bin/env python

from __future__ import annotations

import signal
import sys

import matplotlib.pyplot as plt
import numpy as np

from cvxpy_mpc.utils import (
    detect_obstacle_camera,

    update_path_obstacles,
)
import time
from mpc_python.cvxpy_mpc.cvxpy_mpc_holonomic import HolonomicMPC
from mpc_python.cvxpy_mpc.path import Path
from mpc_python.cvxpy_mpc.state import HolonomicState
from mpc_python.sim import Sim, SimPlot


# Classes
class HolonomicMPCSim:
    def __init__(self) -> None:

        self.sim = Sim()
        self.mpc = HolonomicMPC("config/mpc_holonomic.yaml")


        # [x, y, theta, vx, vy, omega]
        self.state = HolonomicState(state_dim=6, control_dim=3, dt = self.mpc.dt, horizon_time=self.mpc.horizon_time)

        self.state.start_state(self.sim.config)

        # Path from waypoint interpolation

        self.path = Path(
                        self.state,
                         self.sim.config["path"]["waypoints_x"],
                        self.sim.config["path"]["waypoints_y"],
                        self.sim.config["path"]["interpolation_step"])


        self.sim = Sim()
        self.plot = SimPlot(self.sim, self.path, self.mpc)





    def run(self) -> None:
        self.plot.plot_sim(self.state)
        try:
            while plt.fignum_exists(self.plot.fig.number):

                if self.plot.check_completion(self.path.goal_dist()):
                    return

                # External obstacle detection pipeline
                if self.sim.path_obstacles:
                    dynamic_obs = update_path_obstacles(
                        self.sim.path_obstacles, self.path.path, self.mpc.dt
                    )
                    self.detected_obs = detect_obstacle_camera(
                        dynamic_obs,
                        self.state.x,
                        self.state.y,
                        self.state.theta,
                self.sim.sensor_max_range,
                        self.sim.sensor_fov_deg,
                    )
                else:
                    self.sim.detected_obs = None

                # Get Reference_traj -> inputs are in worldframe

                target = self.path.get_holonomic_ref_trajectory(
                    self.sim.target_speed)


                # dynamycs w.r.t robot frame
                curr_state = self.state.ego_state()

                # Transform global obstacle to ego frame
                if self.detected_obs is not None:
                    gx, gy, r, vx, vy = self.detected_obs
                    dx = gx - self.state.x
                    dy = gy - self.state.y
                    ct, st = np.cos(-self.state.theta), np.sin(-self.state.theta)
                    obs_ego = (
                        dx * ct - dy * st,
                        dy * ct + dx * st,
                        r,
                        vx * ct - vy * st,
                        vy * ct + vx * st,
                    )
                else:
                    obs_ego = None

                t0 = time.perf_counter()
                x_mpc, u_mpc = self.mpc.solve(
                    # self.state,
                    curr_state,
                    target,
                    verbose=False,
                    obstacle=obs_ego,
                )
                self.sim.mpc_solve_time = time.perf_counter() - t0
                # only the first one is used to advance the simulation

                self.state.control[:] = self.state.world_control(u_mpc)

                # Convert MPC preview from ego->world BEFORE advancing state,
                # so it's anchored to the state it was computed for
                self.sim.optimized_trajectory = self.state.ego_to_world_traj( x_mpc)

                self.state.ode_predict_next_state()


                self.sim.time += self.mpc.dt
                self.state.update_history()
                self.plot.plot_sim(self.state)
        except KeyboardInterrupt:
            print("\nInterrupted by user (CTRL-C). Exiting...")








def do_sim() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sim = HolonomicMPCSim()
    try:
        sim.run()
    except Exception as e:
        sys.exit(e)


if __name__ == "__main__":
    do_sim()
