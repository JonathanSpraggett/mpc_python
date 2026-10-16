from __future__ import annotations

import pathlib
import signal
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer
from cvxpy_mpc import MPC
from cvxpy_mpc.utils import (
    detect_obstacle_camera,
    update_path_obstacles,
)

import yaml
from cvxpy_mpc.cvxpy_mpc_holonomic import HolonomicMPC
from mpc_python.cvxpy_mpc.path import Path
from mpc_python.cvxpy_mpc.state import HolonomicState
from mpc_python.sim import Sim
from mpc_python.sim_mujoco import SimMujoco, SharedData


# State convention used everywhere in this demo:
#   [x, y, yaw, vx, vy, yaw_rate]
# MPC input:
#   [ax, ay, yaw_acceleration]
# Low-level MuJoCo actuator target:
#   [vx_cmd, vy_cmd, yaw_rate_cmd]





def controller_loop(
    mpc: MPC,
    path: Path,
    state: HolonomicState,
    shared: SharedData,
    goal_threshold: float,
    target_speed: float,
) -> None:

    while True:
        start_time = time.time()

        # (safely) grab the latest state from the simulation
        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            state.state = shared.state.copy()  # Global [X, Y, V, Theta]
            global_obs = (
                shared.obstacle
            )  # from external detection pipeline (global frame)
        state.control = shared.mpc_accel
        elapsed = shared.mpc_elapsed

        # Check goal using absolute global coordinates
        goal_dist = path.goal_dist()
        speed = np.hypot(state.vx, state.vy)

        if goal_dist < goal_threshold and speed < 0.10:
            with shared.lock:
                shared.goal_reached = True
                shared.mpc_accel[:] = 0.0
                shared.mpc_velocity_target[:] = 0.0
            break
        # todo try putting state and path in shared then try the ros
        # Add delay compensation
        # ok why we need this in practice? the optimiser takes some time
        # to compute the next command. The mpc should compute the command for t+delay because that is
        # when it will be applied. the actual delay is the expected computation time (assumed from last one)
        state.predict_next_state(elapsed)


        # NOTE: we convert the state in ego frame and we use a ego target
        # so we the optimization problem is a bit easier and we save some solver time
        # Get reference trajectory
        path.state = state
        target = path.get_holonomic_ref_trajectory(target_speed)
        # print(self.path)


        pred_ego_state = state.ego_state()

        # Transform global obstacle to ego frame using the same pred_state
        # that the rest of the MPC ego frame is built from
        if global_obs is not None:
            gx, gy, r, vx, vy = global_obs
            dx = gx - state.x
            dy = gy - state.y
            ct, st = np.cos(-state.theta), np.sin(-state.theta)
            pred_obstacle = (
                dx * ct - dy * st,
                dy * ct + dx * st,
                r,
                vx * ct - vy * st,
                vy * ct + vx * st,
            )
        else:
            pred_obstacle = None

        x_mpc, u_mpc = mpc.solve(
            pred_ego_state, target, verbose=False, obstacle=pred_obstacle
        )


        # Extract the immediate next optimal control actions

        # accel
        velocity_target = [x_mpc[3, 1], x_mpc[4, 1], x_mpc[5, 1]]
        state.control[:] = [u_mpc[0, 0], u_mpc[1, 0], u_mpc[2, 0]]
        state.control[:] =  state.world_control(u_mpc)
        # print(control)
        # control = u_mpc[:, 0].copy()
        # vel
        # tmp = rot @ np.array([x_mpc[3, 1], x_mpc[4, 1], ])
        # control = [tmp[0], tmp[1], x_mpc[5, 1]]
        # control = [x_mpc[3, 1], x_mpc[4, 1], x_mpc[5, 1]]

        elapsed = time.time() - start_time

        # Safely push results back to the shared object
        with shared.lock:
            shared.mpc_accel[:] = state.control
            shared.mpc_velocity_target[:] = velocity_target
            shared.mpc_elapsed = elapsed
            shared.x_mpc_world = (
                state.ego_to_world_traj(x_mpc) if x_mpc is not None else None
            )

        # Enforce loop frequency
        elapsed_total = time.time() - start_time
        sleep_time = max(0.0, mpc.dt - elapsed_total)
        time.sleep(sleep_time)





class HolonomicMPCMujoco:
    def __init__(self) -> None:
        self.sim = Sim()


        self.mpc = HolonomicMPC("config/mpc_holonomic.yaml")

        # [x, y, theta, vx, vy, omega]
        self.state = HolonomicState(state_dim=6, control_dim=3, dt=self.mpc.dt, horizon_time=self.mpc.horizon_time)

        self.state.start_state(self.sim.config)


        self.path = Path(
            self.state,
            self.sim.config["path"]["waypoints_x"],
            self.sim.config["path"]["waypoints_y"],
            self.sim.config["path"]["interpolation_step"])

        # move the obstacles along the path and computes the x and y velocity
        # [x,y,r,vx,vy] just as they would come out from a tracker (e.g. ekf estimate)
        self.dynamic_obstacle_list = update_path_obstacles(self.sim.path_obstacles, self.path.path, 0.0)

        self.mj_sim = SimMujoco(self.sim)
        self.mj_sim.initialize_plant(self.state)

        self.mpc_thread = threading.Thread(
            target=controller_loop,
            args=(self.mpc, self.path, self.state, self.mj_sim.shared, self.sim.goal_threshold, self.sim.target_speed),
            daemon=True,
        )
        # handle CTRL-C for the mpc thread
        self.shutdown_flag = threading.Event()

        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        self.shutdown_flag.set()


    # here we run the sim loop
    def run(self) -> None:

        cte_hist = []
        heading_error_hist = []
        cte_rmse = -1
        heading_rmse = -1
        fps = 60.0
        render_dt = 1.0 / fps
        sim_start_time = time.perf_counter()
        self.mpc_thread.start()

        try:

            while self.mj_sim.viewer.is_running() and not self.shutdown_flag.is_set():

                if self.mj_sim.check_completion(cte_rmse, heading_rmse):
                    break

                elapsed_real_time = time.perf_counter() - sim_start_time

                self.state.state = self.mj_sim.get_state()
                # print(self.state.state)
                # External obstacle detection pipeline (global frame)
                if self.sim.path_obstacles:
                    detected_obs = detect_obstacle_camera(
                        self.dynamic_obstacle_list,
                        self.state.x,
                        self.state.y,
                        self.state.theta,
                        self.sim.sensor_max_range,
                        self.sim.sensor_fov_deg,
                    )
                else:
                    detected_obs = None

                # Sync with MPC Thread
                with self.mj_sim.shared.lock:
                    self.mj_sim.shared.state[:] = self.state.state
                    self.mj_sim.shared.obstacle = detected_obs
                    mpc_elapsed = self.mj_sim.shared.mpc_elapsed
                    mpc_accel = self.mj_sim.shared.mpc_accel.copy()
                    mpc_velocity_target = self.mj_sim.shared.mpc_velocity_target.copy()
                    x_mpc_world = self.mj_sim.shared.x_mpc_world

                # Log position etc...
                self.state.update_history()
                cte, heading_err = self.path.compute_errors()
                cte_hist.append(cte)
                heading_error_hist.append(np.degrees(heading_err))
                cte_rmse = np.sqrt(np.mean(np.square(cte_hist)))
                heading_rmse = np.sqrt(np.mean(np.square(heading_error_hist)))

                # Step physics
                while self.mj_sim.data.time < elapsed_real_time and not self.shutdown_flag.is_set():
                    # ZERO-ORDER HOLD (ZOH)
                    # The MPC outputs a target steering angle.
                    # Since the low-level actuator (aka steering wheel PID) usually
                    # tracks position directly, we treat this command as a constant step.
                    # We hold it flat (Zero-Order Hold) across the entire window.
                    # d.ctrl[0] = mpc_steer

                    # FIRST-ORDER HOLD (FOH)
                    # MuJoCo's wheel actuators expect a velocity command,
                    # but the MPC outputs an acceleration.
                    #
                    # If we applied a raw step jump to velocity (speed+=mpc_acc*DT), it would imply infinite acceleration.
                    # Instead, we integrate the acceleration command over every microscopic physics step. This smoothly ramps the
                    # velocity command over time, creating a First-Order Hold that perfectly mimics
                    # a real motor.
                    # d.ctrl[1] += mpc_accel * m.opt.timestep

                    """FOH velocity setpoints from piecewise-constant MPC acceleration.
        
                        This intentionally mirrors the car MuJoCo demo's treatment of longitudinal
                        acceleration: acceleration is held over the MPC interval and integrated at
                        the tiny MuJoCo physics timestep.  Therefore the velocity setpoint ramps
                        linearly rather than jumping once every MPC cycle.
                        """
                    self.mj_sim.apply_velocity(self.state, self.mpc, mpc_accel)

                    if self.sim.path_obstacles:
                        self.dynamic_obstacle_list[:] = update_path_obstacles(
                            self.sim.path_obstacles, self.path.path, self.mj_sim.model.opt.timestep
                        )
                    mujoco.mj_step(self.mj_sim.model, self.mj_sim.data)

                # Update camera position to follow the car
                self.mj_sim.viewer.cam.lookat[:] = [self.state.x, self.state.y, 0.0]

                # re-draw markers
                self.mj_sim.viewer.user_scn.ngeom = 0
                self.mj_sim.draw_path(self.path.path)

                if self.sim.path_obstacles:
                    self.mj_sim.draw_obstacle(self.dynamic_obstacle_list)
                    self.mj_sim.draw_sensor_fov(
                        self.state.x,
                        self.state.y,
                        self.state.theta,
                        self.sim.sensor_max_range,
                        self.sim.sensor_fov_deg,
                    )
                self.mj_sim.draw_trail( self.state.state_history["x"], self.state.state_history["y"])
                if x_mpc_world is not None:
                    self.mj_sim.draw_mpc_preview( x_mpc_world)

                # Update the HUD
                goal_dist =self.path.goal_dist()

                text = f"MPC Demo\n"\
                       f"state:   vx {self.state.vx:+.2f}  vy {self.state.vy:+.2f} m/s  "\
                    f"yaw {np.degrees(self.state.theta):+.1f} deg  w {self.state.omega:+.2f} rad/s\n"\
                    f"MPC u:   ax {mpc_accel[0]:+.2f}  ay {mpc_accel[1]:+.2f} m/s2  "\
                    f"alpha {mpc_accel[2]:+.2f} rad/s2  solve {mpc_elapsed * 1000:.0f} ms\n"\
                    f"next v*: vx {mpc_velocity_target[0]:+.2f}  vy {mpc_velocity_target[1]:+.2f}  "\
                    f"error:  CTE {cte:.3f} m  |  heading {np.degrees(heading_err):.1f} deg\n"\
                    f"RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n"\
                    f"goal:   {goal_dist:.2f} m\n"\
                    f"avoid:  {'YES' if detected_obs is not None else 'no' if self.sim.path_obstacles else 'off'}\n"
                # print(text)
                self.mj_sim.draw_text(text)
                self.mj_sim.viewer.sync()

                # Frame limiting (sleep just enough to hit 60 FPS)
                time_until_next_frame = render_dt - (
                    time.perf_counter() - elapsed_real_time - sim_start_time
                )
                if time_until_next_frame > 0:
                    time.sleep(time_until_next_frame)

        except Exception as e:
            print(e)

        finally:
            self.mj_sim.shared.is_active = False
            self.mpc_thread.join(timeout=1.0)
            if self.mpc_thread.is_alive():
                print("MPC thread is still alive...")

            self.mj_sim.viewer.clear_texts()


def do_sim() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sim = HolonomicMPCMujoco()
    try:
        sim.run()
    except Exception as e:
        sys.exit(e)

if __name__ == "__main__":
    do_sim()

