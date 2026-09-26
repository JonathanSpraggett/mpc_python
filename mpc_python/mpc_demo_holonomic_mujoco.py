#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import signal
import threading
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import yaml

from cvxpy_mpc.holonomic_mpc import HolonomicMPC
from cvxpy_mpc.holonomic_utils import get_holonomic_ref_trajectory
from cvxpy_mpc.utils import (
    compute_errors,
    compute_path_from_wp,
    detect_obstacle_camera,
    update_path_obstacles,
)


# State convention used everywhere in this demo:
#   [x, y, vx, vy, yaw, yaw_rate]
# MPC input:
#   [ax, ay, yaw_acceleration]
# Low-level MuJoCo actuator target:
#   [vx_cmd, vy_cmd, yaw_rate_cmd]


@dataclass(frozen=True)
class PlanarPlantIds:
    x_qpos: int
    y_qpos: int
    yaw_qpos: int
    x_dof: int
    y_dof: int
    yaw_dof: int
    vx_actuator: int
    vy_actuator: int
    yaw_rate_actuator: int
    body: int


class SharedData:
    """Thread-safe interface between the MPC and MuJoCo physics loops."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

        self.state: npt.NDArray[np.float64] = np.zeros(6, dtype=float)
        self.mpc_accel: npt.NDArray[np.float64] = np.zeros(3, dtype=float)
        self.mpc_velocity_target: npt.NDArray[np.float64] = np.zeros(3, dtype=float)
        self.x_mpc_world: npt.NDArray[np.float64] | None = None
        self.obstacle: tuple[float, float, float, float, float] | None = None
        self.mpc_elapsed: float = 0.0

        self.goal_reached = False
        self.is_active = True


def _require_named_id(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    name: str,
) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx == -1:
        raise ValueError(f"MuJoCo object '{name}' was not found")
    return idx


def get_planar_plant_ids(model: mujoco.MjModel) -> PlanarPlantIds:
    x_joint = _require_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "x_joint")
    y_joint = _require_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "y_joint")
    yaw_joint = _require_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw_joint")

    return PlanarPlantIds(
        x_qpos=int(model.jnt_qposadr[x_joint]),
        y_qpos=int(model.jnt_qposadr[y_joint]),
        yaw_qpos=int(model.jnt_qposadr[yaw_joint]),
        x_dof=int(model.jnt_dofadr[x_joint]),
        y_dof=int(model.jnt_dofadr[y_joint]),
        yaw_dof=int(model.jnt_dofadr[yaw_joint]),
        vx_actuator=_require_named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vx_servo"
        ),
        vy_actuator=_require_named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "vy_servo"
        ),
        yaw_rate_actuator=_require_named_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_rate_servo"
        ),
        body=_require_named_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "holonomic_drone"
        ),
    )


def get_state(
    data: mujoco.MjData,
    ids: PlanarPlantIds,
) -> npt.NDArray[np.float64]:
    """Read [x, y, vx, vy, yaw, yaw_rate] directly from planar joints."""
    return np.array(
        [
            data.qpos[ids.x_qpos],
            data.qpos[ids.y_qpos],
            data.qvel[ids.x_dof],
            data.qvel[ids.y_dof],
            data.qpos[ids.yaw_qpos],
            data.qvel[ids.yaw_dof],
        ],
        dtype=float,
    )


def predict_with_constant_acceleration(
    state: npt.NDArray[np.float64],
    acceleration: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Exact constant-acceleration propagation for an arbitrary dt."""
    predicted = state.copy()
    half_dt2 = 0.5 * dt * dt

    predicted[0] += state[2] * dt + acceleration[0] * half_dt2
    predicted[1] += state[3] * dt + acceleration[1] * half_dt2
    predicted[2] += acceleration[0] * dt
    predicted[3] += acceleration[1] * dt
    predicted[4] += state[5] * dt + acceleration[2] * half_dt2
    predicted[5] += acceleration[2] * dt
    return predicted


def controller_loop(
    mpc: HolonomicMPC,
    path: npt.NDArray[np.float64],
    shared: SharedData,
    goal_threshold: float,
    target_speed: float,
) -> None:
    """MPC loop running independently from the high-rate MuJoCo loop."""

    while True:
        loop_start = time.perf_counter()

        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            current_state = shared.state.copy()
            global_obstacle = shared.obstacle
            previous_accel = shared.mpc_accel.copy()
            previous_solve_time = shared.mpc_elapsed

        goal_distance = np.hypot(
            current_state[0] - path[0, -1],
            current_state[1] - path[1, -1],
        )
        speed = np.hypot(current_state[2], current_state[3])

        # Require both position convergence and nearly zero translational speed.
        if goal_distance < goal_threshold and speed < 0.10:
            with shared.lock:
                shared.goal_reached = True
                shared.mpc_accel[:] = 0.0
                shared.mpc_velocity_target[:] = 0.0
            break

        # Same idea as the repository's car MuJoCo demo: compensate for the
        # previous optimizer runtime because the command computed now will be
        # applied slightly after the state measurement was taken.
        pred_state = predict_with_constant_acceleration(
            current_state,
            previous_accel,
            previous_solve_time,
        )

        target = get_holonomic_ref_trajectory(
            pred_state,
            path,
            target_speed,
            mpc.control_horizon * mpc.dt,
            mpc.dt,
        )

        x_mpc, u_mpc = mpc.solve(
            pred_state,
            target,
            verbose=False,
            obstacle=global_obstacle,
        )

        acceleration = u_mpc[:, 0].copy()
        velocity_target = mpc.velocity_command(x_mpc)
        solve_elapsed = time.perf_counter() - loop_start

        with shared.lock:
            shared.mpc_accel[:] = acceleration
            shared.mpc_velocity_target[:] = velocity_target
            shared.x_mpc_world = x_mpc.copy()
            shared.mpc_elapsed = solve_elapsed

        sleep_time = max(0.0, mpc.dt - (time.perf_counter() - loop_start))
        time.sleep(sleep_time)


def draw_path(
    viewer: mujoco.viewer.Handle,
    path: npt.NDArray[np.float64],
) -> None:
    for i in range(path.shape[1] - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        p1 = np.array([path[0, i], path[1, i], 0.015], dtype=np.float64)
        p2 = np.array([path[0, i + 1], path[1, i + 1], 0.015], dtype=np.float64)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.012, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([0.0, 0.6, 1.0, 1.0], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.012,
            p1,
            p2,
        )
        viewer.user_scn.ngeom += 1


def draw_trail(
    viewer: mujoco.viewer.Handle,
    x_hist: list[float],
    y_hist: list[float],
    downsample: int = 8,
) -> None:
    if len(x_hist) < 2:
        return

    for i in range(0, len(x_hist) - 1, downsample):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        j = min(i + downsample, len(x_hist) - 1)
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        p1 = np.array([x_hist[i], y_hist[i], 0.025], dtype=np.float64)
        p2 = np.array([x_hist[j], y_hist[j], 0.025], dtype=np.float64)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.018, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1.0, 0.15, 0.15, 0.7], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.018,
            p1,
            p2,
        )
        viewer.user_scn.ngeom += 1


def draw_obstacles(viewer: mujoco.viewer.Handle, obstacles) -> None:
    for ox, oy, radius, _, _ in obstacles:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([radius, 0.08, 0.0], dtype=np.float64),
            pos=np.array([ox, oy, 0.08], dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1.0, 0.1, 0.1, 0.40], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_mpc_preview(
    viewer: mujoco.viewer.Handle,
    x_mpc: npt.NDArray[np.float64],
) -> None:
    for i in range(x_mpc.shape[1]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.035, 0.0, 0.0], dtype=np.float64),
            pos=np.array([x_mpc[0, i], x_mpc[1, i], 0.04], dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([0.1, 1.0, 0.1, 0.65], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_sensor_fov(
    viewer: mujoco.viewer.Handle,
    x: float,
    y: float,
    yaw: float,
    max_range: float,
    fov_deg: float,
) -> None:
    half_fov = 0.5 * np.radians(fov_deg)
    origin = np.array([x, y, 0.04], dtype=np.float64)
    rgba = np.array([0.95, 0.85, 0.1, 0.25], dtype=np.float32)

    angles = np.linspace(yaw - half_fov, yaw + half_fov, 24)
    points = [
        np.array(
            [x + max_range * np.cos(a), y + max_range * np.sin(a), 0.04],
            dtype=np.float64,
        )
        for a in angles
    ]

    for endpoint in (points[0], points[-1]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.008, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.008,
            origin,
            endpoint,
        )
        viewer.user_scn.ngeom += 1

    for p1, p2 in zip(points[:-1], points[1:]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.006, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.006,
            p1,
            p2,
        )
        viewer.user_scn.ngeom += 1


def initialize_plant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PlanarPlantIds,
    start: dict,
) -> None:
    heading = float(start["heading"])
    speed = float(start.get("velocity", 0.0))

    data.qpos[ids.x_qpos] = float(start["x"])
    data.qpos[ids.y_qpos] = float(start["y"])
    data.qpos[ids.yaw_qpos] = heading

    data.qvel[ids.x_dof] = speed * np.cos(heading)
    data.qvel[ids.y_dof] = speed * np.sin(heading)
    data.qvel[ids.yaw_dof] = 0.0

    # Start the low-level velocity servos at the measured initial velocity.
    data.ctrl[ids.vx_actuator] = data.qvel[ids.x_dof]
    data.ctrl[ids.vy_actuator] = data.qvel[ids.y_dof]
    data.ctrl[ids.yaw_rate_actuator] = data.qvel[ids.yaw_dof]

    mujoco.mj_forward(model, data)


def apply_constant_acceleration_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PlanarPlantIds,
    acceleration: npt.NDArray[np.float64],
    mpc: HolonomicMPC,
) -> None:
    """FOH velocity setpoints from piecewise-constant MPC acceleration.

    This intentionally mirrors the car MuJoCo demo's treatment of longitudinal
    acceleration: acceleration is held over the MPC interval and integrated at
    the tiny MuJoCo physics timestep.  Therefore the velocity setpoint ramps
    linearly rather than jumping once every MPC cycle.
    """
    dt = float(model.opt.timestep)

    data.ctrl[ids.vx_actuator] = np.clip(
        data.ctrl[ids.vx_actuator] + acceleration[0] * dt,
        -mpc.max_vx,
        mpc.max_vx,
    )
    data.ctrl[ids.vy_actuator] = np.clip(
        data.ctrl[ids.vy_actuator] + acceleration[1] * dt,
        -mpc.max_vy,
        mpc.max_vy,
    )
    data.ctrl[ids.yaw_rate_actuator] = np.clip(
        data.ctrl[ids.yaw_rate_actuator] + acceleration[2] * dt,
        -mpc.max_yaw_rate,
        mpc.max_yaw_rate,
    )


def main() -> None:
    root = pathlib.Path(__file__).parent

    model_path = root / "models" / "holonomic" / "holonomic_drone.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ids = get_planar_plant_ids(model)

    sim_config = yaml.safe_load((root / "config" / "simulation.yaml").read_text())
    start = sim_config["start"]
    target_speed = float(sim_config["target_speed"])
    sensor_max_range = float(sim_config["sensor"]["max_range"])
    sensor_fov_deg = float(sim_config["sensor"]["fov_deg"])
    goal_threshold = float(sim_config["goal_threshold"])

    mpc = HolonomicMPC("config/mpc_holonomic.yaml")
    initialize_plant(model, data, ids, start)

    path = compute_path_from_wp(
        sim_config["path"]["waypoints_x"],
        sim_config["path"]["waypoints_y"],
        sim_config["path"]["interpolation_step"],
    )

    # update_path_obstacles mutates each obstacle's distance, so make our own
    # dictionaries instead of reusing the YAML objects elsewhere.
    path_obstacles = [dict(obs) for obs in sim_config.get("obstacles", [])]
    dynamic_obstacles = update_path_obstacles(path_obstacles, path, 0.0)

    shared = SharedData()
    shared.state[:] = get_state(data, ids)

    mpc_thread = threading.Thread(
        target=controller_loop,
        args=(mpc, path, shared, goal_threshold, target_speed),
        daemon=True,
    )

    shutdown_flag = threading.Event()

    def handle_shutdown(signum, frame) -> None:
        del signum, frame
        shutdown_flag.set()

    signal.signal(signal.SIGINT, handle_shutdown)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [float(start["x"]), float(start["y"]), 0.0]
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = -90.0
        viewer.cam.elevation = -55.0

        render_fps = 60.0
        render_dt = 1.0 / render_fps

        x_history: list[float] = []
        y_history: list[float] = []
        cte_history: list[float] = []
        heading_error_history: list[float] = []

        sim_start_time = time.perf_counter()
        mpc_thread.start()

        try:
            while viewer.is_running() and not shutdown_flag.is_set():
                frame_start = time.perf_counter()

                if shared.goal_reached:
                    viewer.set_texts(
                        [
                            (
                                None,
                                None,
                                "GOAL REACHED\n"
                                "Holonomic vehicle stopped at the end of the path.",
                                "",
                            )
                        ]
                    )
                    viewer.sync()
                    while viewer.is_running() and not shutdown_flag.is_set():
                        time.sleep(0.05)
                    break

                current_state = get_state(data, ids)

                if path_obstacles:
                    detected_obstacle = detect_obstacle_camera(
                        dynamic_obstacles,
                        current_state[0],
                        current_state[1],
                        current_state[4],
                        sensor_max_range,
                        sensor_fov_deg,
                    )
                else:
                    detected_obstacle = None

                with shared.lock:
                    shared.state[:] = current_state
                    shared.obstacle = detected_obstacle
                    mpc_accel = shared.mpc_accel.copy()
                    mpc_velocity_target = shared.mpc_velocity_target.copy()
                    x_mpc_world = (
                        None
                        if shared.x_mpc_world is None
                        else shared.x_mpc_world.copy()
                    )
                    mpc_elapsed = shared.mpc_elapsed

                # Tracking metrics: compute_errors expects its legacy
                # [x, y, scalar_speed, heading] layout.
                speed = np.hypot(current_state[2], current_state[3])
                metric_state = np.array(
                    [current_state[0], current_state[1], speed, current_state[4]]
                )
                cte, heading_error = compute_errors(metric_state, path)
                x_history.append(current_state[0])
                y_history.append(current_state[1])
                cte_history.append(cte)
                heading_error_history.append(np.degrees(heading_error))

                cte_rmse = float(np.sqrt(np.mean(np.square(cte_history))))
                heading_rmse = float(
                    np.sqrt(np.mean(np.square(heading_error_history)))
                )

                # Keep MuJoCo approximately synchronized to wall time, exactly
                # like the repository's existing MuJoCo demo.
                elapsed_real_time = time.perf_counter() - sim_start_time
                while data.time < elapsed_real_time and not shutdown_flag.is_set():
                    apply_constant_acceleration_command(
                        model,
                        data,
                        ids,
                        mpc_accel,
                        mpc,
                    )

                    if path_obstacles:
                        dynamic_obstacles[:] = update_path_obstacles(
                            path_obstacles,
                            path,
                            float(model.opt.timestep),
                        )

                    mujoco.mj_step(model, data)

                # Camera follows the vehicle but stays in a top-down oblique view.
                viewer.cam.lookat[:] = [current_state[0], current_state[1], 0.0]

                viewer.user_scn.ngeom = 0
                draw_path(viewer, path)
                draw_trail(viewer, x_history, y_history)
                if path_obstacles:
                    draw_obstacles(viewer, dynamic_obstacles)
                    draw_sensor_fov(
                        viewer,
                        current_state[0],
                        current_state[1],
                        current_state[4],
                        sensor_max_range,
                        sensor_fov_deg,
                    )
                if x_mpc_world is not None:
                    draw_mpc_preview(viewer, x_mpc_world)

                goal_distance = np.hypot(
                    current_state[0] - path[0, -1],
                    current_state[1] - path[1, -1],
                )

                # data.ctrl is the continuously-ramped low-level velocity
                # setpoint. mpc_velocity_target is the endpoint predicted by
                # the MPC for the next MPC node; they should be close near the
                # end of each control interval, but need not be identical at
                # every physics step.
                servo_target = np.array(
                    [
                        data.ctrl[ids.vx_actuator],
                        data.ctrl[ids.vy_actuator],
                        data.ctrl[ids.yaw_rate_actuator],
                    ]
                )

                viewer.set_texts(
                    [
                        (
                            None,
                            None,
                            "Holonomic constant-acceleration MPC\n"
                            f"state:   vx {current_state[2]:+.2f}  vy {current_state[3]:+.2f} m/s  "
                            f"yaw {np.degrees(current_state[4]):+.1f} deg  w {current_state[5]:+.2f} rad/s\n"
                            f"MPC u:   ax {mpc_accel[0]:+.2f}  ay {mpc_accel[1]:+.2f} m/s2  "
                            f"alpha {mpc_accel[2]:+.2f} rad/s2  solve {mpc_elapsed*1000:.0f} ms\n"
                            f"next v*: vx {mpc_velocity_target[0]:+.2f}  vy {mpc_velocity_target[1]:+.2f}  "
                            f"w {mpc_velocity_target[2]:+.2f}\n"
                            f"servo:   vx {servo_target[0]:+.2f}  vy {servo_target[1]:+.2f}  "
                            f"w {servo_target[2]:+.2f}\n"
                            f"error:   CTE {cte:+.3f} m  yaw {np.degrees(heading_error):+.1f} deg  "
                            f"RMSE {cte_rmse:.3f} m / {heading_rmse:.1f} deg\n"
                            f"goal:    {goal_distance:.2f} m   "
                            f"avoid: {'YES' if detected_obstacle is not None else 'no' if path_obstacles else 'off'}",
                            "",
                        )
                    ]
                )

                viewer.sync()

                sleep_time = render_dt - (time.perf_counter() - frame_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        finally:
            with shared.lock:
                shared.is_active = False
            mpc_thread.join(timeout=1.0)
            viewer.clear_texts()


if __name__ == "__main__":
    main()
