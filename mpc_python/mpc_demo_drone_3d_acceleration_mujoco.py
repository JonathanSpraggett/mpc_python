#!/usr/bin/env python3
"""XYZ + yaw acceleration-controlled drone following a path 2 m above ground.

Run this file for the viewer, or pass --headless --duration 10 for a smoke run.
This ideal holonomic model stays level; it does not simulate rotor/attitude dynamics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from cvxpy_mpc.drone_3d_acceleration_mpc import Drone3DAccelerationMPC, build_drone_path
from cvxpy_mpc.utils import detect_obstacle_camera, update_path_obstacles


def get_plant_ids(model):
    """Named addresses in state and command order, independent of XML ordering."""
    joints = [model.joint(name) for name in ("x_joint", "y_joint", "z_joint", "yaw_joint")]
    qpos = np.array([int(j.qposadr[0]) for j in joints])
    dofs = np.array([int(j.dofadr[0]) for j in joints])
    actuators = np.array([model.actuator(name).id for name in
                          ("ax_motor", "ay_motor", "az_motor", "yaw_accel_motor")])
    return qpos, dofs, actuators


def get_state(data, ids):
    qpos, dofs, _ = ids
    positions, velocities = data.qpos[qpos], data.qvel[dofs]
    return np.r_[positions[:3], velocities[:3], positions[3], velocities[3]]


def initialize_plant(model, data, ids, start):
    qpos, dofs, _ = ids
    mujoco.mj_resetData(model, data)
    data.qpos[qpos] = [start["x"], start["y"], start["z"], start["heading"]]
    speed = float(start.get("velocity", 0))
    data.qvel[dofs] = [speed * np.cos(start["heading"]), speed * np.sin(start["heading"]), 0, 0]
    mujoco.mj_forward(model, data)


def apply_acceleration_command(model, data, ids, acceleration, mpc):
    """Hold net world-frame acceleration; the model compensates gravity."""
    command = mpc.acceleration_command(np.asarray(acceleration).reshape(4, 1))
    actuator_ids = ids[2]
    ranges = model.actuator_ctrlrange[actuator_ids]
    limits = np.array([mpc.planar.max_ax, mpc.planar.max_ay, mpc.max_az, mpc.planar.max_yaw_accel])
    if np.any(ranges[:, 0] > -limits) or np.any(ranges[:, 1] < limits):
        raise ValueError("Model ctrlrange must cover the configured MPC acceleration limits")
    data.ctrl[actuator_ids] = command


def draw_lines(viewer, points, color, radius=0.012):
    for p1, p2 in zip(points[:-1], points[1:]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.array([radius, 0., 0.]), np.zeros(3),
                           np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p1, p2)
        viewer.user_scn.ngeom += 1


def draw_scene(viewer, path, history, prediction, obstacles):
    viewer.user_scn.ngeom = 0
    draw_lines(viewer, path[:3].T, [0., 0.6, 1., 1.])
    draw_lines(viewer, np.asarray(history)[::4, :3], [1., 0.15, 0.15, 0.7], 0.018)
    if prediction is not None:
        draw_lines(viewer, prediction[:3].T, [0.1, 1., 0.1, 0.8], 0.02)
    # Render XY obstacle constraints as columns extending above flight altitude.
    height = max(3.0, float(path[2].max()) + 1.0)
    for ox, oy, radius, _, _ in obstacles:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                           np.array([radius, height / 2, 0.]), np.array([ox, oy, height / 2]),
                           np.eye(3).ravel(), np.array([1., 0.1, 0.1, 0.3], dtype=np.float32))
        viewer.user_scn.ngeom += 1


def run(duration=None, headless=False, config_path=None):
    root = Path(__file__).parent
    config = yaml.safe_load(Path(config_path or root / "config/simulation_drone_3d.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(root / "models/holonomic/holonomic_drone_3d_acceleration.xml"))
    data = mujoco.MjData(model)
    ids = get_plant_ids(model)
    initialize_plant(model, data, ids, config["start"])
    mpc = Drone3DAccelerationMPC(root / "config/mpc_holonomic.yaml", **config["vertical"])
    path = build_drone_path(config)
    planar_path = path[[0, 1, 3]]
    obstacles = [dict(o) for o in config.get("obstacles", [])]
    steps = round(mpc.dt / model.opt.timestep)
    if not np.isclose(steps * model.opt.timestep, mpc.dt):
        raise ValueError("MPC dt must be a multiple of the physics timestep")
    history = [get_state(data, ids)]
    goal_reached = False

    def simulate(viewer=None):
        nonlocal goal_reached
        next_control = 0.0
        prediction = None
        clock_start = time.perf_counter()
        while duration is None or data.time < duration:
            if viewer is not None and not viewer.is_running():
                break
            state = get_state(data, ids)
            if data.time + 1e-9 >= next_control:
                dynamic = update_path_obstacles(obstacles, planar_path, 0.0)
                detected = detect_obstacle_camera(dynamic, state[0], state[1], state[6],
                                                  config["sensor"]["max_range"], config["sensor"]["fov_deg"])
                reference = mpc.reference(state, path, config["target_speed"])
                prediction, controls = mpc.solve(state, reference, obstacle=detected)
                apply_acceleration_command(model, data, ids, mpc.acceleration_command(controls), mpc)
                next_control += mpc.dt
                goal_reached = (np.linalg.norm(state[:3] - path[:3, -1]) < config["goal_threshold"]
                                and np.linalg.norm(state[3:6]) < 0.1 and abs(state[7]) < 0.1)
                if headless and goal_reached:
                    break
            # Continue controlling at the goal so the drone remains hovering.
            for _ in range(min(steps, 10) if viewer is not None else steps):
                update_path_obstacles(obstacles, planar_path, float(model.opt.timestep))
                mujoco.mj_step(model, data)
            state = get_state(data, ids)
            history.append(state)
            if viewer is not None:
                dynamic = update_path_obstacles(obstacles, planar_path, 0.0)
                draw_scene(viewer, path, history, prediction, dynamic)
                viewer.cam.lookat[:] = state[:3]
                viewer.set_texts([(None, None,
                    f"3D acceleration MPC | z {state[2]:.2f} m (target 2 m)\n"
                    f"u [ax ay az alpha]: {data.ctrl[ids[2]].round(2)}\n"
                    f"{'GOAL: hovering' if goal_reached else 'Following elevated waypoints'}", "")])
                viewer.sync()
                delay = data.time - (time.perf_counter() - clock_start)
                if delay > 0:
                    time.sleep(min(delay, 0.02))
        return np.asarray(history)

    if headless:
        return simulate()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 7.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = -90
        return simulate(viewer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=None, help="Simulation seconds")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    history = run(args.duration if args.duration is not None else (10.0 if args.headless else None),
                  args.headless, args.config)
    print(f"Simulated {len(history)-1} samples; final XYZ {history[-1, :3]}")


if __name__ == "__main__":
    main()
