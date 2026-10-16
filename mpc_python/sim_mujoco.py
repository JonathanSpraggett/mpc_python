import pathlib
import threading
import time

import numpy as np
from dataclasses import dataclass

import mujoco

from mpc_python.sim import Sim
import numpy.typing as npt


# MPC and sim are on 2 threads
# avoids messy global vars
class SharedData:
    """Encapsulates all data shared between the Physics thread and MPC thread."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()

        # Core control & state
        self.state: npt.NDArray[np.float64] = np.zeros(6, dtype=float)
        self.goal_reached: bool = False
        self.is_active: bool = True
        self.mpc_accel: npt.NDArray[np.float64] = np.zeros(3, dtype=float)
        self.mpc_velocity_target: npt.NDArray[np.float64] = np.zeros(3, dtype=float)

        # Telemetry & visualization
        self.x_mpc_world: npt.NDArray[np.float64] | None = None
        self.mpc_elapsed: float = 0.0

        # Externally detected obstacle (ego frame, set by physics thread)
        self.obstacle: tuple[float, float, float] | None = None

@dataclass(frozen=True)
class PlanarPlantids:
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


class SimMujoco:
    def __init__(self, sim:Sim, ) -> None:
        self.sim = sim
        root = pathlib.Path(__file__).parent

        model_path = root / "models" / "holonomic" / "holonomic_drone.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.ids = self.get_vehicle_ids()

        self.shared = SharedData()

        self.viewer =  mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5.0
        self.viewer.cam.azimuth = -90.0
        self.viewer.cam.elevation = -55.0




    def body_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i == -1:
            raise ValueError(f"Body '{name}' not found")
        return i

    def _require_named_id(self,
            obj_type: mujoco.mjtObj,
            name: str,
    ) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx == -1:
            raise ValueError(f"MuJoCo object '{name}' was not found")
        return idx

    def get_vehicle_ids(self) -> PlanarPlantids:
        x_joint = self._require_named_id( mujoco.mjtObj.mjOBJ_JOINT, "x_joint")
        y_joint = self._require_named_id( mujoco.mjtObj.mjOBJ_JOINT, "y_joint")
        yaw_joint = self._require_named_id( mujoco.mjtObj.mjOBJ_JOINT, "yaw_joint")

        return PlanarPlantids(
            x_qpos=int(self.model.jnt_qposadr[x_joint]),
            y_qpos=int(self.model.jnt_qposadr[y_joint]),
            yaw_qpos=int(self.model.jnt_qposadr[yaw_joint]),
            x_dof=int(self.model.jnt_dofadr[x_joint]),
            y_dof=int(self.model.jnt_dofadr[y_joint]),
            yaw_dof=int(self.model.jnt_dofadr[yaw_joint]),
            vx_actuator=self._require_named_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "vx_servo"
            ),
            vy_actuator=self._require_named_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "vy_servo"
            ),
            yaw_rate_actuator=self._require_named_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_rate_servo"
            ),
            body=self._require_named_id(
                 mujoco.mjtObj.mjOBJ_BODY, "holonomic_drone"
            ),
        )

    def initialize_plant(self, state) -> None:


        self.data.qpos[self.ids.x_qpos] = state.x
        self.data.qpos[self.ids.y_qpos] = state.y
        self.data.qpos[self.ids.yaw_qpos] = state.theta

        self.data.qvel[self.ids.x_dof] = state.vx
        self.data.qvel[self.ids.y_dof] = state.vy
        self.data.qvel[self.ids.yaw_dof] = state.omega

        # Start the low-level velocity servos at the measured initial velocity.
        self.data.ctrl[self.ids.vx_actuator] = self.data.qvel[self.ids.x_dof]
        self.data.ctrl[self.ids.vy_actuator] = self.data.qvel[self.ids.y_dof]
        self.data.ctrl[self.ids.yaw_rate_actuator] = self.data.qvel[self.ids.yaw_dof]

        mujoco.mj_forward(self.model, self.data)

        self.shared.state[:] = self.get_state()
        self.viewer.cam.lookat[:] = [float(self.shared.state[0]), float(self.shared.state[1]), 0.0]

    def get_state(self ) -> npt.NDArray[np.float64]:
        """Read [x, y, yaw, vx, vy, yaw_rate] directly from planar joints."""

        return np.array(
            [
                self.data.qpos[self.ids.x_qpos],
                self.data.qpos[self.ids.y_qpos],
                self.data.qpos[self.ids.yaw_qpos],
                self.data.qvel[self.ids.x_dof],
                self.data.qvel[self.ids.y_dof],
                self.data.qvel[self.ids.yaw_dof],
            ],
            dtype=float,
        )

    def check_completion(self, cte_rmse, heading_rmse):
        # Check for completion
        if self.shared.goal_reached:
            self.viewer.set_texts(
                [
                    (
                        None,
                        None,
                        f"GOAL REACHED\n"
                        f"Final RMSE:   CTE {cte_rmse:.3f} m  |  heading {heading_rmse:.1f} deg\n",
                        "",
                    )
                ]
            )
            self.viewer.sync()
            print(
                "\nGoal reached! Close the viewer window or press CTRL-C to exit."
            )
            # Idle until the user closes the window or forces a KeyboardInterrupt
            while self.viewer.is_running():
                time.sleep(0.1)
            return True
        return False

    def apply_velocity(self, state, mpc, mpc_accel):
        dt = float(self.model.opt.timestep)

        # theta = self.data.qpos[self.ids.yaw_qpos]
        # # theta = state.theta
        # rot = np.array(
        #     [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        # )
        #
        # mpc_accel[:2] = rot.T @ mpc_accel[:2]
        # mpc_a =  mpc_accel[:2]
        # v = rot @ np.array([self.data.ctrl[self.ids.vx_actuator], self.data.ctrl[self.ids.vy_actuator]])
        # print(self.data.ctrl[self.ids.vx_actuator], self.data.qvel[self.ids.x_dof],v[0])
        # print(self.data.ctrl[self.ids.yaw_rate_actuator], self.data.qvel[self.ids.yaw_dof], state.omega,self.data.qpos[self.ids.yaw_qpos], state.theta)

        self.data.ctrl[self.ids.vx_actuator] = np.clip(
            self.data.ctrl[self.ids.vx_actuator] + mpc_accel[0] * dt,
            -mpc.max_vx,
            mpc.max_vx,
        )
        self.data.ctrl[self.ids.vy_actuator] = np.clip(
            self.data.ctrl[self.ids.vy_actuator] + mpc_accel[1] * dt,
            -mpc.max_vy,
            mpc.max_vy,
        )
        self.data.ctrl[self.ids.yaw_rate_actuator] = np.clip(
            self.data.ctrl[self.ids.yaw_rate_actuator] + mpc_accel[2] * dt,
            -mpc.max_yaw_rate,
            mpc.max_yaw_rate,
        )

        # self.data.ctrl[self.ids.yaw_rate_actuator] = 0

    def draw_path(self, path: npt.NDArray[np.float64]) -> None:
        for i in range(path.shape[1] - 1):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                break

            # next geometry slot
            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]

            p1 = np.array([path[0, i], path[1, i], 0.03], dtype=np.float64)
            p2 = np.array([path[0, i + 1], path[1, i + 1], 0.03], dtype=np.float64)

            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array(
                    [0.008, 0.0, 0.0], dtype=np.float64
                ),  # [radius, unused, unused]
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3).ravel(),
                rgba=np.array([0, 0.6, 1, 1], dtype=np.float32),
            )

            # mujoco handles the vector math to stretch it between p1 and p2
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, p1, p2)

            self.viewer.user_scn.ngeom += 1

    def draw_trail(self,
            x_hist: list[float],
            y_hist: list[float],
            downsample: int = 10,
    ) -> None:
        if len(x_hist) < 2:
            return
        for i in range(0, len(x_hist) - 1, downsample):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                break

            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
            alpha = (i + 1) / len(x_hist) * 0.8

            p1 = np.array([x_hist[i], y_hist[i], 0.005], dtype=np.float64)
            p2 = np.array([x_hist[i + 1], y_hist[i + 1], 0.005], dtype=np.float64)

            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3).ravel(),
                rgba=np.array([1, 0, 0, alpha], dtype=np.float32),
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02, p1, p2)
            self.viewer.user_scn.ngeom += 1

    def draw_obstacle(self, obstacles) -> None:
        for idx, (ox, oy, rad, _, _) in enumerate(obstacles):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                return
            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([rad, 0.0, 0.0], dtype=np.float64),
                pos=np.array([ox, oy, 0.0], dtype=np.float64),
                mat=np.eye(3).ravel(),
                rgba=np.array([1, 0, 0, 0.4], dtype=np.float32),
            )
            self.viewer.user_scn.ngeom += 1

    def draw_mpc_preview(
            self, x_mpc_world: npt.NDArray[np.float64]
    ) -> None:
        for i in range(x_mpc_world.shape[1]):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                break

            # next geometry slot
            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]

            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.03, 0.0, 0.0], dtype=np.float64),
                pos=np.array(
                    [x_mpc_world[0, i], x_mpc_world[1, i], 0.01], dtype=np.float64
                ),
                mat=np.eye(3).ravel(),
                rgba=np.array([0, 1, 0, 0.6], dtype=np.float32),
            )
            self.viewer.user_scn.ngeom += 1

    def draw_sensor_fov(
            self,
            x: float,
            y: float,
            heading: float,
            max_range: float,
            fov_deg: float,
    ) -> None:
        fov_rad = np.radians(fov_deg)
        right_angle = heading - fov_rad / 2.0
        left_angle = heading + fov_rad / 2.0

        right_end = np.array(
            [x + max_range * np.cos(right_angle), y + max_range * np.sin(right_angle), 0.0]
        )
        left_end = np.array(
            [x + max_range * np.cos(left_angle), y + max_range * np.sin(left_angle), 0.0]
        )
        origin = np.array([x, y, 0.0])

        rgba = np.array([0.8, 0.8, 0, 0.2], dtype=np.float32)

        for end in (right_end, left_end):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                return
            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3).ravel(),
                rgba=rgba,
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01, origin, end)
            self.viewer.user_scn.ngeom += 1

        num_arc_pts = 20
        prev = None
        for i in range(num_arc_pts + 1):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                return
            t = i / num_arc_pts
            angle = right_angle + t * fov_rad
            p = np.array(
                [x + max_range * np.cos(angle), y + max_range * np.sin(angle), 0.0]
            )
            if prev is not None:
                g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=np.array([0.01, 0.0, 0.0], dtype=np.float64),
                    pos=np.zeros(3, dtype=np.float64),
                    mat=np.eye(3).ravel(),
                    rgba=rgba,
                )
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01, prev, p)
                self.viewer.user_scn.ngeom += 1
            prev = p

    def draw_text(self, text):
        # data.ctrl is the continuously-ramped low-level velocity
        # setpoint. mpc_velocity_target is the endpoint predicted by
        # the MPC for the next MPC node; they should be close near the
        # end of each control interval, but need not be identical at
        # every physics step.
        servo_target = np.array(
            [
                self.data.ctrl[self.ids.vx_actuator],
                self.data.ctrl[self.ids.vy_actuator],
                self.data.ctrl[self.ids.yaw_rate_actuator],
            ]
        )

        self.viewer.set_texts(
            [
                (
                    None,
                    None,
                    text,
                    # f"servo:   vx {servo_target[0]:+.2f}  vy {servo_target[1]:+.2f}  ",
                    # f"w {servo_target[2]:+.2f}\n",
                    "",
                )
            ]
        )
