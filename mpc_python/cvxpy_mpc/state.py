
from dataclasses import dataclass, fields, field
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import odeint


class State(IntEnum):
    """What the gimbal is currently doing."""

    IDLE = auto()
    RUNNING = auto()


@dataclass
class StateData(ABC):
    # State
    X_IDX = 0
    Y_IDX = 1
    THETA_IDX = 2
    VX_IDX = 3
    VY_IDX = 4
    OMEGA_IDX = 5

    # Control
    AX_IDX = 0
    AY_IDX = 1
    ALPHA_IDX = 2


    state_dim : int = 6
    control_dim : int = 3


    _state: npt.NDArray[np.float64]  =  field(init=False)
    _control: npt.NDArray[np.float64] = field(init=False)

    dt : float = 0.2
    horizon_time: float = 4.0
    control_horizon = int(horizon_time / dt)

    state_history: dict =  field(default_factory=dict)
    control_history: dict = field(default_factory=dict)


    def __post_init__(self):
        self._state = np.zeros(self.state_dim, dtype=float)
        self._control = np.zeros(self.control_dim, dtype=float)

        self.state_history["x"] = [self.x]
        self.state_history["y"] = [self.y]
        self.state_history["theta"] = [self.theta]
        self.state_history["vx"] = [self.vx]
        self.state_history["vy"] = [self.vy]
        self.state_history["omega"] = [self.omega]

        self.control_history["ax"] = [0.0]
        self.control_history["ay"] = [0.0]
        self.control_history["alpha"] = [0.0]

    def __getitem__(self, index) -> Any:
        return self.state[index]



    @property
    def state(self) -> npt.NDArray[np.float64]:
        return self._state

    @state.setter
    def state(self, value: npt.NDArray[np.float64]):
        self._state = value

    @property
    def x(self) -> np.float64:
        return self._state[self.X_IDX]

    @property
    def y(self) -> np.float64:
        return self._state[self.Y_IDX]

    @property
    def theta(self) -> np.float64:
        return self._state[self.THETA_IDX]

    @property
    def vx(self) -> np.float64:
        return self._state[self.VX_IDX]

    @property
    def vy(self) -> np.float64:
        return self._state[self.VY_IDX]

    @property
    def omega(self) -> np.float64:
        return self._state[self.OMEGA_IDX]

    @property
    def rot(self) -> npt.NDArray[np.float64]:
        theta = self.theta

        return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    def world_to_ego(self, p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.rot.T @ p

    def ego_to_world(self, p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.rot @ p

    def ego_to_world_traj(self, x_mpc: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        traj = self.ego_to_world(x_mpc[:2, :].copy())
        traj[0, :] += self.x
        traj[1, :] += self.y
        return traj

    @property
    def control(self) -> npt.NDArray[np.float64]:
        return self._control

    @control.setter
    def control(self, value: npt.NDArray[np.float64]):
        self._control = value

    @property
    def ax(self) -> np.float64:
        return self._control[self.AX_IDX]

    @property
    def ay(self) -> np.float64:
        return self._control[self.AY_IDX]

    @property
    def alpha(self) -> np.float64:
        return self._control[self.ALPHA_IDX]

    def update_history(self):
        self.state_history["x"].append(self.x)
        self.state_history["y"].append(self.y)
        self.state_history["theta"].append(self.theta)
        self.state_history["vx"].append(self.vx)
        self.state_history["vy"].append(self.vy)
        self.state_history["omega"].append(self.omega)

        self.control_history["ax"].append(self.ax)
        self.control_history["ay"].append(self.ay)
        self.control_history["alpha"].append(self.alpha)

    def ego_state(self):
        tmp = self.world_to_ego(np.array([self.vx, self.vy]))
        return np.array([0, 0, 0, tmp[0], tmp[1], self.omega])

    def world_control(self, u_mpc: npt.NDArray[np.float64]) :
        tmp = self.ego_to_world(np.array([u_mpc[0, 0], u_mpc[1, 0]]))
        return np.array([tmp[0], tmp[1], u_mpc[2, 0]])



    @abstractmethod
    def ode_predict_next_state(self): # todo could and should split between state, control, model
        pass


@dataclass
class HolonomicState(StateData):
    # State
    X_IDX = 0
    Y_IDX = 1
    THETA_IDX = 2
    VX_IDX = 3
    VY_IDX = 4
    OMEGA_IDX = 5

    # Control
    AX_IDX = 0
    AY_IDX = 1
    ALPHA_IDX = 2


    def start_state(self, config):
        start = config["start"]
        heading = float(start["heading"])
        speed = float(start["velocity"])
        self._state = np.array(
            [
                float(start["x"]),
                float(start["y"]),
                heading,
                speed * np.cos(heading),
                speed * np.sin(heading),
                0.0,
            ],
            dtype=float,
        )

    def ode_predict_next_state(self):
        def kinematic_model(x, t, u):  # t, state, control):
            """
            Holonomic drone kinematic model.

            State:
                state = [x, y, yaw, vx, vy, yaw_rate]

            Control:
                control = [ax, ay, yaw_accel]

            Returns:
                state_dot = [x_dot, y_dot, yaw_dot,
                             vx_dot, vy_dot, yaw_rate_dot]
            """

            x, y, yaw, vx, vy, yaw_rate = x
            ax, ay, yaw_accel = u

            state_dot = np.array([
                vx,  # x_dot
                vy,  # y_dot
                yaw_rate,  # yaw_dot
                ax,  # vx_dot
                ay,  # vy_dot
                yaw_accel  # yaw_rate_dot
            ])

            return state_dot

        # solve ODE
        tspan = [0, self.dt]
        self.state = odeint(kinematic_model, self.state, tspan, args=(self.control[:],))[1]

    @staticmethod
    def _compute_model_matrices(
            dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Exact discrete matrices for piecewise-constant acceleration."""
        A = np.eye(6, dtype=float)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        B = np.zeros((6, 3))  # todo should be less hardcoded
        tmp = 0.5 * dt * dt
        B[0, 0] = tmp
        B[1, 1] = tmp
        B[2, 2] = tmp
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        return np.round(A, 4), np.round(B, 4)

    def predict_next_state(self, dt):
        A, B = self._compute_model_matrices(dt)
        pred_state = self.state.copy()
        half_dt2 = 0.5 * dt * dt
        # control = self.ego_to_world(np.array([self.ax,self.ay]))
        # control = np.array([control[0], control[1], self.alpha])
        # self.state =  A @ pred_state + B @ control #self.control
        self.state = A @ pred_state + B @ self.control
        # print(self.state)
        # Integrate physics forward in global space
        # pred_state[self.X_IDX] += pred_state[3] * dt + self.control[0] * half_dt2
        # pred_state[self.Y_IDX] += pred_state[4] * dt + self.control[1] * half_dt2
        # pred_state[self.THETA_IDX] += pred_state[5] * dt + self.control[2] * half_dt2
        # pred_state[self.VX_IDX] += self.control[0] * dt
        # pred_state[self.VY_IDX] += self.control[1] * dt
        # pred_state[self.OMEGA_IDX] += self.control[2] * dt
        # self.state = pred_state
