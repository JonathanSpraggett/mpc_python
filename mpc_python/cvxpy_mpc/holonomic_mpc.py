from __future__ import annotations

import pathlib

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml


class HolonomicMPC:
    """Linear MPC for a planar holonomic drone-like vehicle.

    State vector (world/local navigation frame):
        x = [px, py, vx, vy, theta, omega]

    Optimization input:
        u = [ax, ay, alpha]

    The input is assumed constant over each MPC timestep.  The exact
    constant-acceleration discrete model is therefore:

        px[k+1]    = px[k]    + vx[k] dt + 0.5 ax[k] dt^2
        py[k+1]    = py[k]    + vy[k] dt + 0.5 ay[k] dt^2
        vx[k+1]    = vx[k]    + ax[k] dt
        vy[k+1]    = vy[k]    + ay[k] dt
        theta[k+1] = theta[k] + omega[k] dt + 0.5 alpha[k] dt^2
        omega[k+1] = omega[k] + alpha[k] dt

    Although acceleration is the MPC input, a velocity-controlled vehicle
    should normally receive the first predicted next-state velocity:

        [vx_cmd, vy_cmd, yaw_rate_cmd] = x_mpc[[2, 3, 5], 1]
    """

    PX = 0
    PY = 1
    VX = 2
    VY = 3
    THETA = 4
    OMEGA = 5

    AX = 0
    AY = 1
    ALPHA = 2

    def __init__(
        self,
        config: str | pathlib.Path | dict,
        horizon_time: float | None = None,
        timestep: float | None = None,
        state_cost: list[float] | None = None,
        final_state_cost: list[float] | None = None,
        input_cost: list[float] | None = None,
        input_rate_cost: list[float] | None = None,
        slack_penalty: float | None = None,
    ) -> None:
        if isinstance(config, (str, pathlib.Path)):
            path = pathlib.Path(config)
            if not path.is_absolute():
                # Matches the original repository behaviour.  From this file,
                # parent.parent is the repository's mpc_python/ directory.
                path = pathlib.Path(__file__).parent.parent / config
            with open(path) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config

        vehicle = config_data["model"]["vehicle"]
        obstacle = config_data["controller"]["obstacle"]
        prediction = config_data["controller"]["prediction"]
        weights = config_data["controller"]["weights"]

        self._state_dim = 6
        self._control_dim = 3

        self.radius = float(vehicle["radius"])

        self.max_vx = float(vehicle["max_vx"])
        self.max_vy = float(vehicle["max_vy"])
        self.max_yaw_rate = float(vehicle["max_yaw_rate"])

        self.max_ax = float(vehicle["max_ax"])
        self.max_ay = float(vehicle["max_ay"])
        self.max_yaw_accel = float(vehicle["max_yaw_accel"])

        # Same idea as max_d_acc in the car controller: rate of change of
        # acceleration = jerk.  Angular equivalent is angular jerk.
        self.max_jerk_x = float(vehicle["max_jerk_x"])
        self.max_jerk_y = float(vehicle["max_jerk_y"])
        self.max_yaw_jerk = float(vehicle["max_yaw_jerk"])

        self.dt = float(
            timestep if timestep is not None else prediction["timestep"]
        )
        horizon = float(
            horizon_time
            if horizon_time is not None
            else prediction["horizon_time"]
        )
        self.control_horizon = int(horizon / self.dt)
        if self.control_horizon < 1:
            raise ValueError("MPC horizon must contain at least one timestep")

        state_cost_weights = (
            state_cost if state_cost is not None else weights["state_cost"]
        )
        final_state_cost_weights = (
            final_state_cost
            if final_state_cost is not None
            else weights["final_state_cost"]
        )
        input_cost_weights = (
            input_cost if input_cost is not None else weights["input_cost"]
        )
        input_rate_cost_weights = (
            input_rate_cost
            if input_rate_cost is not None
            else weights["input_rate_cost"]
        )

        self._validate_weight_length(
            "state_cost", state_cost_weights, self._state_dim
        )
        self._validate_weight_length(
            "final_state_cost", final_state_cost_weights, self._state_dim
        )
        self._validate_weight_length(
            "input_cost", input_cost_weights, self._control_dim
        )
        self._validate_weight_length(
            "input_rate_cost", input_rate_cost_weights, self._control_dim
        )

        self.q_matrix: npt.NDArray[np.float64] = np.diag(state_cost_weights)
        self.qf_matrix: npt.NDArray[np.float64] = np.diag(
            final_state_cost_weights
        )
        self.r_matrix: npt.NDArray[np.float64] = np.diag(input_cost_weights)
        self.rr_matrix: npt.NDArray[np.float64] = np.diag(
            input_rate_cost_weights
        )

        self._safety_margin = float(obstacle["safety_margin"])
        self._slack_penalty = float(
            slack_penalty
            if slack_penalty is not None
            else obstacle["slack_penalty"]
        )
        self._vehicle_buffer = self.radius + self._safety_margin

        self.A, self.B = self._compute_model_matrices(self.dt)

        # Decision variables.
        self._states = opt.Variable(
            (self._state_dim, self.control_horizon + 1), name="states"
        )
        self._controls = opt.Variable(
            (self._control_dim, self.control_horizon), name="accelerations"
        )

        # Runtime parameters.
        self._initial_state = opt.Parameter(self._state_dim, name="x0")
        self._reference = opt.Parameter(
            (self._state_dim, self.control_horizon + 1), name="reference"
        )
        self._last_acceleration = opt.Parameter(
            self._control_dim, name="last_applied_acceleration"
        )

        # Obstacle half-plane parameters.  These are kept compatible with the
        # original repository's obstacle representation:
        # (x, y, radius, vx, vy).
        self._obstacle_normal_x = opt.Parameter(
            self.control_horizon, name="obs_nx"
        )
        self._obstacle_normal_y = opt.Parameter(
            self.control_horizon, name="obs_ny"
        )
        self._obstacle_safe_distance = opt.Parameter(
            self.control_horizon, name="obs_dist"
        )
        self._obstacle_slack = opt.Variable(
            self.control_horizon, nonneg=True, name="obstacle_slacks"
        )

        self._previous_acceleration: npt.NDArray[np.float64] | None = None
        self._previous_trajectory: npt.NDArray[np.float64] | None = None

        self._problem = self._make_mpc_problem()

    @staticmethod
    def _validate_weight_length(name: str, values: list[float], expected: int) -> None:
        if len(values) != expected:
            raise ValueError(
                f"{name} must contain {expected} values, got {len(values)}"
            )

    @staticmethod
    def _compute_model_matrices(
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Exact discrete matrices for piecewise-constant acceleration."""
        A = np.eye(6, dtype=float)
        A[0, 2] = dt
        A[1, 3] = dt
        A[4, 5] = dt

        B = np.zeros((6, 3), dtype=float)
        half_dt2 = 0.5 * dt * dt

        B[0, 0] = half_dt2
        B[2, 0] = dt

        B[1, 1] = half_dt2
        B[3, 1] = dt

        B[4, 2] = half_dt2
        B[5, 2] = dt

        return A, B

    def _make_mpc_problem(self) -> opt.Problem:
        cost = 0
        constraints: list[opt.Constraint] = []

        constraints.append(self._states[:, 0] == self._initial_state)

        for k in range(self.control_horizon):
            # Constant-acceleration dynamics.
            constraints.append(
                self._states[:, k + 1]
                == self.A @ self._states[:, k]
                + self.B @ self._controls[:, k]
            )

            # For a holonomic vehicle, direct x/y tracking is meaningful.
            state_error = self._states[:, k] - self._reference[:, k]
            cost += opt.quad_form(state_error, self.q_matrix)

            # Acceleration control effort.
            cost += opt.quad_form(self._controls[:, k], self.r_matrix)

            # Penalize changes in acceleration (jerk smoothing).
            if k == 0:
                du = self._controls[:, 0] - self._last_acceleration
            else:
                du = self._controls[:, k] - self._controls[:, k - 1]
            cost += opt.quad_form(du, self.rr_matrix)

            # Same linearized obstacle half-plane idea as the original repo.
            constraints.append(
                self._obstacle_normal_x[k] * self._states[self.PX, k + 1]
                + self._obstacle_normal_y[k] * self._states[self.PY, k + 1]
                >= self._obstacle_safe_distance[k] - self._obstacle_slack[k]
            )
            cost += self._slack_penalty * self._obstacle_slack[k]

        terminal_error = self._states[:, -1] - self._reference[:, -1]
        cost += opt.quad_form(terminal_error, self.qf_matrix)

        # Independent translational velocity limits.
        constraints += [
            opt.abs(self._states[self.VX, :]) <= self.max_vx,
            opt.abs(self._states[self.VY, :]) <= self.max_vy,
            opt.abs(self._states[self.OMEGA, :]) <= self.max_yaw_rate,
        ]

        # Acceleration limits.
        constraints += [
            opt.abs(self._controls[self.AX, :]) <= self.max_ax,
            opt.abs(self._controls[self.AY, :]) <= self.max_ay,
            opt.abs(self._controls[self.ALPHA, :]) <= self.max_yaw_accel,
        ]

        # Jerk limits, including the transition from the acceleration that was
        # actually used during the previous MPC cycle.
        constraints += [
            opt.abs(
                self._controls[self.AX, 0]
                - self._last_acceleration[self.AX]
            )
            / self.dt
            <= self.max_jerk_x,
            opt.abs(
                self._controls[self.AY, 0]
                - self._last_acceleration[self.AY]
            )
            / self.dt
            <= self.max_jerk_y,
            opt.abs(
                self._controls[self.ALPHA, 0]
                - self._last_acceleration[self.ALPHA]
            )
            / self.dt
            <= self.max_yaw_jerk,
        ]

        for k in range(1, self.control_horizon):
            constraints += [
                opt.abs(
                    self._controls[self.AX, k]
                    - self._controls[self.AX, k - 1]
                )
                / self.dt
                <= self.max_jerk_x,
                opt.abs(
                    self._controls[self.AY, k]
                    - self._controls[self.AY, k - 1]
                )
                / self.dt
                <= self.max_jerk_y,
                opt.abs(
                    self._controls[self.ALPHA, k]
                    - self._controls[self.ALPHA, k - 1]
                )
                / self.dt
                <= self.max_yaw_jerk,
            ]

        return opt.Problem(opt.Minimize(cost), constraints)

    def _set_obstacle_parameters(
        self,
        target: npt.NDArray[np.float64],
        obstacle: tuple[float, float, float, float, float] | None,
    ) -> None:
        nx = np.zeros(self.control_horizon)
        ny = np.zeros(self.control_horizon)
        safe_distance = np.zeros(self.control_horizon)

        if obstacle is None:
            # Make the constraint trivially satisfied.
            nx[:] = 1.0
            safe_distance[:] = -1.0e6
        else:
            obstacle_x, obstacle_y, obstacle_radius, obstacle_vx, obstacle_vy = obstacle

            for k in range(self.control_horizon):
                # Constraint acts on state k+1, so predict obstacle at k+1 too.
                t = (k + 1) * self.dt
                obs_x = obstacle_x + obstacle_vx * t
                obs_y = obstacle_y + obstacle_vy * t

                dx = target[self.PX, k + 1] - obs_x
                dy = target[self.PY, k + 1] - obs_y
                distance = np.hypot(dx, dy)

                if distance < 1.0e-5:
                    # Degenerate case: use the current vehicle-to-obstacle
                    # direction.  If that is also degenerate, choose +x.
                    dx = float(self._initial_state.value[self.PX] - obs_x)
                    dy = float(self._initial_state.value[self.PY] - obs_y)
                    distance = np.hypot(dx, dy)
                    if distance < 1.0e-5:
                        dx, dy, distance = 1.0, 0.0, 1.0

                nx[k] = dx / distance
                ny[k] = dy / distance

                safe_distance[k] = (
                    nx[k] * obs_x
                    + ny[k] * obs_y
                    + obstacle_radius
                    + self._vehicle_buffer
                )

        self._obstacle_normal_x.value = nx
        self._obstacle_normal_y.value = ny
        self._obstacle_safe_distance.value = safe_distance

    def solve(
        self,
        initial_state: npt.NDArray[np.float64] | list[float],
        target: npt.NDArray[np.float64],
        verbose: bool = False,
        obstacle: tuple[float, float, float, float, float] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        initial_state = np.asarray(initial_state, dtype=float)
        target = np.asarray(target, dtype=float)

        if initial_state.shape != (self._state_dim,):
            raise ValueError(
                "initial_state must be [x, y, vx, vy, theta, omega]"
            )
        expected_shape = (self._state_dim, self.control_horizon + 1)
        if target.shape != expected_shape:
            raise ValueError(
                f"target must have shape {expected_shape}, got {target.shape}"
            )

        self._initial_state.value = initial_state
        self._reference.value = target
        self._last_acceleration.value = (
            self._previous_acceleration[:, 0]
            if self._previous_acceleration is not None
            else np.zeros(self._control_dim)
        )
        self._set_obstacle_parameters(target, obstacle)

        self._problem.solve(
            solver=opt.CLARABEL,
            warm_start=True,
            verbose=verbose,
            canon_backend=opt.SCIPY_CANON_BACKEND,
            enforce_dpp=True,
        )

        if self._states.value is None or self._controls.value is None:
            print("Holonomic MPC failed -> controlled stop fallback")
            return self._controlled_stop(initial_state)

        self._previous_trajectory = np.asarray(self._states.value, dtype=float)
        self._previous_acceleration = np.asarray(self._controls.value, dtype=float)
        return self._previous_trajectory, self._previous_acceleration

    def _controlled_stop(
        self, initial_state: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        controls = np.zeros((self._control_dim, self.control_horizon))
        states = np.zeros((self._state_dim, self.control_horizon + 1))
        states[:, 0] = initial_state

        for k in range(self.control_horizon):
            vx = states[self.VX, k]
            vy = states[self.VY, k]
            omega = states[self.OMEGA, k]

            controls[self.AX, k] = np.clip(
                -vx / self.dt, -self.max_ax, self.max_ax
            )
            controls[self.AY, k] = np.clip(
                -vy / self.dt, -self.max_ay, self.max_ay
            )
            controls[self.ALPHA, k] = np.clip(
                -omega / self.dt,
                -self.max_yaw_accel,
                self.max_yaw_accel,
            )
            states[:, k + 1] = (
                self.A @ states[:, k] + self.B @ controls[:, k]
            )

        self._previous_trajectory = states
        self._previous_acceleration = controls
        return states, controls

    @staticmethod
    def velocity_command(
        predicted_states: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return [vx_cmd, vy_cmd, yaw_rate_cmd] for the next MPC timestep."""
        if predicted_states.shape[0] != 6 or predicted_states.shape[1] < 2:
            raise ValueError("predicted_states must have shape (6, N+1), N >= 1")
        return predicted_states[[2, 3, 5], 1].copy()

    def predict_next_state(
        self,
        state: npt.NDArray[np.float64] | list[float],
        acceleration: npt.NDArray[np.float64] | list[float],
    ) -> npt.NDArray[np.float64]:
        """One exact constant-acceleration propagation step."""
        x = np.asarray(state, dtype=float)
        u = np.asarray(acceleration, dtype=float)
        return self.A @ x + self.B @ u
