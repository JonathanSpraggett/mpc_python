from __future__ import annotations

import pathlib

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml


class HolonomicMPC:
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
            # resolve path
            path = pathlib.Path(config)
            if not path.is_absolute():
                path = pathlib.Path(__file__).parent.parent / config
            with open(path) as f:
                config_data = yaml.safe_load(f)
        else:
            # raw dict
            config_data = config

        vehicle_config = config_data["model"]["vehicle"]
        obstacle_config = config_data["controller"]["obstacle"]
        prediction_config = config_data["controller"]["prediction"]
        weights_config = config_data["controller"]["weights"]

        self._state_dim: int = 6
        self._control_dim: int = 3

        self.radius = float(vehicle_config["radius"])

        self.max_vx = float(vehicle_config["max_vx"])
        self.max_vy = float(vehicle_config["max_vy"])
        self.max_yaw_rate = float(vehicle_config["max_yaw_rate"])

        self.max_ax = float(vehicle_config["max_ax"])
        self.max_ay = float(vehicle_config["max_ay"])
        self.max_yaw_accel = float(vehicle_config["max_yaw_accel"])

        # Same idea as max_d_acc in the car controller: rate of change of
        # acceleration = jerk.  Angular equivalent is angular jerk.
        self.max_jerk_x = float(vehicle_config["max_jerk_x"])
        self.max_jerk_y = float(vehicle_config["max_jerk_y"])
        self.max_yaw_jerk = float(vehicle_config["max_yaw_jerk"])

        self.dt: float = (
            timestep if timestep is not None else prediction_config["timestep"]
        )
        horizon = (
            horizon_time
            if horizon_time is not None
            else prediction_config["horizon_time"]
        )
        self.control_horizon: int = int(horizon / self.dt)

        state_cost_weights = (
            state_cost if state_cost is not None else weights_config["state_cost"]
        )
        terminal_cost_weights = (
            final_state_cost
            if final_state_cost is not None
            else weights_config["final_state_cost"]
        )
        input_cost_weights = (
            input_cost if input_cost is not None else weights_config["input_cost"]
        )
        input_rate_cost_weights = (
            input_rate_cost
            if input_rate_cost is not None
            else weights_config["input_rate_cost"]
        )

        if len(state_cost_weights) != self._state_dim:
            raise ValueError(
                f"State Error cost matrix should be of size {self._state_dim}"
            )
        if len(terminal_cost_weights) != self._state_dim:
            raise ValueError(
                f"End State Error cost matrix should be of size {self._state_dim}"
            )
        if len(input_cost_weights) != self._control_dim:
            raise ValueError(
                f"Control Effort cost matrix should be of size {self._control_dim}"
            )
        if len(input_rate_cost_weights) != self._control_dim:
            raise ValueError(
                "Control Effort Difference cost matrix should be of size "
                f"{self._control_dim}"
            )

        # NOTE: we use sum_squares which is not the same as a quad_form(x,Q), this is for strict DDP compliance
        # But to use sum_squares correctly, you need to pass it a matrix A such that A^T A = Q
        # We can get away with sqrt because matrices are diagonal
        self.q_matrix: npt.NDArray[np.float64] = np.sqrt(np.diag(state_cost_weights))
        self.qf_matrix: npt.NDArray[np.float64] = np.sqrt(
            np.diag(terminal_cost_weights)
        )
        self.r_matrix: npt.NDArray[np.float64] = np.sqrt(np.diag(input_cost_weights))
        self.rr_matrix: npt.NDArray[np.float64] = np.sqrt(
            np.diag(input_rate_cost_weights)
        )

        self._safety_margin: float = obstacle_config["safety_margin"]
        self._slack_penalty: float = (
            slack_penalty
            if slack_penalty is not None
            else obstacle_config["slack_penalty"]
        )

        self.A, self.B = self._compute_model_matrices(self.dt)

        self._vehicle_buffer: float =  self.radius + self._safety_margin

        # CVXPY vars
        self._states: opt.Variable = opt.Variable(
            (self._state_dim, self.control_horizon + 1), name="states"
        )
        self._controls: opt.Variable = opt.Variable(
            (self._control_dim, self.control_horizon), name="actions"
        )

        # CVXPY params (placeholder for run-time data)
        self._initial_state: opt.Parameter = opt.Parameter(self._state_dim, name="x0")
        self._last_command: opt.Parameter = opt.Parameter(
            self._control_dim, name="last_applied_command"
        )

        # Reference params (DPP-compliant placeholders)
        # see https://www.cvxpy.org/tutorial/dpp/index.html
        self._cos_reference = opt.Parameter(self.control_horizon + 1)
        self._sin_reference = opt.Parameter(self.control_horizon + 1)
        self._along_reference = opt.Parameter(self.control_horizon + 1)
        self._cross_reference = opt.Parameter(self.control_horizon + 1)
        self._heading_reference = opt.Parameter(self.control_horizon + 1)
        self._vx_reference = opt.Parameter(self.control_horizon + 1)
        self._vy_reference = opt.Parameter(self.control_horizon + 1)
        self._omega_reference = opt.Parameter(self.control_horizon + 1)

        # Obstacle params (half-plane linearization)
        self._obstacle_normal_x = opt.Parameter(self.control_horizon, name="obs_nx")
        self._obstacle_normal_y = opt.Parameter(self.control_horizon, name="obs_ny")
        self._obstacle_safe_distance = opt.Parameter(
            self.control_horizon, name="obs_dist"
        )
        # In optimization, a "slack" variable is a mathematical fudge factor.
        # Instead of treating the obstacle as an unyielding concrete wall,
        # we treat it as a stiff rubber wall. This variable tracks *how much*
        # we dent the wall if the vehicle is physically forced into it.
        # This allows in practice to turn the obstacle factor from hard(may cause failures) to soft
        self._obstacle_slack: opt.Variable = opt.Variable(
            self.control_horizon, nonneg=True, name="obstacle_slacks"
        )

        self._previous_command: npt.NDArray[np.float64] | None = None
        self._previous_trajectory: npt.NDArray[np.float64] | None = None

        # build the problem ONCE
        self._problem: opt.Problem = self._make_mpc_problem()

    @staticmethod
    def _compute_model_matrices(
            dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Exact discrete matrices for piecewise-constant acceleration."""
        A = np.eye(6, dtype=float)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        B = np.zeros((6, 3)) # todo should be less hardcoded
        tmp = 0.5 * dt * dt
        B[0, 0] = tmp
        B[1, 1] = tmp
        B[2, 2] = tmp
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        return np.round(A, 4), np.round(B, 4)



    def _make_mpc_problem(self) -> opt.Problem:

        cost = 0
        constraints = []

        for k in range(self.control_horizon):
            # Constant-acceleration dynamics.
            constraints += [
                self._states[:, k + 1]
                == self.A @ self._states[:, k]
                + self.B @ self._controls[:, k]
            ]

            # XY tracking does NOT make much sense in autonomous driving...
            # Instead we care how much off to the side we are w.r.t the track
            #
            # The standard cross-track and along-track errors are calculated by projecting position errors onto the track point
            # $\theta_{\text{ref}}$:$$e_{\text{along}} = \cos(\theta_{\text{ref}})(x - x_{\text{ref}}) + \sin(\theta_{\text{ref}})(y - y_{\text{ref}})
            # $e_{\text{cross}} = -\sin(\theta_{\text{ref}})(x - x_{\text{ref}}) + \cos(\theta_{\text{ref}})(y - y_{\text{ref}})$
            # we expand that and get the following Algebraic problem:

            # Algebraic along-track and cross-track expressions
            # We will fill the values when the reference is provided, that is why they are params
            
            # State
            along_track_error = (
                self._cos_reference[k] * self._states[0, k]
                + self._sin_reference[k] * self._states[1, k]
                - self._along_reference[k]
            )
            cross_track_error = (
                -self._sin_reference[k] * self._states[0, k]
                + self._cos_reference[k] * self._states[1, k]
                - self._cross_reference[k]
            )
            error = opt.vstack(
                [
                    along_track_error,
                    cross_track_error,
                    self._states[2, k] - self._heading_reference[k],
                    self._states[3, k] - self._vx_reference[k],
                    self._states[4, k] - self._vy_reference[k],
                    self._states[5, k] - self._omega_reference[k],
                ]
            )
            cost += opt.sum_squares(self.q_matrix @ error)

            # Input
            # Acceleration control effort.
            cost += opt.sum_squares(self.r_matrix @ self._controls[:, k])
            
            # Penalize changes in acceleration (jerk smoothing).
            if k == 0:
                
                du = self.rr_matrix @ (self._controls[:, 0] - self._last_command)
                
            else:
               
                du = self.rr_matrix @ (self._controls[:, k] - self._controls[:, k - 1])
                
            cost += opt.sum_squares(du)
            
            # Obstacle half-plane constraint:
            # obstacle avoidance: (px - pobs)*2 > R  in non-convex :(
            # this is linearised as : dot(px - pbos, n) > R
            # where n is a normal pointing from obstacle center toward the reference trajectory.
            # so the optimiser knows to stay on the same side of the obstacle as the reference
            #
            #      Valid Region
            #            ^
            #            | n = (nx, ny)
            #            * p_ref
            #            |
            #    --------+---------- Half-Plane
            #            | R
            #            * p_obs
            #
            # Hard Constraint:  dot(n, p) >= Safe_Distance
            # Soft Constraint:  dot(n, p) >= Safe_Distance - Slack
            constraints += [
                self._obstacle_normal_x[k] * self._states[0, k + 1]
                + self._obstacle_normal_y[k] * self._states[1, k + 1]
                >= self._obstacle_safe_distance[k] - self._obstacle_slack[k]
            ]
            cost += self._slack_penalty * self._obstacle_slack[k]
            
        terminal_along_track_error = (
            self._cos_reference[-1] * self._states[0, -1]
            + self._sin_reference[-1] * self._states[1, -1]
            - self._along_reference[-1]
        )
        terminal_cross_track_error = (
            -self._sin_reference[-1] * self._states[0, -1]
            + self._cos_reference[-1] * self._states[1, -1]
            - self._cross_reference[-1]
        )
        terminal_error = opt.vstack(
            [
                terminal_along_track_error,
                terminal_cross_track_error,
                self._states[2, -1] - self._heading_reference[-1],
                self._states[3, -1] - self._vx_reference[-1],
                self._states[4, -1] - self._vy_reference[-1],
                self._states[5, -1] - self._omega_reference[-1],
            ]
        )
        cost += opt.sum_squares(self.qf_matrix @ terminal_error)

        # initial state
        constraints += [self._states[:, 0] == self._initial_state]

        # state bounds
        
        # Independent translational velocity limits.
        constraints += [
            opt.abs(self._states[3, :]) <= self.max_vx,
            opt.abs(self._states[4, :]) <= self.max_vy,
            opt.abs(self._states[5, :]) <= self.max_yaw_rate,
        ]
        
        # Acceleration limits.
        constraints += [
            opt.abs(self._controls[0, :]) <= self.max_ax,
            opt.abs(self._controls[1, :]) <= self.max_ay,
            opt.abs(self._controls[2, :]) <= self.max_yaw_accel,
        ]

        # Jerk limits, including the transition from the acceleration that was
        # actually used during the previous MPC cycle.
        constraints += [
            opt.abs(
                self._controls[0, 0]
                - self._last_command[0]
            )
            / self.dt
            <= self.max_jerk_x,
            opt.abs(
                self._controls[1, 0]
                - self._last_command[1]
            )
            / self.dt
            <= self.max_jerk_y,
            opt.abs(
                self._controls[2, 0]
                - self._last_command[2]
            )
            / self.dt
            <= self.max_yaw_jerk,
        ]

        for k in range(1, self.control_horizon):
            constraints += [
                opt.abs(
                    self._controls[0, k]
                    - self._controls[0, k - 1]
                )
                / self.dt
                <= self.max_jerk_x,
                opt.abs(
                    self._controls[1, k]
                    - self._controls[1, k - 1]
                )
                / self.dt
                <= self.max_jerk_y,
                opt.abs(
                    self._controls[2, k]
                    - self._controls[2, k - 1]
                )
                / self.dt
                <= self.max_yaw_jerk,
            ]
        problem = opt.Problem(opt.Minimize(cost), constraints)
        return problem

    def update_obstacles(
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

                dx = target[0, k + 1] - obs_x
                dy = target[1, k + 1] - obs_y
                distance = np.hypot(dx, dy)

                if distance < 1.0e-5:
                    # Degenerate case: use the current vehicle-to-obstacle
                    # direction.  If that is also degenerate, choose +x.
                    dx = float(self._initial_state.value[0] - obs_x)
                    dy = float(self._initial_state.value[1] - obs_y)
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
        assert len(initial_state) == self._state_dim
        assert target.shape == (self._state_dim, self.control_horizon + 1)


        self._initial_state.value = np.array(initial_state)
        self._last_command.value = (
            self._previous_command[:, 0]
            if self._previous_command is not None
            else np.zeros(self._control_dim)
        )

        x_ref, y_ref, theta_ref = target[0, :], target[1, :],  target[2, :]
        vx_ref, vy_ref, omega_ref = target[3, :], target[4, :],  target[5, :]

        # Pre-calculate scalar projections using NumPy vectors
        cos_values = np.cos(theta_ref)
        sin_values = np.sin(theta_ref)
        along_projections = cos_values * x_ref + sin_values * y_ref
        cross_projections = -sin_values * x_ref + cos_values * y_ref

        self._cos_reference.value = cos_values
        self._sin_reference.value = sin_values
        self._along_reference.value = along_projections
        self._cross_reference.value = cross_projections
        self._heading_reference.value = theta_ref
        self._vx_reference.value = vx_ref
        self._vy_reference.value = vy_ref
        self._omega_reference.value = omega_ref

        self.update_obstacles(target, obstacle)



        self._problem.solve(
            solver=opt.CLARABEL,
            enforce_dpp=True,
            warm_start=True,
            verbose=verbose,
            canon_backend=opt.SCIPY_CANON_BACKEND,
            tol_gap_abs=1e-3,  # relax tolerances
            tol_gap_rel=1e-3,
            tol_feas=1e-3,
        )


        if self._states.value is None or self._controls.value is None:
            # the optimiser failed!
            # In this case you want to initialise a recovery behaviour!
            # To make this simple here I just decelerate
            print("MPC failed -> Emergency braking!")
            return self._controlled_stop(initial_state)

        # Store the finalized optimal trajectory for the next control cycle
        self._previous_trajectory = np.asarray(self._states.value, dtype=float)
        self._previous_command = np.asarray(self._controls.value, dtype=float)

        return self._previous_trajectory, self._previous_command

    def _controlled_stop(
        self, initial_state: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        controls = np.zeros((self._control_dim, self.control_horizon))
        states = np.zeros((self._state_dim, self.control_horizon + 1))
        states[:, 0] = initial_state

        for k in range(self.control_horizon):
            vx = states[3, k]
            vy = states[4, k]
            omega = states[5, k]

            controls[0, k] = np.clip(
                -vx / self.dt, -self.max_ax, self.max_ax
            )
            controls[1, k] = np.clip(
                -vy / self.dt, -self.max_ay, self.max_ay
            )
            controls[2, k] = np.clip(
                -omega / self.dt,
                -self.max_yaw_accel,
                self.max_yaw_accel,
            )
            states[:, k + 1] = (
                self.A @ states[:, k] + self.B @ controls[:, k]
            )

        self._previous_trajectory = states
        self._previous_command = controls
        return states, controls
