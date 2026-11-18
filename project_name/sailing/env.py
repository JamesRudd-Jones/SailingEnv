"""
Single boat sailing to a mark upwind, following the BifurcaGym environment contract.

Everything a person sets or reads is in compass degrees, with wind given as the direction it blows from, and the physics
is in maths angles. World x is east and world y is north, so positions need no conversion.
"""

import math
from typing import Any, Dict, Tuple

from bifurcagym import spaces
from bifurcagym.envs import base_env
from bifurcagym.envs.utils import integrate_ode
import chex
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom

from project_name.sailing import dynamics, wind
from project_name.sailing.angles import compass_to_maths, maths_to_compass, wind_velocity, wrap
from project_name.sailing.dynamics import BoatParams


@struct.dataclass
class EnvState(base_env.EnvState):
    boat: chex.Array  # see dynamics.py for the order
    wind_speed: chex.Array  # m/s
    wind_direction: chex.Array  # compass degrees, the direction it blows from
    mark: chex.Array  # m, east and north


@struct.dataclass
class EnvParams:
    boat: BoatParams = BoatParams()

    mean_wind_speed: float = 5.0  # m/s
    mean_wind_direction: float = 0.0  # compass degrees, from the north
    wind_speed_sd: float = 0.75  # m/s
    wind_direction_sd: float = 6.0  # degrees
    wind_speed_time_constant: float = 30.0  # s
    wind_direction_time_constant: float = 60.0  # s

    mark_east: float = 0.0  # m
    mark_north: float = 100.0  # m, directly upwind of the start, so tacking is needed
    # Degrees. Each reset turns the mark about the start by up to this either way, so that training sees marks that need
    # no tacking as well as upwind ones. get_state assumes the unturned mark, so generative_step is exact only at zero
    mark_bearing_spread: float = 0.0
    mark_radius: float = 5.0  # m
    start_spread: float = 15.0  # m
    arena_half_width: float = 200.0  # m, leaving this ends the episode

    max_rudder: float = math.radians(85.0)
    max_sail: float = math.radians(90.0)  # positive is sheet length, negative is backing the main

    progress_reward: float = 0.1  # per metre closer to the mark
    finish_reward: float = 10.0
    time_penalty: float = 0.01  # per step


class SailingCSDA(base_env.BaseEnvironment):

    def __init__(self, dt: float = 0.5, substeps: int = 10, max_steps_in_ep: int = 600, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = dt  # s between decisions, the physics runs at dt / substeps, which must stay at or below 0.05 s
        self.substeps: int = substeps
        self.max_steps_in_ep: int = max_steps_in_ep

        self.requires_float64: bool = False

        rudder_choices = jnp.radians(jnp.array((-30.0, 0.0, 30.0)))
        sheet_choices = jnp.radians(jnp.array((15.0, 45.0, 80.0)))
        self.action_array: chex.Array = jnp.stack(jnp.meshgrid(rudder_choices, sheet_choices, indexing="ij"), axis=-1).reshape(-1, 2)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(self,
                 input_action: chex.Numeric,
                 state: EnvState,
                 params: EnvParams,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        action = self.action_convert(input_action, params)

        # The wind is held steady over one decision step, so it only moves between steps
        control = jnp.concatenate((action, wind_velocity(state.wind_speed, state.wind_direction)))
        boat = integrate_ode(dynamics.ode_rhs, 0.0, state.boat, control, self.dt, self.substeps, params.boat)
        boat = boat.at[dynamics.HEADING].set(wrap(boat[dynamics.HEADING]))

        speed_key, direction_key = jrandom.split(key)
        new_state = EnvState(boat=boat,
                             wind_speed=jnp.maximum(wind.ornstein_uhlenbeck_step(state.wind_speed,
                                                                                 params.mean_wind_speed,
                                                                                 params.wind_speed_sd,
                                                                                 params.wind_speed_time_constant,
                                                                                 self.dt,
                                                                                 speed_key),
                                                    0.5),
                             wind_direction=wind.ornstein_uhlenbeck_step(state.wind_direction,
                                                                         params.mean_wind_direction,
                                                                         params.wind_direction_sd,
                                                                         params.wind_direction_time_constant,
                                                                         self.dt,
                                                                         direction_key),
                             mark=state.mark,
                             time=state.time + 1)

        reward, done = self.reward_and_done_function(input_action, state, new_state, params, key)

        info = {"finished": self._distance_to_mark(new_state) <= params.mark_radius,
                "compass_heading": maths_to_compass(boat[dynamics.HEADING]),
                }

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                reward,
                done,
                info)

    @staticmethod
    def _distance_to_mark(state: EnvState) -> chex.Array:
        offset = state.mark - state.boat[jnp.array((dynamics.X, dynamics.Y))]
        return jnp.sqrt(jnp.sum(offset * offset) + 1e-6)

    def reset_env(self, params: EnvParams, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        position_key, heading_key, speed_key, wind_speed_key, wind_direction_key = jrandom.split(key, 5)
        # Folded from the heading key, so that the other draws, and every existing episode, are unchanged at zero spread
        turn = jnp.radians(params.mark_bearing_spread) * jrandom.uniform(jrandom.fold_in(heading_key, 1), minval=-1.0, maxval=1.0)
        mark = jnp.array((jnp.cos(turn) * params.mark_east - jnp.sin(turn) * params.mark_north,
                          jnp.sin(turn) * params.mark_east + jnp.cos(turn) * params.mark_north))

        position = jrandom.uniform(position_key, (2,), minval=-params.start_spread, maxval=params.start_spread)
        heading = jrandom.uniform(heading_key, minval=-jnp.pi, maxval=jnp.pi)
        forward_speed = jrandom.uniform(speed_key, minval=0.0, maxval=1.5)
        boat = jnp.array((position[0], position[1], heading, forward_speed, 0.0, 0.0, 0.0, 0.0))

        state = EnvState(boat=boat,
                         wind_speed=jnp.maximum(params.mean_wind_speed + params.wind_speed_sd * jrandom.normal(wind_speed_key), 0.5),
                         wind_direction=params.mean_wind_direction + params.wind_direction_sd * jrandom.normal(wind_direction_key),
                         mark=mark,
                         time=0)

        return self.get_obs(state), state

    def reward_and_done_function(self,
                                 input_action_t: chex.Numeric,
                                 state_t: EnvState,
                                 state_tp1: EnvState,
                                 params: EnvParams,
                                 key: chex.PRNGKey = None,
                                 ) -> Tuple[chex.Array, chex.Array]:
        distance = self._distance_to_mark(state_tp1)
        finished = distance <= params.mark_radius

        progress = self._distance_to_mark(state_t) - distance
        reward = params.progress_reward * progress + params.finish_reward * finished - params.time_penalty

        position = state_tp1.boat[jnp.array((dynamics.X, dynamics.Y))]
        outside = jnp.any(jnp.abs(position - state_tp1.mark * 0.5) > params.arena_half_width)
        done = finished | outside | (state_tp1.time >= self.max_steps_in_ep)

        return reward, done

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        return self.action_array[action.squeeze()]

    def get_obs(self, state: EnvState, key: chex.PRNGKey = None) -> chex.Array:
        """Relative angles as sine and cosine, so there are no wraps to learn across. Bow is zero, port is positive."""
        boat = state.boat
        heading = boat[dynamics.HEADING]
        wind_from = compass_to_maths(state.wind_direction)
        true_wind_angle = wind_from - heading

        offset = state.mark - boat[jnp.array((dynamics.X, dynamics.Y))]
        mark_bearing = jnp.arctan2(offset[1], offset[0] + 1e-9) - heading

        # Apparent wind ignoring yaw rate, because a policy only needs a good reading of it, and it is not used in get_state
        velocity_x = wind_velocity(state.wind_speed, state.wind_direction)
        wind_x = jnp.cos(heading) * velocity_x[0] + jnp.sin(heading) * velocity_x[1] - boat[dynamics.U]
        wind_y = jnp.cos(heading) * velocity_x[1] - jnp.sin(heading) * velocity_x[0] - boat[dynamics.V]
        apparent_wind_angle = jnp.arctan2(-wind_y, -wind_x + 1e-9)

        return jnp.stack((jnp.sin(true_wind_angle), jnp.cos(true_wind_angle),
                          jnp.sin(mark_bearing), jnp.cos(mark_bearing),
                          self._distance_to_mark(state),
                          boat[dynamics.U], boat[dynamics.V], boat[dynamics.R],
                          boat[dynamics.BOOM], boat[dynamics.RUDDER],
                          state.wind_speed,
                          jnp.sin(jnp.radians(state.wind_direction)), jnp.cos(jnp.radians(state.wind_direction)),
                          jnp.sin(apparent_wind_angle), jnp.cos(apparent_wind_angle)))

    def get_state(self, obs: chex.Array, params: EnvParams) -> EnvState:
        wind_direction = jnp.degrees(jnp.arctan2(obs[11], obs[12]))
        heading = wrap(compass_to_maths(wind_direction) - jnp.arctan2(obs[0], obs[1]))
        bearing = heading + jnp.arctan2(obs[2], obs[3])
        mark = jnp.array((params.mark_east, params.mark_north))
        position = mark - obs[4] * jnp.stack((jnp.cos(bearing), jnp.sin(bearing)))

        boat = jnp.array((position[0], position[1], heading, obs[5], obs[6], obs[7], obs[8], obs[9]))
        return EnvState(boat=boat, wind_speed=obs[10], wind_direction=wind_direction, mark=mark, time=-1)

    def render_traj(self, trajectory_state: EnvState, params: EnvParams, file_path: str = "../animations"):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import numpy as np

        boats = np.asarray(trajectory_state.boat)
        marks = np.asarray(trajectory_state.mark)
        wind_speeds = np.asarray(trajectory_state.wind_speed)
        wind_directions = np.asarray(trajectory_state.wind_direction)
        times = np.asarray(trajectory_state.time)
        frames = boats.shape[0]

        hull_shape = np.array(((6.0, 0.0), (-3.0, 2.2), (-3.0, -2.2)))  # drawn larger than life so that it can be seen
        boom_length = 5.0

        margin = 25.0
        east_low = min(boats[:, dynamics.X].min(), marks[:, 0].min()) - margin
        east_high = max(boats[:, dynamics.X].max(), marks[:, 0].max()) + margin
        north_low = min(boats[:, dynamics.Y].min(), marks[:, 1].min()) - margin
        north_high = max(boats[:, dynamics.Y].max(), marks[:, 1].max()) + margin

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_facecolor("#e6f2f7")
        ax.set(xlim=(east_low, east_high), ylim=(north_low, north_high), xlabel="East (m)", ylabel="North (m)", aspect="equal")

        trail, = ax.plot([], [], color="#2a78d6", linewidth=1.5)
        mark_point, = ax.plot([], [], "o", color="#eb6834", markersize=10)
        hull = ax.add_patch(Polygon(hull_shape, closed=True, color="#0b0b0b"))
        boom, = ax.plot([], [], color="#eb6834", linewidth=2.5)
        wind_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(color="#52514e", width=1.5, headwidth=8))
        title = ax.set_title("", loc="left")

        def update(frame):
            x, y, heading = boats[frame, dynamics.X], boats[frame, dynamics.Y], boats[frame, dynamics.HEADING]
            episode_start = int(np.max(np.where(times[:frame + 1] == 0)[0], initial=0))
            trail.set_data(boats[episode_start:frame + 1, dynamics.X], boats[episode_start:frame + 1, dynamics.Y])
            mark_point.set_data([marks[frame, 0]], [marks[frame, 1]])

            rotation = np.array(((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading))))
            hull.set_xy((rotation @ hull_shape.T).T + np.array((x, y)))

            # The boom points aft of the mast, and a positive angle is out to starboard
            boom_direction = heading + np.pi + boats[frame, dynamics.BOOM]
            boom.set_data([x, x + boom_length * np.cos(boom_direction)], [y, y + boom_length * np.sin(boom_direction)])

            from_rad = np.radians(wind_directions[frame])
            head = np.array((east_low + 15.0, north_high - 15.0))
            tail = head + 10.0 * np.array((np.sin(from_rad), np.cos(from_rad)))
            wind_arrow.set_position(tail)
            wind_arrow.xy = head

            speed = np.hypot(boats[frame, dynamics.U], boats[frame, dynamics.V])
            compass = (90.0 - np.degrees(heading)) % 360.0
            title.set_text(f"Heading {compass:03.0f}°  {speed:.2f} m/s  "
                           f"wind {wind_speeds[frame]:.1f} m/s from {wind_directions[frame] % 360.0:03.0f}°  t={frame * self.dt:.0f} s")
            return trail, mark_point, hull, boom, wind_arrow, title

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=self.dt * 1000, blit=False)
        anim.save(f"{file_path}_{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "Sailing-v0"

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.action_array.shape[0])

    def observation_space(self, params: EnvParams) -> spaces.Box:
        reach = 2.0 * jnp.sqrt(2.0) * params.arena_half_width
        low = jnp.array((-1.0, -1.0, -1.0, -1.0, 0.0, -4.0, -5.0, -3.0, -jnp.pi / 2, -params.max_rudder, 0.0,
                         -1.0, -1.0, -1.0, -1.0))
        high = jnp.array((1.0, 1.0, 1.0, 1.0, reach, 8.0, 5.0, 3.0, jnp.pi / 2, params.max_rudder, 20.0,
                          1.0, 1.0, 1.0, 1.0))
        return spaces.Box(low, high, (15,), dtype=jnp.float32)


class SailingCSCA(SailingCSDA):
    """The action is (rudder, sail) in radians. A positive sail command is sheet length, a negative one backs the main."""

    def action_convert(self, action: chex.Numeric, params: EnvParams) -> chex.Numeric:
        limit = jnp.array((params.max_rudder, params.max_sail))
        return jnp.clip(action, -limit, limit).squeeze()

    def action_space(self, params: EnvParams) -> spaces.Box:
        limit = jnp.array((params.max_rudder, params.max_sail))
        return spaces.Box(-limit, limit, (2,), dtype=jnp.float32)
