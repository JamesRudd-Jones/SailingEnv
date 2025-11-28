import jax
import jax.numpy as jnp
import jax.random as jrandom
from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
import chex
from typing import Dict, Tuple, Any, Union, Optional
import matplotlib.pyplot as plt


@struct.dataclass
class EnvState(base_env.EnvState):
    boat_pos: jnp.ndarray
    boat_vel: jnp.ndarray
    boat_heading: jnp.ndarray
    boat_heading_rate: jnp.ndarray
    rudder_angle: jnp.ndarray
    sail_angle: jnp.ndarray
    time: int


class SailingEnvCSCA(base_env.BaseEnvironment):
    """
    wind x y is global
    boat has a local x y
    """
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.1

        self.max_steps_in_episode: int = 500
        self.max_action: float = 1.0

        self.wind_dir: float = 0.0  # deg
        self.wind_speed: float = 5.0  # ms^-1
        self.wind_vel: jnp.ndarray = -self.wind_speed * jnp.array((jnp.cos(jnp.radians(self.wind_dir)),
                                                                   jnp.sin(jnp.radians(self.wind_dir))))  # ms^-1

        self.mass: float = 120.0  # kg

        self.gravity = 9.81  # ms^-2

        self.rudder_area: float = 0.13  # m^2
        self.rudder_aspect_ratio: float = 2.326923076923077  # self.rudder_span / self.rudder_chord  # TODO aka rudder stretching in the paper
        self.rudder_dist_to_cog: float = 1.24  # m
        self.min_rudder: jnp.ndarray = jnp.radians(-35)  # rads
        self.max_rudder: jnp.ndarray = jnp.radians(35)  # rads

        self.air_kinematic_viscosity: float = 0.0000171  # Pa * s
        self.air_density: float =  1.3  # kgm^-3

        self.sail_area: float = 6.4  # m^2
        self.sail_chord: float = 1  # m
        self.sail_span: float = 6.2   # m
        # self.sail_aspect_ratio: float = 0.961
        self.sail_aspect_ratio = self.sail_span / self.sail_chord  # TODO changed this for testing
        self.sail_dist_cog_to_pressure_point: float = 0.43  # m
        self.sail_dist_mast_to_pressure_point: float = 0.68  # m
        self.min_sail: jnp.ndarray = jnp.radians(0)  # rads
        self.max_sail: jnp.ndarray = jnp.radians(90)  # rads

        self.water_kinematic_viscosity: float = 0.0000001  # Pa * s
        self.water_density: float = 1000  # kgm^-3

        self.cboard_area: float = 0.5  # m^2
        self.cboard_chord: float = 2  # m
        self.cboard_span: float = 0.55  # m
        # self.cboard_aspect_ratio: float = 0.605
        self.cboard_aspect_ratio = self.cboard_span / self.cboard_chord  # TODO change it back
        self.cboard_dist_cog_to_pressure_point: float = 0.24  # m
        self.cboard_dist_cog_to_middle: float = self.cboard_dist_cog_to_pressure_point - 0.7  # m # TODO sort all these COG distances out

        self.waterline_length: float = 4  # m
        self.hull_speed: jnp.ndarray = 2.5  # 0.4 * jnp.sqrt(self.gravity * self.waterline_length)  # ms^-1  # TODO return this back to equation
        self.lateral_area: float = 2.5  # m^2
        self.wave_resistance_param: float = 1

        self.along_damping: float = 15
        self.transverse_damping: float = 5
        self.heading_time_constant: float = 5
        self.moi_z: float = 1066  # kgm^2

        self.damping_invariant_x: float = -self.mass / self.along_damping
        self.damping_invariant_y: float = -self.mass / self.transverse_damping
        self.damping_invariant_heading: float = -(self.moi_z / self.heading_time_constant)
        self.wave_impedance_invariant: float = (self.water_density / 2) * self.lateral_area

        self.screen_width: int = 100  # m
        self.screen_height: int = 100  # m

        self.init_pos: jnp.ndarray = jnp.array((self.screen_width / 4, self.screen_height / 2))

        self.marks: jnp.ndarray = jnp.array(((3 * self.screen_width / 4, self.screen_height / 2),))
        # TODO to deal with multiple marks, could jnp.roll once done a conditional
        # self.reward_gate: jnp.ndarray = jnp.array((10, 10))

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:

        # Adjust and apply actions
        action = self.action_convert(input_action)
        # sail_angle = jnp.radians(20.0)  # can range between 0 and 90
        # rudder_angle = -0.02  # action
        # rudder_angle = action[0]
        # sail_angle = action[1]

        delta_rudder = -2 * (action[0] - state.rudder_angle)
        max_rudder_speed = jnp.pi / 20
        delta_rudder = jnp.clip(delta_rudder, -max_rudder_speed, max_rudder_speed)

        delta_sail = -2 * (action[1] - state.sail_angle)
        max_sail_speed = jnp.pi
        delta_sail = jnp.clip(delta_sail, -max_sail_speed, max_sail_speed)

        rudder_angle = jnp.clip(state.rudder_angle + delta_rudder * self.dt, self.min_rudder, self.max_rudder)
        sail_angle = jnp.clip(state.sail_angle + delta_sail * self.dt, self.min_sail, self.max_sail)
        # TODO is there a better way to do this?

        def get_derivatives(curr_state, r_angle, s_angle):
            # Convert wind to boat and find apparent wind
            transformed_wind = self.global_to_boat(curr_state.boat_heading, self.wind_vel)
            apparent_wind = transformed_wind - curr_state.boat_vel
            apparent_wind_angle = jnp.arctan2(-apparent_wind[1], -apparent_wind[0])
            apparent_wind_speed = jnp.sqrt(apparent_wind[0] ** 2 + apparent_wind[1] ** 2)

            # Calc aoa
            true_sail_angle = jnp.sign(apparent_wind_angle) * jnp.abs(s_angle)
            aoa = apparent_wind_angle - true_sail_angle

            # Calc sail force
            aoa = jax.lax.select(aoa * true_sail_angle < 0, 0.0, aoa)

            eff_aoa = jnp.select([aoa < -jnp.pi / 2, aoa > jnp.pi / 2],
                                 [jnp.pi + aoa, -jnp.pi + aoa],
                                 aoa)

            aero_friction = jax.lax.select(apparent_wind_speed != 0,
                                           2.66 / jnp.sqrt((apparent_wind_speed * self.sail_chord) / self.air_kinematic_viscosity),
                                           0.0)

            # coeff_lift = (2 * jnp.pi * eff_aoa) / (1 + 2 / self.sail_aspect_ratio)
            # coeff_drag = aero_friction * coeff_lift ** 2

            coeff_lift = 2 * jnp.pi * eff_aoa
            coeff_drag = aero_friction + (coeff_lift ** 2 / (jnp.pi * self.sail_aspect_ratio))

            aero_pressure = 0.5 * self.air_density * apparent_wind_speed ** 2
            sail_lift = aero_pressure * self.sail_area * coeff_lift
            sail_drag = aero_pressure * self.sail_area * coeff_drag

            sail_force_x = -sail_drag * jnp.cos(apparent_wind_angle) + sail_lift * jnp.sin(apparent_wind_angle)
            sail_force_y = -sail_lift * jnp.cos(apparent_wind_angle) - sail_drag * jnp.sin(apparent_wind_angle)

            aero_separation = 1 - jnp.exp(-jnp.square(jnp.abs(eff_aoa) / jnp.radians(25)))

            sail_separated_force_x = jnp.sign(aoa) * aero_pressure * self.sail_area * jnp.sin(aoa) ** 2 * jnp.sin(true_sail_angle)
            sail_separated_force_y = -jnp.sign(aoa) * aero_pressure * self.sail_area * jnp.sin(aoa) ** 2 * jnp.cos(true_sail_angle)

            sail_x = (1 - aero_separation) * sail_force_x + aero_separation * sail_separated_force_x
            sail_y = (1 - aero_separation) * sail_force_y + aero_separation * sail_separated_force_y

            # Calc centreboard force
            boat_speed = jnp.sqrt(curr_state.boat_vel[0] ** 2 + curr_state.boat_vel[1] ** 2)
            leeway_angle = jnp.arctan2(curr_state.boat_vel[1], curr_state.boat_vel[0])

            eff_leeway_angle = jnp.select([leeway_angle < -jnp.pi / 2, leeway_angle > jnp.pi / 2],
                                          [jnp.pi + leeway_angle, -jnp.pi + leeway_angle],
                                          default=leeway_angle)

            hydro_friction = jax.lax.select(boat_speed != 0,
                                            2.66 / jnp.sqrt((boat_speed * self.cboard_chord) / self.water_kinematic_viscosity),
                                            0.0)

            # coeff_lift = (2 * jnp.pi * eff_leeway_angle) / (1 + 2 / self.cboard_aspect_ratio)
            # coeff_drag = hydro_friction * coeff_lift ** 2

            coeff_lift = 2 * jnp.pi * eff_leeway_angle
            coeff_drag = hydro_friction * (coeff_lift ** 2 / (jnp.pi * self.cboard_aspect_ratio))

            hydro_pressure = 0.5 * self.water_density * boat_speed ** 2
            cboard_lift = hydro_pressure * self.cboard_area * coeff_lift
            cboard_drag = hydro_pressure * self.cboard_area * coeff_drag

            cboard_force_x = -cboard_drag * jnp.cos(leeway_angle) - cboard_lift * jnp.sin(leeway_angle)
            cboard_force_y = -cboard_lift * jnp.cos(leeway_angle) - cboard_drag * jnp.sin(leeway_angle)

            hydro_separation = 1 - jnp.exp(-jnp.square(jnp.abs(eff_leeway_angle) / jnp.radians(25)))

            cboard_separated_force_y = -jnp.sign(leeway_angle) * hydro_pressure * self.cboard_area * jnp.sin(leeway_angle) ** 2

            cboard_x = (1 - hydro_separation) * cboard_force_x
            cboard_y = (1 - hydro_separation) * cboard_force_y + hydro_separation * cboard_separated_force_y

            # Calc rudder force
            rudder_x = -(((4 * jnp.pi) / self.rudder_aspect_ratio) * r_angle ** 2) * hydro_pressure * self.rudder_area
            rudder_y = 2 * jnp.pi * hydro_pressure * self.rudder_area * r_angle
            # TODO this assumes rudder doesn't stall, it would be interesting if possible to add this as a feature perhaps

            # Calc damping
            damping_x = self.damping_invariant_x * curr_state.boat_vel[0]
            damping_y = self.damping_invariant_y * curr_state.boat_vel[1]
            damping_heading = self.damping_invariant_heading * curr_state.boat_heading_rate

            # Calc wave impedance
            # wave_impedance = -jnp.sign(curr_state.boat_vel[0]) * self.wave_resistance_param * hydro_pressure * self.lateral_area * (boat_speed / self.hull_speed) ** 4
            wave_impedance = jnp.zeros(())

            # Sum up forces and turn to velocities
            delta_pos = self.boat_to_global(curr_state.boat_heading, curr_state.boat_vel)
            delta_heading = curr_state.boat_heading_rate

            # delta_vel_x = (sail_x + cboard_x + rudder_x + damping_x + wave_impedance) / self.mass
            # delta_vel_y = (sail_y + cboard_y + rudder_y + damping_y) / self.mass
            delta_vel_x = delta_heading * curr_state.boat_vel[1] + (sail_x + cboard_x + rudder_x + damping_x + wave_impedance) / self.mass
            delta_vel_y = -delta_heading * curr_state.boat_vel[0] + (sail_y + cboard_y + rudder_y + damping_y) / self.mass
            # delta_heading_rate = (damping_heading
            #                       - rudder_y * self.rudder_dist_to_cog
            #                       + sail_y * self.sail_dist_cog_to_pressure_point
            #                       + sail_x * jnp.sin(true_sail_angle) * self.sail_dist_mast_to_pressure_point
            #                       + cboard_y * (self.cboard_dist_cog_to_pressure_point * (1 - hydro_separation)
            #                                     + self.cboard_dist_cog_to_middle * hydro_separation)) / self.moi_z
            delta_heading_rate = (damping_heading - rudder_y * self.rudder_dist_to_cog) / self.moi_z

            return ((delta_pos, jnp.array((delta_vel_x, delta_vel_y)), delta_heading, delta_heading_rate),
                    (true_sail_angle, sail_x, sail_y, cboard_x, cboard_y, hydro_separation))

        deltas, (true_sail_angle, sail_x, sail_y, cboard_x, cboard_y, hydro_separation) = get_derivatives(state,
                                                                                                          state.rudder_angle,
                                                                                                          state.sail_angle)

        """ Euler discretisation step """
        new_boat_pos = state.boat_pos + deltas[0] * self.dt
        new_boat_vel = state.boat_vel + deltas[1] * self.dt
        new_boat_heading = state.boat_heading + deltas[2] * self.dt
        new_boat_heading_rate = state.boat_heading_rate + deltas[3] * self.dt
        # new_boat_heading = state.boat_heading
        # new_boat_heading_rate = state.boat_heading_rate
        """ End Euler """

        # """ RK4 discretisation step """
        # k1_rates, (true_sail_angle, sail_x, sail_y, cboard_x, cboard_y, hydro_separation) = get_derivatives(state,
        #                                                                                                     state.rudder_angle,
        #                                                                                                     state.sail_angle)
        #
        # state_k2 = EnvState(boat_pos=state.boat_pos + k1_rates[0] * self.dt / 2,
        #                     boat_vel=state.boat_vel + k1_rates[1] * self.dt / 2,
        #                     boat_heading=state.boat_heading + k1_rates[2] * self.dt / 2,
        #                     boat_heading_rate=state.boat_heading_rate + k1_rates[3] * self.dt / 2,
        #                     rudder_angle=rudder_angle,
        #                     sail_angle=sail_angle,
        #                     time=state.time)
        # k2_rates, _ = get_derivatives(state_k2, state.rudder_angle, state.sail_angle)
        #
        # state_k3 = EnvState(boat_pos=state.boat_pos + k2_rates[0] * self.dt / 2,
        #                     boat_vel=state.boat_vel + k2_rates[1] * self.dt / 2,
        #                     boat_heading=state.boat_heading + k2_rates[2] * self.dt / 2,
        #                     boat_heading_rate=state.boat_heading_rate + k2_rates[3] * self.dt / 2,
        #                     rudder_angle=rudder_angle,
        #                     sail_angle=sail_angle,
        #                     time=state.time)
        # k3_rates, _ = get_derivatives(state_k3, state.rudder_angle, state.sail_angle)
        #
        # state_k4 = EnvState(boat_pos=state.boat_pos + k3_rates[0] * self.dt,
        #                     boat_vel=state.boat_vel + k3_rates[1] * self.dt,
        #                     boat_heading=state.boat_heading + k3_rates[2] * self.dt,
        #                     boat_heading_rate=state.boat_heading_rate + k3_rates[3] * self.dt,
        #                     rudder_angle=rudder_angle,
        #                     sail_angle=sail_angle,
        #                     time=state.time)
        # k4_rates, _ = get_derivatives(state_k4, state.rudder_angle, state.sail_angle)
        #
        # # Final RK4 update
        # new_boat_pos = state.boat_pos + self.dt / 6 * (k1_rates[0] + 2 * k2_rates[0] + 2 * k3_rates[0] + k4_rates[0])
        # new_boat_vel = state.boat_vel + self.dt / 6 * (k1_rates[1] + 2 * k2_rates[1] + 2 * k3_rates[1] + k4_rates[1])
        # new_boat_heading = state.boat_heading + self.dt / 6 * (k1_rates[2] + 2 * k2_rates[2] + 2 * k3_rates[2] + k4_rates[2])
        # new_boat_heading_rate = state.boat_heading_rate + self.dt / 6 * (k1_rates[3] + 2 * k2_rates[3] + 2 * k3_rates[3] + k4_rates[3])
        # """ End RK4 """

        # TODO is it worth adding in acceleration into this?

        new_state = EnvState(boat_pos=new_boat_pos,
                             boat_vel=new_boat_vel,
                             boat_heading=new_boat_heading,
                             boat_heading_rate=new_boat_heading_rate,
                             rudder_angle=rudder_angle,
                             sail_angle=sail_angle,
                             time=state.time + 1,
                             )

        reward = self.reward_function(action, state, new_state, key)

        # TODO same calcs are in get obs and reward and done, can we combine?

        info = {"discount": self.discount(new_state),
                "sail_angle": true_sail_angle,
                "sail_force_x": sail_x,
                "sail_force_y": sail_y,
                "centreboard_force_x": cboard_x,
                "centreboard_force_y": cboard_y,
                "hydro_separation": hydro_separation,
                }

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                jnp.array(reward),
                self.is_done(new_state),
                info,
                )

    @staticmethod
    def boat_to_global(boat_heading, boat_val):
        global_x = boat_val[0] * jnp.cos(boat_heading) - boat_val[1] * jnp.sin(boat_heading)
        global_y = boat_val[1] * jnp.cos(boat_heading) + boat_val[0] * jnp.sin(boat_heading)
        return jnp.array((global_x.squeeze(), global_y.squeeze()))

    @staticmethod
    def global_to_boat(boat_heading, global_val):
        boat_x = global_val[0] * jnp.cos(boat_heading) + global_val[1] * jnp.sin(boat_heading)
        boat_y = global_val[1] * jnp.cos(boat_heading) - global_val[0] * jnp.sin(boat_heading)
        return jnp.array((boat_x.squeeze(), boat_y.squeeze()))

    @staticmethod
    def unit_vector(angle):  # TODO check this func
        return jnp.array((jnp.sin(angle), jnp.cos(angle)))

    @staticmethod
    def perpendicular(angle):  # TODO check this func
        return jnp.array((-angle[1], angle[0]))

    def angle_to_wind(self, heading):  # TODO check this func
        angle_diff = heading - self.wind_dir
        return (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi  # Ensure the angle difference is between -pi and pi

    def angle_to_mark(self, state):  # TODO check this func
        abs_angle = jnp.arctan2(self.marks[0, 0] - state.boat_pos[0], self.marks[0, 1] - state.boat_pos[1])
        # TODO hardcoded just to do the first of the marks
        relative_angle = abs_angle - state.boat_heading
        normalised = (relative_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return normalised

    def dist_to_mark(self, state):  # TODO check this func
        return self.marks[0] - state.boat_pos  # TODO hardcoded just to do the first of the marks

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # init_state = jrandom.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        init_pos = self.init_pos
        init_dir = jnp.radians(40)
        init_boat_vel = jnp.array((0.0, 0.0))
        state = EnvState(boat_pos=init_pos,
                         boat_vel=init_boat_vel.squeeze(),
                         boat_heading=init_dir,
                         boat_heading_rate=jnp.zeros(()),
                         rudder_angle=jnp.radians(0),
                         sail_angle=jnp.radians(10),
                         time=0,
                         )
        return self.get_obs(state), state

    def reward_function(self,
                        input_action_t: Union[jnp.int_, jnp.float_, chex.Array],
                        state_t: EnvState,
                        state_tp1: EnvState,
                        key: chex.PRNGKey = None,
                        ) -> chex.Array:
        done_x = jax.lax.select(jnp.logical_or(state_tp1.boat_pos[0] < 0, state_tp1.boat_pos[0] > self.screen_width),
                                jnp.array(True), jnp.array(False))
        done_y = jax.lax.select(jnp.logical_or(state_tp1.boat_pos[1] < 0, state_tp1.boat_pos[1] > self.screen_height),
                                jnp.array(True), jnp.array(False))
        done_boundaries = jnp.logical_or(done_x, done_y)
        done_time = jax.lax.select(state_tp1.time >= 3000, jnp.array(True), jnp.array(False))
        overall_done = jnp.logical_or(done_time, done_boundaries)
        # reward_dist = -jnp.linalg.norm(self.dist_to_mark(state_tp1), 8)#  / jnp.sqrt(jnp.square(self.screen_width) + jnp.square(self.screen_height))
        reward_dist = 1.0 * (jnp.linalg.norm(self.dist_to_mark(state_t), 8) - jnp.linalg.norm(self.dist_to_mark(state_tp1), 8))
        # reward_dist = -0.001 * (jnp.linalg.norm(self.dist_to_mark(state_t), 2))
        reward = jax.lax.select(overall_done, -100.0, reward_dist)

        return reward

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, jnp.array((self.min_rudder, self.min_sail)), jnp.array((self.max_rudder, self.max_sail))).squeeze()

    def get_obs(self, state, key: chex.PRNGKey = None) -> chex.Array:  # TODO sort this out
        boat_speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_heading))
        angle_to_wind = self.angle_to_wind(state.boat_heading)
        angle_to_mark = self.angle_to_mark(state)
        dist_to_mark = self.dist_to_mark(state)
        obs = jnp.array([boat_speed,
                         angle_to_wind,
                         state.rudder_angle,
                         state.sail_angle,
                         state.boat_heading_rate,
                         angle_to_mark,
                         jnp.linalg.norm(dist_to_mark),
                        ])
        return obs

    def get_state(self, obs: chex.Array) -> EnvState:
        raise NotImplementedError

    def is_done(self, state: EnvState) -> chex.Array:
        # """Check whether state is terminal."""
        dist_to_mark = self.dist_to_mark(state)
        done_dist = jax.lax.select(jnp.linalg.norm(dist_to_mark) <= 1, jnp.array(True), jnp.array(False))
        done_time = jax.lax.select(state.time >= 3000, jnp.array(True), jnp.array(False))
        done_x = jax.lax.select(jnp.logical_or(state.boat_pos[0] < 0, state.boat_pos[0] > self.screen_width),
                                jnp.array(True), jnp.array(False))
        done_y = jax.lax.select(jnp.logical_or(state.boat_pos[1] < 0, state.boat_pos[1] > self.screen_height),
                                jnp.array(True), jnp.array(False))
        done_boundaries = jnp.logical_or(done_x, done_y)
        done_inter = jnp.logical_or(done_dist, done_time)
        done = jnp.logical_or(done_boundaries, done_inter)

        return done

    @staticmethod
    def create_boat_marker():
        import matplotlib.path as mpath
        # A triangle pointing to the right (0 degrees)
        verts = [(-0.5, -0.25),
                 (-0.5, 0.25),
                 (0.25, 0),
                 (-0.5, -0.25),
        ]

        codes = [
            mpath.Path.MOVETO,
            mpath.Path.LINETO,
            mpath.Path.LINETO,
            mpath.Path.CLOSEPOLY,
        ]

        boat_path = mpath.Path(verts, codes)

        return boat_path

    def render_traj(self, trajectory_state: EnvState, info: dict):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.markers as markers

        def unit_circle_to_compass(unit_heading):  # must be in radians
            return jnp.mod(0.5 * jnp.pi - unit_heading, 2 * jnp.pi)

        def compass_to_unit_circle(compass_heading):   # must be in radians
            return jnp.mod(0.5 * jnp.pi - compass_heading, 2 * jnp.pi)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(self.name)
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(0, self.screen_height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        ax.set_facecolor((0.8, 1.0, 1.0))
        # ax.grid(True)

        # Draw wind angle
        wind_length = 7
        sail_length = 3
        wind_x = self.screen_width / 2
        wind_y = self.screen_height
        dx = float(-wind_length * jnp.cos(unit_circle_to_compass(jnp.radians(self.wind_dir))))
        dy = float(-wind_length * jnp.sin(unit_circle_to_compass(jnp.radians(self.wind_dir))))
        ax.arrow(wind_x, wind_y, dx, dy, width=1, label="Wind Direction")

        boat_marker_path = self.create_boat_marker()

        line, = ax.plot([], [], 'r-', lw=1.5, label='Agent Trail')
        boat_plot, = ax.plot([], [], marker=boat_marker_path, markersize=25, linestyle='None', color='purple', label='Boat')
        sail, = ax.plot([], [], color="black", label='Sail')

        agent_path_history = jnp.expand_dims(jnp.array((self.init_pos[1], self.init_pos[0])), axis=0) # TODO how to sort out this hardcoded values

        # Draw marks
        for i in range(self.marks.shape[0]):
            ax.plot(self.marks[i, 1], self.marks[i, 0], color="orange", marker="o", markersize=12, label='Marks')

        # # Draw Speed Text
        # font = pygame.font.Font(None, 30)
        # speed_in_fwd_dir = state.boat_vel[0] * jnp.sin(state.boat_dir) + state.boat_vel[1] * jnp.cos(state.boat_dir)
        # speed_text = font.render(f"Speed: {jnp.squeeze(speed_in_fwd_dir):.2f} knots", True, (0, 0, 0))
        # screen.blit(speed_text, (10, 10))

        def update(frame):
            nonlocal agent_path_history

            y = jnp.expand_dims(trajectory_state.boat_pos[frame, 0], axis=0)
            x = jnp.expand_dims(trajectory_state.boat_pos[frame, 1], axis=0)

            # TODO need to add boat heading to the image

            if  x == self.init_pos[0] and y == self.init_pos[1]:  # TODO figure out how to sort out this hardcoded fix
                agent_path_history = jnp.expand_dims(jnp.array((self.init_pos[1], self.init_pos[0])), axis=0)
            else:
                xy = jnp.concatenate((jnp.expand_dims(x, 0), jnp.expand_dims(y, 0)), axis=-1)
                agent_path_history = jnp.concatenate((agent_path_history, xy), axis=0)

            t = markers.MarkerStyle(marker=boat_marker_path)
            t._transform = t.get_transform().rotate_deg(jnp.degrees(unit_circle_to_compass(trajectory_state.boat_heading[frame])))

            # Update the boat's data and transformation
            boat_plot.set_data(x, y)
            boat_plot.set_marker(t)

            sail_angle_global = unit_circle_to_compass(trajectory_state.boat_heading[frame] + jnp.pi + info["sail_angle"][frame])
            sail.set_data([x, x + sail_length * jnp.cos(sail_angle_global)],
                          [y, y + sail_length * jnp.sin(sail_angle_global)])

            line.set_data(agent_path_history[:, 0], agent_path_history[:, 1])

            reward = self.reward_function(jnp.zeros(1,),
                                         jax.tree.map(lambda x: x[frame], trajectory_state),
                                         jax.tree.map(lambda x: x[frame+1], trajectory_state),
                                         jrandom.key(42))
            # TODO can we just feed in the usual reward?

            # ax.set_title(f"Reward = {reward:.3f}, Sail FX = {info['sail_force_x'][frame]:.2f}, Sail FY = {info['sail_force_y'][frame]:.2f}, CB FX = {info['centreboard_force_x'][frame]:.2f}, CB FY = {info['centreboard_force_y'][frame]:.2f}")
            boat_vel = jnp.sqrt(trajectory_state.boat_vel[frame, 0] ** 2 + trajectory_state.boat_vel[frame, 1] ** 2)
            ax.set_title(f"Boat Vel = {boat_vel:.3f}, Sail FX = {info['sail_force_x'][frame]:.2f}, Sail FY = {info['sail_force_y'][frame]:.2f}, CB FX = {info['centreboard_force_x'][frame]:.2f}, CB FY = {info['centreboard_force_y'][frame]:.2f}")

            return line, boat_plot, sail

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=trajectory_state.time.shape[0],
                                       interval=self.dt * 1000,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"../animations/{self.name}.gif")
        plt.close()

    @property
    def name(self) -> str:
        return "SailingEnv-v0"

    def action_space(self) -> spaces.Box:
        low = jnp.array((-self.min_rudder,
                         self.min_sail))
        hi = jnp.array((-self.max_rudder,
                         self.max_sail))
        return spaces.Box(low, hi, (2,), dtype=jnp.float32)

    def observation_space(self) -> spaces.Box:
        max_dist = jnp.sqrt(jnp.square(self.screen_width) + jnp.square(self.screen_height))
        # TODO sort out the above to be a bit better
        low = jnp.array([0.0,
                         -jnp.pi,
                         self.min_rudder,
                         self.min_sail,
                         0.0,
                         -jnp.pi,
                         0.0,
                         ])
        high = jnp.array([100,  # TODO random for now
                          jnp.pi,
                          self.max_rudder,
                          self.max_sail,
                          100,  # TODO check this is correct
                          jnp.pi,
                          max_dist,
                          ])
        return spaces.Box(-low, high, (7,), dtype=jnp.float32)


class SailingEnvCSDA(SailingEnvCSCA):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.action_array: chex.Array = jnp.array((0.0, 1.0, -1.0))

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return self.action_array[action] * self.max_action

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_array))

if __name__ == '__main__':
    print(jax.extend.backend.get_backend().platform)

    with jax.disable_jit(disable=False):
        key = jrandom.key(42)

        env = SailingEnvCSCA()

        with jax.disable_jit(disable=False):
            key, _key = jrandom.split(key)
            obs, state = env.reset(_key)

            time_steps = 200#0#0#0

            def _step(runner_state, unused):
                obs, state, key = runner_state
                key, _key = jrandom.split(key)
                # action = env.action_space().sample(_key)
                action = jnp.zeros(2,)
                # action = jnp.ones(2,)
                action = jnp.array((0.1, jnp.radians(10)))

                key, _key = jrandom.split(key)
                nobs, delta_obs, nstate, reward, done, info = env.step(action, state, _key)

                return (nobs, nstate, key), (state, info)

            _, (traj_state, traj_info) = jax.lax.scan(_step, (obs, state, key), None, time_steps)
        # env.render_traj(traj_state, traj_info)

        forward_speed = traj_state.boat_vel[:, 0]
        leeway = traj_state.boat_vel[:, 1]
        hydro_sep = traj_info["hydro_separation"]
        time = traj_state.time
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0, 0].plot(time, forward_speed, color="blue", label="Forward")
        axs[0, 0].plot(time, leeway, color="red", label="Leeway")
        axs[0, 0].plot(time, hydro_sep, color="green", label="Hydro Separation")
        # plt.xlim(0, 3 // env.dt)
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].legend()

        axs[1, 0].plot(time, traj_state.boat_heading, color="purple", label="Boat Heading")
        axs[1, 0].plot(time, traj_state.boat_heading_rate, color="pink", label="Boat Heading Rate")
        axs[1, 0].set_ylim(0, 10)
        axs[1, 0].legend()

        axs[0, 1].plot(traj_state.boat_pos[:, 1], traj_state.boat_pos[:, 0])
        axs[0, 1].set_xlabel("X")
        axs[0, 1].set_ylabel("Y")
        axs[0, 1].set_xlim(0, 100)
        axs[0, 1].set_ylim(0, 100)

        # boat_pos = new_boat_pos,
        # boat_vel = new_boat_vel,
        # boat_heading = new_boat_heading,
        # boat_heading_rate = new_boat_heading_rate,

        plt.tight_layout()
        plt.show()

        print(forward_speed)
        print(hydro_sep)

