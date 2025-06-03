from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
from typing import Dict, Tuple, Any, Union, Optional


@struct.dataclass
class EnvState(base_env.EnvState):
    boat_pos: jnp.ndarray
    boat_vel: jnp.ndarray
    boat_heading: jnp.ndarray
    boat_heading_rate: jnp.ndarray
    time: int


class SailingEnvCSCA(base_env.BaseEnvironment):
    """
    0 degrees is the top of the screen or defined as north
    wind x y is global
    boat has a local x y
    """
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.1

        self.max_steps_in_episode: int = 500
        self.max_action: float = 1.0
        self.mass: float = 120.0  # in kg  # used to be 120

        self.wind_dir: float = 180.0  # deg
        self.wind_speed: float = 5.0  # in ms^-1
        self.wind_vel: jnp.ndarray = self.wind_speed * jnp.array((jnp.sin(jnp.radians(self.wind_dir)),
                                                                  jnp.cos(jnp.radians(self.wind_dir))))  # in ms^-1

        self.mass: float = 350.0  # in kg  # used to be 120
        self.max_rudder_angle = jnp.radians(45.0)
        self.water_density = 1000  # [kg/m^3]
        self.rudder_stretching = 2.326923076923077
        self.rudder_blade_area = 0.13  # [m^2]
        self.air_density =  1.3 # [kg/m^3]
        self.air_viscosity = 0.0000171 # [Pa * s]
        self.sail_length = 1 # [m]
        self.sail_stretching = 0.961
        self.sail_area = 6.2 # [m]
        self.water_viscosity = 0.0000001 # [Pa *
        self.keel_length = 2 # [m]
        self.keel_stretching = 0.605
        self.lateral_area = 2.5 # [m^2]
        self.hull_speed = 2.5
        self.distance_cog_rudder = 1.24 # [m]
        self.distance_cog_sail_pressure_point = 0.24 # [m]
        self.distance_mast_sail_pressure_point = 0.68 # [m]
        self.distance_cog_keel_pressure_point = 0.24 # [m]
        self.distance_cog_keel_middle = self.distance_cog_keel_pressure_point - .7

        self.along_damping = 15
        self.transverse_damping = 5
        self.yaw_timeconstant = 5
        self.moi_z = 1066

        self.damping_invariant_x = -self.mass / self.along_damping
        self.damping_invariant_y = -self.mass / self.transverse_damping
        self.damping_invariant_yaw = -(self.moi_z / self.yaw_timeconstant)
        self.wave_impedance_invariant = (self.water_density / 2) * self.lateral_area

        self.screen_width: int = 100 # in m
        self.screen_height: int = 100 # in m

        self.marks: jnp.ndarray = jnp.array(((self.screen_width/2, 80),))
        # TODO to deal with multiple marks, could jnp.roll once done a conditional
        # self.reward_gate: jnp.ndarray = jnp.array((10, 10))

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        # 1) Get important values
        speed = jnp.sqrt(state.boat_vel[0] ** 2 + state.boat_vel[1] ** 2)

        transformed_x = self.wind_vel[0] * jnp.sin(state.boat_heading) - self.wind_vel[1] * jnp.cos(state.boat_heading)
        transformed_y = self.wind_vel[1] * jnp.sin(state.boat_heading) + self.wind_vel[0] * jnp.cos(state.boat_heading)
        # TODO check the positive and negatives for vy

        apparent_wind_x = transformed_x - state.boat_vel[0]
        apparent_wind_y = transformed_y - state.boat_vel[1]
        apparent_wind_angle = jnp.arctan2(-apparent_wind_y, -apparent_wind_x)
        apparent_wind_speed = jnp.sqrt(apparent_wind_x ** 2 + apparent_wind_y ** 2)

        # # 2) Update boat heading based on action
        action = self.action_convert(input_action)
        #
        # rudder_angle = jnp.clip(action, -self.max_rudder_angle, self.max_rudder_angle)
        # TODO assume that can just change rudder as desired, there is no latency involved

        sail_angle = jnp.radians(45.0)  # can range between -90 and +90 generally but only needs to be between 0 and 90

        # # 3) Calc rudder force
        # pressure = (self.water_density / 2) * speed ** 2
        # rudder_force_x = -(((4 * jnp.pi) / self.rudder_stretching) * rudder_angle ** 2) * pressure * self.rudder_blade_area
        # rudder_force_y = 2 * jnp.pi * pressure * self.rudder_blade_area * rudder_angle

        true_sail_angle = jnp.sign(apparent_wind_angle) * abs(sail_angle)
        aoa_old = apparent_wind_angle - true_sail_angle

        true_sail_angle = sail_angle  # TODO sort the above out at some point, the above is old and just tryna figure it out
        aoa = -apparent_wind_angle - true_sail_angle

        # 4) Calc sail force
        aoa = jax.lax.select(aoa * true_sail_angle < 0, 0.0, aoa)
        # if aoa * true_sail_angle < 0:
        #     aoa = 0

        eff_aoa = aoa  # eff_aoa : effective angle of attack
        eff_aoa = jnp.where(aoa < -jnp.pi / 2, jnp.pi + aoa, eff_aoa)
        eff_aoa = jnp.where(aoa > jnp.pi / 2, -jnp.pi + aoa, eff_aoa)
        # TODO is the above correct hmmm
        # if aoa < -jnp.pi / 2:
        #     eff_aoa = jnp.pi + aoa
        # elif aoa > jnp.pi / 2:
        #     eff_aoa = -jnp.pi + aoa

        pressure = (self.air_density / 2) * apparent_wind_speed ** 2 * jnp.cos(jnp.cos(true_sail_angle)) ** 2

        aero_friction = jax.lax.select(apparent_wind_speed != 0,
                                       3.55 * jnp.sqrt(self.air_viscosity / (apparent_wind_speed * self.sail_length)),
                                       0.0)

        # # # With separation ##########################################################################################
        # separation = 1 - jnp.exp(-((abs(eff_aoa)) / (jnp.pi / 180 * 25)) ** 2)
        #
        # coeff_lift = 2 * jnp.pi * eff_aoa
        # coeff_drag = aero_friction + (4 * jnp.pi * eff_aoa ** 2 * separation) / self.sail_stretching
        #
        # propulsion = pressure * self.sail_area * (coeff_lift * jnp.sin(apparent_wind_angle) - coeff_drag * jnp.cos(apparent_wind_angle))
        # transverse_force = pressure * self.sail_area * (-coeff_lift * jnp.cos(apparent_wind_angle) - coeff_drag * jnp.sin(apparent_wind_angle))
        #
        # separated_propulsion = jnp.sign(aoa) * pressure * self.sail_area * jnp.sin(aoa) ** 2 * jnp.sin(true_sail_angle)
        # separated_transverse_force = -jnp.sign(aoa) * pressure * self.sail_area * jnp.sin(aoa) ** 2 * jnp.cos(true_sail_angle)
        #
        # sail_force_x = (1 - separation) * propulsion + separation * separated_propulsion
        # sail_force_y = (1 - separation) * transverse_force + separation * separated_transverse_force
        ################################################################################################################

        # # # Without separation #######################################################################################
        coeff_lift = 2 * jnp.pi * eff_aoa
        coeff_drag = aero_friction + coeff_lift ** 2 / (jnp.pi * self.sail_stretching)

        propulsion = pressure * self.sail_area * (jnp.sin(apparent_wind_angle) * coeff_lift - jnp.cos(apparent_wind_angle) * coeff_drag)
        transverse_force = pressure * self.sail_area * -(jnp.sin(apparent_wind_angle) * coeff_drag + jnp.cos(apparent_wind_angle) * coeff_lift)

        sail_force_x = propulsion
        sail_force_y = transverse_force
        ################################################################################################################

        # 5) Calculate lateral force
        hydro_friction = jax.lax.select(speed != 0,
                                        2.66 * jnp.sqrt(self.water_viscosity / (speed * self.keel_length)),
                                        0.0)

        #     aoa :           angle of attack
        # eff_aoa : effective angle of attack
        eff_aoa = aoa = jnp.arctan2(state.boat_vel[1], state.boat_vel[0])
        eff_aoa = jnp.where(aoa < -jnp.pi / 2, jnp.pi + aoa, eff_aoa)
        eff_aoa = jnp.where(aoa > jnp.pi / 2, -jnp.pi + aoa, eff_aoa)
        # TODO is the above correct hmmm
        # if aoa < -jnp.pi / 2:
        #     eff_aoa = jnp.pi + aoa
        # elif aoa > jnp.pi / 2:
        #     eff_aoa = -jnp.pi + aoa

        lateral_separation = 1 - jnp.exp(-((abs(eff_aoa)) / (jnp.pi / 180 * 25)) ** 2)

        # Identical calculation for x and y
        tmp = -(hydro_friction + (4 * jnp.pi * eff_aoa ** 2 * lateral_separation) / self.keel_stretching)

        separated_transverse_force = -jnp.sign(aoa) * pressure * self.sail_area * jnp.sin(aoa) ** 2

        lateral_force_x = (1 - lateral_separation) * (tmp * jnp.cos(aoa) + 2 * jnp.pi * eff_aoa * jnp.sin(aoa)) * pressure * self.lateral_area
        lateral_force_y = (1 - lateral_separation) * (tmp * jnp.sin(aoa) - 2 * jnp.pi * eff_aoa * jnp.cos(aoa)) * pressure * self.lateral_area + lateral_separation * separated_transverse_force

        # # 6) Calc damping
        # damping_x = self.damping_invariant_x * state.boat_vel[0]
        # damping_y = self.damping_invariant_y * state.boat_vel[1]
        # damping_yaw = self.damping_invariant_yaw * state.boat_heading_rate
        #
        # # 7) Calc wave impedance
        # wave_impedance = -jnp.sign(state.boat_vel[0]) * speed**2 * (speed / self.hull_speed)**2 * self.wave_impedance_invariant

        delta_pos_x = state.boat_vel[0] * jnp.sin(state.boat_heading) - state.boat_vel[1] * jnp.cos(state.boat_heading)
        delta_pos_y = state.boat_vel[1] * jnp.sin(state.boat_heading) + state.boat_vel[0] * jnp.cos(state.boat_heading)
        # TODO check the positive and negatives for vy

        delta_yaw = state.boat_heading_rate  # * jnp.sin(roll) + pitch_rate * jnp.cos(roll)

        # delta_vel_x = delta_yaw * state.boat_vel[1] + (sail_force_x + lateral_force_x + rudder_force_x + damping_x + wave_impedance) / self.mass
        delta_vel_x = delta_yaw * state.boat_vel[1] + (sail_force_x + lateral_force_x) / self.mass
        # delta_vel_y = -delta_yaw * state.boat_vel[0] + ((sail_force_y + lateral_force_y + rudder_force_y) + damping_y) / self.mass
        delta_vel_y = -delta_yaw * state.boat_vel[0] + (sail_force_y + lateral_force_y) / self.mass

        # delta_yaw_rate = (damping_yaw
        #                   # + hydrostatic_force.z * hydrostatic_force.x * jnp.cos(roll)
        #                   - rudder_force_y * self.distance_cog_rudder
        #                   + sail_force_y * self.distance_cog_sail_pressure_point
        #                   + sail_force_x * jnp.cos(true_sail_angle) * self.distance_mast_sail_pressure_point
        #                   + lateral_force_y * (self.distance_cog_keel_pressure_point * (1 - lateral_separation) + self.distance_cog_keel_middle * lateral_separation)) / self.moi_z

        # 8) Apply differential step
        new_boat_pos = state.boat_pos + jnp.array((delta_pos_x, delta_pos_y)) * self.dt
        # new_boat_heading = state.boat_heading + delta_yaw * self.dt
        new_boat_vel = state.boat_vel + jnp.array((delta_vel_x, delta_vel_y)) * self.dt
        # new_boat_heading_rate = state.boat_heading_rate + delta_yaw_rate * self.dt

        # TODO do I have to normalise heading to ensure between a set range

        # Update state dict and evaluate termination conditions
        new_state = EnvState(boat_pos=new_boat_pos,
                             boat_heading=state.boat_heading,
                             boat_vel=new_boat_vel,
                             boat_heading_rate=state.boat_heading_rate,
                             time=state.time + 1,
                             )

        reward = self.reward_function(action, state, new_state, key)

        # TODO same calcs are in get obs and reward and done, can we combine?

        return (jax.lax.stop_gradient(self.get_obs(new_state)),
                jax.lax.stop_gradient(self.get_obs(new_state) - self.get_obs(state)),
                jax.lax.stop_gradient(new_state),
                jnp.array(reward),
                self.is_done(new_state),
                {"discount": self.discount(new_state)},
                )

    @staticmethod
    def vector_decomp(magnitude, angle):
        return magnitude * jnp.array((jnp.sin(angle), jnp.cos(angle)))

    @staticmethod
    def unit_vector(angle):
        return jnp.array((jnp.sin(angle), jnp.cos(angle)))

    @staticmethod
    def perpendicular(angle):
        return jnp.array((-angle[1], angle[0]))

    def angle_to_wind(self, heading):
        angle_diff = heading - self.wind_dir
        return (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi  # Ensure the angle difference is between -pi and pi

    def angle_to_mark(self, state):
        abs_angle = jnp.arctan2(self.marks[0, 0] - state.boat_pos[0], self.marks[0, 1] - state.boat_pos[1])
        # TODO hardcoded just to do the first of the marks
        relative_angle = abs_angle - state.boat_heading
        normalised = (relative_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return normalised

    def dist_to_mark(self, state):
        return self.marks[0] - state.boat_pos  # TODO hardcoded just to do the first of the marks

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # init_state = jrandom.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        init_pos = jnp.array(((self.screen_width/2,), (25,)))
        init_dir = jnp.radians(jnp.ones(1,) * 90)
        init_boat_vel = jnp.array((1.0, 0.0))
        state = EnvState(boat_pos=init_pos.squeeze(),
                         boat_vel=init_boat_vel.squeeze(),
                         boat_heading=init_dir.squeeze(),
                         boat_heading_rate=jnp.zeros(()),
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
        return jnp.clip(action, -self.max_action, self.max_action).squeeze()

    def get_obs(self, state, key: chex.PRNGKey = None) -> chex.Array:
        boat_speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_heading))
        angle_to_wind = self.angle_to_wind(state.boat_heading)
        angle_to_mark = self.angle_to_mark(state)
        dist_to_mark = self.dist_to_mark(state)
        obs = jnp.array([boat_speed,
                         angle_to_wind,
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

    def render_traj(self, trajectory_state: EnvState):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

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
        wind_x = self.screen_width / 2
        wind_y = self.screen_height
        dx = wind_length * jnp.sin(self.wind_dir)
        dy = wind_length * -jnp.cos(self.wind_dir)
        ax.arrow(wind_x, wind_y, dx, dy, width=1, label="Wind Direction")

        line, = ax.plot([], [], 'r-', lw=1.5, label='Agent Trail')
        boat, = ax.plot([], [], color="purple", marker="o", markersize=12, label='Current State')

        agent_path_history = jnp.array(((self.screen_width/2,), (25.0,)))  # TODO how to sort out this hardcoded values

        # Draw marks
        for i in range(self.marks.shape[0]):
            ax.plot(self.marks[i, 0], self.marks[i, 1], color="orange", marker="o", markersize=12, label='Marks')

        # # Draw Speed Text
        # font = pygame.font.Font(None, 30)
        # speed_in_fwd_dir = state.boat_vel[0] * jnp.sin(state.boat_dir) + state.boat_vel[1] * jnp.cos(state.boat_dir)
        # speed_text = font.render(f"Speed: {jnp.squeeze(speed_in_fwd_dir):.2f} knots", True, (0, 0, 0))
        # screen.blit(speed_text, (10, 10))

        def update(frame):

            global agent_path_history

            # boat_angle = jnp.squeeze(state.boat_dir[frame])
            x, y = (jnp.expand_dims(trajectory_state.boat_pos[frame, 0], axis=0),
                    jnp.expand_dims(trajectory_state.boat_pos[frame, 1], axis=0))

            if x == self.screen_width/2 and y == 25.0:  # TODO figure out how to sort out this hardcoded fix
                agent_path_history = jnp.array(((self.screen_width/2,), (25.0,)))
            else:
                xy = jnp.concatenate((jnp.expand_dims(x, 0), jnp.expand_dims(y, 0)))
                agent_path_history = jnp.concatenate((agent_path_history, xy), axis=-1)

            boat.set_data(x, y)

            line.set_data(agent_path_history[0], agent_path_history[1])

            reward = self.reward_function(jnp.zeros(1,),
                                         jax.tree.map(lambda x: x[frame], trajectory_state),
                                         jax.tree.map(lambda x: x[frame+1], trajectory_state),
                                         jrandom.key(42))
            ax.set_title(f"Reward = {reward:.3f}")

            return line, boat

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
        return spaces.Box(-self.max_action, self.max_action, (1,), dtype=jnp.float32)

    def observation_space(self) -> spaces.Box:
        max_dist = jnp.sqrt(jnp.square(self.screen_width) + jnp.square(self.screen_height))
        # TODO sort out the above to be a bit better
        low = jnp.array([0.0,
                         -jnp.pi,
                         0.0,
                         -jnp.pi,
                         0.0,
                         ])
        high = jnp.array([self.max_speed,
                          jnp.pi,
                          self.acceleration,  # TODO check this is correct
                          jnp.pi,
                          max_dist,
                          ])
        return spaces.Box(-low, high, (5,), dtype=jnp.float32)


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
    with jax.disable_jit(disable=False):
        key = jrandom.PRNGKey(42)

        env = SailingEnvCSCA()

        with jax.disable_jit(disable=True):
            key, _key = jrandom.split(key)
            obs, state = env.reset(_key)

            time_steps = 80#0#0#0

            def _step(runner_state, unused):
                obs, state, key = runner_state
                key, _key = jrandom.split(key)
                # action = env.action_space().sample(_key)
                action = jnp.zeros(1,)
                # action = jnp.ones(1,)

                key, _key = jrandom.split(key)
                nobs, delta_obs, nstate, reward, done, _ = env.step(action, state, _key)

                return (nobs, nstate, key), state


            _, traj_state = jax.lax.scan(_step, (obs, state, key), None, time_steps)
        env.render_traj(traj_state)

