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
    boat_dir: jnp.ndarray
    boat_angular_acc: jnp.ndarray
    boat_angular_vel: jnp.ndarray
    boat_vel: jnp.ndarray
    time: int


class SailingEnvCSCA(base_env.BaseEnvironment):
    """
    0 degrees is the top of the screen or defined as north
    """
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 0.1

        self.max_steps_in_episode: int = 500
        self.wind_dir: float = 0.0
        self.wind_vel: jnp.ndarray = jnp.array((0.0, -7.0)) # in ms^-1
        self.max_action: float = 1.0
        self.max_speed: float = 5.0  # Max boat speed in ms^-1
        self.max_angular_vel: float = jnp.pi / 2.0  # Max angular velocity (rad/s), roughly 90 deg/s
        self.max_angular_acc: float = jnp.pi  # Max angular acceleration (rad/s^2), roughly 180 deg/s^2
        self.max_speed: float = 3.0  # in ms^-1
        self.acceleration: float = 10.0  # in ms^-2
        self.deceleration: float = 20.0  # in ms^-2
        # TODO adjust the accels above

        self.mass: float = 120.0  # in kg

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
        # 1. Update boat heading based on action
        action = self.action_convert(input_action)

        RUDDER_MAX_FORCE_COEFF: float = 50.0

        rudder_force_magnitude = action * RUDDER_MAX_FORCE_COEFF * (jnp.linalg.norm(state.boat_vel) ** 2)
        angular_acc = rudder_force_magnitude / self.mass
        angular_acc = jnp.clip(angular_acc,
                               -self.max_angular_vel,
                               self.max_angular_vel)

        angular_vel = state.boat_angular_vel + angular_acc * self.dt
        angular_vel = jnp.clip(angular_vel,
                               -self.max_angular_vel,
                               self.max_angular_vel)

        # speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_dir))
        # current_speed_magnitude = jnp.linalg.norm(state.boat_vel)
        # signed_speed = jax.lax.select(speed > 0, current_speed_magnitude, -current_speed_magnitude)
        #
        # angular_acc_due_to_act = action.squeeze() * signed_speed * 0.5  # Scale action effect, adjust constant
        # # TODO if speed is zero then action has no effect, it may be good to have small constant so boat doesn't get "stuck" although this is realistic
        # new_boat_angular_vel = state.boat_angular_acc * 0.97 + angular_acc_due_to_act * self.dt
        # # TODO some decel modifier, maybe better way to state it
        # new_boat_angular_acc = jnp.clip(new_boat_angular_vel,
        #                                 -self.max_angular_vel,
        #                                 self.max_angular_vel)

        new_heading = state.boat_dir + angular_vel * self.dt
        new_heading = jnp.mod(new_heading, 2 * jnp.pi)  # Wrap heading to be within 0 and 2*pi

        unit_heading = self.unit_vector(new_heading)
        unit_perp = self.perpendicular(unit_heading)

        # 2. Calculate the angle between the boat heading and wind direction.
        # angle_diff = self.angle_to_wind(new_heading)

        # 3. Calculate the speed multiplier based on the polar curve.
        # speed_multiplier = self.polar_curve(jnp.abs(angle_diff))  # TODO assuming polar curve is the same on both tacks
        apparent_wind = self.wind_vel - state.boat_vel  # TODO is this correct?
        apparent_wind_speed = jnp.linalg.norm(apparent_wind)
        apparent_wind_angle = jnp.arctan2(apparent_wind[1], apparent_wind[0])
        angle_to_apparent_wind = self.angle_to_wind(apparent_wind_angle)
        sail_forward_coeff = self.polar_curve(angle_to_apparent_wind)
        sail_side_coeff = jnp.zeros(())  # TODO sort this out from polar curve at some point

        # 4. Update boat speed, accounting for acceleration/deceleration.
        SAIL_DRIVE_COEFF = 8.0
        SAIL_SIDE_COEFF = 0.5
        fdrive = sail_forward_coeff * apparent_wind_speed ** 2 * SAIL_DRIVE_COEFF * unit_heading
        fside_sail = sail_side_coeff * apparent_wind_speed ** 2 * SAIL_SIDE_COEFF * jnp.sign(angle_to_apparent_wind) * unit_perp

        v_forward = jnp.dot(state.boat_vel, unit_heading) * unit_heading
        v_perp = state.boat_vel - v_forward

        DRAG_COEFF_FORWARD = 5.0  # Drag along the boat's direction
        DRAG_COEFF_KEEL = 100.0  # High drag perpendicular to boat's direction (from keel)

        fdrag_forward = -v_forward * jnp.linalg.norm(v_forward) * DRAG_COEFF_FORWARD
        fdrag_keel = -v_perp * jnp.linalg.norm(v_perp) * DRAG_COEFF_KEEL

        total_force = fdrive + fside_sail + fdrag_forward + fdrag_keel

        acceleration_vector = total_force / self.mass
        new_boat_vel = state.boat_vel + acceleration_vector * self.dt

        # Optionally clip boat speed
        # current_vel_mag = jnp.linalg.norm(new_boat_vel)
        # new_boat_vel = jax.lax.select(current_vel_mag > self.max_speed,
        #                               new_boat_vel * (self.max_speed / current_vel_mag),
        #                               new_boat_vel)

        # 5. Update boat position based on heading and speed.
        new_boat_pos = state.boat_pos + new_boat_vel * self.dt

        # Update state dict and evaluate termination conditions
        new_state = EnvState(boat_pos=new_boat_pos,
                             boat_dir=new_heading,
                             boat_angular_acc=angular_acc,
                             boat_angular_vel=angular_vel,
                             boat_vel=new_boat_vel,
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

    def polar_curve_tesy(self, angle_to_apparent_wind: float) -> Tuple[float, float]:
        """
        Simulated polar curve. Returns (forward_coeff, side_coeff) based on the
        absolute angle to the apparent wind.
        angle_to_apparent_wind is in [0, pi].

        This version uses jax.lax.cond with correct operand passing for nested conditions.
        """
        # Ensure angle_to_apparent_wind is a scalar.
        angle = angle_to_apparent_wind

        # Define some key angles in radians
        CLOSE_HAULED_ANGLE = jnp.pi / 4.0 # 45 degrees (too close to wind)
        BEAM_REACH_ANGLE = jnp.pi / 2.0 # 90 degrees (best speed)

        # Define the functions for jax.lax.cond
        # Each function takes one argument, which is the 'operand' passed to cond
        # In the outermost cond, the operand is 'angle'.
        # In the inner cond, the operand is also 'angle'.

        def _close_hauled_branch(current_angle): # Receives 'angle' as 'current_angle'
            forward = jnp.array(0.0)
            side = jnp.cos(current_angle) * 0.7
            return forward, side

        def _not_close_hauled_branch(current_angle): # Receives 'angle' as 'current_angle'
            # This branch contains another jax.lax.cond
            def _beam_reach_sub_branch(inner_angle): # Receives 'current_angle' as 'inner_angle'
                forward = jnp.sin(inner_angle * 2) * 0.8
                side = (1 - jnp.sin(inner_angle * 2)) * 0.3
                return forward, side

            def _running_sub_branch(inner_angle): # Receives 'current_angle' as 'inner_angle'
                forward = jnp.sin(inner_angle) * 0.5
                side = jnp.array(0.1)
                return forward, side

            return jax.lax.cond(
                current_angle < BEAM_REACH_ANGLE, # Condition for inner branch
                _beam_reach_sub_branch,           # Function if true
                _running_sub_branch,              # Function if false
                current_angle                     # OPERAND PASSED TO INNER FUNCTIONS
            )

        forward_coeff, side_coeff = jax.lax.cond(
            angle < CLOSE_HAULED_ANGLE, # Condition for outer branch
            _close_hauled_branch,       # Function if true
            _not_close_hauled_branch,   # Function if false
            angle                       # OPERAND PASSED TO OUTER FUNCTIONS
        )

        return forward_coeff, side_coeff

    @staticmethod
    def polar_curve(theta):
        def vel(theta, theta_0=0, theta_dead=jnp.pi / 12):
            return 1 - jnp.exp(-(theta - theta_0) ** 2 / theta_dead)

        def rew(theta, theta_0=0, theta_dead=jnp.pi / 12):
            return vel(theta, theta_0, theta_dead) * jnp.cos(theta)

        def line_2(theta):
            return theta / (theta + 1) * 1.64

        def line_3(theta):
            return theta / (theta - 0.2) * 0.975

        def line_4(theta):
            return theta / (theta - 0.8) * 0.704

        boundaries = jnp.array([0, 7 * jnp.pi / 36, 5 * jnp.pi / 8, 3 * jnp.pi / 4, jnp.pi])
        functions = [rew, line_2, line_3, line_4]

        mask = (theta >= boundaries[:-1]) & (theta < boundaries[1:])

        result = jnp.sum(jnp.stack([jnp.where(mask, f(theta), 0) for mask, f in zip(mask, functions)]), axis=0)
        result = jnp.where(theta == boundaries[-1], functions[-1](theta), result)

        return result

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
        relative_angle = abs_angle - state.boat_dir
        normalised = (relative_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return normalised

    def dist_to_mark(self, state):
        return self.marks[0] - state.boat_pos  # TODO hardcoded just to do the first of the marks

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # init_state = jrandom.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        init_pos = jnp.array(((self.screen_width/2,), (25,)))
        init_dir = jnp.radians(jnp.ones(1,) * 45)
        boat_speed = 1.0
        init_boat_vel = self.vector_decomp(boat_speed, init_dir)
        state = EnvState(boat_pos=init_pos.squeeze(axis=-1),
                         boat_dir=init_dir.squeeze(),
                         boat_angular_acc=jnp.zeros(1,).squeeze(),
                         boat_angular_vel=jnp.zeros(1,).squeeze(),
                         boat_vel=init_boat_vel.squeeze(axis=-1),
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
        boat_speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_dir))
        angle_to_wind = self.angle_to_wind(state.boat_dir)
        angle_to_mark = self.angle_to_mark(state)
        dist_to_mark = self.dist_to_mark(state)
        obs = jnp.array([boat_speed,
                         angle_to_wind,
                         state.boat_angular_acc,
                         state.boat_angular_vel,
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
        ax.set_xlabel("Y (m)")
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

        key, _key = jrandom.split(key)
        obs, state = env.reset(_key)

        time_steps = 100#0#0#0

        def _step(runner_state, unused):
            obs, state, key = runner_state
            key, _key = jrandom.split(key)
            # action = env.action_space().sample(_key)
            # action = jnp.zeros(1,)
            action = jnp.ones(1,)

            key, _key = jrandom.split(key)
            nobs, delta_obs, nstate, reward, done, _ = env.step(action, state, _key)

            return (nobs, nstate, key), state

        with jax.disable_jit(disable=True):
            _, traj_state = jax.lax.scan(_step, (obs, state, key), None, time_steps)
        env.render_traj(traj_state)

