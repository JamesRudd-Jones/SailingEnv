from bifurcagym.envs import base_env
from bifurcagym import spaces
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
from typing import Dict, Tuple, Any, Union, Optional
import time
import pygame
from pygame import gfxdraw


@struct.dataclass
class EnvState(base_env.EnvState):
    boat_pos: jnp.ndarray
    boat_dir: jnp.ndarray
    boat_dir_acc: jnp.ndarray
    boat_vel: jnp.ndarray
    boat_path: jnp.ndarray
    time: int


class SailingEnvCSCA(base_env.BaseEnvironment):
    """
    0 degrees is the top of the screen or defined as north
    """
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)

        self.dt: float = 1.0

        self.max_steps_in_episode: int = 500
        self.wind_dir: float = 0.0
        self.wind_vel: jnp.ndarray = jnp.array((0.0, -50.0)) * self.dt
        self.max_action: float = 0.001
        self.max_heading_vel: float = 300.0 / 360.0 * 2 * jnp.pi * self.dt
        self.max_speed: float = 10.0
        self.acceleration: float = 1.0
        self.deceleration: float = 2.0

        self.mass: float = 3000.0

        self.screen_width: int = 800
        self.screen_height: int = 600
        # boat_path_length: int = 30

        self.marks: jnp.ndarray = jnp.array(((400, 500),))
        # TODO to deal with multiple marks, could jnp.roll once done a conditional
        self.reward_gate: jnp.ndarray = jnp.array((10, 10))

    def step_env(self,
                 input_action: Union[jnp.int_, jnp.float_, chex.Array],
                 state: EnvState,
                 key: chex.PRNGKey,
                 ) -> Tuple[chex.Array, chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        # 1. Update boat heading based on action
        action = self.action_convert(input_action)
        speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_dir))
        sqrtspeed = jax.lax.select(speed > 0,
                                   jnp.sqrt(jnp.linalg.norm(state.boat_vel)),
                                   -jnp.sqrt(jnp.linalg.norm(state.boat_vel)))
        new_boat_dir_acc = state.boat_dir_acc * 0.97  # TODO some decel modifier, maybe better way to state it
        new_boat_dir_acc = jnp.clip(new_boat_dir_acc + action.squeeze() * sqrtspeed,
                                      -self.max_heading_vel,
                                      self.max_heading_vel)
        new_heading = state.boat_dir + new_boat_dir_acc
        new_heading = jnp.mod(new_heading, 2 * jnp.pi)  # Wrap heading to be within 0 and 2*pi

        fcentripetal = new_boat_dir_acc * self.mass

        unit_heading_2 = self.unit_vector(new_heading)
        unit_perp_2 = self.perpendicular(unit_heading_2)

        # 2. Calculate the angle between the boat heading and wind direction.
        angle_diff = self.angle_to_wind(new_heading)

        # 3. Calculate the speed multiplier based on the polar curve.
        speed_multiplier = self.polar_curve(jnp.abs(angle_diff))  # TODO assuming polar curve is the same on both tacks
        apparent_wind_2 = self.wind_vel - state.boat_vel
        apparent_wind_speed = jnp.linalg.norm(apparent_wind_2)

        # 4. Update boat speed, accounting for acceleration/deceleration.
        SAILCOEFF = 7.0
        fdrive_2 = speed_multiplier * apparent_wind_speed * SAILCOEFF * unit_heading_2

        vforward_2 = jnp.dot(state.boat_vel, unit_heading_2) * unit_heading_2
        vperpendicular_2 = state.boat_vel - vforward_2

        fdrag_2 = -vforward_2 * jnp.linalg.norm(vforward_2) * 100.0  # opposite to direction of movement
        fkeel_2 = -vperpendicular_2 * jnp.linalg.norm(vperpendicular_2) * 1200.0
        fperp_2 = unit_perp_2 * fcentripetal * jnp.linalg.norm(state.boat_vel)

        new_boat_vel_2 = state.boat_vel + (fdrive_2 + fdrag_2 + fkeel_2 + fperp_2) / self.mass

        # 5. Update boat position based on heading and speed.
        new_boat_pos_2 = state.boat_pos + new_boat_vel_2 * self.dt

        # 6. Update boat path
        old_path = jnp.roll(state.boat_path, 1, axis=-1)
        new_boat_path = old_path.at[:, 0].set(new_boat_pos_2)

        # Update state dict and evaluate termination conditions
        new_state = EnvState(boat_pos=new_boat_pos_2,
                         boat_dir=new_heading,
                         boat_dir_acc=new_boat_dir_acc,
                         boat_vel=new_boat_vel_2,
                         boat_path=new_boat_path,
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
        # Ensure the angle difference is between -pi and pi
        return (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def angle_to_mark(self, state):
        abs_angle = jnp.arctan2(self.marks[0, 0] - state.boat_pos[0], self.marks[0, 1] - state.boat_pos[1])
        relative_angle = abs_angle - state.boat_dir
        normalised = (relative_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return normalised

    def dist_to_mark(self, state):
        return self.marks[0] - state.boat_pos[0]

    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # init_state = jrandom.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        init_pos = jnp.array(((400.0,), (100.0,)))
        init_dir = jnp.radians(jnp.ones(1,) * 90)
        boat_speed = 1
        init_boat_vel = self.vector_decomp(boat_speed, init_dir)
        state = EnvState(boat_pos=init_pos.squeeze(axis=-1),
                         boat_dir=init_dir.squeeze(),
                         boat_dir_acc=jnp.zeros(1,).squeeze(),
                         boat_vel=init_boat_vel.squeeze(axis=-1),
                         # boat_path=jnp.repeat(init_pos, self.boat_path_length, axis=1),
                         boat_path=jnp.repeat(init_pos, 40, axis=1),
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
        # reward_dist = -jnp.linalg.norm(self.dist_to_mark(state, self), 8)#  / jnp.sqrt(jnp.square(self.screen_width) + jnp.square(self.screen_height))
        reward_dist = jnp.linalg.norm(self.dist_to_mark(state_t), 8) - jnp.linalg.norm(self.dist_to_mark(state_tp1), 8)
        reward = jax.lax.select(overall_done, -100.0, reward_dist)

        return reward

    def action_convert(self,
                       action: Union[jnp.int_, jnp.float_, chex.Array]) -> Union[jnp.int_, jnp.float_, chex.Array]:
        return jnp.clip(action, -self.max_action, self.max_action)

    def get_obs(self, state, key: chex.PRNGKey = None) -> chex.Array:
        boat_speed = jnp.dot(state.boat_vel, self.unit_vector(state.boat_dir))
        angle_to_wind = self.angle_to_wind(state.boat_dir)
        angle_to_mark = self.angle_to_mark(state)
        dist_to_mark = self.dist_to_mark(state)
        obs = jnp.array([boat_speed,
                         angle_to_wind,
                         state.boat_dir_acc,
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
        # ax.set_xlabel("X")
        ax.set_ylim(0, self.screen_height)
        # ax.set_ylabel("Y")
        # ax.set_aspect('equal')
        # ax.grid(True)

        # Draw wind angle
        wind_length = 50
        wind_x = self.screen_width / 2
        wind_y = self.screen_height
        end_x = wind_x + wind_length + jnp.sin(self.wind_dir)
        end_y = wind_y + wind_length - jnp.cos(self.wind_dir)
        ax.plot([wind_x, end_x], [wind_y, end_y])

        line, = ax.plot([], [], 'r-', lw=1.5, label='Agent Trail')
        dot, = ax.plot([], [], color="purple", marker="o", markersize=12, label='Current State')

        # # Draw marks
        # for i in range(self.marks.shape[0]):
        #     x, y = to_screen_coords(self.marks[i, 0], self.marks[i, 1])
        #     gfxdraw.aacircle(screen, x, y, 4, (0, 0, 0))
        #     gfxdraw.filled_circle(screen, x, y, 4, (0, 0, 0))
        #
        # # Draw Speed Text
        # font = pygame.font.Font(None, 30)
        # speed_in_fwd_dir = state.boat_vel[0] * jnp.sin(state.boat_dir) + state.boat_vel[1] * jnp.cos(state.boat_dir)
        # speed_text = font.render(f"Speed: {jnp.squeeze(speed_in_fwd_dir):.2f} knots", True, (0, 0, 0))
        # screen.blit(speed_text, (10, 10))
        #
        # # Draw Time Text
        # time_text = font.render(f"Time: {state.time}", True, (0, 0, 0))
        # screen.blit(time_text, (10, 40))
        #
        # # Draw Position Text
        # pos_text = font.render(f"Position: ({state.boat_pos[0]:.2f}, {state.boat_pos[1]:.2f})", True, (0, 0, 0))
        # screen.blit(pos_text, (10, 70))

        def update(frame):
            # boat_angle = jnp.squeeze(state.boat_dir[frame])
            x, y = (trajectory_state.boat_pos[frame, 0], trajectory_state.boat_pos[frame, 1])
            dot.set_data(jnp.expand_dims(x, axis=0), jnp.expand_dims(y, axis=0))
            # line.set_data(agent_path_history[0], agent_path_history[1])
            return line, dot

        # Create the animation
        anim = animation.FuncAnimation(fig,
                                       update,
                                       frames=trajectory_state.time.shape[0],
                                       interval=self.dt * 10,  # Convert dt to milliseconds
                                       blit=True
                                       )
        anim.save(f"../animations/{self.name}.gif")
        plt.close()

    def render(self, state: EnvState, render_delay: float = 0.0):
        """
        Remember width is left to right increasing
        BUT height is top to bottom increasing
        """
        if not hasattr(self, '_display'):
            pygame.init()
            self._display = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Sailing Simulator")
        screen = self._display
        screen.fill((240, 240, 240))  # Light gray background

        # Convert state variables to screen coordinates
        def to_screen_coords(x, y):
            # x_offset = screen_width / 2
            # y_offset = screen_height / 2
            # scale = 50  # Adjust this scaling factor as needed
            # return (int(x_offset + x * scale), int((1 - y) * screen_height / 2))  # y is flipped in screen coords
            flip_y = (1 - (y / self.screen_height)) * self.screen_height
            return int(x), int(flip_y)

        # Draw Boat Path
        path_length = 30
        for i in range(path_length):
            p = state.boat_path[:, i]
            x, y = to_screen_coords(p[0], p[1])
            gfxdraw.aacircle(screen, x, y, 1, (0, 0, 255))
            gfxdraw.filled_circle(screen, x, y, 1, (0, 0, 255))

        # Draw Boat
        boat_angle = jnp.squeeze(state.boat_dir)
        boat_x_screen, boat_y_screen = to_screen_coords(state.boat_pos[0], state.boat_pos[1])

        # Load and rotate the boat image.
        boat_image_path = "boaty_boat.png"
        boat_image = pygame.image.load(boat_image_path).convert_alpha()
        # Scale the image
        original_boat_width, original_boat_height = boat_image.get_size()
        boat_scale = 0.02
        new_boat_width = int(original_boat_width * boat_scale)
        new_boat_height = int(original_boat_height * boat_scale)
        scaled_boat_image = pygame.transform.scale(boat_image, (new_boat_width, new_boat_height))
        # Rotate the image.  pygame rotation is counter-clockwise, so we negate the angle.
        rotated_boat_image = pygame.transform.rotate(scaled_boat_image, -jnp.degrees(boat_angle))
        # Get the center of the rotated image.
        boat_rect = rotated_boat_image.get_rect(center=(boat_x_screen, boat_y_screen))
        # Blit the rotated image onto the screen.
        screen.blit(rotated_boat_image, boat_rect)

        boat_length = 10
        end_x = boat_x_screen + boat_length * jnp.sin(boat_angle)
        end_y = boat_y_screen - boat_length * jnp.cos(boat_angle)
        pygame.draw.line(screen, (255, 0, 0), (int(boat_x_screen), int(boat_y_screen)), (int(end_x), int(end_y)), 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(boat_x_screen), int(boat_y_screen)), boat_length/2, 1)

        # Draw Wind Arrow
        wind_angle = self.wind_dir
        wind_length = 50
        wind_x = self.screen_width / 2
        wind_y = 0
        end_x = wind_x + wind_length * jnp.sin(wind_angle)
        end_y = wind_y + wind_length * jnp.cos(wind_angle)

        # Draw the wind arrow using gfxdraw for antialiasing
        pygame.draw.line(screen, (255, 0, 0), (wind_x, wind_y), (int(end_x), int(end_y)), 2)
        arrow_head_size = 10
        arrow_tip = (int(end_x), int(end_y))
        arrow_left = (int(end_x + arrow_head_size * jnp.sin(wind_angle - jnp.pi / 6)),
                      int(end_y - arrow_head_size * jnp.cos(wind_angle - jnp.pi / 6)))
        arrow_right = (int(end_x + arrow_head_size * jnp.sin(wind_angle + jnp.pi / 6)),
                       int(end_y - arrow_head_size * jnp.cos(wind_angle + jnp.pi / 6)))
        gfxdraw.aapolygon(screen, [arrow_tip, arrow_left, arrow_right], (255, 0, 0))
        gfxdraw.filled_polygon(screen, [arrow_tip, arrow_left, arrow_right], (255, 0, 0))
        # TODO check all the above actually works

        # Draw marks
        for i in range(self.marks.shape[0]):
            x, y = to_screen_coords(self.marks[i, 0], self.marks[i, 1])
            gfxdraw.aacircle(screen, x, y, 4, (0, 0, 0))
            gfxdraw.filled_circle(screen, x, y, 4, (0, 0, 0))

        # Draw Speed Text
        font = pygame.font.Font(None, 30)
        speed_in_fwd_dir = state.boat_vel[0] * jnp.sin(state.boat_dir) + state.boat_vel[1] * jnp.cos(state.boat_dir)
        speed_text = font.render(f"Speed: {jnp.squeeze(speed_in_fwd_dir):.2f} knots", True, (0, 0, 0))
        screen.blit(speed_text, (10, 10))

        # Draw Time Text
        time_text = font.render(f"Time: {state.time}", True, (0, 0, 0))
        screen.blit(time_text, (10, 40))

        # Draw Position Text
        pos_text = font.render(f"Position: ({state.boat_pos[0]:.2f}, {state.boat_pos[1]:.2f})", True, (0, 0, 0))
        screen.blit(pos_text, (10, 70))

        pygame.display.flip()
        pygame.event.pump()  # Process events to prevent freezing
        time.sleep(render_delay)

        # return screen

    @property
    def name(self) -> str:
        return "SailingEnv-v0"

    def action_space(self) -> spaces.Box:
        return spaces.Box(-self.max_action, self.max_action, (1,), dtype=jnp.float32)

    def observation_space(self) -> spaces.Box:
        max_speed = 2
        max_dist = jnp.sqrt(jnp.square(self.screen_width) + jnp.square(self.screen_height))
        max_accel = 2.0
        # TODO sort out the above to be a bit better
        low = jnp.array([0.0,
                         -jnp.pi,
                         0.0,
                         -jnp.pi,
                         0.0,
                         ])
        high = jnp.array([max_speed,
                          jnp.pi,
                          max_accel,
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

        # Instantiate the environment & its settings.
        env = SailingEnvCSCA()

        # Reset the environment.
        key, _key = jrandom.split(key)
        obs, state = env.reset(_key)

        time_steps = 30#00  # 500
        # start_time = time.time()
        returns = 0
        # for _ in range(time_steps):
        #     # Sample a random action.
        #     key, _key = jrandom.split(key)
        #     # action = env.action_space(env_self).sample(_key)
        #     # action = jnp.zeros(1,)
        #     action = jnp.ones(1,) * -0.001
        #
        #     # env.render(state, env_self, render_delay=0.05)
        #     env.render(state)
        #
        #     # Perform the step transition.
        #     key, _key = jrandom.split(key)
        #     obs, delta_obs, state, reward, done, _ = env.step(action, state, _key)
        #     returns += reward
        #     print(returns)
        #
        #     if done:
        #         break

        def _step(runner_state, unused):
            obs, state, key = runner_state
            key, _key = jrandom.split(key)
            action = env.action_space().sample(_key)
            # action = jnp.zeros(1,)
            # action = jnp.ones(1,) * -0.001

            # env.render(state, env_self, render_delay=0.05)
            # env.render(state)

            # Perform the step transition.
            key, _key = jrandom.split(key)
            nobs, delta_obs, nstate, reward, done, _ = env.step(action, state, _key)

            return (nobs, nstate, key), state

        _, traj_state = jax.lax.scan(_step, (obs, state, key), None, time_steps)
        env.render_traj(traj_state)

