"""
Simulation helpers for checking the dynamics, shared by the tests and the calibration script.

Polars here are a test output, the dynamics never read them.
"""

from typing import Callable

from bifurcagym.envs.utils import integrate_ode
import chex
import jax
import jax.numpy as jnp
import numpy as np

from project_name.sailing import dynamics
from project_name.sailing.angles import wind_velocity, wrap
from project_name.sailing.dynamics import BoatParams


SUBSTEP = 0.05  # s


def start_state(heading: float, u: float = 0.0, v: float = 0.0, r: float = 0.0, boom: float = 0.0,
                rudder: float = 0.0) -> chex.Array:
    return jnp.array((0.0, 0.0, heading, u, v, r, boom, rudder))


def step(boat: chex.Array, action: chex.Array, wind_world: chex.Array, params: BoatParams, dt: float = SUBSTEP,
         substeps: int = 1) -> chex.Array:
    control = jnp.concatenate((action, wind_world))
    return integrate_ode(dynamics.ode_rhs, 0.0, boat, control, dt, substeps, params)


def rollout(boat: chex.Array, policy: Callable, wind_world: chex.Array, params: BoatParams, duration: float,
            dt: float = SUBSTEP) -> chex.Array:
    """policy(boat, t) gives the action. Returns the state after every step."""
    def body(state, t):
        new_state = step(state, policy(state, t), wind_world, params, dt)
        return new_state, new_state

    return jax.lax.scan(body, boat, jnp.arange(0.0, duration, dt))[1]


GRID_NAMES = ("speed", "leeway", "heading", "rudder", "sail", "wind")
GRID_AXES = (jnp.array((0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),  # the polars top out near 4.4 m/s in 12 m/s of wind
             jnp.radians(jnp.array((0.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 89.0, 135.0, 165.0, 175.0))),
             jnp.radians(jnp.array((30.0, 120.0))),
             jnp.radians(jnp.array((0.0, 15.0, 30.0, 80.0))),
             jnp.radians(jnp.array((45.0, -70.0))),
             jnp.array((5.0, 10.0, 20.0)))


def stability_grid(params: BoatParams, dt: float = SUBSTEP) -> np.ndarray:
    """Spectral radius of one integration step over GRID_AXES, where above 1 means small errors grow."""
    def radius(speed, leeway, heading, rudder, sail, wind_speed):
        boat = start_state(heading, speed * jnp.cos(leeway), speed * jnp.sin(leeway), boom=0.5, rudder=rudder)
        action = jnp.stack((rudder, sail))
        wind = wind_velocity(wind_speed, 0.0)
        jacobian = jax.jacfwd(lambda b: step(b, action, wind, params, dt))(boat)
        return jnp.max(jnp.abs(jnp.linalg.eigvals(jacobian)))

    mesh = jnp.meshgrid(*GRID_AXES, indexing="ij")
    radii = jax.vmap(radius)(*[axis.ravel() for axis in mesh])
    return np.array(radii).reshape(mesh[0].shape)


def heading_hold(target_heading: chex.Array, sail_command: chex.Array) -> Callable:
    """Simple rudder controller, used to hold a heading while measuring steady performance."""
    def policy(boat, t):
        rudder = jnp.clip(1.0 * wrap(boat[dynamics.HEADING] - target_heading) + 0.6 * boat[dynamics.R], -0.6, 0.6)
        return jnp.stack((rudder, sail_command))

    return policy


def steady_performance(true_wind_angle_deg: float, wind_speed: float, sheet_rad: chex.Array, params: BoatParams,
                       duration: float = 90.0) -> chex.Array:
    """Mean (u, v) over the last 15 s on a held heading. Wind is from compass north, so heading is taken off that."""
    heading = jnp.radians(90.0 - true_wind_angle_deg)
    wind = wind_velocity(wind_speed, 0.0)
    trajectory = rollout(start_state(heading, u=1.0), heading_hold(heading, sheet_rad), wind, params, duration)
    tail = trajectory[-int(15.0 / SUBSTEP):]
    return jnp.mean(tail[:, (dynamics.U, dynamics.V)], axis=0)


def polar(true_wind_angles_deg: chex.Array, wind_speed: float, params: BoatParams) -> dict:
    """Best forward speed over all sheet settings at each true wind angle, with its leeway and sheet."""
    sheets = jnp.radians(jnp.arange(0.0, 91.0, 5.0))

    def at_angle(angle):
        velocities = jax.vmap(lambda s: steady_performance(angle, wind_speed, s, params))(sheets)
        best = jnp.argmax(velocities[:, 0])
        return velocities[best], sheets[best]

    velocities, best_sheets = jax.jit(jax.vmap(at_angle))(true_wind_angles_deg)
    speed = jnp.hypot(velocities[:, 0], velocities[:, 1])
    return {"true_wind_angle": true_wind_angles_deg,
            "speed": speed,
            "leeway": jnp.degrees(jnp.arctan2(velocities[:, 1], velocities[:, 0])),
            "sheet": jnp.degrees(best_sheets),
            "vmg": velocities[:, 0] * jnp.cos(jnp.radians(true_wind_angles_deg)),
            }
