"""
A hand-written tactician: sail close-hauled on the current tack until the mark can be laid, then head straight for it.

It proves the task can be done, and gives a score for a learned policy to beat.
"""

import chex
import jax.numpy as jnp

from project_name.sailing import dynamics
from project_name.sailing.angles import compass_to_maths, wind_velocity, wrap
from project_name.sailing.env import EnvParams, EnvState


CLOSE_HAULED = jnp.radians(45.0)
ANGLE_OF_ATTACK = jnp.radians(15.0)
# A big course change, such as a tack, gets the helm hard over until near the new course, as in analysis.tack_helm.
# The proportional controller alone eases the helm through the turn, and the boat stalls head to wind
HARD_OVER = jnp.radians(45.0)
HARD_OVER_BEYOND = jnp.radians(15.0)


def scripted_action(state: EnvState, params: EnvParams) -> chex.Array:
    """Returns (rudder, sail) in radians."""
    boat = state.boat
    wind_from = compass_to_maths(state.wind_direction)
    offset = state.mark - boat[jnp.array((dynamics.X, dynamics.Y))]
    mark_bearing = jnp.arctan2(offset[1], offset[0] + 1e-9)

    mark_off_wind = wrap(mark_bearing - wind_from)
    current_tack = jnp.where(wrap(boat[dynamics.HEADING] - wind_from) >= 0.0, 1.0, -1.0)
    target = jnp.where(jnp.abs(mark_off_wind) < CLOSE_HAULED, wind_from + current_tack * CLOSE_HAULED, mark_bearing)

    error = wrap(boat[dynamics.HEADING] - target)  # positive rudder reduces the maths heading
    rudder = jnp.where(jnp.abs(error) > HARD_OVER_BEYOND, jnp.sign(error) * HARD_OVER,
                       jnp.clip(1.0 * error + 0.6 * boat[dynamics.R], -0.6, 0.6))

    wind = wind_velocity(state.wind_speed, state.wind_direction)
    apparent_angle = dynamics.forces_and_moment(boat, wind, params.boat)["apparent_wind_angle"]
    sheet = jnp.clip(jnp.abs(apparent_angle) - ANGLE_OF_ATTACK, jnp.radians(5.0), jnp.radians(85.0))

    return jnp.stack((rudder, sheet))
