"""
The only place that converts between compass and maths angles.

Physics uses maths angles: world x east, world y north, heading in radians anticlockwise from +x, body x forward and
body y to port. Anything a person sets or reads uses compass degrees, clockwise from north, with wind given as the
direction it blows from.
"""

import chex
import jax.numpy as jnp


def wrap(angle: chex.Array) -> chex.Array:
    """Wraps radians to [-pi, pi]."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def compass_to_maths(compass_deg: chex.Array) -> chex.Array:
    """Compass heading in degrees to maths heading in radians."""
    return wrap(jnp.radians(90.0 - compass_deg))


def maths_to_compass(maths_rad: chex.Array) -> chex.Array:
    """Maths heading in radians to compass heading in degrees, in [0, 360)."""
    return jnp.mod(90.0 - jnp.degrees(maths_rad), 360.0)


def wind_velocity(speed: chex.Array, from_compass_deg: chex.Array) -> chex.Array:
    """World-frame velocity of the air, for wind blowing from the given compass direction."""
    from_rad = jnp.radians(from_compass_deg)
    return -speed * jnp.stack((jnp.sin(from_rad), jnp.cos(from_rad)))
