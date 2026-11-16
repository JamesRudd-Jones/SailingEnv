"""
Base wind: speed and direction that wander over time. The wake between boats will be added here for multiple boats.
"""

import chex
import jax.numpy as jnp
import jax.random as jrandom


def ornstein_uhlenbeck_step(value: chex.Array,
                            mean: chex.Array,
                            stationary_sd: chex.Array,
                            time_constant: chex.Array,
                            dt: float,
                            key: chex.PRNGKey,
                            ) -> chex.Array:
    """Exact update, so that the spread of the wind does not depend on the timestep."""
    decay = jnp.exp(-dt / time_constant)
    return mean + decay * (value - mean) + stationary_sd * jnp.sqrt(1.0 - decay * decay) * jrandom.normal(key)
