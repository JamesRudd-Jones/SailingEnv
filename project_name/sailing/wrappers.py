"""
BifurcaGym's normalised wrappers call jnp.ones_like on a spaces.Box rather than on its bounds, which raises a ValueError
for every environment, including its own NPendulum-v0 in tests/test_wrappers.py. These subclasses correct just that.
Once it is fixed in BifurcaGym, use its wrappers directly and delete this file.
"""

from bifurcagym import spaces
from bifurcagym.envs import EnvParams
from bifurcagym.wrappers import NormalisedWrapperCSCA, NormalisedWrapperCSDA
import jax.numpy as jnp


def _unit_box(space: spaces.Box) -> spaces.Box:
    return spaces.Box(low=-jnp.ones_like(space.low), high=jnp.ones_like(space.high), shape=space.shape, dtype=space.dtype)


class NormalisedSailingCSDA(NormalisedWrapperCSDA):
    def observation_space(self, params: EnvParams) -> spaces.Box:
        return _unit_box(self.unnorm_obs_space(params))


class NormalisedSailingCSCA(NormalisedWrapperCSCA):
    def observation_space(self, params: EnvParams) -> spaces.Box:
        return _unit_box(self.unnorm_obs_space(params))

    def action_space(self, params: EnvParams) -> spaces.Box:
        return _unit_box(self.unnorm_action_space(params))
