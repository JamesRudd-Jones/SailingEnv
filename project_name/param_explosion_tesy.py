import jax
import jax.numpy as jnp
import jax.random as jrandom
# import cmocean
from SailingEnv import SailingEnvCSCA, EnvState
import matplotlib.pyplot as plt


env = SailingEnvCSCA()


def test_leeway(param, disable_jit=False):
    key = jrandom.key(42)
    rudder_angle = jnp.radians(0)
    sail_angle = jnp.radians(10)
    action = jnp.array((rudder_angle, sail_angle))

    state = EnvState(boat_pos=env.init_pos,
                     boat_vel=jnp.array((1.0, param)).squeeze(),
                     boat_heading=jnp.radians(45),
                     boat_heading_rate=jnp.zeros(()),
                     rudder_angle=rudder_angle,
                     sail_angle=sail_angle,
                     time=0,
                     )

    with jax.disable_jit(disable=disable_jit):
        _, _, nstate, _, _, _ = env.step(action, state, key)
    return nstate.boat_vel[1]


im, axs = plt.subplots(2, 1, sharex=True)

limit = 2.2
test_vals = jnp.linspace(-limit, limit, 101)
y_vals = jax.vmap(test_leeway)(test_vals)
axs[0].plot(test_vals, y_vals)
axs[0].set_ylabel("Leeway")


# if leeway is 2 and x is 1 then it crashes


axs[1].plot(test_vals, jnp.arctan2(test_vals, 1))
axs[1].set_xlabel("Test Val")
axs[1].set_ylabel("Arctan2")
plt.show()


test_leeway(2.2, True)