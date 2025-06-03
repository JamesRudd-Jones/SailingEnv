import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


wind_dir: float = 180.0  # deg
wind_speed: float = 5.0  # in ms^-1
wind_vel: jnp.ndarray = wind_speed * jnp.array((jnp.sin(jnp.radians(wind_dir)),
                                                                  jnp.cos(jnp.radians(wind_dir))))  # in ms^-1
init_dir = jnp.radians(jnp.ones(1,) * 90)
init_boat_vel = jnp.array((1.0, 0.0))

plt.arrow(-float(wind_vel[0]), -float(wind_vel[1]), float(wind_vel[0]), float(wind_vel[1]), head_width=0.4, label="Wind")
plt.arrow(0, 0, float(init_boat_vel[0]), float(init_boat_vel[1]), head_width=0.4, label="Boat Vel")
screen_size = 10
plt.xlim(-screen_size, screen_size)
plt.ylim(-screen_size, screen_size)
plt.legend()
plt.show()


"https://www.sailingworld.com/how-to/angles-of-attack/"

"https://tuprints.ulb.tu-darmstadt.de/8471/7/sailboat_model_irsc.pdf"