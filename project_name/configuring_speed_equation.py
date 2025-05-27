import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import jax


def piecewise_function(theta):
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



# Generate theta values
num_points = 400
theta_values = jnp.linspace(0, jnp.pi, num_points)
piecewise_output_jittable = jax.vmap(piecewise_function)(theta_values)

# Plotting
plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), projection='polar')

ax.plot(theta_values, piecewise_output_jittable)
# ax.plot(radians(90), 0, 'go', label='Reference Point')

ax.set_rmin(0)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
plt.title("Jittable Piecewise Function in Polar Coordinates", fontsize=18)
plt.show()