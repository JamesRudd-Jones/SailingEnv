import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


wind_dir = 0.0
wind_speed = 5
wind_vel = wind_speed * jnp.array((jnp.sin(jnp.radians(wind_dir)),
                                   jnp.cos(jnp.radians(wind_dir))))  # in ms^-1

boat_heading = jnp.radians(45)
boat_vel = jnp.array((1.0, 0.0))

true_wind_to_boat_x = wind_vel[0] * jnp.sin(boat_heading) - wind_vel[1] * jnp.cos(boat_heading)
true_wind_to_boat_y = wind_vel[1] * jnp.sin(boat_heading) + wind_vel[0] * jnp.cos(boat_heading)
true_wind_angle_boat = jnp.arctan2(-true_wind_to_boat_y, -true_wind_to_boat_x)
print(f"True Wind Angle to Boat = {jnp.degrees(true_wind_angle_boat)}")

apparent_wind_to_boat_x = true_wind_to_boat_x - boat_vel[0]
apparent_wind_to_boat_y = true_wind_to_boat_y - boat_vel[1]
apparent_wind_angle_boat = jnp.arctan2(-apparent_wind_to_boat_y, -apparent_wind_to_boat_x)
print(f"Apparent Wind Angle to Boat = {jnp.degrees(apparent_wind_angle_boat)}")
apparent_wind_speed = jnp.sqrt(apparent_wind_to_boat_x ** 2 + apparent_wind_to_boat_y ** 2)
print(f"Apparent Wind Speed = {apparent_wind_speed}")

delta_pos_x = boat_vel[0] * jnp.sin(boat_heading) - boat_vel[1] * jnp.cos(boat_heading)
delta_pos_y = boat_vel[1] * jnp.sin(boat_heading) + boat_vel[0] * jnp.cos(boat_heading)

apparent_wind_x = apparent_wind_to_boat_x * jnp.sin(boat_heading) + apparent_wind_to_boat_y * jnp.cos(boat_heading)
apparent_wind_y = apparent_wind_to_boat_y * jnp.sin(boat_heading) - apparent_wind_to_boat_x * jnp.cos(boat_heading)
apparent_wind_angle = jnp.arctan2(-apparent_wind_y, -apparent_wind_x)

plt.arrow(5, 10, -wind_vel[0], -wind_vel[1], width=0.1, color="Green", label="True Wind")
plt.arrow(5 - delta_pos_x, 4.5 - delta_pos_y/2, delta_pos_x, delta_pos_y, width=0.1, color="Red", label="Boat")
plt.arrow(5, 10, apparent_wind_x, -apparent_wind_y, width=0.1, color="Blue", label="Apparent Wind")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
# plt.show()

print(f"True Wind Angle = {wind_dir}")
print(f"Apparent Wind Angle = {jnp.degrees(apparent_wind_angle)}")
