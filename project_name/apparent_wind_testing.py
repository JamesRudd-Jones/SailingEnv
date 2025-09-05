import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


wind_dir = 0.0
wind_speed = 5
wind_vel = -wind_speed * jnp.array((jnp.cos(jnp.radians(wind_dir)),
                                   jnp.sin(jnp.radians(wind_dir))))  # in ms^-1

boat_heading = jnp.radians(90)
boat_vel = 2 * jnp.array((1.0, 0.0))
sail_angle = jnp.radians(45)  # can range between 0 and 90

true_wind_to_boat_x = wind_vel[0] * jnp.cos(boat_heading) - wind_vel[1] * jnp.sin(boat_heading)
true_wind_to_boat_y = wind_vel[1] * jnp.cos(boat_heading) - wind_vel[0] * jnp.sin(boat_heading)
true_wind_angle_boat = jnp.arctan2(-true_wind_to_boat_y, -true_wind_to_boat_x)
print(true_wind_to_boat_x, true_wind_to_boat_y)
print(f"True Wind Angle to Boat = {jnp.degrees(true_wind_angle_boat)}")

apparent_wind_to_boat_x = true_wind_to_boat_x - boat_vel[0]
apparent_wind_to_boat_y = true_wind_to_boat_y - boat_vel[1]
apparent_wind_angle_boat = jnp.arctan2(-apparent_wind_to_boat_y, -apparent_wind_to_boat_x)
print(f"Apparent Wind Angle to Boat = {jnp.degrees(apparent_wind_angle_boat)}")
apparent_wind_speed = jnp.sqrt(apparent_wind_to_boat_x ** 2 + apparent_wind_to_boat_y ** 2)
print(f"Apparent Wind Speed = {apparent_wind_speed}")

true_sail_angle = jnp.copysign(sail_angle, apparent_wind_angle_boat)  # prevents sign of 0 when AWA == 0
print(f"True Sail Angle = {jnp.degrees(true_sail_angle)}")
aoa = apparent_wind_angle_boat - true_sail_angle
print(f"AOA = {jnp.degrees(aoa)}")

aoa = jax.lax.select(aoa * jnp.sign(true_sail_angle) < 0, 0.0, aoa)
print(f"AOA True = {jnp.degrees(aoa)}")




delta_pos_x = boat_vel[0] * jnp.cos(boat_heading) - boat_vel[1] * jnp.sin(boat_heading)
delta_pos_y = boat_vel[1] * jnp.cos(boat_heading) + boat_vel[0] * jnp.sin(boat_heading)

apparent_wind_x = (apparent_wind_to_boat_x * jnp.cos(boat_heading) - apparent_wind_to_boat_y * jnp.sin(boat_heading))
apparent_wind_y = (apparent_wind_to_boat_y * jnp.cos(boat_heading) + apparent_wind_to_boat_x * jnp.sin(boat_heading))
apparent_wind_angle = jnp.arctan2(-apparent_wind_y, -apparent_wind_x)

sail_x = -jnp.cos(boat_heading + true_sail_angle)
sail_y = -jnp.sin(boat_heading + true_sail_angle)

sail_length = 1
plt.arrow(5 - wind_vel[0], 5 - wind_vel[1], wind_vel[0], wind_vel[1], width=0.1, length_includes_head=True, color="Green", label="True Wind")
plt.plot([5, 5 + sail_x], [5, 5 + sail_y], color="Black", label="Sail")
plt.arrow(5 - delta_pos_x/2, 5 - delta_pos_y/2, delta_pos_x, delta_pos_y, width=0.1, length_includes_head=True, color="Red", label="Boat")
plt.arrow(5 - apparent_wind_x, 5 - apparent_wind_y, apparent_wind_x, apparent_wind_y, width=0.1, length_includes_head=True, color="Blue", label="Apparent Wind")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.show()

print(f"True Wind Angle = {wind_dir}")
print(f"Apparent Wind Angle = {jnp.degrees(apparent_wind_angle)}")
print(f"{jnp.degrees(apparent_wind_angle_boat) - jnp.degrees(true_wind_angle_boat)} should equal {jnp.degrees(apparent_wind_angle)}")
