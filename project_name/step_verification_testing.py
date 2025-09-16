import jax
import jax.numpy as jnp


air_kinematic_viscosity = 0.0000171  # [Pa * s]
air_density =  1.3 # [kg/m^3]
water_kinematic_viscosity = 0.0000001 # [Pa * s]
water_density = 1000  # [kg/m^3]

sail_area = 6.2 # [m^2]
sail_chord = 2  # [m]  # TODO sort these values out
sail_span = sail_area / sail_chord  # [m]
sail_aspect_ratio = sail_span / sail_chord

centreboard_area = 0.5  # [m^2]
centreboard_chord = 0.5  # [m]
centreboard_span = centreboard_area / centreboard_chord  # [m]
centreboard_aspect_ratio = centreboard_span / centreboard_chord
lateral_area = 2.5 # [m^2]


wind_dir: float = 0.0  # deg
wind_speed: float = 5.0  # in ms^-1
wind_vel: jnp.ndarray = -wind_speed * jnp.array((jnp.cos(jnp.radians(wind_dir)),
                                                 jnp.sin(jnp.radians(wind_dir))))  # in ms^-1


sail_angle = jnp.radians(30.0)  # can range between 0 and 90
boat_heading = jnp.radians(45.0)
boat_vel = jnp.array((1.0, 0.0))

# Convert wind to boat and find apparent wind
true_wind_to_boat_x = wind_vel[0] * jnp.cos(boat_heading) - wind_vel[1] * jnp.sin(boat_heading)
true_wind_to_boat_y = wind_vel[1] * jnp.cos(boat_heading) - wind_vel[0] * jnp.sin(boat_heading)
transformed_wind = jnp.array((true_wind_to_boat_x, true_wind_to_boat_y))
apparent_wind = transformed_wind - boat_vel
apparent_wind_angle = jnp.arctan2(-apparent_wind[1], -apparent_wind[0])
apparent_wind_speed = jnp.sqrt(apparent_wind[0] ** 2 + apparent_wind[1] ** 2)

# Calc aoa
true_sail_angle = jnp.copysign(sail_angle, apparent_wind_angle)  # prevents sign of 0 when AWA == 0
# reversed_apparent_wind_angle = jnp.copysign((jnp.pi - jnp.abs(apparent_wind_angle)), apparent_wind_angle)  # TODO a way to ensure correct direction?
# aoa = reversed_apparent_wind_angle - true_sail_angle
aoa = apparent_wind_angle - true_sail_angle
# TODO check the above

# Calc sail force
aoa = jax.lax.select(aoa * true_sail_angle < 0, 0.0, aoa)
# if aoa * true_sail_angle < 0:
#     aoa = 0

eff_aoa = aoa  # eff_aoa : effective angle of attack
eff_aoa = jnp.where(aoa < -jnp.pi / 2, jnp.pi + aoa, eff_aoa)
eff_aoa = jnp.where(aoa > jnp.pi / 2, -jnp.pi + aoa, eff_aoa)
# TODO is the above correct hmmm
# if aoa < -jnp.pi / 2:
#     eff_aoa = jnp.pi + aoa
# elif aoa > jnp.pi / 2:
#     eff_aoa = -jnp.pi + aoa

aero_friction = jax.lax.select(apparent_wind_speed != 0,
                               3.55 * jnp.sqrt(air_kinematic_viscosity / (apparent_wind_speed * sail_span)),
                               0.0)

coeff_lift = (2 * jnp.pi * eff_aoa) / (1 + 2 / sail_aspect_ratio)
coeff_drag = aero_friction * coeff_lift ** 2

sail_lift = 0.5 * air_density * apparent_wind_speed ** 2 * sail_area * coeff_lift
sail_drag = 0.5 * air_density * apparent_wind_speed ** 2 * sail_area * coeff_drag

sail_force_x = -sail_drag * jnp.cos(apparent_wind_angle) + sail_lift * jnp.sin(apparent_wind_angle)
sail_force_y = -sail_lift * jnp.cos(apparent_wind_angle) - sail_drag * jnp.sin(apparent_wind_angle)


# Calc centreboard force
boat_speed = jnp.sqrt(boat_vel[0] ** 2 + boat_vel[1] ** 2)
eff_leeway_angle = leeway_angle = jnp.arctan2(boat_vel[1], boat_vel[0])
eff_leeway_angle = jnp.where(leeway_angle < -jnp.pi / 2, jnp.pi + leeway_angle, eff_leeway_angle)
eff_leeway_angle = jnp.where(leeway_angle > jnp.pi / 2, -jnp.pi + leeway_angle, eff_leeway_angle)

pressure = 0.5 * water_density * boat_speed ** 2

hydro_friction = jax.lax.select(boat_speed != 0,
                                2.66 / jnp.sqrt((boat_speed * centreboard_chord) / water_kinematic_viscosity),
                                0.0)

separation = 1 - jnp.exp(-((jnp.abs(eff_leeway_angle)) / (jnp.pi / 180 * 25)) ** 2)

coeff_lift = 2 * jnp.pi * eff_leeway_angle
coeff_drag = hydro_friction + ((coeff_lift ** 2 * separation) / (jnp.pi * centreboard_aspect_ratio))

separated_force_y = -jnp.sign(leeway_angle) * pressure * sail_area * jnp.sin(leeway_angle) ** 2
# TODO unsure the sin above is correct

centreboard_lift = pressure * centreboard_area * coeff_lift
centreboard_drag = pressure * centreboard_area * coeff_drag

# centreboard_force_x = (1 - separation) * (-centreboard_lift * jnp.sin(leeway_angle) - centreboard_drag * jnp.cos(leeway_angle))
# centreboard_force_y = (1 - separation) * (centreboard_drag * jnp.sin(leeway_angle) - centreboard_lift * jnp.cos(leeway_angle)) + separation * separated_force_y
centreboard_force_x = -centreboard_drag * jnp.cos(leeway_angle) + centreboard_lift * jnp.sin(leeway_angle)
centreboard_force_y = -centreboard_lift * jnp.cos(leeway_angle) - centreboard_drag * jnp.sin(leeway_angle)

print("HERE")