"""
Based off the following: https://github.com/simonkohaut/stda-sailboat-simulator/blob/master/src/simulation.py
"""


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Arc
import math
import matplotlib


def create_boat_marker():
    import matplotlib.path as mpath
    # A triangle pointing to the right (0 degrees)
    verts = [(-0.5, -0.25),
             (-0.5, 0.25),
             (0.25, 0),
             (-0.5, -0.25),
             ]

    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.LINETO,
        mpath.Path.CLOSEPOLY,
    ]

    boat_path = mpath.Path(verts, codes)

    return boat_path


def get_angle_plot_1(line1, line2, boat_heading, aw_angle):

    vec1 = line1[1] - line1[0]
    vec2 = line2[1] - line2[0]

    def get_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    intersection = get_intersection(line1[0], line1[1], line2[0], line2[1])

    dot_product = jnp.dot(vec1, vec2)

    norm1 = jnp.linalg.norm(vec1)
    norm2 = jnp.linalg.norm(vec2)

    cosine_angle = jnp.abs(dot_product) / (norm1 * norm2)
    clipped_cosine_angle = jnp.clip(cosine_angle, -1.0, 1.0)
    angle_radians = jnp.arccos(clipped_cosine_angle)
    angle_degrees = jnp.degrees(angle_radians)

    # The Arc patch needs the angles in degrees
    theta1 = jnp.abs(jnp.degrees(float(boat_heading.squeeze())) - 90 - 180)
    theta2 = theta1 + jnp.copysign(angle_degrees, aw_angle)

    return Arc(intersection, width=4, height=4, angle=0, theta1=min(theta1, theta2), theta2=max(theta1, theta2), color='purple', linewidth=2,
               linestyle='-', label="%0.2f"%float(angle_degrees)+u"\u00b0")


def get_angle_plot_2(line1, line2, boat_heading, aw_angle, sail_angle, aoa):

    vec1 = line1[1] - line1[0]
    vec2 = line2[1] - line2[0]

    def get_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    intersection = get_intersection(line1[0], line1[1], line2[0], line2[1])

    dot_product = jnp.dot(vec1, vec2)

    norm1 = jnp.linalg.norm(vec1)
    norm2 = jnp.linalg.norm(vec2)

    cosine_angle = dot_product / (norm1 * norm2)
    clipped_cosine_angle = jnp.clip(cosine_angle, -1.0, 1.0)
    angle_radians = jnp.arccos(clipped_cosine_angle)
    angle_degrees = jnp.degrees(angle_radians)

    # The Arc patch needs the angles in degrees
    theta0 = jnp.abs(jnp.degrees(float(boat_heading.squeeze())) - 90 - 180)
    theta1 = theta0 + jnp.copysign(jnp.degrees(sail_angle), aw_angle) - 180
    # theta_inter = jnp.sign(aoa) * jnp.sign(aw_angle) * angle_degrees
    theta_inter = jnp.copysign(jnp.copysign(angle_degrees, aw_angle), aoa)
    theta2 = theta1 - theta_inter if boat_heading >= jnp.radians(180.0) else theta1 + theta_inter
    # TODO a dodgy fix for now

    return Arc(intersection, width=4, height=4, angle=0, theta1=min(theta1, theta2), theta2=max(theta1, theta2), color='orange', linewidth=2,
               linestyle='-', label="%0.2f"%float(jnp.degrees(aoa))+u"\u00b0")


def get_angle_text(angle_plot):
    angle = angle_plot.get_label()[:-1]
    angle = "%0.2f"%float(angle)+u"\u00b0" # Display angle upto 2 decimal places

    # Get the vertices of the angle arc
    vertices = angle_plot.get_verts()

    # Get the midpoint of the arc extremes
    x_width = (vertices[0][0] + vertices[-1][0]) / 2.0
    y_width = (vertices[0][1] + vertices[-1][1]) / 2.0

    #print x_width, y_width

    separation_radius = max(x_width/2.0, y_width/2.0)

    return [ x_width + separation_radius, y_width + separation_radius, angle]


vertices = [(0, 0), (-1, 1), (1, 1), (2, 0), (1, -1), (-1, -1), (0, 0)]
p = Path(vertices,[1,2,2,2,2,2,79])


wind_dir: float = 0.0  # deg
wind_speed: float = 5.0  # in ms^-1
wind_vel: jnp.ndarray = -wind_speed * jnp.array((jnp.cos(jnp.radians(wind_dir)),
                                                 jnp.sin(jnp.radians(wind_dir))))  # in ms^-1
init_dir = jnp.radians(jnp.ones(1,) * 30)
init_boat_vel = jnp.array((1.0, 0.0))

sail_angle = jnp.radians(0.0)  # between 0 and 90

def boat_to_global(boat_heading, val):
    delta_pos_x = val[0] * jnp.cos(boat_heading) - val[1] * jnp.sin(boat_heading)
    delta_pos_y = val[1] * jnp.cos(boat_heading) + val[0] * jnp.sin(boat_heading)
    return jnp.array((delta_pos_x.squeeze(), delta_pos_y.squeeze()))

def global_to_boat(boat_heading, val):  # TODO fill this in
    delta_pos_x = val[0] * jnp.cos(boat_heading) + val[1] * jnp.sin(boat_heading)
    delta_pos_y = val[1] * jnp.cos(boat_heading) - val[0] * jnp.sin(boat_heading)
    return jnp.array((delta_pos_x.squeeze(), delta_pos_y.squeeze()))

def apparent_wind(boat_heading, boat_vel, wind_vel):
    transformed_wind = global_to_boat(boat_heading, wind_vel)

    apparent_wind = transformed_wind - boat_vel
    apparent_wind_angle = jnp.arctan2(-apparent_wind[1], -apparent_wind[0])
    apparent_wind_speed = jnp.sqrt(apparent_wind[0] ** 2 + apparent_wind[1] ** 2)

    return apparent_wind, apparent_wind_speed, apparent_wind_angle

def get_sail_angle_plot(boat_dir, sail_angle, apparent_wind_angle): # prevents sign of 0 when AWA == 0
    true_sail_angle = jnp.sign(apparent_wind_angle) * jnp.abs(sail_angle) # prevents sign of 0 when AWA == 0

    new_angle_rad = boat_dir + jnp.pi + true_sail_angle

    sail_length = 1

    end_point_x = sail_length * jnp.cos(new_angle_rad)
    end_point_y = sail_length * jnp.sin(new_angle_rad)

    # aoa = apparent_wind_angle - jnp.copysign(jnp.pi, apparent_wind_angle) - true_sail_angle
    aoa = apparent_wind_angle - true_sail_angle

    return jnp.array((end_point_x.squeeze(), end_point_y.squeeze())), true_sail_angle, aoa

def calc_sail_forces(true_sail_angle, aoa, apparent_wind_speed, apparent_wind_angle):
    air_kinematic_viscosity = 0.0000171  # [Pa * s]
    air_density = 1.3  # [kg/m^3]
    sail_area = 6.2  # [m^2]
    sail_chord = 2  # [m]
    sail_span = sail_area / sail_chord  # [m]
    sail_aspect_ratio = sail_span / sail_chord

    # Calc sail force
    aoa = jax.lax.select(aoa * true_sail_angle < 0, 0.0, aoa)
    # if aoa * true_sail_angle < 0:
    #     aoa = 0

    eff_aoa = aoa  # eff_aoa : effective angle of attack
    eff_aoa = jnp.where(aoa < -jnp.pi / 2, jnp.pi + aoa, eff_aoa)
    eff_aoa = jnp.where(aoa > jnp.pi / 2, -jnp.pi + aoa, eff_aoa)
    # # if aoa < -jnp.pi / 2:
    # #     eff_aoa = jnp.pi + aoa
    # # elif aoa > jnp.pi / 2:
    # #     eff_aoa = -jnp.pi + aoa
    # TODO sort out the eff_aoa code

    aero_friction = jax.lax.select(apparent_wind_speed != 0,
                                   2.66 / jnp.sqrt((apparent_wind_speed * sail_chord) / air_kinematic_viscosity),
                                   0.0)

    coeff_lift = (2 * jnp.pi * eff_aoa) / (1 + 2 / sail_aspect_ratio)
    coeff_drag = aero_friction * coeff_lift ** 2

    print(f"AoA : {aoa}, Eff AoA : {eff_aoa}, Coeff of Lift : {coeff_lift}")

    pressure = 0.5 * air_density * apparent_wind_speed ** 2
    sail_lift = pressure * sail_area * coeff_lift
    sail_drag = pressure * sail_area * coeff_drag

    print(f"Sail Lift : {sail_lift}, Sail Drag : {sail_drag}")

    sail_force_x = -sail_drag * jnp.cos(apparent_wind_angle) + sail_lift * jnp.sin(apparent_wind_angle)
    sail_force_y = -sail_lift * jnp.cos(apparent_wind_angle) - sail_drag * jnp.sin(apparent_wind_angle)

    separation = 1 - jnp.exp(-(abs(eff_aoa) / (jnp.radians(25))) ** 2)

    separated_force_x = jnp.sign(aoa) * pressure * sail_area * jnp.sin(aoa) ** 2 * jnp.sin(true_sail_angle)
    separated_force_y = -jnp.sign(aoa) * pressure * sail_area * jnp.sin(aoa) ** 2 * jnp.cos(true_sail_angle)

    x = (1 - separation) * sail_force_x + separation * separated_force_x
    y = (1 - separation) * sail_force_y + separation * separated_force_y

    return jnp.array((x, y))

def calc_centreboard_forces(boat_vel):
    water_kinematic_viscosity = 0.0000001  # [Pa * s]
    water_density = 1000  # [kg/m^3]
    centreboard_area = 0.5  # [m^2]
    centreboard_chord = 0.5  # [m]
    centreboard_span = centreboard_area / centreboard_chord  # [m]
    centreboard_aspect_ratio = centreboard_span / centreboard_chord

    # Calc centreboard force
    boat_speed = jnp.sqrt(boat_vel[0] ** 2 + boat_vel[1] ** 2)
    eff_leeway_angle = leeway_angle = jnp.arctan2(-boat_vel[1], -boat_vel[0])
    eff_leeway_angle = jnp.where(leeway_angle < -jnp.pi / 2, jnp.pi + leeway_angle, eff_leeway_angle)
    eff_leeway_angle = jnp.where(leeway_angle > jnp.pi / 2, -jnp.pi + leeway_angle, eff_leeway_angle)

    hydro_friction = jax.lax.select(boat_speed != 0,
                                    2.66 / jnp.sqrt((boat_speed * centreboard_chord) / water_kinematic_viscosity),
                                    0.0)

    coeff_lift = (2 * jnp.pi * eff_leeway_angle) / (1 + 2 / centreboard_aspect_ratio)
    coeff_drag = hydro_friction * coeff_lift ** 2

    pressure = 0.5 * water_density * boat_speed ** 2
    centreboard_lift = pressure * centreboard_area * coeff_lift
    centreboard_drag = pressure * centreboard_area * coeff_drag

    centreboard_force_x = -centreboard_drag * jnp.cos(leeway_angle) + centreboard_lift * jnp.sin(leeway_angle)
    centreboard_force_y = -centreboard_lift * jnp.cos(leeway_angle) - centreboard_drag * jnp.sin(leeway_angle)
    # centreboard_force_x = 0.0
    # centreboard_force_y = 0.0

    separation = 1 - jnp.exp(-(abs(eff_leeway_angle) / (jnp.radians(25))) ** 2)

    return jnp.array((centreboard_force_x, centreboard_force_y))

boat_delta = boat_to_global(init_dir, init_boat_vel)
aw_boat, aw_boat_speed, aw_boat_angle = apparent_wind(init_dir, init_boat_vel, wind_vel)
tw_boat = global_to_boat(init_dir, wind_vel)
tw_boat_angle = jnp.arctan2(-tw_boat[1], -tw_boat[0])
print(aw_boat)
print(f"AW Boat Angle : {jnp.degrees(aw_boat_angle)}")
print(f"TW Boat Angle : {jnp.degrees(tw_boat_angle)}")

aw = boat_to_global(init_dir, aw_boat)

sail, true_sail_angle, aoa = get_sail_angle_plot(init_dir, sail_angle, aw_boat_angle)
print(f"AOA : {jnp.degrees(aoa)}")

sail_force_boat = calc_sail_forces(true_sail_angle, aoa, aw_boat_speed, aw_boat_angle)
sail_force_global_x = boat_to_global(init_dir, jnp.array((sail_force_boat[0], 0)))
sail_force_global_y = boat_to_global(init_dir, jnp.array((0, sail_force_boat[1])))

centreboard_force_boat = calc_centreboard_forces(init_boat_vel)
centreboard_force_global = boat_to_global(init_dir, centreboard_force_boat)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1)

wind_scaler = 1.5
ax.arrow(-float(wind_vel[0]) * wind_scaler, -float(wind_vel[1]) * wind_scaler, float(wind_vel[0]), float(wind_vel[1]), head_width=0.4, color="green", label="Wind", head_starts_at_zero=True)
ax.arrow(-float(aw[0]) * wind_scaler, -float(aw[1]) * wind_scaler, float(aw[0]), float(aw[1]), head_width=0.4, color="blue", label="Apparent Wind Global", head_starts_at_zero=True)
# ax.arrow(0, 0, float(boat_delta[0]) * 2, float(boat_delta[1]) * 2, head_width=0.4, color="pink", label="Boat Vel", head_starts_at_zero=True, zorder=2)

# plt.axline([0, 0], [float(boat_delta[0]), float(boat_delta[1])], linestyle='--', color="pink", zorder=1)
plt.plot([0, sail[0]], [0, sail[1]], color="black", label="Sail", zorder=1)
# plt.axline([0, 0], [sail[0], sail[1]], linestyle='--', color="black", zorder=1)

sail_force_plot_scaler = 0.1
ax.arrow(0, 0, float(sail_force_global_x[0]) * sail_force_plot_scaler, float(sail_force_global_x[1]) * sail_force_plot_scaler, label="Sail Force X", head_width=0.4, color="purple")
ax.arrow(0, 0, float(sail_force_global_y[0]) * sail_force_plot_scaler, float(sail_force_global_y[1]) * sail_force_plot_scaler, label="Sail Force Y", head_width=0.4, color="pink")

# centreboard_force_plot_scaler = 1
# ax.arrow(0, 0, float(centreboard_force_global[0]) * centreboard_force_plot_scaler, 0, label="CBoard Force X", head_width=0.4, color="orange")
# ax.arrow(0, 0, 0, float(centreboard_force_global[1]) * centreboard_force_plot_scaler, label="CBoard Force Y", head_width=0.4, color="yellow")

# angle_plot = get_angle_plot_1(jnp.array(((0, 0), (float(boat_delta[0]), float(boat_delta[1])))),
#                              jnp.array(((0, 0), (sail[0], sail[1]))),
#                              init_dir,
#                              aw_boat_angle)
# angle_text = get_angle_text(angle_plot)
# ax.add_patch(angle_plot) # To display the angle arc
# ax.text(*angle_text) # To display the angle value
#
# angle_plot = get_angle_plot_2(jnp.array(((0, 0), (sail[0], sail[1]))),
#                               jnp.array(((0, 0), (aw[0], aw[1]))),
#                               init_dir,
#                               aw_boat_angle,
#                               sail_angle,
#                               aoa)
# angle_text = get_angle_text(angle_plot)
# ax.add_patch(angle_plot) # To display the angle arc
# ax.text(*angle_text) # To display the angle value

t = matplotlib.markers.MarkerStyle(marker=create_boat_marker())
t._transform = t.get_transform().rotate_deg(jnp.degrees(init_dir.squeeze()))
ax.plot(0, 0, label="Boat", marker=t, markersize=30, zorder=0)
# plt.arrow(0, 0, float(init_boat_vel[0]), float(init_boat_vel[1]), head_width=0.4, color="pink", label="Boat Vel")
screen_size = 10
ax.set_xlim(-screen_size, screen_size)
ax.set_ylim(-screen_size, screen_size)
# plt.axis('equal')
# plt.scatter(0, 0, marker=p, s=400)
ax.arrow(-screen_size+2, screen_size-2, 0, 1, head_width=0.2, color="black", label="Global Y")
ax.arrow(-screen_size+2, screen_size-2, 1, 0, head_width=0.2, color="black", label="Global X")
plt.legend()
plt.show()


"https://www.sailingworld.com/how-to/angles-of-attack/"

"https://tuprints.ulb.tu-darmstadt.de/8471/7/sailboat_model_irsc.pdf"