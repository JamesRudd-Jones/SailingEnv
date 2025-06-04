import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Arc
import math


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


wind_dir: float = 180.0  # deg
wind_speed: float = 5.0  # in ms^-1
wind_vel: jnp.ndarray = wind_speed * jnp.array((jnp.sin(jnp.radians(wind_dir)),
                                                                  jnp.cos(jnp.radians(wind_dir))))  # in ms^-1
init_dir = jnp.radians(jnp.ones(1,) * 200)
init_boat_vel = jnp.array((1.0, 0.0))

sail_angle = jnp.radians(45.0)  # between 0 and 90

def boat_to_global(boat_heading, val):
    delta_pos_x = val[0] * jnp.sin(boat_heading) - val[1] * jnp.cos(boat_heading)
    delta_pos_y = val[1] * jnp.sin(boat_heading) + val[0] * jnp.cos(boat_heading)
    return jnp.array((delta_pos_x.squeeze(), delta_pos_y.squeeze()))

def global_to_boat(boat_heading, val):  # TODO fill this in
    delta_pos_x = val[0] * jnp.sin(boat_heading) + val[1] * jnp.cos(boat_heading)
    delta_pos_y = val[1] * jnp.sin(boat_heading) - val[0] * jnp.cos(boat_heading)

    return jnp.array((delta_pos_x.squeeze(), delta_pos_y.squeeze()))

def apparent_wind(boat_heading, boat_vel, wind_vel):
    transformed_wind = global_to_boat(boat_heading, wind_vel)

    apparent_wind = transformed_wind - boat_vel
    apparent_wind_angle = -jnp.arctan2(apparent_wind[1], apparent_wind[0])
    apparent_wind_speed = jnp.sqrt(apparent_wind[0] ** 2 + apparent_wind[1] ** 2)

    return apparent_wind, apparent_wind_angle, apparent_wind_speed

def get_sail_angle_plot(boat_vel_global, sail_angle, apparent_wind_angle):
    initial_angle_rad = jnp.arctan2(-boat_vel_global[1], -boat_vel_global[0])

    true_sail_angle = -jnp.copysign(sail_angle, apparent_wind_angle)  # prevents sign of 0 when AWA == 0

    new_angle_rad = initial_angle_rad - true_sail_angle

    sail_length = 2

    end_point_x = sail_length * jnp.cos(new_angle_rad)
    end_point_y = sail_length * jnp.sin(new_angle_rad)

    # aoa = apparent_wind_angle - jnp.copysign(jnp.pi, apparent_wind_angle) - true_sail_angle
    aoa = (jnp.pi - jnp.abs(apparent_wind_angle)) - sail_angle

    return jnp.array((end_point_x, end_point_y)), aoa

boat_delta = boat_to_global(init_dir, init_boat_vel)
aw_boat, aw_boat_angle, aw_boat_speed = apparent_wind(init_dir, init_boat_vel, wind_vel)
tw_boat = global_to_boat(init_dir, wind_vel)
tw_boat_angle = -jnp.arctan2(tw_boat[1], tw_boat[0])
print(aw_boat)
print(f"AW Boat Angle : {jnp.degrees(aw_boat_angle)}")
print(f"TW Boat Angle : {jnp.degrees(tw_boat_angle)}")

aw = boat_to_global(init_dir, aw_boat)

sail, aoa = get_sail_angle_plot(boat_delta, sail_angle, aw_boat_angle)
print(f"AOA : {jnp.degrees(aoa)}")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1)

ax.arrow(-float(wind_vel[0]), -float(wind_vel[1])+0.9, float(wind_vel[0]), float(wind_vel[1]), head_width=0.4, color="green", label="Wind", head_starts_at_zero=True)
ax.arrow(-aw[0]*1.2, -aw[1]*1.2, aw[0], aw[1], head_width=0.4, color="blue", label="Apparent Wind Global", head_starts_at_zero=True)
ax.arrow(0, 0, float(boat_delta[0]) * 2, float(boat_delta[1]) * 2, head_width=0.4, color="pink", label="Boat Vel", head_starts_at_zero=True, zorder=2)


plt.axline([0, 0], [float(boat_delta[0]), float(boat_delta[1])], linestyle='--', color="pink", zorder=1)
plt.plot([0, sail[0]], [0, sail[1]], color="black", label="Sail", zorder=-1)
plt.axline([0, 0], [sail[0], sail[1]], linestyle='--', color="black", zorder=1)

angle_plot = get_angle_plot_1(jnp.array(((0, 0), (float(boat_delta[0]), float(boat_delta[1])))),
                             jnp.array(((0, 0), (sail[0], sail[1]))),
                             init_dir,
                             aw_boat_angle)
angle_text = get_angle_text(angle_plot)
ax.add_patch(angle_plot) # To display the angle arc
ax.text(*angle_text) # To display the angle value

angle_plot = get_angle_plot_2(jnp.array(((0, 0), (sail[0], sail[1]))),
                              jnp.array(((0, 0), (aw[0], aw[1]))),
                              init_dir,
                              aw_boat_angle,
                              sail_angle,
                              aoa)
angle_text = get_angle_text(angle_plot)
ax.add_patch(angle_plot) # To display the angle arc
ax.text(*angle_text) # To display the angle value

ax.scatter(0, 0, label="Boat", zorder=0)
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