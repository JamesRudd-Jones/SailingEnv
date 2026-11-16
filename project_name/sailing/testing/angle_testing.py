"""
Angles: physics works in maths angles, while people set and read compass angles.

Checks the conversion, and that the two agree physically: where the wind lands on the boat, and which way the rudder
turns it. The plot shows four compass headings in a wind from north, then a turn on positive rudder.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import analysis, dynamics
from project_name.sailing.angles import compass_to_maths, maths_to_compass, wind_velocity
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES, TEXT_SECONDARY


def main() -> Checks:
    checks = Checks("Angles: compass outside, maths inside")
    params = BoatParams()
    wind = wind_velocity(5.0, 0.0)

    compass = jnp.arange(0.0, 360.0, 15.0)
    error = float(jnp.max(jnp.abs(maths_to_compass(compass_to_maths(compass)) - compass)))
    checks.check("Compass to maths and back returns the same angle", error < 1e-3, f"worst error {error:.1e} deg")

    east = float(compass_to_maths(jnp.array(90.0)))
    north = float(compass_to_maths(jnp.array(0.0)))
    checks.check("East is +x and north is +y", abs(east) < 1e-6 and abs(north - np.pi / 2) < 1e-6,
                 f"east {np.degrees(east):.1f}, north {np.degrees(north):.1f} deg in maths")

    checks.check("Wind from north blows south", np.allclose(wind, (0.0, -5.0), atol=1e-5), f"velocity {np.round(wind, 3)}")

    fig, (ax_wind, ax_turn) = plt.subplots(1, 2, figsize=(11, 5))
    ax_wind.set_title("Wind angle by compass heading, wind from north")
    ax_wind.annotate("", xy=(0, -1.6), xytext=(0, 1.6), arrowprops=dict(color=TEXT_SECONDARY, width=1.5, headwidth=8))
    ax_wind.text(0.08, 1.45, "wind", color=TEXT_SECONDARY)

    expected = {0.0: 0.0, 90.0: 90.0, 180.0: 180.0, 270.0: -90.0}
    for heading_compass, target in expected.items():
        heading = compass_to_maths(jnp.array(heading_compass))
        boat = analysis.start_state(float(heading))
        awa = float(jnp.degrees(dynamics.forces_and_moment(boat, wind, params)["apparent_wind_angle"]))
        difference = abs((awa - target + 180.0) % 360.0 - 180.0)
        side = {0.0: "head to wind", 90.0: "port beam", 180.0: "dead astern", 270.0: "starboard beam"}[heading_compass]
        checks.check(f"Heading {heading_compass:03.0f} puts the wind {side}", difference < 0.5, f"AWA {awa:.1f} deg")

        cx, cy = 3.0 * np.cos(float(heading)), 3.0 * np.sin(float(heading))
        dx, dy = 0.8 * np.cos(float(heading)), 0.8 * np.sin(float(heading))
        ax_wind.annotate("", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                         arrowprops=dict(color=SERIES[0], width=3, headwidth=11))
        beside_arrow = 1.0 if abs(cx) < 1.0 else 0.0  # north and south arrows take the label beside them, not on top
        below_arrow = 0.0 if abs(cx) < 1.0 else 1.0
        ax_wind.text(cx + beside_arrow, cy - below_arrow, f"{heading_compass:03.0f}°\nAWA {awa:.0f}°", ha="center", va="center",
                     color=TEXT_SECONDARY)
    ax_wind.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="East (m)", ylabel="North (m)", aspect="equal")

    start = compass_to_maths(jnp.array(90.0))
    command = jnp.radians(jnp.array((15.0, 45.0)))
    trajectory = np.array(analysis.rollout(analysis.start_state(float(start), u=2.0), lambda b, t: command, wind, params, 8.0))
    heading_compass = np.unwrap(np.array(maths_to_compass(trajectory[:, dynamics.HEADING])), period=360.0)
    checks.check("Positive rudder turns to starboard, raising compass heading", heading_compass[-1] > 90.0,
                 f"090 to {heading_compass[-1]:.0f} deg after 8 s")

    ax_turn.set_title("Positive rudder from heading 090")
    ax_turn.plot(trajectory[:, dynamics.X], trajectory[:, dynamics.Y], color=SERIES[0])
    ax_turn.plot(trajectory[0, dynamics.X], trajectory[0, dynamics.Y], "o", color=SERIES[0], markersize=8)
    ax_turn.text(trajectory[0, dynamics.X], trajectory[0, dynamics.Y] + 0.4, "start", color=TEXT_SECONDARY)
    ax_turn.set(xlabel="East (m)", ylabel="North (m)", aspect="equal")

    return checks.finish(fig, "angles")


if __name__ == "__main__":
    main()
