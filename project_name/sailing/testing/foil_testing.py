"""
Foils: one function for the sail, centreboard and rudder, with no polars or lookup tables.

Checks that a foil stalls, that it keeps working in every direction of flow, and that no angle causes a force jump,
because jumps at particular angles were one source of the old blow-ups.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import dynamics
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES


def main() -> Checks:
    checks = Checks("Foils: stall and 360 degree inflow")
    params = BoatParams()
    foils = {"rudder": (params.rudder, params.water_density, 2.5),
             "centreboard": (params.board, params.water_density, 2.5),
             "sail": (params.sail, params.air_density, 5.0)}

    fig, (ax_rudder, ax_coeff, ax_all) = plt.subplots(1, 3, figsize=(15, 4.6))

    degrees = jnp.arange(0.0, 90.5, 0.5)
    fx, fy = jax.vmap(lambda d: dynamics.foil_force(-2.5, 0.0, jnp.radians(d), params.rudder, params.water_density))(degrees)
    side, drag = np.array(fy), -np.array(fx)
    peak = int(np.argmax(side))
    at_45 = side[int(45.0 / 0.5)]
    checks.check("Rudder side force peaks near its stall angle", 10.0 <= degrees[peak] <= 30.0,
                 f"peak {side[peak]:.0f} N at {float(degrees[peak]):.1f} deg")
    checks.check("Rudder side force clearly falls after the stall", at_45 < 0.8 * side[peak],
                 f"{at_45:.0f} N at 45 deg, {at_45 / side[peak]:.0%} of peak")
    checks.check("Rudder side force reaches zero with the blade square to the flow", abs(side[-1]) < 1.0, f"{side[-1]:.2f} N")
    checks.check("Rudder drag keeps rising to 90 degrees, so it can brake", np.all(np.diff(drag) > 0.0),
                 f"{drag[0]:.0f} N at 0 deg, {drag[-1]:.0f} N at 90 deg")

    ax_rudder.set_title("Rudder force at 2.5 m/s")
    ax_rudder.plot(degrees, side, color=SERIES[0], label="Side force")
    ax_rudder.plot(degrees, drag, color=SERIES[1], label="Drag")
    ax_rudder.axvline(float(degrees[peak]), color=SERIES[0], linewidth=1, linestyle=":")
    ax_rudder.set(xlabel="Rudder angle (deg)", ylabel="Force (N)")
    ax_rudder.legend()

    inflow = jnp.radians(jnp.arange(0.0, 360.0, 0.25))
    for index, (name, (foil, density, speed)) in enumerate(foils.items()):
        def force(angle):
            flow_x, flow_y = -speed * jnp.cos(angle), speed * jnp.sin(angle)
            return jnp.stack(dynamics.foil_force(flow_x, flow_y, 0.0, foil, density))

        forces = np.array(jax.vmap(force)(inflow))
        magnitude = np.linalg.norm(forces, axis=-1)
        jump = np.max(np.linalg.norm(np.diff(forces, axis=0), axis=-1)) / magnitude.max()
        checks.check(f"No force jump around 360 degrees of inflow: {name}", jump < 0.05,
                     f"largest change per 0.25 deg is {jump:.2%} of peak")

        flow = np.stack((-speed * np.cos(inflow), speed * np.sin(inflow)), axis=-1)
        work = np.sum(forces * flow, axis=-1)
        checks.check(f"Force always takes energy from the flow, never adds it: {name}", np.all(work > 0.0),
                     f"smallest power into the foil {work.min():.2f} W")

        normalised = magnitude / (0.5 * density * foil.area * speed ** 2)
        ax_all.plot(np.degrees(inflow), normalised, color=SERIES[index], label=name.capitalize())

    ax_all.set_title("Force coefficient over all inflow angles")
    ax_all.set(xlabel="Inflow angle (deg)", ylabel="Force / (dynamic pressure x area)", xticks=np.arange(0, 361, 90))
    ax_all.legend(loc="upper center")

    ax_coeff.set_title("Rudder side force against drag, by angle")
    ax_coeff.plot(drag, side, color=SERIES[0])
    for label_degrees in (5.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        i = int(label_degrees / 0.5)
        ax_coeff.plot(drag[i], side[i], "o", color=SERIES[0], markersize=7)
        ax_coeff.annotate(f"{label_degrees:.0f}°", (drag[i], side[i]), textcoords="offset points", xytext=(6, 6))
    ax_coeff.set(xlabel="Drag (N)", ylabel="Side force (N)")

    return checks.finish(fig, "foils")


if __name__ == "__main__":
    main()
