"""
Manoeuvres: the behaviour that should emerge from the foils and the boom rather than being programmed.

Tacking, gybing, braking with extreme rudder, backing the main, and getting out of irons. None of these has any code of
its own in the dynamics, so each one passing is evidence that the physics is doing the work.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import analysis, dynamics
from project_name.sailing.angles import wind_velocity, wrap
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES


DT = analysis.SUBSTEP


def main() -> Checks:
    checks = Checks("Manoeuvres")
    params = BoatParams()
    wind = wind_velocity(5.0, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    (ax_tack, ax_gybe, ax_brake), (ax_back, ax_irons, ax_astern) = axes

    # Head to wind
    north = jnp.array(np.pi / 2)
    head_to_wind = np.array(analysis.rollout(analysis.start_state(north, u=2.0), analysis.heading_hold(north, jnp.radians(10.0)),
                                             wind, params, 30.0))
    checks.check("Cannot sail directly into the wind", head_to_wind[-1, dynamics.U] < 0.2,
                 f"2.00 m/s falls to {head_to_wind[-1, dynamics.U]:.2f} m/s in 30 s")

    # Tack, with the helm hard over and the sheet where it is fastest close-hauled
    start, end = jnp.radians(90.0 - 45.0), jnp.radians(90.0 + 45.0)
    tack_policy = analysis.tack_helm(start, end, 60.0, jnp.radians(15.0))
    tack = np.array(analysis.rollout(analysis.start_state(start, u=1.0), tack_policy, wind, params, 100.0))
    time_axis = np.arange(len(tack)) * DT
    before, during = tack[1199], tack[1200:1400]
    kept = during[:, dynamics.U].min() / before[dynamics.U]
    checks.check("Tack carries way through the wind", during[:, dynamics.U].min() > 0.0,
                 f"{before[dynamics.U]:.2f} falls to {during[:, dynamics.U].min():.2f} m/s, ends {tack[-1, dynamics.U]:.2f} m/s")
    # Provisional, like the polar targets: a sailor's figure for a Firefly tack should replace it
    checks.check("Tack keeps at least a quarter of its speed", kept > 0.25, f"keeps {100 * kept:.0f}% at the slowest")
    settled = np.argmax(np.abs(np.array(wrap(jnp.array(tack[1200:, dynamics.HEADING]) - end))) < np.radians(5.0)) * DT
    checks.check("Tack is on the new course within 8 s", 0.0 < settled < 8.0, f"within 5 deg after {settled:.1f} s")
    checks.check("Boom crosses by itself in the tack", before[dynamics.BOOM] * tack[-1, dynamics.BOOM] < 0.0,
                 f"{np.degrees(before[dynamics.BOOM]):.0f} to {np.degrees(tack[-1, dynamics.BOOM]):.0f} deg")
    ax_tack.set_title("Speed through a tack and a gybe")
    ax_tack.plot(time_axis - 60.0, tack[:, dynamics.U], color=SERIES[0], label="Tack, 45° each side of the wind")
    ax_tack.set(xlim=(-5, 25), xlabel="Time from the turn starting (s)", ylabel="Forward speed (m/s)")
    ax_gybe.set_title("Boom through a tack and a gybe")
    ax_gybe.plot(time_axis - 60.0, np.degrees(tack[:, dynamics.BOOM]), color=SERIES[0], label="Tack, sheet 15°, helm 45°")
    ax_gybe.set(xlim=(-5, 25), xlabel="Time from the turn starting (s)", ylabel="Boom angle (deg), positive to starboard")

    # Gybe
    run_start, run_end = jnp.radians(90.0 - 150.0), jnp.radians(90.0 + 150.0)

    def gybe_policy(boat, t):
        return analysis.heading_hold(jnp.where(t < 30.0, run_start, run_end), jnp.radians(70.0))(boat, t)

    gybe = np.array(analysis.rollout(analysis.start_state(run_start, u=1.0), gybe_policy, wind, params, 70.0))
    checks.check("Boom gybes when the wind crosses the stern", gybe[599, dynamics.BOOM] * gybe[-1, dynamics.BOOM] < 0.0,
                 f"{np.degrees(gybe[599, dynamics.BOOM]):.0f} to {np.degrees(gybe[-1, dynamics.BOOM]):.0f} deg")
    gybe_time = np.arange(len(gybe)) * DT - 30.0
    ax_tack.plot(gybe_time, gybe[:, dynamics.U], color=SERIES[1], label="Gybe, 150° each side of the wind")
    ax_tack.legend()
    ax_gybe.plot(gybe_time, np.degrees(gybe[:, dynamics.BOOM]), color=SERIES[1], label="Gybe, sheet 70°")
    ax_gybe.legend()

    # Steady reach, the starting point for the rest
    reach = analysis.rollout(analysis.start_state(0.0, u=1.0), analysis.heading_hold(jnp.array(0.0), jnp.radians(45.0)),
                             wind, params, 60.0)[-1]
    print(f"  Starting from a steady beam reach at {float(reach[dynamics.U]):.2f} m/s, true wind 5 m/s")

    # Braking with the rudder
    ax_brake.set_title("Holding the rudder over from a steady reach")
    after_5s = {}
    for index, degrees in enumerate((0.0, 30.0, 60.0, 85.0)):
        command = jnp.radians(jnp.array((degrees, 45.0)))
        trajectory = np.array(analysis.rollout(reach, lambda b, t: command, wind, params, 5.0))
        speed = np.hypot(trajectory[:, dynamics.U], trajectory[:, dynamics.V])
        after_5s[degrees] = speed[-1]
        ax_brake.plot(np.arange(len(speed)) * DT, speed, color=SERIES[index], label=f"{degrees:.0f}°")
    ax_brake.set(xlabel="Time (s)", ylabel="Boat speed (m/s)")
    ax_brake.legend(title="Rudder")
    checks.check("Extreme rudder sheds more speed than moderate rudder", after_5s[85.0] < after_5s[30.0] < after_5s[0.0],
                 ", ".join(f"{d:.0f} deg: {s:.2f} m/s" for d, s in after_5s.items()))

    # Backing the main
    ax_back.set_title("Backing the main from a steady reach, rudder centred")
    slowest = {}
    for index, degrees in enumerate((45.0, -30.0, -60.0, -85.0)):
        command = jnp.array((0.0, jnp.radians(degrees)))
        trajectory = np.array(analysis.rollout(reach, lambda b, t: command, wind, params, 20.0))
        slowest[degrees] = trajectory[:, dynamics.U].min()
        label = "Sheeted, not backed" if degrees > 0 else f"Backed {abs(degrees):.0f}°"
        ax_back.plot(np.arange(len(trajectory)) * DT, trajectory[:, dynamics.U], color=SERIES[index], label=label)
    ax_back.axhline(0.0, color=SERIES[0], linewidth=0.6)
    ax_back.set(xlabel="Time (s)", ylabel="Forward speed (m/s)")
    ax_back.legend()
    checks.check("Backing the main stops the boat and sends it astern", slowest[-60.0] < -0.1, f"lowest u {slowest[-60.0]:.2f} m/s")
    checks.check("Going astern stays at a sensible speed", slowest[-60.0] > -1.5, f"lowest u {slowest[-60.0]:.2f} m/s")

    # Irons
    ax_irons.set_title("In irons: stopped, 5 degrees off the wind")
    trials = {"Eased, rudder centred": (0.0, 90.0),
              "Backed, rudder centred": (0.0, -70.0),
              "Backed, rudder to port": (-35.0, -70.0),
              "Backed, rudder to starboard": (35.0, -70.0)}
    off_wind = {}
    bow = jnp.radians(90.0 - 5.0)
    for index, (name, (rudder, sail)) in enumerate(trials.items()):
        command = jnp.radians(jnp.array((rudder, sail)))
        trajectory = np.array(analysis.rollout(analysis.start_state(bow), lambda b, t: command, wind, params, 20.0))
        degrees_off = np.degrees(np.array(wrap(jnp.array(np.pi / 2) - trajectory[:, dynamics.HEADING])))
        off_wind[name] = degrees_off[-1]
        ax_irons.plot(np.arange(len(trajectory)) * DT, degrees_off, color=SERIES[index], label=name)
    ax_irons.set(xlabel="Time (s)", ylabel="Degrees off the wind")
    ax_irons.legend()
    checks.check("Backing with rudder gets the bow away from the wind",
                 abs(off_wind["Backed, rudder to port"]) > 30.0 and abs(off_wind["Backed, rudder to starboard"]) > 30.0,
                 f"{off_wind['Backed, rudder to port']:.0f} and {off_wind['Backed, rudder to starboard']:.0f} deg")
    # Not a target: a sailor uses the rudder for this. With it centred the boat simply goes astern, holding its heading
    print(f"  Note: backed with the rudder centred, the bow stays {off_wind['Backed, rudder centred']:.1f} deg off the wind")

    # Rudder going astern
    speeds = jnp.linspace(-2.0, 2.0, 81)
    rates = np.array([float(dynamics.boat_derivative(analysis.start_state(0.0, u=float(u), rudder=jnp.radians(15.0)),
                                                     jnp.zeros(2), jnp.zeros(2), params)[dynamics.R]) for u in speeds])
    checks.check("Same rudder turns the boat the opposite way going astern", rates[0] > 0.0 > rates[-1],
                 f"yaw acceleration {rates[0]:.2f} at -2 m/s, {rates[-1]:.2f} at +2 m/s")
    ax_astern.set_title("Yaw acceleration from 15 degrees of rudder")
    ax_astern.plot(np.array(speeds), rates, color=SERIES[0])
    ax_astern.set(xlabel="Forward speed u (m/s)", ylabel="Yaw acceleration (rad/s²)")

    return checks.finish(fig, "manoeuvres")


if __name__ == "__main__":
    main()
