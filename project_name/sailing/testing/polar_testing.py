"""
Performance: how the boat sails at each angle to the wind, measured by simulating it.

The dynamics never read a polar. This runs the boat on each heading at every sheet setting until it settles and keeps
the best, so a polar is only the output. No measured Firefly polars could be found, so apart from the sourced hull speed
these targets are provisional, from general sailing knowledge.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import analysis
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES


ANGLES = jnp.array((20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 75.0, 90.0, 110.0, 135.0, 150.0, 175.0))
WIND_SPEEDS = (2.5, 5.0, 8.0, 12.0)


def main() -> Checks:
    checks = Checks("Performance: simulated polars")
    params = BoatParams()
    angles = np.array(ANGLES)

    fig = plt.figure(figsize=(13, 5.5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_vmg = fig.add_subplot(1, 2, 2)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_thetamax(180.0)
    ax_polar.set_title("Boat speed (m/s) against true wind angle")

    results = {}
    for index, wind_speed in enumerate(WIND_SPEEDS):
        result = {k: np.array(v) for k, v in analysis.polar(ANGLES, wind_speed, params).items()}
        results[wind_speed] = result
        print(f"  True wind {wind_speed} m/s\n     TWA  speed  leeway  sheet    VMG")
        for i in range(len(angles)):
            print(f"     {angles[i]:3.0f}  {result['speed'][i]:5.2f}  {result['leeway'][i]:6.1f}  {result['sheet'][i]:5.0f}"
                  f"  {result['vmg'][i]:5.2f}")
        ax_polar.plot(np.radians(angles), result["speed"], "o-", color=SERIES[index], label=f"{wind_speed} m/s", markersize=4)
        ax_vmg.plot(angles, result["vmg"], "o-", color=SERIES[index], label=f"{wind_speed} m/s", markersize=4)

    ax_polar.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1))
    ax_vmg.set_title("Speed made good towards the wind")
    ax_vmg.axhline(0.0, color=SERIES[0], linewidth=0)
    ax_vmg.set(xlabel="True wind angle (deg)", ylabel="VMG (m/s), negative is downwind")
    ax_vmg.legend()

    five = results[5.0]
    upwind = angles < 90.0
    best = angles[upwind][int(np.argmax(five["vmg"][upwind]))]
    checks.check("Best upwind angle is around 30 to 50 degrees", 30.0 <= best <= 50.0, f"{best:.0f} deg at 5 m/s")

    leeway = np.abs(five["leeway"][list(angles).index(45.0)])
    checks.check("Leeway is a few degrees close-hauled", 2.0 <= leeway <= 8.0, f"{leeway:.1f} deg at TWA 45, 5 m/s")

    reach = [results[w]["speed"][list(angles).index(90.0)] for w in WIND_SPEEDS]
    checks.check("Stronger wind means more speed", np.all(np.diff(reach) > 0.0),
                 "beam reach " + ", ".join(f"{s:.2f}" for s in reach) + " m/s")

    # Speed alone is a poor test of pinching, as a sweep of drag and luff showed: it barely moves the ratio to 45 degrees.
    # What pinching should do is lose ground to windward and slip sideways
    at_45 = list(angles).index(45.0)
    checks.check("Pinching at 20 degrees loses ground to windward", five["vmg"][0] < 0.95 * five["vmg"].max(),
                 f"VMG {five['vmg'][0]:.2f} against best {five['vmg'].max():.2f} m/s")
    checks.check("Pinching at 20 degrees slips far more sideways", abs(five["leeway"][0]) > 2.0 * abs(five["leeway"][at_45]),
                 f"leeway {abs(five['leeway'][0]):.1f} deg against {abs(five['leeway'][at_45]):.1f} deg at TWA 45")

    # The one sourced speed: 4.6 kn hull speed. Below it in light wind, past it once the dinghy can plane
    reach_light, reach_fresh = results[2.5]["speed"].max(), results[12.0]["speed"].max()
    checks.check("Under hull speed in light wind", reach_light < params.hull_speed, f"{reach_light:.2f} m/s in 2.5 m/s of wind")
    checks.check("Past hull speed in fresh wind", reach_fresh > params.hull_speed, f"{reach_fresh:.2f} m/s in 12 m/s of wind")

    return checks.finish(fig, "polars")


if __name__ == "__main__":
    main()
