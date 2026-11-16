"""
Sweeps the least certain sail coefficients, showing the effect on performance and on stability together.

Parasitic drag: two crew sitting out are about 1.2 m^2 at a drag coefficient near 1, which is about 0.14 of the 8.36 m^2
sail area. Hull and rig add perhaps 0.03, and the sail's own profile drag about 0.03, so around 0.2 in all.
Luff angle: how far a soft sail must be from the wind before it holds its shape.
"""

import jax.numpy as jnp
import numpy as np

from project_name.sailing import analysis
from project_name.sailing.dynamics import BoatParams


ANGLES = jnp.array((20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 90.0))


def describe(params: BoatParams, label: str) -> None:
    result = {k: np.array(v) for k, v in analysis.polar(ANGLES, 5.0, params).items()}
    radii = analysis.stability_grid(params)
    index = np.unravel_index(int(radii.argmax()), radii.shape)
    where = [round(float(np.degrees(axis[i])), 0) if name not in ("speed", "wind") else float(axis[i])
             for name, axis, i in zip(analysis.GRID_NAMES, analysis.GRID_AXES, index)]
    upwind = np.array(ANGLES) < 90.0
    best = int(np.argmax(np.where(upwind, result["vmg"], -np.inf)))
    print(f"{label:24s} speed@20={result['speed'][0]:.2f} @45={result['speed'][3]:.2f} @90={result['speed'][6]:.2f} "
          f"ratio20/45={result['speed'][0] / result['speed'][3]:.0%} leeway@20={result['leeway'][0]:5.1f} "
          f"bestVMG={result['vmg'][best]:.2f}@{float(ANGLES[best]):.0f} radius={radii.max():.3f} at {where}")


if __name__ == "__main__":
    print(f"Radius worst state is {analysis.GRID_NAMES}, true wind 5 m/s for performance")
    base = BoatParams()
    # A main and jib together hold their lift to larger angles, and to a higher peak, than a single soft sail
    for stall, width in ((22.0, 0.10), (26.0, 0.08), (28.0, 0.06), (30.0, 0.06)):
        sail = base.sail.replace(cd0=0.20, luff_angle=jnp.radians(10.0), stall_angle=jnp.radians(stall), stall_width=width)
        describe(base.replace(sail=sail), f"sail stall {stall:.0f} w{width:.2f}")

    # The worst growth is the rudder stalling at speeds beyond what this boat reaches
    sail = base.sail.replace(cd0=0.20, luff_angle=jnp.radians(10.0), stall_angle=jnp.radians(28.0), stall_width=0.06)
    for width in (0.15, 0.18):
        rudder = base.rudder.replace(stall_width=width)
        describe(base.replace(sail=sail, rudder=rudder), f"rudder width {width:.2f}")
