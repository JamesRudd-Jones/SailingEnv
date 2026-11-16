"""
Stability: the question that started all this. Does it blow up, and if so why?

The spectral radius of one integration step says how much a small error grows per step, so above 1 means growth.
Numerical instability gets worse as the timestep grows, while real growth in the model tends to exp(rate * dt), which
goes to 1 as the timestep shrinks. The plots separate the two, and show that large leeway is not where it fails.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import analysis, dynamics
from project_name.sailing.angles import wind_velocity
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES


def main() -> Checks:
    checks = Checks("Stability and numerics")
    params = BoatParams()

    radii = analysis.stability_grid(params)
    worst = float(radii.max())
    index = np.unravel_index(int(radii.argmax()), radii.shape)
    where = {}
    for name, axis, i in zip(analysis.GRID_NAMES, analysis.GRID_AXES, index):
        value = float(axis[i])
        where[name] = round(value, 1) if name in ("speed", "wind") else round(float(np.degrees(value)), 1)
    checks.check("Worst growth per 0.05 s step is small across all states and actions", worst < 1.1,
                 f"radius {worst:.3f} at {where}")

    sharp = params.replace(rudder=params.rudder.replace(stall_width=0.03), board=params.board.replace(stall_width=0.03))
    timesteps = (0.5, 0.2, 0.1, 0.05, 0.025, 0.0125)
    current = [float(analysis.stability_grid(params, dt).max()) for dt in timesteps]
    too_sharp = [float(analysis.stability_grid(sharp, dt).max()) for dt in timesteps]
    checks.check("Growth shrinks towards 1 as the timestep shrinks", current[4] < current[3] and current[5] < current[4],
                 "radius " + ", ".join(f"{r:.3f}" for r in current[3:]))
    checks.check("A 0.5 s decision step needs substeps", current[0] > 1.5, f"radius {current[0]:.3g} at 0.5 s without substeps")

    def steady_speed(dt):
        policy = analysis.heading_hold(jnp.array(0.0), jnp.radians(45.0))
        trajectory = analysis.rollout(analysis.start_state(0.0, u=1.0), policy, wind_velocity(5.0, 0.0), params, 60.0, dt)
        return float(trajectory[-1, dynamics.U])

    coarse, fine = steady_speed(0.05), steady_speed(0.005)
    checks.check("Steady speed does not depend on the timestep", abs(coarse / fine - 1.0) < 0.01,
                 f"{coarse:.3f} m/s at 0.05 s, {fine:.3f} m/s at 0.005 s")

    def random_rollout(key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        wind = wind_velocity(jrandom.uniform(k1, minval=1.0, maxval=20.0), 0.0)
        velocity = jrandom.uniform(k3, (3,), minval=jnp.array((-3.0, -3.0, -1.0)), maxval=jnp.array((6.0, 3.0, 1.0)))
        boat = analysis.start_state(jrandom.uniform(k2, minval=-3.14, maxval=3.14), *velocity)
        actions = jnp.repeat(jrandom.uniform(k4, (60, 2), minval=-1.5, maxval=1.5), 40, axis=0)

        def body(state, action):
            new_state = analysis.step(state, action, wind, params)
            return new_state, jnp.hypot(new_state[dynamics.U], new_state[dynamics.V])

        final, speeds = jax.lax.scan(body, boat, actions)
        return jnp.all(jnp.isfinite(final)), jnp.max(speeds[200:])

    finite, top_speed = jax.jit(jax.vmap(random_rollout))(jrandom.split(jrandom.key(2), 8192))
    checks.check("8192 boats with random hard actions for 120 s stay finite", jnp.all(finite),
                 f"{int(jnp.sum(~finite))} failed, top speed after 10 s {float(jnp.max(top_speed)):.2f} m/s")

    @jax.jit
    def throughput_run(boats):
        action, wind = jnp.array((0.0, 0.5)), wind_velocity(5.0, 0.0)
        return jax.lax.scan(lambda z, _: (jax.vmap(lambda b: analysis.step(b, action, wind, params, 0.1, 2))(z), None),
                            boats, None, 1000)[0]

    boats = jnp.tile(analysis.start_state(1.0, u=1.0), (4096, 1))
    throughput_run(boats).block_until_ready()
    started = time.time()
    throughput_run(boats).block_until_ready()
    rate = 4096 * 1000 / (time.time() - started)
    print(f"  Throughput on {jax.devices()[0].platform}, physics only, 0.1 s step with 2 substeps: {rate:,.0f} boat steps/s")

    fig, (ax_map, ax_dt) = plt.subplots(1, 2, figsize=(13, 5))

    by_speed_leeway = radii.max(axis=tuple(range(2, radii.ndim)))
    speeds, leeways = np.array(analysis.GRID_AXES[0]), np.degrees(np.array(analysis.GRID_AXES[1]))
    image = ax_map.imshow(by_speed_leeway, aspect="auto", origin="lower", cmap="Blues", vmin=1.0, vmax=1.1)
    ax_map.set_xticks(range(len(leeways)), [f"{a:.0f}" for a in leeways])
    ax_map.set_yticks(range(len(speeds)), [f"{s:.1f}" for s in speeds])
    ax_map.grid(False)
    ax_map.set_title("Worst growth per step: speed against leeway")
    ax_map.set(xlabel="Leeway (deg)", ylabel="Boat speed (m/s)")
    fig.colorbar(image, ax=ax_map, label="Spectral radius")

    ax_dt.set_title("Worst growth against timestep")
    ax_dt.plot(timesteps, current, "o-", color=SERIES[0], label="Current stall width, 0.12", markersize=8)
    ax_dt.plot(timesteps, too_sharp, "o-", color=SERIES[1], label="Sharp stall, width 0.03", markersize=8)
    ax_dt.axhline(1.0, color=SERIES[2], linewidth=1, label="No growth")
    ax_dt.axvline(0.05, color=SERIES[2], linewidth=1, linestyle=":", label="Substep used")
    ax_dt.set(xscale="log", yscale="log", xlabel="Timestep (s)", ylabel="Spectral radius")
    ax_dt.legend()

    return checks.finish(fig, "stability")


if __name__ == "__main__":
    main()
