"""
Rigid body and hull: the equations of motion before any wind matters.

Checks that the velocity terms do no work, that port and starboard behave identically, and that hull drag has no jump as
forward speed crosses zero. The old wave term jumped there when the boat was also moving sideways, which is what caused
the explosions when reversing. The plot compares the two directly.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import analysis, dynamics
from project_name.sailing.angles import wind_velocity
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES


def old_wave_force(u: np.ndarray, v: float) -> np.ndarray:
    """The wave term from SailingEnv.py, kept here only for comparison."""
    speed_squared = u ** 2 + v ** 2
    return -np.sign(u) * 0.5 * 1000.0 * speed_squared * 2.5 * (np.sqrt(speed_squared) / 2.5) ** 4


def main() -> Checks:
    checks = Checks("Rigid body and hull")
    params = BoatParams()
    momentum_scale = jnp.array((params.mass_x, params.mass_y, params.inertia_z))

    def velocity_terms_only(nu):
        return dynamics.rigid_body_acceleration(nu[0], nu[1], nu[2], 0.0, 0.0, 0.0, params)

    def rk4(nu, dt=0.05):
        k1 = velocity_terms_only(nu)
        k2 = velocity_terms_only(nu + 0.5 * dt * k1)
        k3 = velocity_terms_only(nu + 0.5 * dt * k2)
        k4 = velocity_terms_only(nu + dt * k3)
        return nu + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    powers = jax.vmap(lambda nu: jnp.dot(momentum_scale * nu, velocity_terms_only(nu)))(
        jrandom.uniform(jrandom.key(0), (1024, 3), minval=-5.0, maxval=5.0))
    worst = float(jnp.max(jnp.abs(powers)))
    checks.check("Velocity terms do no work", worst < 1e-2, f"worst power {worst:.1e} W")

    _, history = jax.lax.scan(lambda nu, _: (rk4(nu), rk4(nu)), jnp.array((2.0, 0.5, 0.3)), None, 1200)
    energy = np.array(0.5 * jnp.sum(momentum_scale * history ** 2, axis=-1))
    drift = np.abs(energy / energy[0] - 1.0)
    checks.check("Kinetic energy held for 60 s with all forces off", drift.max() < 1e-3, f"worst drift {drift.max():.1e}")

    def mirror_difference(key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        boat = jrandom.uniform(k1, (dynamics.NUM_STATES,), minval=-1.5, maxval=1.5)
        rudder, sail = jrandom.uniform(k2, minval=-1.0, maxval=1.0), jrandom.uniform(k3, minval=-1.5, maxval=1.5)
        wind = jrandom.uniform(k4, (2,), minval=-8.0, maxval=8.0)
        flip = jnp.array((1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0))
        base = dynamics.boat_derivative(boat, jnp.stack((rudder, sail)), wind, params)
        mirrored = dynamics.boat_derivative(flip * boat, jnp.stack((-rudder, sail)), wind * jnp.array((1.0, -1.0)), params)
        return jnp.max(jnp.abs(flip * mirrored - base))

    asymmetry = float(jnp.max(jax.vmap(mirror_difference)(jrandom.split(jrandom.key(1), 512))))
    checks.check("Port and starboard are mirror images", asymmetry < 1e-2, f"worst difference {asymmetry:.1e}")

    jacobian = jax.jacfwd(lambda b: dynamics.boat_derivative(b, jnp.zeros(2), jnp.zeros(2), params))
    checks.check("Gradients are finite at rest in no wind", jnp.all(jnp.isfinite(jacobian(jnp.zeros(dynamics.NUM_STATES)))))

    surge_speeds = np.linspace(-0.5, 0.5, 2001)
    new_surge = np.array(jax.vmap(lambda u: dynamics.hull_force(u, 1.5, 0.0, params)[0])(jnp.array(surge_speeds)))
    new_jump = float(np.max(np.abs(np.diff(new_surge))))
    old_surge = old_wave_force(surge_speeds, 1.5)
    old_jump = float(np.max(np.abs(np.diff(old_surge))))
    checks.check("Hull surge force has no jump as u crosses zero while moving sideways", new_jump < 1.0,
                 f"largest step {new_jump:.3f} N, old wave term {old_jump:.0f} N")

    fig, (ax_energy, ax_mirror, ax_jump) = plt.subplots(1, 3, figsize=(15, 4.6))

    ax_energy.set_title("Kinetic energy with all forces off")
    ax_energy.plot(np.arange(1200) * 0.05, energy / energy[0], color=SERIES[0])
    ax_energy.set(xlabel="Time (s)", ylabel="Energy / starting energy", ylim=(0.95, 1.05))

    wind = wind_velocity(5.0, 0.0)
    ax_mirror.set_title("Port and starboard tacks, wind from north")
    for index, (name, compass) in enumerate((("Starboard tack", -45.0), ("Port tack", 45.0))):
        heading = jnp.radians(90.0 - compass)
        trajectory = np.array(analysis.rollout(analysis.start_state(heading, u=1.0),
                                               analysis.heading_hold(heading, jnp.radians(20.0)), wind, params, 30.0))
        ax_mirror.plot(trajectory[:, dynamics.X], trajectory[:, dynamics.Y], color=SERIES[index], label=name)
        if index == 0:
            first = trajectory
        else:
            gap = np.max(np.abs(first[:, dynamics.X] + trajectory[:, dynamics.X]))
            checks.check("The two tacks trace mirror image tracks", gap < 1e-2, f"worst gap {gap:.1e} m")
    ax_mirror.set(xlabel="East (m)", ylabel="North (m)", aspect="equal")
    ax_mirror.legend()

    ax_jump.set_title("Surge force as u crosses zero, with v = 1.5 m/s")
    ax_jump.plot(surge_speeds, old_surge, color=SERIES[1], label="Old wave term")
    ax_jump.plot(surge_speeds, new_surge, color=SERIES[0], label="New hull drag")
    ax_jump.set(xlabel="Forward speed u (m/s)", ylabel="Surge force (N)")
    ax_jump.legend()

    return checks.finish(fig, "rigid_body")


if __name__ == "__main__":
    main()
