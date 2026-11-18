"""
Tacking in detail: where the speed goes, what limits the turn, and how much helm a tack wants.

The manoeuvre section checks that a tack works. This one breaks a tack down, so that a change to the hull, the foils or
the sail can be judged by what it does to the speed lost and the time taken. The surge budget splits the force along
the boat into its sources, the last being the transfer of sway momentum into surge as the boat turns, m_y v r.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import analysis, dynamics
from project_name.sailing.angles import wind_velocity, wrap
from project_name.sailing.dynamics import BoatParams
from project_name.sailing.testing.common import Checks, SERIES, TEXT_SECONDARY


DT = analysis.SUBSTEP
TURN_TIME = 60.0
TURN_STEP = int(TURN_TIME / DT)
START, END = jnp.radians(90.0 - 45.0), jnp.radians(90.0 + 45.0)  # 45 degrees each side of a north wind
SHEET = jnp.radians(15.0)
HELMS_DEG = (25.0, 35.0, 45.0, 60.0)


def tack(policy, wind, params) -> np.ndarray:
    return np.array(analysis.rollout(analysis.start_state(START, u=1.0), policy, wind, params, TURN_TIME + 30.0))


def summary(trajectory: np.ndarray) -> dict:
    before = trajectory[TURN_STEP - 1, dynamics.U]
    after_turn = trajectory[TURN_STEP:]
    slowest = int(np.argmin(after_turn[:, dynamics.U]))
    off_course = np.abs(np.array(wrap(jnp.array(after_turn[:, dynamics.HEADING]) - END)))
    on_course = np.argmax(off_course < np.radians(5.0))
    return {"before": before,
            "slowest": after_turn[slowest, dynamics.U],
            "slowest_at": slowest * DT,
            "kept": after_turn[slowest, dynamics.U] / before,
            "on_course_at": on_course * DT if off_course[on_course] < np.radians(5.0) else np.inf,
            "slowest_step": TURN_STEP + slowest}


def budgets(trajectory: np.ndarray, wind, params) -> dict:
    """Surge forces and yaw moments by source, in N and N m, at every step."""
    def at(boat):
        loads = dynamics.forces_and_moment(boat, wind, params)
        u, v, r, boom = boat[dynamics.U], boat[dynamics.V], boat[dynamics.R], boat[dynamics.BOOM]
        hull_x, _, hull_n = dynamics.hull_force(u, v, r, params)
        effort_x = params.mast_x - params.boom_to_centre_of_effort * jnp.cos(boom)
        effort_y = -params.boom_to_centre_of_effort * jnp.sin(boom)
        return {"surge": {"Sail": loads["sail_x"], "Board": loads["board_x"], "Rudder": loads["rudder_x"],
                          "Hull": hull_x, "Sway into surge, m_y v r": params.mass_y * v * r},
                "yaw": {"Sail": effort_x * loads["sail_y"] - effort_y * loads["sail_x"],
                        "Board": params.board_x * loads["board_y"],
                        "Rudder": params.rudder_x * loads["rudder_y"],
                        "Hull": hull_n,
                        "Munk": -(params.mass_y - params.mass_x) * u * v}}

    return jax.tree_util.tree_map(np.array, jax.vmap(at)(jnp.array(trajectory)))


def main() -> Checks:
    checks = Checks("Tacking in detail")
    params = BoatParams()
    wind = wind_velocity(5.0, 0.0)

    hard_over = tack(analysis.tack_helm(START, END, TURN_TIME, SHEET), wind, params)
    easing = tack(lambda b, t: analysis.heading_hold(jnp.where(t < TURN_TIME, START, END), SHEET)(b, t), wind, params)
    good, eased = summary(hard_over), summary(easing)
    print(f"  Hard-over helm: {good['before']:.2f} to {good['slowest']:.2f} m/s after {good['slowest_at']:.1f} s, "
          f"on the new course after {good['on_course_at']:.1f} s")
    print(f"  Easing helm:    {eased['before']:.2f} to {eased['slowest']:.2f} m/s after {eased['slowest_at']:.1f} s, "
          f"on the new course after {eased['on_course_at']:.1f} s")
    checks.check("Hard-over helm keeps more speed than easing the helm through the turn", good["kept"] > eased["kept"],
                 f"{100 * good['kept']:.0f}% against {100 * eased['kept']:.0f}%")
    checks.check("Hard-over helm reaches the new course sooner", good["on_course_at"] < eased["on_course_at"],
                 f"{good['on_course_at']:.1f} s against {eased['on_course_at']:.1f} s")

    # The budget must add up, or it is not showing where the speed went
    loads = budgets(hard_over, wind, params)
    window = slice(TURN_STEP, good["slowest_step"])
    impulses = {name: float(np.sum(force[window]) * DT) for name, force in loads["surge"].items()}
    momentum_lost = params.mass_x * (hard_over[window.stop, dynamics.U] - hard_over[TURN_STEP - 1, dynamics.U])
    checks.check("Surge budget accounts for the speed lost", abs(sum(impulses.values()) - momentum_lost) < 0.05 * abs(momentum_lost),
                 f"sum of sources {sum(impulses.values()):.0f} N s, momentum change {momentum_lost:.0f} N s")
    print("  Impulse to the slowest point (N s): " + ", ".join(f"{name} {value:.0f}" for name, value in impulses.items()))

    # With the helm hard over the rudder ends up almost trailing: the turn is near the tightest the boat can make
    peak = TURN_STEP + int(np.argmax(np.abs(hard_over[TURN_STEP:TURN_STEP + 200, dynamics.R])))
    u, v, r, rudder = hard_over[peak, (dynamics.U, dynamics.V, dynamics.R, dynamics.RUDDER)]
    # Angle of attack as foil_force defines it, from the water velocity relative to the blade
    flow_x, flow_y = -u, -(v + r * params.rudder_x)
    speed = np.hypot(flow_x, flow_y)
    sin_aoa = (flow_y * np.cos(rudder) - flow_x * np.sin(rudder)) / speed
    cos_aoa = -(flow_x * np.cos(rudder) + flow_y * np.sin(rudder)) / speed
    print(f"  At peak yaw rate {np.degrees(abs(r)):.0f} deg/s the rudder meets the water at "
          f"{abs(np.degrees(np.arctan2(sin_aoa, cos_aoa))):.0f} deg, with {abs(np.degrees(rudder)):.0f} deg of helm")

    sweep = {helm: summary(tack(analysis.tack_helm(START, END, TURN_TIME, SHEET, helm=float(np.radians(helm))),
                                wind, params)) for helm in HELMS_DEG}
    for helm, result in sweep.items():
        print(f"  Helm {helm:.0f} deg: keeps {100 * result['kept']:.0f}%, on the new course after {result['on_course_at']:.1f} s")
    best_kept = max(result["kept"] for result in sweep.values())
    checks.check("The default 45 degrees of helm keeps close to the most speed", sweep[45.0]["kept"] > best_kept - 0.03,
                 f"{100 * sweep[45.0]['kept']:.0f}% against the best {100 * best_kept:.0f}%")
    checks.check("More helm turns faster", all(sweep[a]["on_course_at"] > sweep[b]["on_course_at"]
                                               for a, b in zip(HELMS_DEG, HELMS_DEG[1:])),
                 ", ".join(f"{helm:.0f} deg {result['on_course_at']:.1f} s" for helm, result in sweep.items()))

    fig, ((ax_speed, ax_budget), (ax_yaw, ax_sweep)) = plt.subplots(2, 2, figsize=(13, 8.5))
    time_axis = np.arange(len(hard_over)) * DT - TURN_TIME

    ax_speed.set_title("Forward speed through the tack")
    ax_speed.plot(time_axis, hard_over[:, dynamics.U], color=SERIES[0], label="Helm hard over, tack_helm")
    ax_speed.plot(time_axis, easing[:, dynamics.U], color=SERIES[1], label="Helm eased, heading_hold")
    ax_speed.set(xlim=(-2, 20), xlabel="Time from the turn starting (s)", ylabel="Forward speed u (m/s)")
    ax_speed.legend()

    ax_budget.set_title("Where the speed goes, to the slowest point")
    names = list(impulses)
    ax_budget.barh(names, [impulses[name] for name in names], color=SERIES[0])
    ax_budget.axvline(0.0, color=TEXT_SECONDARY, linewidth=0.8)
    ax_budget.invert_yaxis()
    ax_budget.set(xlabel=f"Surge impulse (N s), against {params.mass_x * good['before']:.0f} N s of momentum before")

    ax_yaw.set_title("Yaw moments, helm hard over")
    for index, name in enumerate(("Rudder", "Hull", "Munk")):
        ax_yaw.plot(time_axis, loads["yaw"][name], color=SERIES[index], label=name)
    ax_yaw.set(xlim=(-2, 10), xlabel="Time from the turn starting (s)", ylabel="Yaw moment (N m), positive to port")
    ax_yaw.legend()

    ax_sweep.set_title("Amount of helm in the tack")
    ax_sweep.plot(HELMS_DEG, [100 * sweep[h]["kept"] for h in HELMS_DEG], color=SERIES[0], marker="o")
    ax_sweep.set(xlabel="Helm (deg)", ylabel="Speed kept at the slowest (%)")
    ax_time = ax_sweep.twinx()
    ax_time.plot(HELMS_DEG, [sweep[h]["on_course_at"] for h in HELMS_DEG], color=SERIES[1], marker="s")
    ax_time.set_ylabel("Time to the new course (s)", color=SERIES[1])
    ax_time.grid(False)

    return checks.finish(fig, "tack")


if __name__ == "__main__":
    main()
