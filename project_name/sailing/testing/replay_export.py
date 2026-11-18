"""
Exports episodes of the trained PPO policy and the scripted tactician, sailed in the same wind, for the replay viewer.

Loads the checkpoint that ppo_testing saved, so run that first. Writes testing/plots/replay.json:
    pdm run python -m project_name.sailing.testing.replay_export
"""

import json

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from project_name.sailing import dynamics
from project_name.sailing.angles import maths_to_compass
from project_name.sailing.env import SailingCSCA
from project_name.sailing.scripted import scripted_action
from project_name.sailing.testing.common import PLOTS
from project_name.sailing.testing.env_testing import evaluate
from project_name.sailing.testing.ppo_testing import CHECKPOINTS, DEFAULT_CONFIG, ActorCritic
from project_name.sailing.wrappers import NormalisedSailingCSCA


UPWIND_EPISODES = 8
TURNED_EPISODES = 6


def load_policy(env, env_params, seed: int = 0):
    network = ActorCritic(env.action_space(env_params).shape[0], DEFAULT_CONFIG["hidden"], DEFAULT_CONFIG["initial_log_std"])
    template = network.init(jax.random.key(0), jnp.zeros(env.observation_space(env_params).shape))
    params = serialization.from_bytes(template, (CHECKPOINTS / f"ppo_upwind_seed{seed}.msgpack").read_bytes())
    return lambda obs, state: network.apply(params, obs)[0].mode()


def episode(result: dict, index: int, dt: float) -> dict:
    end = int(result["ended"][index]) + 1
    states = jax.tree.map(lambda x: np.asarray(x[index, :end]), result["states"])
    boat = states.boat
    return {"finished": bool(result["success"][index]),
            "time": round(end * dt, 1),
            "x": np.round(boat[:, dynamics.X], 2).tolist(),
            "y": np.round(boat[:, dynamics.Y], 2).tolist(),
            "heading": np.round(np.asarray(maths_to_compass(jnp.array(boat[:, dynamics.HEADING]))), 1).tolist(),
            "speed": np.round(np.hypot(boat[:, dynamics.U], boat[:, dynamics.V]), 3).tolist(),
            "boom": np.round(np.degrees(boat[:, dynamics.BOOM]), 1).tolist(),
            "rudder": np.round(np.degrees(boat[:, dynamics.RUDDER]), 1).tolist(),
            "wind_speed": np.round(states.wind_speed, 2).tolist(),
            "wind_direction": np.round(states.wind_direction, 1).tolist()}


def main():
    raw, normalised = SailingCSCA(), NormalisedSailingCSCA(SailingCSCA())
    upwind_params = raw.default_params
    ppo = load_policy(normalised, upwind_params)

    sets = []
    for name, params, count in (("Upwind mark, as evaluated", upwind_params, UPWIND_EPISODES),
                                ("Mark at other bearings, as trained", upwind_params.replace(mark_bearing_spread=180.0),
                                 TURNED_EPISODES)):
        learned = evaluate(normalised, params, ppo)
        scripted = evaluate(raw, params, lambda obs, state, p=params: scripted_action(state, p))
        for index in range(count):
            mark = np.asarray(learned["states"].mark[index, 0])
            sets.append({"group": name,
                         "mark": np.round(mark, 2).tolist(),
                         "ppo": episode(learned, index, raw.dt),
                         "scripted": episode(scripted, index, raw.dt)})
        print(f"  {name}: PPO {learned['finish_rate']:.1%}, scripted {scripted['finish_rate']:.1%} over 512 episodes")

    path = PLOTS / "replay.json"
    path.write_text(json.dumps({"dt": raw.dt, "mark_radius": upwind_params.mark_radius, "episodes": sets}))
    print(f"  Wrote {len(sets)} episodes to {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
