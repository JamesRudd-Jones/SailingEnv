"""
Single boat environment: does it honour the BifurcaGym contract, and can the task be done at all?

The contract checks follow BifurcaGym's own tests/test_env.py. A hand-written tactician then sails the course, which shows
the task is solvable and gives a score for PPO to beat.
"""

import time

from bifurcagym.wrappers import AutoResetWrapper, MetricsWrapper
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from project_name.sailing import dynamics
from project_name.sailing.env import SailingCSCA, SailingCSDA
from project_name.sailing.scripted import scripted_action
from project_name.sailing.testing.common import Checks, PLOTS, SERIES
from project_name.sailing.wrappers import NormalisedSailingCSCA


def contract_problem(env, env_params, key) -> str:
    """Empty when every step honours the contract, otherwise says what broke."""
    obs, state = env.reset(env_params, key)
    obs_again, state_again = env.reset(env_params, key)
    chex.assert_trees_all_equal((obs, state), (obs_again, state_again))
    chex.assert_shape(obs, env.observation_space(env_params).shape)

    for _ in range(100):
        key, action_key, step_key = jrandom.split(key, 3)
        action = env.action_space(env_params).sample(action_key)
        generative = env.generative_step(action, obs, env_params, step_key)
        nobs, delta, nstate, reward, done, _ = env.step(action, state, env_params, step_key)

        chex.assert_shape(reward, ())
        chex.assert_shape(done, ())
        chex.assert_tree_all_finite(nobs)
        chex.assert_trees_all_close(env.apply_delta_obs(obs, delta, env_params), nobs, atol=1e-3)

        fn_reward, fn_done = env.reward_and_done_function(action, state, nstate, env_params, step_key)
        chex.assert_trees_all_close(fn_reward, reward, atol=1e-4)
        chex.assert_trees_all_equal(fn_done, done)

        chex.assert_trees_all_close(generative[0], nobs, atol=1e-3)
        chex.assert_trees_all_close(generative[2].replace(time=nstate.time), nstate, atol=1e-3)
        chex.assert_trees_all_close(generative[3], reward, atol=1e-3)

        obs, state = nobs, nstate
        if done:
            obs, state = env.reset(env_params, step_key)
    return ""


def run_scripted(env, env_params, key):
    def body(carry, step_key):
        obs, state = carry
        action = scripted_action(state, env_params)
        new_obs, _, new_state, reward, done, info = env.step(action, state, env_params, step_key)
        return (new_obs, new_state), (state, reward, done, info["finished"])

    reset_key, steps_key = jrandom.split(key)
    carry = env.reset(env_params, reset_key)
    _, history = jax.lax.scan(body, carry, jrandom.split(steps_key, env.max_steps_in_ep))
    return history


def main() -> Checks:
    checks = Checks("Single boat environment")

    for name, env in (("continuous actions", SailingCSCA()), ("discrete actions", SailingCSDA())):
        params = env.default_params
        try:
            problem = contract_problem(env, params, jrandom.key(0))
        except AssertionError as error:
            problem = str(error).splitlines()[0][:150]
        checks.check(f"Contract holds with {name}", problem == "", problem)

    env = SailingCSCA()
    params = env.default_params

    samples = jax.vmap(env.action_space(params).sample)(jrandom.split(jrandom.key(1), 1000))
    checks.check("Sampled actions cover both sides of zero", samples.min(axis=0).max() < 0.0 < samples.max(axis=0).min(),
                 f"rudder {np.degrees(samples[:, 0].min()):.0f} to {np.degrees(samples[:, 0].max()):.0f} deg, "
                 f"sail {np.degrees(samples[:, 1].min()):.0f} to {np.degrees(samples[:, 1].max()):.0f} deg")

    obs_space = env.observation_space(params)
    checks.check("Observation bounds are the right way round", jnp.all(obs_space.low < obs_space.high))

    # All the wrappers together, batched and scanned, as the RL pipeline runs them
    wrapped = MetricsWrapper(AutoResetWrapper(NormalisedSailingCSCA(SailingCSCA())))

    def random_policy_run(key):
        reset_key, steps_key = jrandom.split(key)
        carry = wrapped.reset(params, reset_key)

        def body(inner, step_key):
            obs, state = inner
            action_key, env_key = jrandom.split(step_key)
            action = jrandom.uniform(action_key, (2,), minval=-1.0, maxval=1.0)
            new_obs, _, new_state, reward, done, _ = wrapped.step(action, state, params, env_key)
            return (new_obs, new_state), (new_obs, reward, done)

        return jax.lax.scan(body, carry, jrandom.split(steps_key, 1000))[1]

    obs_history, rewards, dones = jax.jit(jax.vmap(random_policy_run))(jrandom.split(jrandom.key(2), 64))
    checks.check("Runs under vmap and scan with normalised, autoreset and metrics wrappers",
                 jnp.all(jnp.isfinite(obs_history)) and jnp.all(jnp.isfinite(rewards)),
                 f"{int(dones.sum())} episodes ended, worst normalised obs {float(jnp.max(jnp.abs(obs_history))):.2f}")

    @jax.jit
    def throughput_run(key):
        return jax.vmap(random_policy_run)(jrandom.split(key, 4096))[1].sum()

    throughput_run(jrandom.key(3)).block_until_ready()
    started = time.time()
    throughput_run(jrandom.key(4)).block_until_ready()
    rate = 4096 * 1000 / (time.time() - started)
    print(f"  Throughput on {jax.devices()[0].platform}, full env with wrappers: {rate:,.0f} decision steps/s, "
          f"or {rate * env.dt:,.0f} simulated boat seconds per second")

    # The tactician
    states, rewards, dones, finished = jax.jit(jax.vmap(lambda k: run_scripted(env, params, k)))(jrandom.split(jrandom.key(5), 512))
    states, rewards, dones, finished = jax.tree.map(np.array, (states, rewards, dones, finished))

    first_done = np.argmax(dones, axis=1)
    ever_done = dones.any(axis=1)
    ended = np.where(ever_done, first_done, env.max_steps_in_ep - 1)
    success = finished[np.arange(len(ended)), ended] & ever_done
    returns = np.array([rewards[i, :ended[i] + 1].sum() for i in range(len(ended))])
    finish_times = (ended[success] + 1) * env.dt

    checks.check("Scripted tactician reliably reaches the mark upwind", success.mean() > 0.95,
                 f"{success.mean():.1%} finish, median {np.median(finish_times):.0f} s, mean return {returns.mean():.2f}")

    made_good = 100.0 / np.median(finish_times)
    checks.check("Its speed to windward agrees with the polars", 0.6 < made_good < 1.6,
                 f"about {made_good:.2f} m/s made good, against about 1.1 m/s best VMG at 5 m/s")

    tacks = []
    for i in np.where(success)[0][:64]:
        relative = np.sin(states.boat[i, :ended[i], dynamics.HEADING] - np.pi / 2)
        tacks.append(int(np.sum(np.abs(np.diff(np.sign(relative))) > 0)))
    checks.check("It has to tack to get there", np.median(tacks) >= 1, f"median {np.median(tacks):.0f} tacks")

    fig, (ax_tracks, ax_speed) = plt.subplots(1, 2, figsize=(12, 6))
    ax_tracks.set_title("Scripted tactician, wind from north")
    for i in range(12):
        ax_tracks.plot(states.boat[i, :ended[i] + 1, dynamics.X], states.boat[i, :ended[i] + 1, dynamics.Y],
                       color=SERIES[0], linewidth=1.2)
    ax_tracks.plot(*np.array((params.mark_east, params.mark_north)), "o", color=SERIES[1], markersize=12, label="Mark")
    ax_tracks.set(xlabel="East (m)", ylabel="North (m)", aspect="equal")
    ax_tracks.legend()

    ax_speed.set_title("Time taken to reach the mark")
    ax_speed.hist(finish_times, bins=25, color=SERIES[0], rwidth=0.9)
    ax_speed.set(xlabel="Time to finish (s)", ylabel="Episodes")

    PLOTS.mkdir(exist_ok=True)
    env.render_traj(jax.tree.map(lambda x: x[0, :ended[0] + 1], states), params, str(PLOTS / "scripted"))
    print(f"  Animation: {PLOTS / 'scripted'}_{env.name}.gif")

    return checks.finish(fig, "env")


if __name__ == "__main__":
    main()
