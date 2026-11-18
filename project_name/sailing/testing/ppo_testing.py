"""
PPO to the upwind mark, scored against the scripted tactician on the same episodes.

A local version of BifurcaGym's baselines/ppo_baseline.py, itself after PureJaxRL, because bifurcagym.make is a
hard-coded list of environments. It differs in being one jittable function that vmaps over seeds, so a GPU runs many
at once, and in using tanh layers with orthogonal initialisation as PureJaxRL does for continuous control.

The wrappers are those of env_testing: metrics around autoreset around normalisation, using the local
NormalisedSailingCSCA until the BifurcaGym fix. Training is too slow for run_all, so run it alone:
    pdm run python -m project_name.sailing.testing.ppo_testing
    pdm run python -m project_name.sailing.testing.ppo_testing --total-timesteps 50000000 --num-seeds 4
"""

import argparse
import time
from pathlib import Path
from typing import NamedTuple

from bifurcagym.wrappers import AutoResetWrapper, MetricsWrapper
import distrax
import flax.linen as nn
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax

from project_name.sailing import dynamics
from project_name.sailing.env import SailingCSCA
from project_name.sailing.scripted import scripted_action
from project_name.sailing.testing.common import Checks, PLOTS, SERIES
from project_name.sailing.testing.env_testing import describe, evaluate
from project_name.sailing.wrappers import NormalisedSailingCSCA


CHECKPOINTS = Path(__file__).parent / "checkpoints"

DEFAULT_CONFIG = {"total_timesteps": 10_000_000,
                  "num_envs": 128,
                  "rollout_length": 128,  # decision steps, 64 s of sailing
                  "update_epochs": 4,
                  "num_minibatches": 8,
                  "lr": 3e-4,
                  "anneal_lr": True,
                  "gamma": 0.99,  # a horizon of about 100 steps, 50 s, against about 270 steps to finish
                  "gae_lambda": 0.95,
                  "clip_eps": 0.2,
                  "ent_coef": 0.01,  # keeps the action noise from shrinking before tacking is found
                  # Training marks are turned up to this many degrees either way, so most can be reached without tacking
                  # and upwind marks are the hard end of a natural curriculum. Evaluation always uses the upwind mark
                  "mark_bearing_spread": 180.0,
                  "vf_coef": 0.5,
                  "max_grad_norm": 0.5,
                  "hidden": 128,
                  "initial_log_std": -0.5,  # std 0.6 in normalised action, so about 50 degrees of rudder
                  }


class ActorCritic(nn.Module):
    action_dim: int
    hidden: int
    initial_log_std: float

    @nn.compact
    def __call__(self, x):
        def dense(features, scale):
            return nn.Dense(features, kernel_init=nn.initializers.orthogonal(scale), bias_init=nn.initializers.zeros)

        actor = nn.tanh(dense(self.hidden, np.sqrt(2))(x))
        actor = nn.tanh(dense(self.hidden, np.sqrt(2))(actor))
        mean = dense(self.action_dim, 0.01)(actor)
        log_std = self.param("log_std", nn.initializers.constant(self.initial_log_std), (self.action_dim,))

        critic = nn.tanh(dense(self.hidden, np.sqrt(2))(x))
        critic = nn.tanh(dense(self.hidden, np.sqrt(2))(critic))
        value = dense(1, 1.0)(critic)
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), value.squeeze(axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    episode_return: jnp.ndarray
    finished: jnp.ndarray


def make_env():
    return MetricsWrapper(AutoResetWrapper(NormalisedSailingCSCA(SailingCSCA())))


def make_train(config: dict):
    env = make_env()
    env_params = env.default_params.replace(mark_bearing_spread=config["mark_bearing_spread"])
    num_envs, rollout_length = config["num_envs"], config["rollout_length"]
    num_updates = config["total_timesteps"] // (num_envs * rollout_length)
    batch_size = num_envs * rollout_length
    network = ActorCritic(env.action_space(env_params).shape[0], config["hidden"], config["initial_log_std"])

    report_every = max(num_updates // 20, 1)

    def report_progress(update, mean_return, finish_rate, std):
        """Printed from inside the jitted training, about 20 times per seed. Under vmap, once per seed."""
        if int(update) % report_every == 0 or int(update) == num_updates:
            print(f"  update {int(update)}/{num_updates}, {int(update) * batch_size:,} steps: return {float(mean_return):.2f}, "
                  f"finish {float(finish_rate):.0%}, action std {np.array2string(np.asarray(std), precision=2)}", flush=True)

    def schedule(count):
        return config["lr"] * (1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / num_updates)

    def train(key):
        key, init_key, reset_key = jrandom.split(key, 3)
        params = network.init(init_key, jnp.zeros(env.observation_space(env_params).shape))
        tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                         optax.adam(schedule if config["anneal_lr"] else config["lr"], eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obs, env_state = jax.vmap(env.reset, in_axes=(None, 0))(env_params, jrandom.split(reset_key, num_envs))

        def update_step(runner_state, unused):
            def env_step(carry, unused):
                train_state, env_state, obs, key = carry
                key, action_key, step_key = jrandom.split(key, 3)
                pi, value = network.apply(train_state.params, obs)
                action = pi.sample(seed=action_key)
                new_obs, _, new_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, None, 0))(
                    action, env_state, env_params, jrandom.split(step_key, num_envs))
                transition = Transition(done, action, value, reward, pi.log_prob(action), obs,
                                        info["returned_episode_returns"], info["finished"])
                return (train_state, new_env_state, new_obs, key), transition

            runner_state, batch = jax.lax.scan(env_step, runner_state, None, rollout_length)
            train_state, env_state, last_obs, key = runner_state
            _, last_value = network.apply(train_state.params, last_obs)

            def gae_step(carry, transition):
                gae, next_value = carry
                not_done = 1.0 - transition.done
                delta = transition.reward + config["gamma"] * next_value * not_done - transition.value
                gae = delta + config["gamma"] * config["gae_lambda"] * not_done * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(gae_step, (jnp.zeros_like(last_value), last_value), batch, reverse=True, unroll=16)
            targets = advantages + batch.value

            def update_epoch(carry, unused):
                train_state, key = carry
                key, permute_key = jrandom.split(key)

                def loss_fn(params, minibatch, gae, target):
                    pi, value = network.apply(params, minibatch.obs)
                    log_prob = pi.log_prob(minibatch.action)

                    value_clipped = minibatch.value + (value - minibatch.value).clip(-config["clip_eps"], config["clip_eps"])
                    value_loss = 0.5 * jnp.maximum(jnp.square(value - target), jnp.square(value_clipped - target)).mean()

                    ratio = jnp.exp(log_prob - minibatch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss = -jnp.minimum(ratio * gae,
                                              jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae).mean()
                    entropy = pi.entropy().mean()
                    return actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy, (value_loss, actor_loss)

                def update_minibatch(train_state, minibatch):
                    grads = jax.grad(loss_fn, has_aux=True)(train_state.params, *minibatch)[0]
                    return train_state.apply_gradients(grads=grads), None

                flat = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (batch, advantages, targets))
                shuffled = jax.tree.map(lambda x: jnp.take(x, jrandom.permutation(permute_key, batch_size), axis=0), flat)
                minibatches = jax.tree.map(lambda x: x.reshape((config["num_minibatches"], -1) + x.shape[1:]), shuffled)
                train_state, _ = jax.lax.scan(update_minibatch, train_state, minibatches)
                return (train_state, key), None

            (train_state, key), _ = jax.lax.scan(update_epoch, (train_state, key), None, config["update_epochs"])

            # Per update: mean return and finish rate of the episodes that ended during this rollout, NaN when none did
            ended = batch.done.sum()
            none = jnp.where(ended > 0, 0.0, jnp.nan)
            metrics = {"episodes": ended,
                       "mean_return": jnp.sum(batch.episode_return * batch.done) / jnp.maximum(ended, 1) + none,
                       "finish_rate": jnp.sum(batch.finished & batch.done) / jnp.maximum(ended, 1) + none}

            update = train_state.step // (config["num_minibatches"] * config["update_epochs"])
            jax.debug.callback(report_progress, update, metrics["mean_return"], metrics["finish_rate"],
                               jnp.exp(train_state.params["params"]["log_std"]))
            return (train_state, env_state, last_obs, key), metrics

        runner_state = (train_state, env_state, obs, key)
        runner_state, metrics = jax.lax.scan(update_step, runner_state, None, num_updates)
        return runner_state[0].params, metrics

    return train, network, num_updates


def main(config: dict, num_seeds: int, seed: int) -> Checks:
    checks = Checks("PPO to the upwind mark")
    print(f"  Device {jax.devices()[0].platform}, {num_seeds} seed(s), {config['total_timesteps']:,} steps each", flush=True)

    train, network, num_updates = make_train(config)
    started = time.time()
    params, metrics = jax.block_until_ready(jax.jit(jax.vmap(train))(jrandom.split(jrandom.key(seed), num_seeds)))
    elapsed = time.time() - started
    metrics = jax.tree.map(np.array, metrics)
    print(f"  Trained in {elapsed:.0f} s, compile included: {num_seeds * config['total_timesteps'] / elapsed:,.0f} steps/s")

    checks.check("Training stays finite", all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(params)))

    # Scored on the episodes env_testing uses, without autoreset, on the mean action. The tactician acts in radians, so
    # it runs on the raw env, which gives the same episodes because they depend only on the keys
    env = NormalisedSailingCSCA(SailingCSCA())
    env_params = env.default_params
    scripted = evaluate(SailingCSCA(), env_params, lambda obs, state: scripted_action(state, env_params))
    print(f"  Scripted tactician: {describe(scripted)}")

    learned = []
    for index in range(num_seeds):
        seed_params = jax.tree.map(lambda x: x[index], params)
        result = evaluate(env, env_params, lambda obs, state: network.apply(seed_params, obs)[0].mode())
        learned.append(result)
        print(f"  PPO seed {index}: {describe(result)}")
        CHECKPOINTS.mkdir(exist_ok=True)
        (CHECKPOINTS / f"ppo_upwind_seed{index}.msgpack").write_bytes(serialization.to_bytes(seed_params))
    best = max(range(num_seeds), key=lambda i: learned[i]["mean_return"])

    tenth = max(num_updates // 10, 1)
    early, late = np.nanmean(metrics["mean_return"][:, :tenth]), np.nanmean(metrics["mean_return"][:, -tenth:])
    checks.check("Return improves during training", late > early, f"{early:.2f} in the first tenth, {late:.2f} in the last")
    checks.check("PPO reaches the mark most of the time", learned[best]["finish_rate"] > 0.5,
                 f"best seed {learned[best]['finish_rate']:.1%}")
    checks.check("PPO beats the scripted tactician's mean return", learned[best]["mean_return"] > scripted["mean_return"],
                 f"{learned[best]['mean_return']:.2f} against {scripted['mean_return']:.2f}")
    checks.check("PPO finishes at least as often as the scripted tactician",
                 learned[best]["finish_rate"] >= scripted["finish_rate"],
                 f"{learned[best]['finish_rate']:.1%} against {scripted['finish_rate']:.1%}")

    fig, (ax_curve, ax_tracks, ax_times) = plt.subplots(1, 3, figsize=(18, 6))
    steps = (np.arange(num_updates) + 1) * config["num_envs"] * config["rollout_length"]
    ax_curve.set_title("Training, episodes ending in each rollout")
    for index in range(num_seeds):
        has_episodes = np.isfinite(metrics["mean_return"][index])
        ax_curve.plot(steps[has_episodes], metrics["mean_return"][index][has_episodes], color=SERIES[0], alpha=0.8 if index == best else 0.35,
                      label="PPO" if index == best else None)
    ax_curve.axhline(scripted["mean_return"], color=SERIES[1], linestyle="--", label="Scripted, evaluation")
    ax_curve.set(xlabel="Decision steps per seed", ylabel="Mean episode return")
    ax_curve.legend()

    ax_tracks.set_title(f"PPO seed {best}, wind from north")
    states, ended = learned[best]["states"], learned[best]["ended"]
    for i in range(12):
        ax_tracks.plot(states.boat[i, :ended[i] + 1, dynamics.X], states.boat[i, :ended[i] + 1, dynamics.Y],
                       color=SERIES[0], linewidth=1.2)
    ax_tracks.plot(env_params.mark_east, env_params.mark_north, "o", color=SERIES[1], markersize=12, label="Mark")
    ax_tracks.set(xlabel="East (m)", ylabel="North (m)", aspect="equal")
    ax_tracks.legend()

    ax_times.set_title("Time taken to reach the mark")
    bins = np.linspace(0.0, env.max_steps_in_ep * env.dt, 40)
    ax_times.hist(scripted["finish_times"], bins=bins, color=SERIES[1], alpha=0.6, label="Scripted")
    ax_times.hist(learned[best]["finish_times"], bins=bins, color=SERIES[0], alpha=0.6, label=f"PPO seed {best}")
    ax_times.set(xlabel="Time to finish (s)", ylabel="Episodes")
    ax_times.legend()

    PLOTS.mkdir(exist_ok=True)
    env.render_traj(jax.tree.map(lambda x: x[0, :ended[0] + 1], states), env_params, str(PLOTS / "ppo"))
    print(f"  Animation: {PLOTS / 'ppo'}_{env.name}.gif, checkpoints in {CHECKPOINTS}")
    return checks.finish(fig, "ppo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name, value in DEFAULT_CONFIG.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    arguments = vars(parser.parse_args())
    num_seeds, seed = arguments.pop("num_seeds"), arguments.pop("seed")
    main(arguments, num_seeds, seed)
