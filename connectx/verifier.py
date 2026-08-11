"""
On-policy Monte Carlo value-head training -- no oracle, no bootstrapping,
no target network. Walk the environment with the current value head's own
epsilon-greedy policy, and label every visited state along a walk that
actually reached a solved state with its REALIZED return (what happened,
not a network's own possibly-wrong estimate of what happens next). This is
the only value-training method used for ConnectX, since the real 7x6 board
has no tractable exact solver to regress against.

`unsolved_penalty` is what makes this work for an ADVERSARIAL domain
specifically: `env.step` can return `done=True` on a LOSS (the opponent
won) or a draw, not just our own win. Discarding every one of those walks
(the natural default for a single-agent puzzle, where "unsolved" just means
"further away") would mean the value head never sees a single labeled
example of "this leads to losing" -- exactly the signal an adversarial
domain needs to learn to avoid bad moves.
"""
import collections

import torch
import torch.nn as nn

from .train_utils import states_to_tensor, DEVICE


def _observed_states_to_tensor(env, states):
    return states_to_tensor([env.observe(s) for s in states])


def _greedy_walk_action(env, model, normalizer, state, rng, epsilon):
    """Epsilon-greedy action choice using the value head's own 1-step
    lookahead, scored the same way `search.plan_action`'s depth=1 does
    (`-reward + value`, using the REAL reward/next-state from `env.step`,
    not an imagined one) -- so training optimizes for the exact decision
    rule actually used at inference."""
    legal = [a for a in range(env.num_actions) if env.is_legal(state, a)]
    if rng.random() < epsilon:
        return rng.choice(legal)
    next_states, rewards = [], []
    for a in legal:
        ns, r, _done = env.step(state, a)
        next_states.append(ns)
        rewards.append(r)
    z = model.encode(normalizer.normalize(_observed_states_to_tensor(env, next_states).to(DEVICE)))
    values = model.evaluate(z)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    scores = -rewards_t + values
    return legal[torch.argmin(scores).item()]


def generate_mc_walks(env, model, normalizer, rng, n_problems, max_steps, epsilon,
                       unsolved_penalty=None, **problem_kwargs):
    """Complete walks (up to max_steps) using the current value head's
    epsilon-greedy policy. A walk that reaches `is_solved` labels every
    visited state with its real steps-remaining. A walk that doesn't
    (a loss, a draw, or simply running out of steps) is discarded UNLESS
    `unsolved_penalty` is set, in which case every state along it gets that
    fixed, uniform label instead -- deliberately not scaled by how early or
    late the walk went wrong; a state that's actually fine keeps
    reappearing in OTHER (solved) walks too, so its label self-corrects
    over many rounds rather than staying pinned to one bad walk's worst
    case."""
    labeled = []
    for _ in range(n_problems):
        state, _ = env.random_problem(rng, **problem_kwargs)
        path_states = [state]
        for _ in range(max_steps):
            if env.is_solved(state):
                break
            a = _greedy_walk_action(env, model, normalizer, state, rng, epsilon)
            state, _reward, done = env.step(state, a)
            path_states.append(state)
            if done:
                break
        if env.is_solved(path_states[-1]):
            T = len(path_states) - 1
            for t, s in enumerate(path_states):
                labeled.append((s, float(T - t)))
        elif unsolved_penalty is not None:
            for s in path_states:
                labeled.append((s, float(unsolved_penalty)))
    return labeled


def train_mc_value_onpolicy(env, model, normalizer, rng, n_rounds=15, n_problems_per_round=400,
                             max_steps=10, epochs_per_round=40, lr=1e-3,
                             epsilon_start=1.0, epsilon_end=0.05, warmup_rounds=3,
                             replay_capacity=20000, min_replay_before_train=30,
                             unsolved_penalty=None, verbose_every=5, **problem_kwargs):
    """`epsilon_start=1.0` + `warmup_rounds`: a freshly-initialized value
    head's "greedy" choice is pure noise, which can be WORSE than uniform
    random at stumbling into a solved state by chance. Holding epsilon at
    1.0 (pure random walk) for the first `warmup_rounds` guarantees a
    baseline solve rate to bootstrap training from, before annealing toward
    exploitation."""
    opt = torch.optim.Adam(model.value.parameters(), lr=lr)
    buffer = collections.deque(maxlen=replay_capacity)

    for round_idx in range(n_rounds):
        if round_idx < warmup_rounds:
            epsilon = 1.0
        else:
            progress = (round_idx - warmup_rounds) / max(1, n_rounds - 1 - warmup_rounds)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        labeled = generate_mc_walks(env, model, normalizer, rng, n_problems_per_round, max_steps, epsilon,
                                     unsolved_penalty=unsolved_penalty, **problem_kwargs)
        buffer.extend(labeled)
        if len(buffer) < min_replay_before_train:
            if verbose_every:
                print(f"  [mc-onpolicy] round {round_idx+1}/{n_rounds}  epsilon={epsilon:.2f}  "
                      f"only {len(buffer)} labeled states so far (need {min_replay_before_train}) -- skipping fit")
            continue

        all_data = list(buffer)
        states_t = _observed_states_to_tensor(env, [s for s, _ in all_data]).to(DEVICE)
        returns_t = torch.tensor([r for _, r in all_data], dtype=torch.float32, device=DEVICE)

        model.value_target_mean.copy_(returns_t.mean())
        model.value_target_std.copy_(returns_t.std().clamp(min=1e-3))
        returns_norm = (returns_t - model.value_target_mean) / model.value_target_std

        with torch.no_grad():
            z_states = model.encode(normalizer.normalize(states_t))

        n = len(all_data)
        for _epoch in range(epochs_per_round):
            perm = torch.randperm(n, device=DEVICE)
            pred_norm = model.value(z_states[perm])
            loss = nn.functional.mse_loss(pred_norm, returns_norm[perm])
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose_every and (round_idx + 1) % verbose_every == 0:
            print(f"  [mc-onpolicy] round {round_idx+1}/{n_rounds}  epsilon={epsilon:.2f}  "
                  f"buffer_size={n}  loss={loss.item():.4f}  return_mean={model.value_target_mean.item():.2f}")
