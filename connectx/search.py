"""
Latent-space lookahead solver ("Baseline A" at depth=1, real branching
search at depth>1) -- this is the ORIGINAL search approach this codebase
started with, kept here as the comparison baseline `adversarial_search.py`
(the real approach actually deployed) is measured against. See the
whitepaper's results table for why real board-space search decisively beats
this for an adversarial domain.

The neurosymbolic decode-gate (`caution` below) exists because an earlier
version that let the dynamics model imagine arbitrarily deep with no
legality check at all let it extrapolate to state/action combinations it
never saw during training, corrupting even the very first move's score.
With probability `caution`, each beam entry's latent is decoded back to an
estimated real state (via the trained decoder) and the REAL legal-action
mask is computed from that -- decoding is used only to filter which moves
are allowed, never to make the value judgement itself, which stays fully
latent.
"""
import random

import torch

from .model import WorldModel
from .train_utils import StateNormalizer, states_to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = WorldModel(
        state_dim=ckpt["state_dim"],
        num_actions=ckpt["num_actions"],
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    normalizer = StateNormalizer.__new__(StateNormalizer)
    normalizer.mean = ckpt["norm_mean"].to(DEVICE)
    normalizer.std = ckpt["norm_std"].to(DEVICE)
    return model, normalizer


class VisitedStates:
    """Plain hash-set cycle detector -- ConnectX is fully discrete, so this
    is always an O(1) membership check."""

    def __init__(self, initial_state):
        self._set = {initial_state}

    def __contains__(self, state):
        return state in self._set

    def add(self, state):
        self._set.add(state)


def _evaluate_with_memory(model, cand_z, memory, memory_weight, memory_k):
    """model.evaluate(cand_z), optionally blended with an EpisodicMemory's
    k-NN lookup at the same latents. `memory=None` is the exact original
    behavior with zero overhead. Blend weight is scaled by the memory's own
    self-calibrated `trust` (~1 for a genuinely close match, ~0 for nothing
    similar ever stored), so a distant, irrelevant neighbor doesn't get
    blended in at the same weight as a close one."""
    values = model.evaluate(cand_z)
    if memory is not None and len(memory) > 0 and memory_weight > 0:
        blended, trust = memory.query_batch(cand_z, k=memory_k)
        w = memory_weight * trust
        values = (1 - w) * values + w * blended
    return values


@torch.no_grad()
def plan_action(env, model, normalizer, real_state, depth=3, beam_width=8, caution=1.0, rng=None,
                 memory=None, memory_weight=0.25, memory_k=5):
    """Best first action for real_state, chosen by beam search over chained
    imagined latents, with environment-defined legality gating the branching
    at every ply -- NOT a purely-latent search (see the module docstring's
    decode-gate paragraph): the rollout (`imagine_step`) and the value
    judgement stay fully latent, but the root ply's allowed set comes from
    `env.is_legal(real_state, ...)` and deeper plies DECODE each beam entry's
    latent to an estimated real state and call `env.is_legal` on that
    whenever the `caution` draw succeeds (caution=1.0, the default, means
    always). Accurately: latent dynamics and latent value evaluation, with
    decoded-state, environment-defined legality gating at deeper plies.
    depth=1 reduces exactly to Baseline A (no lookahead beyond the immediate
    predicted next state -- and, having no step>0 plies, the one config that
    genuinely never decodes)."""
    rng = rng or random

    root_legal = [a for a in range(env.num_actions) if env.is_legal(real_state, a)]
    if not root_legal:
        return None

    z0 = normalizer.normalize(states_to_tensor([env.observe(real_state)]).to(DEVICE))
    z0 = model.encode(z0)[0]

    # beam entries: (predicted_z, action_seq, cumulative_predicted_reward)
    beam = [(z0, [], 0.0)]

    for step in range(depth):
        if step == 0:
            allowed_per_entry = [root_legal]
        elif rng.random() < caution:
            zs = torch.stack([z for z, _seq, _r in beam])
            decoded = normalizer.denormalize(model.reconstruct(zs)).round()
            allowed_per_entry = []
            for row in decoded.tolist():
                decoded_state = tuple(int(x) for x in row)
                legal = [a for a in range(env.num_actions) if env.is_legal(decoded_state, a)]
                allowed_per_entry.append(legal if legal else env.always_legal_actions)
        else:
            allowed_per_entry = [env.always_legal_actions] * len(beam)

        candidates = []
        for (z, seq, cum_r), allowed in zip(beam, allowed_per_entry):
            z_batch = z.unsqueeze(0).repeat(len(allowed), 1)
            a_batch = torch.tensor(allowed, dtype=torch.long, device=DEVICE)
            next_z_batch, reward_batch = model.imagine_step(z_batch, a_batch)
            for i, a in enumerate(allowed):
                candidates.append((next_z_batch[i], seq + [a], cum_r + reward_batch[i].item()))

        if not candidates:
            break

        cand_z = torch.stack([c[0] for c in candidates])
        values = _evaluate_with_memory(model, cand_z, memory, memory_weight, memory_k)
        cum_rewards = torch.tensor([c[2] for c in candidates], dtype=torch.float32, device=DEVICE)
        scores = -cum_rewards + values  # value is a cost estimate; combine with accumulated reward

        k = min(beam_width, len(candidates))
        top_idx = torch.topk(scores, k, largest=False).indices.tolist()
        beam = [(candidates[i][0], candidates[i][1], candidates[i][2]) for i in top_idx]

    final_z = torch.stack([b[0] for b in beam])
    final_values = _evaluate_with_memory(model, final_z, memory, memory_weight, memory_k)
    final_cum_rewards = torch.tensor([b[2] for b in beam], dtype=torch.float32, device=DEVICE)
    final_scores = -final_cum_rewards + final_values
    best_idx = torch.argmin(final_scores).item()
    return beam[best_idx][1][0]


def solve_with_search_counted(env, model, normalizer, state, depth, beam_width, caution=1.0, max_total_steps=12,
                               memory=None, memory_weight=0.25, memory_k=5):
    """Runs `plan_action` step by step against the real environment,
    tracking visited states to fail fast on a cycle.

    The `is_solved` check on a `done` step is NOT a redundant safety net --
    it's a real, previously-fixed bug class: `done=True` fires on a LOSS or
    a DRAW in this domain, not just a win (unlike every single-agent puzzle
    domain, where a dead end is simply never marked done, so `done` and
    `is_solved` always agreed). Trusting `done` alone here silently counted
    real losses as wins."""
    cur = state
    visited = VisitedStates(state)
    for i in range(max_total_steps):
        if env.is_solved(cur):
            return True, i
        a = plan_action(env, model, normalizer, cur, depth=depth, beam_width=beam_width, caution=caution,
                         memory=memory, memory_weight=memory_weight, memory_k=memory_k)
        if a is None:
            return False, None
        next_state, _, done = env.step(cur, a)
        if next_state in visited:
            return False, None
        visited.add(next_state)
        cur = next_state
        if done:
            return env.is_solved(cur), (i + 1 if env.is_solved(cur) else None)
    return env.is_solved(cur), (max_total_steps if env.is_solved(cur) else None)


def evaluate(env, model, normalizer, problems, depth, beam_width, caution=1.0, max_total_steps=12, label="",
             memory=None, memory_weight=0.25, memory_k=5):
    solved = 0
    total_steps = 0
    for state, _answer in problems:
        ok, steps = solve_with_search_counted(env, model, normalizer, state, depth, beam_width, caution,
                                               max_total_steps, memory=memory, memory_weight=memory_weight,
                                               memory_k=memory_k)
        if ok:
            solved += 1
            total_steps += steps
    n = len(problems)
    avg_steps = total_steps / solved if solved else float("nan")
    print(f"{label:30s} solve_rate={solved/n:.3f} ({solved}/{n})  avg_steps_when_solved={avg_steps:.2f}")
    return solved / n, avg_steps
