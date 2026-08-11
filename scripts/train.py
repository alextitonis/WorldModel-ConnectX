"""
Trains a WorldModel checkpoint on the real Kaggle ConnectX board (7x6,
win_len=4). No oracle exists at this scale (see env.py's BFS_MAX_CELLS), so
this is a genuinely no-oracle recipe throughout:

1. Stage 1 (dynamics + decoder): self-supervised, `train_utils.train_stage1`
   on random transitions/rollouts -- never depends on an oracle for any
   domain.
2. Stage 2 (value head): `verifier.train_mc_value_onpolicy`, on-policy
   Monte Carlo, `unsolved_penalty=max_steps` -- this is an adversarial
   domain, where a training walk can end in a LOSS, not just run out of
   steps; without this penalty every losing walk is silently discarded and
   the value head never learns to avoid losing moves at all (confirmed as
   the direct cause of a real 0%, worse-than-random win rate before this
   was added).
3. Self-play fine-tune: each round freezes the current model as an
   opponent (viewed from the opponent's side, via `_swap_agent_opponent`)
   and trains further against a small, capped POOL of past frozen
   snapshots (not just the latest one) mixed with the original fixed
   heuristic -- every opponent used above is fixed and non-learning, which
   is the actual ceiling on how strong a non-self-play policy can get.

Evaluation has no oracle-ceiling comparison to report against (none exists
at this scale) -- instead: win rate against the ORIGINAL fixed opponent
(deterministic, `opponent_epsilon=0`) the model is ultimately graded
against, plus a random-legal-play win rate as the required floor.
"""
import copy
import random

import torch

from connectx.env import ConnectXEnv, AGENT, OPPONENT, EMPTY, _encode_board
from connectx.model import WorldModel
from connectx.train_utils import DEVICE, StateNormalizer, generate_transitions, generate_rollout_sequences, \
    train_stage1, eval_stage1, eval_multistep_rollout
from connectx.verifier import train_mc_value_onpolicy
from connectx.search import evaluate
from connectx.adversarial_search import real_adversarial_plan_action


def _swap_agent_opponent(cells):
    """AGENT<->OPPONENT relabeling, EMPTY unchanged -- lets a model that
    was only ever trained to play as AGENT evaluate a position from the
    OTHER side's perspective, by pretending that side is AGENT instead."""
    return [c if c == EMPTY else (OPPONENT if c == AGENT else AGENT) for c in cells]


def make_selfplay_pool_opponent_fn(frozen_pool, frozen_normalizer, opponent_env):
    """Builds an `opponent_policy_fn(cells) -> column` (see env.py's
    `opponent_policy_fn` extension point) that plays using a frozen
    snapshot of this same architecture's own trained judgment, not a
    hand-written heuristic. Each call picks a snapshot uniformly at random
    from `frozen_pool` (a small population of past snapshots, not just the
    latest one -- a coarse approximation of real population-based self-
    play/fictitious play, so the live policy can't narrowly overfit to
    counter-play against whatever the single latest snapshot happens to
    do) and uses the real adversarial search (rounds=1 -- this function is
    called on the order of 100K+ times across a training run, so a slower
    multi-round search would balloon total training time)."""

    def opponent_policy_fn(cells):
        frozen_model = random.choice(frozen_pool)
        swapped_state = tuple(_encode_board(_swap_agent_opponent(cells)))
        with torch.no_grad():
            a = real_adversarial_plan_action(opponent_env, frozen_model, frozen_normalizer, swapped_state, rounds=1)
        return a

    return opponent_policy_fn


def random_baseline_win_rate(env, problems, max_steps, rng):
    """Required control: an agent picking uniformly among its legal
    non-PASS columns. The floor the trained model needs to beat -- not a
    certified ceiling (no oracle exists at this scale), just the honest
    "did training do anything at all" check."""
    wins = 0
    for state, _ in problems:
        cur = state
        for _ in range(max_steps):
            if env.is_solved(cur):
                break
            legal = [a for a in range(env.num_actions) if env.is_legal(cur, a)]
            non_pass = [a for a in legal if a != env.width]
            a = rng.choice(non_pass or legal)
            cur, _r, done = env.step(cur, a)
            if done:
                break
        if env.is_solved(cur):
            wins += 1
    n = len(problems)
    print(f"{'Random-legal-play baseline':30s} win_rate={wins/n:.3f} ({wins}/{n})")
    return wins / n


def main(seed=0, ckpt_path="checkpoints/connectx_checkpoint.pt", latent_dim=96, hidden_dim=256,
         n_problems=2000, walk_len=8, mc_rounds=25, mc_problems_per_round=400,
         opponent_epsilon=0.15, opponent_strong_epsilon=0.0,
         selfplay_rounds=5, selfplay_epsilon=0.4, selfplay_mc_rounds_per_iter=15,
         selfplay_pool_size=5):
    # Two env instances, same board, different opponent determinism: `env`
    # (opponent_epsilon=0, the pure deterministic opponent) is what
    # evaluation is graded against. `train_env` (opponent_epsilon>0) is
    # used ONLY for generating training data -- a perfectly deterministic
    # opponent means every training walk from a matching starting side is
    # the SAME exact game, a real, diagnosed weakness (the model
    # reproducibly lost as first player against this exact opponent while
    # winning as second player).
    env = ConnectXEnv(width=7, height=6, win_len=4)
    train_env = ConnectXEnv(width=7, height=6, win_len=4, opponent_epsilon=opponent_epsilon,
                             opponent_strong_epsilon=opponent_strong_epsilon)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    # Also seed the GLOBAL random module: env.py's opponent_epsilon/
    # opponent_strong_epsilon rolls read `random.random()`/`random.choice()`
    # directly, not this function's own seeded `rng` -- without this,
    # "same seed" runs are silently not reproducible whenever opponent
    # stochasticity is enabled.
    random.seed(seed)
    max_steps = (env.width * env.height) // 2 + 2

    print(f"Domain: ConnectX (real board), {env.width}x{env.height}, win_len={env.win_len}, "
          f"num_actions={env.num_actions}, state_dim={env.state_dim}, "
          f"train opponent_epsilon={opponent_epsilon}, opponent_strong_epsilon={opponent_strong_epsilon}\n")
    assert env.bfs_solve(env.random_problem(rng)[0]) is None, \
        "expected no oracle at real-board scale -- see env.py's BFS_MAX_CELLS"

    print("Generating stage-1 (dynamics) data...")
    train_transitions = generate_transitions(train_env, rng, n_problems=n_problems, walk_len=walk_len)
    val_transitions = generate_transitions(env, rng, n_problems=300, walk_len=walk_len)
    print(f"  {len(train_transitions)} train transitions, {len(val_transitions)} val transitions")

    unroll_k = 4
    train_sequences = generate_rollout_sequences(train_env, rng, n_problems=n_problems, k=unroll_k)
    val_sequences = generate_rollout_sequences(env, rng, n_problems=300, k=unroll_k)
    print(f"  {len(train_sequences)} train sequences, {len(val_sequences)} val sequences (k={unroll_k})")

    all_states_for_norm = [t[0] for t in train_transitions] + [t[2] for t in train_transitions]
    normalizer = StateNormalizer(all_states_for_norm).to(DEVICE)

    model = WorldModel(env.state_dim, env.num_actions, latent_dim=latent_dim, hidden_dim=hidden_dim).to(DEVICE)

    print("\nStage 1: training encoder + dynamics + decoder...")
    train_stage1(model, normalizer, train_transitions, sequences=train_sequences, k=unroll_k)

    print("\nStage-1 val metrics:")
    print(" ", eval_stage1(model, normalizer, val_transitions))
    print(" ", eval_multistep_rollout(model, normalizer, val_sequences, k=unroll_k))

    print("\nStage 2: no-oracle value training (on-policy Monte Carlo, bfs_solve never called)...")
    train_mc_value_onpolicy(train_env, model, normalizer, rng, n_rounds=mc_rounds,
                             n_problems_per_round=mc_problems_per_round, max_steps=max_steps,
                             unsolved_penalty=max_steps)

    # Self-play fine-tune: only the value head trains during
    # train_mc_value_onpolicy (encoder/dynamics/decoder stay fixed from
    # stage 1), so each frozen snapshot's encoder/dynamics are identical
    # to the live model's -- only the value judgment (and therefore the
    # self-play opponent's move choices) differs round to round.
    # `selfplay_pool_size` keeps a capped, small population of past
    # snapshots (drops the oldest once full) rather than only the single
    # latest one, or an unbounded pool that would let early, still-weak
    # snapshots dominate forever.
    frozen_pool = []
    for sp_round in range(selfplay_rounds):
        frozen_model = copy.deepcopy(model).eval()
        for p in frozen_model.parameters():
            p.requires_grad_(False)
        frozen_pool.append(frozen_model)
        if len(frozen_pool) > selfplay_pool_size:
            frozen_pool.pop(0)
        print(f"\nSelf-play fine-tune round {sp_round + 1}/{selfplay_rounds} "
              f"(opponent_selfplay_epsilon={selfplay_epsilon}, pool_size={len(frozen_pool)})...")
        opponent_env = ConnectXEnv(width=7, height=6, win_len=4)  # plain -- used only for is_legal/num_actions
        selfplay_fn = make_selfplay_pool_opponent_fn(frozen_pool, normalizer, opponent_env)
        selfplay_train_env = ConnectXEnv(width=7, height=6, win_len=4,
                                          opponent_epsilon=opponent_epsilon,
                                          opponent_selfplay_epsilon=selfplay_epsilon,
                                          opponent_policy_fn=selfplay_fn)
        train_mc_value_onpolicy(selfplay_train_env, model, normalizer, rng, n_rounds=selfplay_mc_rounds_per_iter,
                                 n_problems_per_round=mc_problems_per_round, max_steps=max_steps,
                                 unsolved_penalty=max_steps)

    if ckpt_path:
        torch.save({
            "model_state": model.state_dict(),
            "norm_mean": normalizer.mean.cpu(),
            "norm_std": normalizer.std.cpu(),
            "state_dim": env.state_dim,
            "num_actions": env.num_actions,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "board_width": env.width,
            "board_height": env.height,
            "win_len": env.win_len,
        }, ckpt_path)
        print(f"\nSaved checkpoint to {ckpt_path}")

    print("\n" + "=" * 20 + " EVALUATION (no oracle -- vs. random-legal-play baseline only) " + "=" * 20)
    eval_rng = random.Random(999)
    problems = [env.random_problem(eval_rng) for _ in range(150)]
    random_baseline_win_rate(env, problems, max_steps, random.Random(1000))
    evaluate(env, model, normalizer, problems, depth=1, beam_width=8, max_total_steps=max_steps,
             label="Baseline A (model, depth=1)")
    evaluate(env, model, normalizer, problems, depth=3, beam_width=8, max_total_steps=max_steps,
             label="Search (model, depth=3)")

    return model, normalizer


if __name__ == "__main__":
    main()
