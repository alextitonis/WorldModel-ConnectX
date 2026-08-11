"""
Builds an EpisodicMemory offline, at packaging time, by playing self-play
games against a MIXED opponent (weak heuristic + random + the stronger
1-ply-deeper heuristic) using the model's own real adversarial search.
WON games are stored as positive (attraction) examples; LOST/drawn games
are ALSO stored, as negative (repulsion) examples -- deliberately mixed
opposition, not just the single fixed weak training opponent, so a loss
has to have actually lost to a real mix of opposition before it gets
stored as "this is bad," rather than encoding one narrow opponent's
particular blind spots as universal truth.
"""
import torch

from .env import ConnectXEnv
from .adversarial_search import real_adversarial_plan_action
from .episodic_memory import EpisodicMemory, add_trajectory_from_real_path, add_negative_trajectory_from_real_path


@torch.no_grad()
def build_episodic_memory(env, model, normalizer, rng, n_games=500, opponent_epsilon=0.2,
                           opponent_strong_epsilon=0.3, adversarial_rounds=2, penalty=None):
    """`penalty` (default 2x max_steps, deliberately bigger than the value
    head's own unsolved_penalty): these are discrete stored memory points,
    not a training-loss target, so being a bit more emphatic buys sharper
    repulsion without the overfitting risk more gradient steps would
    carry."""
    memory = EpisodicMemory()
    diverse_env = ConnectXEnv(width=env.width, height=env.height, win_len=env.win_len,
                               opponent_epsilon=opponent_epsilon, opponent_strong_epsilon=opponent_strong_epsilon)
    max_steps = (env.width * env.height) // 2 + 2
    if penalty is None:
        penalty = 2 * max_steps
    wins, losses = 0, 0
    for _ in range(n_games):
        state, _ = diverse_env.random_problem(rng)
        path_states = [state]
        for _ in range(max_steps):
            if diverse_env.is_solved(state):
                break
            a = real_adversarial_plan_action(diverse_env, model, normalizer, state, rounds=adversarial_rounds)
            if a is None:
                break
            state, _r, done = diverse_env.step(state, a)
            path_states.append(state)
            if done:
                break
        if diverse_env.is_solved(state):
            wins += 1
            add_trajectory_from_real_path(model, normalizer, memory, path_states, env=diverse_env)
        else:
            losses += 1
            add_negative_trajectory_from_real_path(model, normalizer, memory, path_states, penalty, env=diverse_env)
    print(f"  built memory from {wins} won + {losses} lost self-play games "
          f"(mixed opponent) -> {len(memory)} stored states")
    return memory
