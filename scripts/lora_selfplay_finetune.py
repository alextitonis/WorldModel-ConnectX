"""
LoRA self-play fine-tune of an ALREADY-TRAINED checkpoint's value head --
this is the step that actually produced the deployed checkpoint, not a
fresh training run. Starts from the real, working checkpoint (encoder/
dynamics/decoder frozen, matching `train.py`'s own self-play stage, which
never touches them either) and fine-tunes the value head under self-play
through a LoRA-constrained delta rather than an unconstrained further Adam
update -- directly tests whether constraining HOW MUCH the value head can
move (not just how much data/how many rounds) helps it absorb self-play
signal without regressing calibration.

`opponent_strong_epsilon` (the curriculum variant): mixes the stronger
1-ply-deeper heuristic into training ALONGSIDE the self-play snapshots and
the original weak heuristic -- without this, self-play rounds only ever
train against [self-play snapshots, weak heuristic], and never the harder
fixed opponent at all. This is what produced this project's best-confirmed
result (see the whitepaper's results table).

rank=4/alpha=4.0: found to work well for a value head elsewhere in this
project's development on this same architecture; not independently
re-tuned for ConnectX specifically.
"""
import copy
import random

import torch

from connectx.env import ConnectXEnv
from scripts.train import make_selfplay_pool_opponent_fn, random_baseline_win_rate
from connectx.verifier import train_mc_value_onpolicy
from connectx.lora import LoRALinear, apply_lora
from connectx.search import load_checkpoint, evaluate


def _merge_and_unwrap(module):
    """Merge every LoRALinear's delta into its frozen base weight, then
    replace the wrapper with the plain (now-merged) nn.Linear -- so the
    saved checkpoint is an ORDINARY WorldModel state_dict, loadable by
    every existing caller with zero LoRA-awareness needed downstream."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            child.merge_into_base()
            setattr(module, name, child.linear)
        else:
            _merge_and_unwrap(child)


def main(ckpt_path="checkpoints/connectx_checkpoint.pt", seed=0, rank=4, alpha=4.0,
         selfplay_rounds=5, selfplay_epsilon=0.4, selfplay_mc_rounds_per_iter=15,
         mc_problems_per_round=400, selfplay_pool_size=5,
         opponent_epsilon=0.15, opponent_strong_epsilon=0.0,
         save_path="checkpoints/connectx_checkpoint_lora_selfplay.pt"):
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, normalizer = load_checkpoint(ckpt_path)

    rng = random.Random(seed)
    torch.manual_seed(seed)
    random.seed(seed)  # env.py's opponent stochasticity reads the GLOBAL
    # random module, not this local rng -- see train.py's own note

    env = ConnectXEnv(width=raw_ckpt["board_width"], height=raw_ckpt["board_height"],
                       win_len=raw_ckpt["win_len"])
    max_steps = (env.width * env.height) // 2 + 2

    lora_params = apply_lora(model.value, rank=rank, alpha=alpha)
    n_lora = sum(p.numel() for p in lora_params)
    n_frozen = sum(p.numel() for p in model.value.parameters()) - n_lora
    print(f"LoRA rank={rank} alpha={alpha} on model.value: "
          f"{n_lora} trainable params, {n_frozen} frozen (base value head)")

    frozen_pool = []
    for sp_round in range(selfplay_rounds):
        frozen_model = copy.deepcopy(model).eval()
        for p in frozen_model.parameters():
            p.requires_grad_(False)
        frozen_pool.append(frozen_model)
        if len(frozen_pool) > selfplay_pool_size:
            frozen_pool.pop(0)
        print(f"\nSelf-play LoRA fine-tune round {sp_round + 1}/{selfplay_rounds} "
              f"(opponent_selfplay_epsilon={selfplay_epsilon}, pool_size={len(frozen_pool)})...")
        opponent_env = ConnectXEnv(width=env.width, height=env.height, win_len=env.win_len)
        selfplay_fn = make_selfplay_pool_opponent_fn(frozen_pool, normalizer, opponent_env)
        selfplay_train_env = ConnectXEnv(width=env.width, height=env.height, win_len=env.win_len,
                                          opponent_epsilon=opponent_epsilon,
                                          opponent_strong_epsilon=opponent_strong_epsilon,
                                          opponent_selfplay_epsilon=selfplay_epsilon,
                                          opponent_policy_fn=selfplay_fn)
        # Adam(model.value.parameters()) inside train_mc_value_onpolicy
        # naturally trains ONLY the LoRA deltas here: the wrapped base
        # linears have requires_grad=False (set by apply_lora), so their
        # .grad stays None and Adam's step() skips them.
        train_mc_value_onpolicy(selfplay_train_env, model, normalizer, rng,
                                 n_rounds=selfplay_mc_rounds_per_iter,
                                 n_problems_per_round=mc_problems_per_round,
                                 max_steps=max_steps, unsolved_penalty=max_steps)

    _merge_and_unwrap(model.value)

    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": normalizer.mean.cpu(),
        "norm_std": normalizer.std.cpu(),
        "state_dim": raw_ckpt["state_dim"],
        "num_actions": raw_ckpt["num_actions"],
        "latent_dim": raw_ckpt["latent_dim"],
        "hidden_dim": raw_ckpt["hidden_dim"],
        "board_width": env.width,
        "board_height": env.height,
        "win_len": env.win_len,
    }, save_path)
    print(f"\nSaved LoRA-self-play-fine-tuned checkpoint to {save_path}")

    print("\n" + "=" * 20 + " EVALUATION (no oracle -- vs. random-legal-play baseline only) " + "=" * 20)
    eval_rng = random.Random(999)
    problems = [env.random_problem(eval_rng) for _ in range(150)]
    random_baseline_win_rate(env, problems, max_steps, random.Random(1000))
    evaluate(env, model, normalizer, problems, depth=1, beam_width=8, max_total_steps=max_steps,
             label="Baseline A (model, depth=1)")


if __name__ == "__main__":
    main()
