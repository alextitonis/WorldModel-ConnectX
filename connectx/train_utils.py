"""
Stage 1: self-supervised training of encoder + dynamics + decoder on random
(state, action, next_state, reward) transitions and multi-step rollouts --
no labels, no oracle, works for any domain implementing `environment.py`'s
`Environment` interface.

(Value-head training -- "stage 2" -- lives in `verifier.py`, since ConnectX
uses on-policy Monte Carlo value learning rather than an oracle-labeled
regression: there's no tractable exact solver at the real 7x6 board scale.)
"""
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_legal_action(env, rng, state):
    legal = [i for i in range(env.num_actions) if env.is_legal(state, i)]
    return rng.choice(legal)


def generate_transitions(env, rng, n_problems=4000, walk_len=6, **problem_kwargs):
    """Random-walk from random problem starts, collecting (s, a, s', r)
    along the way -- covers realistic reachable states, not just problem
    starts."""
    transitions = []
    for _ in range(n_problems):
        state, _ = env.random_problem(rng, **problem_kwargs)
        for _ in range(walk_len):
            if env.is_solved(state):
                break
            a_idx = sample_legal_action(env, rng, state)
            next_state, reward, done = env.step(state, a_idx)
            transitions.append((state, a_idx, next_state, reward))
            state = next_state
            if done:
                break
    return transitions


def generate_rollout_sequences(env, rng, n_problems=4000, k=3, **problem_kwargs):
    """K-step (states, actions) sequences for the unrolled/open-loop
    training objective below."""
    sequences = []
    for _ in range(n_problems):
        state, _ = env.random_problem(rng, **problem_kwargs)
        states = [state]
        actions = []
        cur = state
        for _ in range(k):
            a_idx = sample_legal_action(env, rng, cur)
            cur, _, _ = env.step(cur, a_idx)
            actions.append(a_idx)
            states.append(cur)
        sequences.append((states, actions))
    return sequences


class StateNormalizer:
    def __init__(self, states):
        t = torch.tensor(states, dtype=torch.float32)
        self.mean = t.mean(dim=0)
        self.std = t.std(dim=0).clamp_min(1e-3)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, state_tensor):
        return (state_tensor - self.mean) / self.std

    def denormalize(self, norm_tensor):
        return norm_tensor * self.std + self.mean


def states_to_tensor(states):
    return torch.tensor([list(s) for s in states], dtype=torch.float32)


def train_stage1(model, normalizer, transitions, sequences=None, k=3,
                  epochs=400, batch_size=256, lr=1e-3):
    """`sequences` (see generate_rollout_sequences) adds a k-step UNROLLED
    loss: the dynamics model is chained k times, feeding its own predicted
    latent back in as input at each step (never re-encoding the real
    intermediate state). This matches how the model is actually used at
    inference (chained multi-step search) -- training on 1-step transitions
    alone leaves compounding rollout error unaddressed."""
    state_dim = normalizer.mean.shape[0]
    states = states_to_tensor([t[0] for t in transitions]).to(DEVICE)
    actions = torch.tensor([t[1] for t in transitions], dtype=torch.long).to(DEVICE)
    next_states = states_to_tensor([t[2] for t in transitions]).to(DEVICE)
    rewards = torch.tensor([t[3] for t in transitions], dtype=torch.float32).to(DEVICE)

    norm_states = normalizer.normalize(states)
    norm_next_states = normalizer.normalize(next_states)

    if sequences is not None:
        seq_states_raw = torch.stack([states_to_tensor(s) for s, _ in sequences]).to(DEVICE)  # [N, k+1, D]
        seq_actions = torch.tensor([a for _, a in sequences], dtype=torch.long).to(DEVICE)  # [N, k]
        norm_seq_states = normalizer.normalize(seq_states_raw.view(-1, state_dim)).view(seq_states_raw.shape)

    n = states.shape[0]
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.dynamics.parameters()) + list(model.decoder.parameters()),
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s, a, s_next, r = norm_states[idx], actions[idx], norm_next_states[idx], rewards[idx]

            z = model.encode(s)
            z_next_target = model.encode(s_next)
            pred_next_z, pred_reward = model.imagine_step(z, a)

            recon = model.reconstruct(z)
            recon_next_from_dynamics = model.reconstruct(pred_next_z)

            loss = (
                mse(recon, s)
                + mse(pred_next_z, z_next_target)
                + mse(recon_next_from_dynamics, s_next)
                + mse(pred_reward, r)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]

        if sequences is not None:
            n_seq = norm_seq_states.shape[0]
            seq_perm = torch.randperm(n_seq, device=DEVICE)
            unrolled_total = 0.0
            for i in range(0, n_seq, batch_size):
                sidx = seq_perm[i:i + batch_size]
                seq_s = norm_seq_states[sidx]  # [B, k+1, D]
                seq_a = seq_actions[sidx]  # [B, k]

                z = model.encode(seq_s[:, 0, :])
                loss_unrolled = 0.0
                for step in range(k):
                    z, _pred_r = model.imagine_step(z, seq_a[:, step])
                    target = seq_s[:, step + 1, :]
                    target_z = model.encode(target)
                    loss_unrolled = loss_unrolled + mse(z, target_z) + mse(model.reconstruct(z), target)
                loss_unrolled = loss_unrolled / k

                opt.zero_grad()
                loss_unrolled.backward()
                opt.step()
                unrolled_total += loss_unrolled.item() * sidx.shape[0]

        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            msg = f"  [stage1] epoch {epoch+1:3d}/{epochs}  loss={total_loss/n:.4f}"
            if sequences is not None:
                msg += f"  unrolled_loss={unrolled_total/n_seq:.4f}"
            print(msg)


@torch.no_grad()
def eval_stage1(model, normalizer, transitions):
    """Decoder reconstruction / dynamics-rollout exact-match accuracy (after
    rounding to the nearest integer) -- the diagnostic that checks the
    latent isn't collapsing to something the decoder can't read back out."""
    states = states_to_tensor([t[0] for t in transitions]).to(DEVICE)
    actions = torch.tensor([t[1] for t in transitions], dtype=torch.long).to(DEVICE)
    next_states = states_to_tensor([t[2] for t in transitions]).to(DEVICE)

    norm_states = normalizer.normalize(states)
    z = model.encode(norm_states)
    recon = normalizer.denormalize(model.reconstruct(z))
    pred_next_z, _ = model.imagine_step(z, actions)
    recon_next = normalizer.denormalize(model.reconstruct(pred_next_z))

    recon_acc = (recon.round() == states).all(dim=1).float().mean().item()
    dyn_acc = (recon_next.round() == next_states).all(dim=1).float().mean().item()
    return {"decoder_recon_exact_acc": recon_acc, "dynamics_rollout_exact_acc": dyn_acc}


@torch.no_grad()
def eval_multistep_rollout(model, normalizer, sequences, k):
    """Chained (open-loop) rollout exact-match accuracy at each step 1..k --
    exposes compounding error the way a single-step eval can't."""
    state_dim = normalizer.mean.shape[0]
    seq_states_raw = torch.stack([states_to_tensor(s) for s, _ in sequences]).to(DEVICE)
    seq_actions = torch.tensor([a for _, a in sequences], dtype=torch.long).to(DEVICE)
    norm_seq_states = normalizer.normalize(seq_states_raw.view(-1, state_dim)).view(seq_states_raw.shape)

    z = model.encode(norm_seq_states[:, 0, :])
    results = {}
    for step in range(k):
        z, _ = model.imagine_step(z, seq_actions[:, step])
        recon = normalizer.denormalize(model.reconstruct(z))
        real = seq_states_raw[:, step + 1, :]
        acc = (recon.round() == real).all(dim=1).float().mean().item()
        results[f"k={step+1}"] = acc
    return results
