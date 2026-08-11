"""
A k-NN lookup table over real (encoded state, outcome) pairs from actually-
played self-play games, consulted alongside the learned value head at
decision time (see `search._evaluate_with_memory` / `adversarial_search.py`).
Distinct from the trained model itself -- nothing here is learned, it's a
cache of real experience the model can fall back on when its own value
estimate might be shaky.

Precedented by episodic control (Blundell et al., "Model-Free Episodic
Control"; Pritzel et al., "Neural Episodic Control") and case-based
reasoning, not a novel mechanism.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EpisodicMemory:
    """Stores (z, outcome) pairs -- z is a REAL encoded state's latent
    (never a predicted/imagined one, so memory never compounds its own
    errors), outcome is that state's real remaining-steps-to-win (for a
    won game) or a fixed penalty (for a lost/drawn one) -- same label scale
    as the value head, so blending stays consistent."""

    def __init__(self):
        self._zs = []
        self._outcomes = []
        self._state_keys = []
        self._seen = {}  # state_key -> index, for exact-match dedup
        self._trust_scale = None

    def __len__(self):
        return len(self._zs)

    def add(self, z, outcome, state_key=None):
        """`state_key`: an optional hashable identity for the real state
        this (z, outcome) came from. When given and already stored, this
        is a revisit of the exact same state -- keep whichever copy has
        the better (smaller) outcome instead of appending a duplicate."""
        if state_key is not None and state_key in self._seen:
            idx = self._seen[state_key]
            if outcome < self._outcomes[idx]:
                self._zs[idx] = z.detach().to("cpu")
                self._outcomes[idx] = float(outcome)
                self._trust_scale = None
            return
        self._zs.append(z.detach().to("cpu"))
        self._outcomes.append(float(outcome))
        self._state_keys.append(state_key)
        if state_key is not None:
            self._seen[state_key] = len(self._zs) - 1
        self._trust_scale = None

    def _stacked(self):
        return torch.stack(self._zs).to(DEVICE), torch.tensor(self._outcomes, device=DEVICE)

    @torch.no_grad()
    def trust_scale(self, sample_size=300):
        """A self-calibrating distance scale for this memory's own latent
        space: the median nearest-OTHER-neighbor distance among a random
        subsample of stored points. A query distance much smaller than
        this means "genuinely close match found"; much larger means
        "nothing like this was ever stored." Self-calibrating per
        checkpoint/latent-dim rather than a hand-picked constant."""
        if self._trust_scale is not None:
            return self._trust_scale
        n = len(self._zs)
        if n < 2:
            self._trust_scale = 1.0
            return self._trust_scale
        Z, _outcomes = self._stacked()
        if n > sample_size:
            idx = torch.randperm(n, device=DEVICE)[:sample_size]
            sample = Z[idx]
        else:
            sample = Z
        dists = torch.cdist(sample, Z)
        dists = torch.where(dists > 1e-6, dists, torch.full_like(dists, float("inf")))
        nn_dist = dists.min(dim=1).values
        nn_dist = nn_dist[torch.isfinite(nn_dist)]
        self._trust_scale = nn_dist.median().item() if len(nn_dist) > 0 else 1.0
        return self._trust_scale

    @torch.no_grad()
    def query_batch(self, zs, k=5):
        """zs: [B, latent_dim]. Returns (blended_estimates [B], trust [B]).
        Each row's blend weights its k nearest stored neighbors by inverse
        distance. `trust` is `exp(-mean_distance / trust_scale)` -- a 0..1
        confidence already normalized against this memory's own typical
        spacing, so callers can scale their blend weight by it directly."""
        Z, outcomes = self._stacked()
        zq = zs.detach().to(DEVICE)
        dists = torch.cdist(zq, Z)
        k = min(k, len(self._zs))
        topk_dists, topk_idx = torch.topk(dists, k, largest=False, dim=1)
        topk_outcomes = outcomes[topk_idx]
        weights = 1.0 / (topk_dists + 1e-2)
        weights = weights / weights.sum(dim=1, keepdim=True)
        blended = (weights * topk_outcomes).sum(dim=1)
        mean_dist = topk_dists.mean(dim=1)
        trust = torch.exp(-mean_dist / self.trust_scale())
        return blended, trust


@torch.no_grad()
def add_trajectory_from_real_path(model, normalizer, memory, path_states, env=None):
    """Adds every state of an actually-played, WON trajectory as a positive
    (attraction) example -- outcome = real distance to the end of this
    path."""
    from train_utils import states_to_tensor

    dedup = env is not None and env.discrete_state
    T = len(path_states) - 1
    for t, s in enumerate(path_states):
        observed = env.observe(s) if env is not None else s
        z = normalizer.normalize(states_to_tensor([observed]).to(DEVICE))
        z = model.encode(z)[0]
        memory.add(z, T - t, state_key=s if dedup else None)


@torch.no_grad()
def add_negative_trajectory_from_real_path(model, normalizer, memory, path_states, penalty, env=None):
    """Adds every state of a LOST/drawn trajectory as a negative
    (repulsion) example -- every state gets the SAME fixed penalty label,
    deliberately uniform across the whole walk. No new retrieval mechanism
    needed: `EpisodicMemory.query_batch`'s existing k-NN blend already
    treats a nearby HIGH-outcome entry as repulsion by construction, the
    exact mirror of how a low one acts as attraction."""
    from train_utils import states_to_tensor

    dedup = env is not None and env.discrete_state
    for s in path_states:
        observed = env.observe(s) if env is not None else s
        z = normalizer.normalize(states_to_tensor([observed]).to(DEVICE))
        z = model.encode(z)[0]
        memory.add(z, float(penalty), state_key=s if dedup else None)
