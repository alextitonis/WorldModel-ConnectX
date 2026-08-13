"""
Packages connectx_checkpoint.pt into a SINGLE self-contained submission.py
Kaggle can actually run -- REWRITTEN 2026-08-10, per explicit user
direction ("let's combine all these in the submission file with
memories, the simulations, the weak learner etc") to fold in THREE
confirmed pieces from this session's follow-up work, replacing the
previous latent-beam-search-only version entirely:

1. **Real adversarial search** (was: latent beam search). Per
   connectx_adversarial_search.py's confirmed result (95-100%/48.3% vs
   random/weak/stronger, beating latent search on every metric by a wide
   margin, confirmed on 2 checkpoints): `env.step()` bundles agent+
   opponent-reply into one transition, so the trained dynamics model was
   never shown "the board right after my move, before their reply" --
   it structurally can't imagine that state. Since ConnectX's rules ARE
   exactly known, this ply is done in REAL board space instead (exact
   enumeration of our moves, exact enumeration of the opponent's
   worst-case real reply), with the learned value head used ONLY as the
   leaf evaluator. This means `dynamics`/`decoder` are NO LONGER NEEDED
   at all (the old latent search's neurosymbolic decode-gate is
   structurally unnecessary once every ply is real, not imagined) --
   only `encoder`+`value` weights are embedded now, a smaller submission.
2. **Episodic memory (positive + negative)**, built OFFLINE (this
   script, at build time) from self-play games against a MIXED opponent
   (weak heuristic + random + the stronger 1-ply-deeper heuristic, per
   explicit user caution -- "so if the opponent is weak it doesn't learn
   the bad ways too" -- see memory_build.py). Won games
   stored as positive (low remaining-steps) examples, lost/drawn games
   as negative (high, fixed-penalty) examples -- one EpisodicMemory,
   blended into every leaf evaluation via the exact same k-NN
   inverse-distance/trust-scaled formula as episodic_memory.py's
   `query_batch`, replicated here in plain torch (no project import,
   this file must stay standalone).
3. **Best-effort online learning ("the weak learner")** -- value-head-
   ONLY updates (matching this session's own confirmed finding: decoder
   updates regressed structured-opponent performance at this data scale,
   so the decoder is excluded entirely here, consistent with "the
   working side only"), applied incrementally as real games are played,
   mirroring continuous_learner.py's confirmed-safe recipe (small
   replay buffer, EMA-updated value_target_mean/std, a few Adam steps
   per update, lr=1e-5) -- reimplemented here in plain torch since this
   file can't import continuous_learner.py.

   **Honest, load-bearing caveat, stated plainly rather than oversold**:
   Kaggle's `agent(observation, configuration)` interface gives no
   direct "episode ended, here's the result" callback -- this file
   infers a completed episode two ways, both using ONLY information
   actually available across calls: (a) our own move immediately wins
   or draws (directly observable -- we know the board we just produced),
   or (b) the NEXT call arrives with a completely empty board while a
   previous episode's trajectory is still buffered -- inferred as a LOSS
   (we didn't win/draw it ourselves, so it must have ended on the
   opponent's move). This whole mechanism is a NO-OP, gracefully, unless
   Kaggle's real evaluation infrastructure reuses the same process across
   multiple episodes for this submission over time (its own rules page,
   read earlier this session, doesn't confirm or deny this -- see
   [[project_connectx_kaggle]]) -- if each episode gets a fresh process,
   this buffer simply starts empty every time and nothing is lost, no
   crash, no wasted budget beyond one negligible check.
"""
import base64
import io

import torch

CKPT_PATH = "checkpoints/connectx_checkpoint.pt"
OUT_PATH = "submission.py"


def _encode_tensor_blob(ck, memory_zs, memory_outcomes):
    """encoder+value weights only (see module docstring -- dynamics/
    decoder are no longer needed by the real adversarial search), plus
    value_target_mean/std (top-level buffers, not nested under a
    submodule prefix) and the offline-built episodic memory's raw
    (z, remaining_steps) pairs."""
    keep = {k: v for k, v in ck["model_state"].items()
            if k.startswith("encoder.") or k.startswith("value.")
            or k in ("value_target_mean", "value_target_std")}
    payload = {
        "weights": keep,
        "norm_mean": ck["norm_mean"],
        "norm_std": ck["norm_std"],
        "state_dim": ck["state_dim"],
        "num_actions": ck["num_actions"],
        "latent_dim": ck["latent_dim"],
        "hidden_dim": ck["hidden_dim"],
        "board_width": ck["board_width"],
        "board_height": ck["board_height"],
        "win_len": ck["win_len"],
        "memory_zs": torch.stack(memory_zs) if memory_zs else torch.zeros(0, ck["latent_dim"]),
        "memory_outcomes": torch.tensor(memory_outcomes, dtype=torch.float32),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


SUBMISSION_TEMPLATE = '''\
"""
Auto-generated by build_submission.py -- DO NOT hand-edit
(regenerate instead). Self-contained Kaggle ConnectX submission: no
imports beyond torch/base64/io, so it runs standalone in Kaggle's
evaluation sandbox.

Policy: ONE ROUND of REAL adversarial search (exact enumeration of our
legal moves, exact enumeration of the opponent's real legal replies,
worst-case-for-us selected -- a genuine minimax over EXACTLY KNOWN board
dynamics, not an imagined latent transition) -- the learned value head
is used ONLY as the leaf evaluator on a real, never-imagined state,
optionally blended with an offline-built episodic memory (won AND lost
self-play games, see module docstring). A best-effort online value-head
update also runs across real games as they're played -- see module
docstring's honest caveat about when this can/can't actually do
anything, given Kaggle's evaluation interface.

**Honest, named limitation** (see connectx_env.py / [[project_connectx_kaggle]]):
the base checkpoint was trained via self-play against a small set of
fixed/self-generated opponents, not against Kaggle's real matchmaking
pool -- see that project's memory entry for the full picture, including
this session's confirmed numbers against synthetic test opponents.
"""
import base64
import collections
import io
import time

import torch

_MEMORY_WEIGHT = {memory_weight}
_MEMORY_K = {memory_k}
_ONLINE_LR = {online_lr}
_ONLINE_UPDATES_PER_EPISODE = {online_updates_per_episode}
_ONLINE_BATCH_SIZE = {online_batch_size}
_UNSOLVED_PENALTY_MULT = {unsolved_penalty_mult}  # x max_steps, matches this session's convention
_ADV_ROUNDS = {adv_rounds}  # real adversarial search rounds -- see _adversarial_plan_action's docstring for timing
# `_ENDGAME_MAX_COLS`/`_ENDGAME_TIME_BUDGET` (added 2026-08-11): below
# this many legal columns, `_exact_endgame_solve` (a real, no-NN,
# alpha-beta minimax to the true end of the game) is tried FIRST and used
# directly if it finishes in time -- see that function's own docstring
# for the calibration and the exact failure mode (a zugzwang/parity trap
# invisible to any bounded-depth search) this targets. `_ENDGAME_MAX_COLS
# = 0` disables this path entirely.
_ENDGAME_MAX_COLS = {endgame_max_cols}
_ENDGAME_TIME_BUDGET = {endgame_time_budget}
# `_DEEPER_ROUNDS`/`_DEEPER_MAX_BRANCHING`/`_DEEPER_TIME_BUDGET`: real,
# mined-from-real-games evidence showed `_ADV_ROUNDS` sometimes sees ZERO
# danger on a position (every column looks equally safe) 2-4 plies before
# a trap that one round DEEPER already narrows down to exactly one safe
# column -- `_ADV_ROUNDS` isn't wrong about what it can see, it just can't
# see far enough to avoid a fork the opponent is setting up. A deeper
# search is provably too slow to run on EVERY move (measured 8-13s at a
# 6-7-legal-column branching factor) -- so this is a SAFE, opportunistic
# escalation, not a blanket depth increase: `_DEEPER_ROUNDS = None`
# disables it entirely, reproducing the original `_ADV_ROUNDS`-only
# behavior byte-for-byte. When enabled, AFTER computing the normal-
# `_ADV_ROUNDS` answer (always -- the guaranteed-safe fallback), a
# `_DEEPER_ROUNDS`-round search is attempted under a hard
# `_DEEPER_TIME_BUDGET` deadline; if it finishes in time its answer is
# used instead (strictly more information, never less), if it times out
# the original answer is returned completely unchanged. Calibrated via a
# 180-game regression suite (random/weak/stronger opponents): zero
# win-rate regression, max observed single-move time 1.641s --
# comfortably under Kaggle's 2s budget.
_DEEPER_ROUNDS = {deeper_rounds}
_DEEPER_MAX_BRANCHING = {deeper_max_branching}
_DEEPER_TIME_BUDGET = {deeper_time_budget}
# `_ONLINE_ENABLED` (added 2026-08-10, right before submitting -- explicit
# user decision after reading the competition's own rule "An Agent's sole
# purpose is to generate an action. Activities/code which do not directly
# contribute to this will be considered malicious...": the online "weak
# learner"'s gradient updates are arguably in service of generating BETTER
# actions, not unrelated activity, but it's a genuine judgment call with
# real (if likely small) risk, not a zero-risk one -- played safe rather
# than assume it's fine. False disables it CLEANLY (no buffer/episode-
# tracking side-state at all when off, not just a no-op update call) so
# a disabled submission's `agent()` genuinely does nothing but generate
# an action, matching the rule as literally as possible.
_ONLINE_ENABLED = {online_enabled}

_BLOB_B64 = (
{blob_literal}
)


def _load():
    payload = torch.load(io.BytesIO(base64.b64decode(_BLOB_B64)), map_location="cpu")
    return payload


_P = _load()
_W = _P["weights"]
_NORM_MEAN = _P["norm_mean"]
_NORM_STD = _P["norm_std"]
_LATENT_DIM = _P["latent_dim"]
_NUM_ACTIONS = _P["num_actions"]  # includes the training-time PASS action (index WIDTH)
_WIDTH = _P["board_width"]
_HEIGHT = _P["board_height"]
_WIN_LEN = _P["win_len"]
_PASS_ACTION = _WIDTH
_CELL_WIDTH = 3
_EMPTY, _AGENT, _OPPONENT = 0, 1, 2
_MAX_STEPS = (_WIDTH * _HEIGHT) // 2 + 2
# `_UNSOLVED_PENALTY` (used ONLY by the online learner's episode-ending
# label, matching continuous_learner.py's own 1x-max_steps convention)
# and `_LOSS_PENALTY` (used ONLY by the adversarial search's "opponent
# wins" terminal case) are DELIBERATELY SEPARATE constants -- a real bug
# found and fixed 2026-08-10, right after this build was already live:
# an earlier version used _UNSOLVED_PENALTY (1x max_steps) for BOTH,
# which meant the search scored "the opponent wins outright" EXACTLY
# THE SAME as "it's a mere draw" -- losing must be unambiguously worse
# than a draw for the search to reliably prioritize blocking a real
# threat over a merely-mediocre move, matching connectx_adversarial_search.py's
# original, correct 2x convention. Confirmed as the direct, mechanistic
# cause of a real observed failure: the deployed agent missed blocking
# an opponent's obvious 3-in-a-column vertical threat, scoring the
# blocking move WORSE (23.463) than a non-blocking move that let the
# opponent win outright (23.000, since the loss was scored at only
# max_steps=23, indistinguishable from ordinary mediocre play).
_UNSOLVED_PENALTY = _UNSOLVED_PENALTY_MULT * _MAX_STEPS
_LOSS_PENALTY = 2 * _MAX_STEPS

# Memory tensors (offline-built, see module docstring) -- fixed, never
# grow at runtime (only the ONLINE value-head buffer below does).
_MEMORY_Z = _P["memory_zs"]
_MEMORY_OUTCOMES = _P["memory_outcomes"]
if _MEMORY_Z.shape[0] >= 2:
    _d = torch.cdist(_MEMORY_Z, _MEMORY_Z)
    _d = torch.where(_d > 1e-6, _d, torch.full_like(_d, float("inf")))
    _nn = _d.min(dim=1).values
    _nn = _nn[torch.isfinite(_nn)]
    _MEMORY_TRUST_SCALE = _nn.median().item() if len(_nn) > 0 else 1.0
else:
    _MEMORY_TRUST_SCALE = 1.0

# --- Value head params made trainable for the online "weak learner"
# (see module docstring's honest caveat) -- encoder stays FROZEN
# (never in this optimizer), matching continuous_learner.py's confirmed
# recipe: only the value head updates online. When `_ONLINE_ENABLED` is
# False, NONE of this setup happens at all (no optimizer, no
# requires_grad, no buffers) -- `agent()` genuinely does nothing but
# generate an action in that case, not just a disabled-but-present
# mechanism. ---
if _ONLINE_ENABLED:
    _VALUE_PARAM_KEYS = [k for k in _W if k.startswith("value.")]
    for _k in _VALUE_PARAM_KEYS:
        _W[_k].requires_grad_(True)
    # Buffers, not trained parameters (EMA-updated in-place under
    # no_grad, matching continuous_learner.py's own convention) -- never
    # added to the optimizer below.
    _VALUE_TARGET_MEAN = _W.get("value_target_mean", torch.tensor(0.0)).clone()
    _VALUE_TARGET_STD = _W.get("value_target_std", torch.tensor(1.0)).clone()
    _ONLINE_OPT = torch.optim.Adam([_W[k] for k in _VALUE_PARAM_KEYS], lr=_ONLINE_LR)
    _REPLAY_BUFFER = collections.deque(maxlen=2000)  # (state_vec: list[float], label: float)
    _EPISODE_STATES = []  # real one-hot state vectors seen/produced so far THIS episode
    _EPISODE_LAST_PIECES = None  # total board piece count as of our last recorded state THIS episode
else:
    _VALUE_TARGET_MEAN = _W.get("value_target_mean", torch.tensor(0.0))
    _VALUE_TARGET_STD = _W.get("value_target_std", torch.tensor(1.0))


def _linear(x, w_key, b_key):
    return torch.nn.functional.linear(x, _W[w_key], _W[b_key])


def _mlp3(x, prefix):
    """Replicates model.py's `mlp([in, hidden, hidden, out])`: Linear ->
    ReLU -> Linear -> ReLU -> Linear (params at Sequential indices
    0/2/4, confirmed against the actual saved state_dict keys)."""
    h = torch.relu(_linear(x, f"{{prefix}}.net.0.weight", f"{{prefix}}.net.0.bias"))
    h = torch.relu(_linear(h, f"{{prefix}}.net.2.weight", f"{{prefix}}.net.2.bias"))
    return _linear(h, f"{{prefix}}.net.4.weight", f"{{prefix}}.net.4.bias")


def _encode(state_vec):
    return _mlp3(state_vec, "encoder")


def _value_raw(z):
    return _mlp3(z, "value").squeeze(-1)


def _value(z):
    """Real-scale value estimate (remaining steps), see model.py's
    WorldModel.evaluate -- denormalizes the network's raw prediction."""
    return _value_raw(z) * _VALUE_TARGET_STD + _VALUE_TARGET_MEAN


def _memory_blend(z_batch, raw_values):
    """Same k-NN inverse-distance/trust-scaled blend as
    episodic_memory.py's EpisodicMemory.query_batch -- replicated here
    in plain torch (this file can't import that module)."""
    if _MEMORY_Z.shape[0] == 0 or _MEMORY_WEIGHT <= 0:
        return raw_values
    dists = torch.cdist(z_batch, _MEMORY_Z)  # [B, N]
    k = min(_MEMORY_K, _MEMORY_Z.shape[0])
    topk_dists, topk_idx = torch.topk(dists, k, largest=False, dim=1)
    topk_outcomes = _MEMORY_OUTCOMES[topk_idx]
    weights = 1.0 / (topk_dists + 1e-2)
    weights = weights / weights.sum(dim=1, keepdim=True)
    blended = (weights * topk_outcomes).sum(dim=1)
    mean_dist = topk_dists.mean(dim=1)
    trust = torch.exp(-mean_dist / _MEMORY_TRUST_SCALE)
    w = _MEMORY_WEIGHT * trust
    return (1 - w) * raw_values + w * blended


# --- Plain-Python board helpers (no torch) -- mirrors connectx_env.py's
# free functions exactly, duplicated here (not imported) since this file
# must be standalone. ---

def _onehot(idx, n):
    v = [0] * n
    v[idx] = 1
    return v


def _rc(row, col):
    return row * _WIDTH + col


def _encode_board(cells):
    out = []
    for c in cells:
        out.extend(_onehot(c, _CELL_WIDTH))
    return out


def _lowest_empty_row(cells, col):
    for row in range(_HEIGHT - 1, -1, -1):
        if cells[_rc(row, col)] == _EMPTY:
            return row
    return None


def _legal_columns(cells):
    return [c for c in range(_WIDTH) if _lowest_empty_row(cells, c) is not None]


def _wins_for(cells, mark):
    for row in range(_HEIGHT):
        for col in range(_WIDTH):
            if cells[_rc(row, col)] != mark:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = row + dr * (_WIN_LEN - 1), col + dc * (_WIN_LEN - 1)
                if not (0 <= er < _HEIGHT and 0 <= ec < _WIDTH):
                    continue
                if all(cells[_rc(row + dr * k, col + dc * k)] == mark for k in range(_WIN_LEN)):
                    return True
    return False


def _board_full(cells):
    return all(c != _EMPTY for c in cells)


def _apply_move(cells, col, mark):
    row = _lowest_empty_row(cells, col)
    new_cells = list(cells)
    new_cells[_rc(row, col)] = mark
    return new_cells


def _kaggle_board_to_cells(board, mark):
    """Kaggle's board: flat list, row-major, 0=empty/1=P1/2=P2, row 0 =
    top -- SAME convention connectx_env.py already uses, confirmed
    against kaggle_environments' own connectx.json. `mark` tells us
    which of Kaggle's 1/2 is US."""
    opponent_mark = 2 if mark == 1 else 1
    cells = []
    for v in board:
        if v == 0:
            cells.append(_EMPTY)
        elif v == mark:
            cells.append(_AGENT)
        else:
            assert v == opponent_mark
            cells.append(_OPPONENT)
    return cells


def _leaf_batch_values(states):
    if not states:
        return {{}}
    state_t = torch.tensor(states, dtype=torch.float32)
    norm_t = (state_t - _NORM_MEAN) / _NORM_STD
    z = _encode(norm_t)
    vals = _memory_blend(z, _value(z))
    return dict(zip(states, vals.tolist()))


def _narrow_to_center(legal_cols, max_branching):
    """Prunes a legal-column list down to `max_branching` columns closest
    to the board's center -- free, real Connect-4 domain knowledge (a
    center column touches more potential 4-in-a-row lines than an edge
    one, same theory as the empty-board opening hint). `max_branching=
    None` is a no-op -- exact, unpruned enumeration. Only ever applied to
    OUR OWN follow-up move choices at the deeper-escalation's round 2+
    (see `_DEEPER_ROUNDS`'s docstring) -- never to `_ADV_ROUNDS`'s own
    (always-unpruned) path, and never to the opponent's reply enumeration
    at ANY round (that's what makes this a genuine worst-case
    guarantee -- narrowing it would mean silently ignoring some of the
    opponent's real threats)."""
    if max_branching is None or len(legal_cols) <= max_branching:
        return legal_cols
    center = (_WIDTH - 1) / 2
    return sorted(legal_cols, key=lambda c: abs(c - center))[:max_branching]


class _RoundSearchTimeout(Exception):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.time() > deadline:
        raise _RoundSearchTimeout()


def _collect_leaves(cells1, remaining_rounds, leaf_cache, max_branching=None, deadline=None):
    _check_deadline(deadline)
    if _board_full(cells1):
        return
    for opp_col in _legal_columns(cells1):
        cells2 = _apply_move(cells1, opp_col, _OPPONENT)
        if _wins_for(cells2, _OPPONENT) or _board_full(cells2):
            continue
        if remaining_rounds <= 1:
            leaf_cache[tuple(_encode_board(cells2))] = None
        else:
            for a2 in _narrow_to_center(_legal_columns(cells2), max_branching):
                cells3 = _apply_move(cells2, a2, _AGENT)
                if _wins_for(cells3, _AGENT):
                    continue
                _collect_leaves(cells3, remaining_rounds - 1, leaf_cache, max_branching, deadline)


def _score_after_our_move(cells1, remaining_rounds, leaf_cache, max_branching=None, deadline=None):
    """cells1: real board right after OUR move (caller already ruled out
    an immediate win here). Returns our worst-case score -- opponent
    picks whichever real reply hurts us most. Reads leaf values from
    `leaf_cache` (already populated by ONE upfront batched call over the
    WHOLE tree -- see _adversarial_plan_action) instead of calling the
    value head again at every node."""
    if _board_full(cells1):
        return float(_MAX_STEPS)
    vals = []
    for opp_col in _legal_columns(cells1):
        cells2 = _apply_move(cells1, opp_col, _OPPONENT)
        if _wins_for(cells2, _OPPONENT):
            vals.append(float(_LOSS_PENALTY))  # opponent wins -- worse than a mere draw, see _LOSS_PENALTY's comment
        elif _board_full(cells2):
            vals.append(float(_MAX_STEPS))
        elif remaining_rounds <= 1:
            vals.append(leaf_cache[tuple(_encode_board(cells2))])
        else:
            vals.append(_score_after_opponent_move(cells2, remaining_rounds - 1, leaf_cache, max_branching, deadline))
    return max(vals)


def _score_after_opponent_move(cells2, remaining_rounds, leaf_cache, max_branching=None, deadline=None):
    """cells2: real board after the opponent's move, our turn again.
    Returns OUR best achievable worst-case score from here."""
    _check_deadline(deadline)
    our_legal = _narrow_to_center(_legal_columns(cells2), max_branching)
    if not our_legal:
        return float(_MAX_STEPS)
    best = None
    for a in our_legal:
        cells3 = _apply_move(cells2, a, _AGENT)
        if _wins_for(cells3, _AGENT):
            return -float(_MAX_STEPS)  # a forced win exists deeper -- short-circuit
        s = _score_after_our_move(cells3, remaining_rounds, leaf_cache, max_branching, deadline)
        if best is None or s < best:
            best = s
    return best


class _EndgameTimeout(Exception):
    pass


def _exact_endgame_solve(cells0, mover, deadline):
    """Exact (no NN) alpha-beta minimax to the true end of the game --
    see adversarial_search.py's identical function for
    the full docstring/calibration; this is a plain-torch-free, standalone
    port (same convention as every other function in this file) so the
    packaged submission never imports the project. Returns
    `(best_action, value)` (value from `mover`'s own perspective, +1/-1/0)
    or `(None, None)` if `deadline` was hit first."""
    memo = {{}}
    center = (_WIDTH - 1) / 2

    def solve(cells, to_move, alpha, beta):
        if time.time() > deadline:
            raise _EndgameTimeout()
        key = (tuple(cells), to_move)
        cached = memo.get(key)
        if cached is not None:
            return cached
        other = _OPPONENT if to_move == _AGENT else _AGENT
        legal = sorted(_legal_columns(cells), key=lambda c: abs(c - center))
        if not legal:
            memo[key] = 0.0
            return 0.0
        if to_move == _AGENT:
            best = -2.0
            for c in legal:
                nxt = _apply_move(cells, c, to_move)
                if _wins_for(nxt, to_move):
                    val = 1.0
                elif _board_full(nxt):
                    val = 0.0
                else:
                    val = solve(nxt, other, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = 2.0
            for c in legal:
                nxt = _apply_move(cells, c, to_move)
                if _wins_for(nxt, to_move):
                    val = -1.0
                elif _board_full(nxt):
                    val = 0.0
                else:
                    val = solve(nxt, other, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)
                if alpha >= beta:
                    break
        memo[key] = best
        return best

    root_legal = _legal_columns(cells0)
    if not root_legal:
        return None, None
    root_legal = sorted(root_legal, key=lambda c: abs(c - center))
    other = _OPPONENT if mover == _AGENT else _AGENT
    try:
        best_a, best_val = None, None
        for c in root_legal:
            nxt = _apply_move(cells0, c, mover)
            if _wins_for(nxt, mover):
                val = 1.0 if mover == _AGENT else -1.0
            elif _board_full(nxt):
                val = 0.0
            else:
                val = solve(nxt, other, -1.0, 1.0)
            if best_val is None or (mover == _AGENT and val > best_val) or (mover == _OPPONENT and val < best_val):
                best_a, best_val = c, val
            if (mover == _AGENT and best_val == 1.0) or (mover == _OPPONENT and best_val == -1.0):
                break
        return best_a, best_val
    except _EndgameTimeout:
        return None, None


def _hands_immediate_win(cells1):
    """True if the OPPONENT has an immediate (1-ply) winning reply
    available from `cells1` -- the real board right after OUR candidate
    move. Added 2026-08-13, ported from the private tree's identical
    fix -- see `_run_search`'s own comment for the real regression this
    fixes: found by mining fresh Kaggle replays for a deployed
    submission whose public score dropped after adding a deeper-search
    escalation. Most affected losses were already genuine forced losses
    regardless of our move (a deeper search simply SEES that sooner,
    correctly scoring every candidate identically badly), but with
    every candidate tied on score, `_run_search`'s plain `<` comparison
    silently kept whichever move came first in center-out order --
    which is NOT necessarily one that avoids handing the opponent an
    immediate win THIS move, even when a tied alternative exists that
    does. Against a real, imperfect opponent pool (not a perfect
    solver), an immediate giveaway forfeits every chance of a mistake;
    delaying the loss does not, so this is a real improvement even in
    an already-lost position, not just cosmetic. Cheap: one extra
    O(width) legal-move scan per candidate."""
    for opp_col in _legal_columns(cells1):
        if _wins_for(_apply_move(cells1, opp_col, _OPPONENT), _OPPONENT):
            return True
    return False


def _run_search(surviving_actions, action_cells1, search_rounds, max_branching=None, deadline=None):
    """One full leaf-collect + batched-eval + minimax pass at a given
    (rounds, max_branching) setting -- factored out so it can be called
    at two different depths, see `_DEEPER_ROUNDS`'s docstring above.
    `deadline`: propagated into `_collect_leaves`/`_score_after_opponent_
    move` (checked at both exponential-blowup recursion points) AND
    checked again here, immediately around the ONE batched NN forward
    pass -- that call is otherwise UNGUARDED/uninterruptible once
    started, so bailing out right before it (rather than only inside the
    pure-Python recursion) avoids ever starting an expensive tensor op
    with no time budget left for it.

    Tie-break (added 2026-08-13, see `_hands_immediate_win`'s docstring):
    among candidates tied on `s`, prefer one that does NOT hand the
    opponent an immediate 1-ply win. Strictly refines the prior
    center-out-only tie-break -- never overrides a genuinely BETTER
    score, only breaks ties among equally-scored candidates."""
    leaf_cache = {{}}
    for a in surviving_actions:
        _collect_leaves(action_cells1[a], search_rounds, leaf_cache, max_branching, deadline)
    _check_deadline(deadline)
    if leaf_cache:
        leaf_cache.update(_leaf_batch_values(list(leaf_cache.keys())))
    _check_deadline(deadline)  # don't walk the tree on a stale/over-budget result either

    best_a, best_score, best_hands_win = None, None, None
    for a in surviving_actions:
        s = _score_after_our_move(action_cells1[a], search_rounds, leaf_cache, max_branching, deadline)
        hands_win = _hands_immediate_win(action_cells1[a])
        key = (s, hands_win)
        if best_score is None or key < (best_score, best_hands_win):
            best_a, best_score, best_hands_win = a, s, hands_win
    return best_a


@torch.no_grad()
def _adversarial_plan_action(cells0):
    """`_ADV_ROUNDS` real adversarial rounds (our move, then the
    opponent's worst-case real reply, repeated) before falling back to
    the learned value head + memory blend as the leaf evaluator -- every
    transition at every round is EXACT (real board simulation, never
    imagined). Root action never returns PASS.

    **Two-phase, GLOBALLY batched leaf evaluation** (fixed 2026-08-10,
    same day, right before submitting -- a real timing bug caught just
    in time, see connectx_adversarial_search.py's identical fix for the
    full story): calling the leaf evaluator separately at every node in
    the tree (the first version of `rounds>1`) measured up to 2.3s/move
    against the offline-built ~2600-state memory -- OVER Kaggle's 2s
    budget. Fixed by walking the tree TWICE (pure Python, cheap): once
    to collect every non-terminal leaf across the WHOLE tree into one
    deduplicated set (transpositions collapse for free), then ONE single
    batched value+memory call, then a second walk doing the actual
    minimax from the precomputed lookup. Re-measured after the fix
    across 60 diverse positions (including the maximal-branching empty-
    board case): rounds=1 max 0.427s, rounds=2 max 0.375s -- comfortably
    (~5x) under budget again."""
    root_legal = _legal_columns(cells0)
    if not root_legal:
        return None

    if _ENDGAME_MAX_COLS and len(root_legal) <= _ENDGAME_MAX_COLS:
        exact_a, _exact_val = _exact_endgame_solve(cells0, _AGENT, deadline=time.time() + _ENDGAME_TIME_BUDGET)
        if exact_a is not None:
            return exact_a
        # else: timed out -- fall through to the round-based search below
        # exactly as if this check had never happened.

    # Center-out root ordering -- NOT a pruning change (every legal column
    # is still considered, nothing narrowed), only fixes which column wins
    # a TIE. The scoring loop below uses strict `<`, so the first action
    # seen at a given score silently wins ties; left-to-right order made
    # that default to the LEFTMOST column, an arbitrary, exploitable bias
    # with no game-theoretic basis (unlike the player-1 opening hint,
    # which deliberately picks center for a real reason). Center columns
    # are the real stronger choice under a tie (more potential 4-in-a-row
    # lines pass through them, same fact `_narrow_to_center` already uses
    # for pruning) -- found from a direct user-observed pattern in real
    # play ("when we are second we put in left going to right").
    _center = (_WIDTH - 1) / 2
    root_legal = sorted(root_legal, key=lambda c: abs(c - _center))

    surviving_actions, action_cells1 = [], {{}}
    for a in root_legal:
        cells1 = _apply_move(cells0, a, _AGENT)
        if _wins_for(cells1, _AGENT):
            return a  # immediate win -- take it, no need to consider anything else
        surviving_actions.append(a)
        action_cells1[a] = cells1

    base_a = _run_search(surviving_actions, action_cells1, _ADV_ROUNDS)  # always computed -- guaranteed-safe fallback

    if _DEEPER_ROUNDS is not None:
        try:
            return _run_search(surviving_actions, action_cells1, _DEEPER_ROUNDS,
                                max_branching=_DEEPER_MAX_BRANCHING,
                                deadline=time.time() + _DEEPER_TIME_BUDGET)
        except _RoundSearchTimeout:
            pass  # didn't finish in time -- fall back to base_a exactly as if _DEEPER_ROUNDS were None

    return base_a


def _online_update(path_states, label):
    """A FEW Adam steps on a mixed old+new batch from the persisted
    replay buffer -- value head ONLY (encoder frozen), mirrors
    continuous_learner.py's confirmed-safe recipe exactly (small
    updates, EMA-scaled value targets, never a full retrain on just the
    latest episode). `label`: either "steps" (a real win -- each state
    labeled with its real remaining-step count) or a fixed penalty
    (loss/draw -- every state in the walk labeled uniformly bad, same
    convention as this session's `unsolved_penalty`). Only ever called
    from `agent()`'s `_ONLINE_ENABLED`-guarded blocks, but a defensive
    no-op guard here too -- never trust a single call site alone for
    something this load-bearing."""
    global _VALUE_TARGET_MEAN, _VALUE_TARGET_STD
    if not _ONLINE_ENABLED:
        return
    if label == "steps":
        T = len(path_states) - 1
        for t, s in enumerate(path_states):
            _REPLAY_BUFFER.append((list(s), float(T - t)))
    else:
        for s in path_states:
            _REPLAY_BUFFER.append((list(s), float(label)))

    if len(_REPLAY_BUFFER) < 8:
        return
    pool = list(_REPLAY_BUFFER)
    states_t = torch.tensor([s for s, _r in pool], dtype=torch.float32)
    returns_t = torch.tensor([r for _s, r in pool], dtype=torch.float32)

    momentum = 0.98
    new_mean, new_std = returns_t.mean(), returns_t.std().clamp(min=1e-3)
    with torch.no_grad():
        _VALUE_TARGET_MEAN.mul_(momentum).add_(new_mean, alpha=1 - momentum)
        _VALUE_TARGET_STD.mul_(momentum).add_(new_std, alpha=1 - momentum)
    returns_norm = (returns_t - _VALUE_TARGET_MEAN) / _VALUE_TARGET_STD

    norm_states_t = (states_t - _NORM_MEAN) / _NORM_STD
    with torch.no_grad():
        z_all = _encode(norm_states_t)

    n = len(pool)
    bs = min(_ONLINE_BATCH_SIZE, n)
    for _ in range(_ONLINE_UPDATES_PER_EPISODE):
        idx = torch.randperm(n)[:bs]
        pred = _value_raw(z_all[idx])
        loss = torch.nn.functional.mse_loss(pred, returns_norm[idx])
        _ONLINE_OPT.zero_grad()
        loss.backward()
        _ONLINE_OPT.step()


def agent(observation, configuration):
    global _EPISODE_STATES, _EPISODE_LAST_PIECES
    board = list(observation.board)
    mark = observation.mark
    cells = _kaggle_board_to_cells(board, mark)

    # See _ONLINE_ENABLED's own comment above -- when False, NONE of the
    # episode-tracking/online-update machinery below runs at all, not
    # just a no-op call: `agent()` genuinely does nothing but pick a
    # move in that case.
    if _ONLINE_ENABLED:
        cur_pieces = sum(1 for v in board if v != 0)
        # See module docstring's honest caveat -- detecting "a previous
        # episode ended without us ever winning/drawing it ourselves"
        # needs care: checking for an ALL-EMPTY board only works when we
        # happen to be the FIRST mover in the new episode -- as the
        # second mover, the very first board we see already has the
        # opponent's first piece on it, so that check would silently
        # miss the boundary and keep appending to a STALE trajectory
        # from the already-ended previous episode (a real bug, caught
        # before submission: our own test harness alternates which side
        # we play, exactly the condition that triggers it). Robust fix:
        # within one genuinely continuing episode, the board's total
        # piece count increases by EXACTLY 1 between our own consecutive
        # calls (one opponent move happened since we last acted) -- any
        # other delta means a new episode has started, whichever side we
        # were on. Infer a LOSS (the only remaining possibility -- our
        # own win/draw is caught below, right after our own move).
        #
        # `_LOSS_PENALTY`, NOT `_UNSOLVED_PENALTY` (fixed 2026-08-10,
        # follow-up session -- found from a direct user-observed real-game
        # pattern, "one move before losing, ours plays leftmost"): this is
        # the exact same mistake as the already-fixed "attacks but never
        # defends" search bug, just unfixed in a SECOND place. The two
        # penalties were introduced specifically so the SEARCH treats an
        # opponent win as worse than a mere draw -- but the online
        # learner's own training label here used `_UNSOLVED_PENALTY` (the
        # DRAW value) for a genuine LOSS too, teaching the value head that
        # losing and drawing are equally bad. Confirmed via real losses
        # mined from actual Kaggle replays: the fresh (never-online-
        # updated) search correctly blocks in all 3 traced cases, but the
        # live, online-drifted process played the losing move instead --
        # this conflated label is the direct mechanism.
        if _EPISODE_STATES and cur_pieces != _EPISODE_LAST_PIECES + 1:
            _online_update(_EPISODE_STATES, float(_LOSS_PENALTY))
            _EPISODE_STATES = []
        if not _EPISODE_STATES:
            _EPISODE_STATES.append(tuple(_encode_board(cells)))

    legal_cols = _legal_columns(cells)
    if not legal_cols:
        return 0  # should never happen -- Kaggle only calls us on a non-terminal state

    # Free, EXACT domain knowledge (same "neurosymbolic gate" philosophy
    # as every other domain's hand-given hint in this project): on a
    # completely empty board, the center column is the known-best
    # Connect-4 opening. Costs nothing, never worse than guessing.
    if all(c == _EMPTY for c in cells):
        best_action = _WIDTH // 2
    else:
        best_action = _adversarial_plan_action(cells)
        if best_action is None:
            return legal_cols[0]

    if not _ONLINE_ENABLED:
        return int(best_action)

    post_cells = _apply_move(cells, best_action, _AGENT)
    _EPISODE_STATES.append(tuple(_encode_board(post_cells)))
    _EPISODE_LAST_PIECES = sum(1 for v in board if v != 0) + 1

    if _wins_for(post_cells, _AGENT):
        _online_update(_EPISODE_STATES, "steps")
        _EPISODE_STATES = []
    elif _board_full(post_cells):
        _online_update(_EPISODE_STATES, float(_UNSOLVED_PENALTY))
        _EPISODE_STATES = []

    return int(best_action)
'''


def main(ckpt_path=CKPT_PATH, memory_ckpt_path=None, n_memory_games=500,
         memory_opponent_epsilon=0.2, memory_opponent_strong_epsilon=0.3,
         memory_weight=0.25, memory_k=5, online_lr=1e-5, online_updates_per_episode=4,
         online_batch_size=256, unsolved_penalty_mult=1.0, adv_rounds=2, seed=0,
         online_enabled=False, endgame_max_cols=5, endgame_time_budget=1.2,
         deeper_rounds=None, deeper_max_branching=4, deeper_time_budget=0.6):
    import random
    from connectx.env import ConnectXEnv
    from connectx.memory_build import build_episodic_memory
    from connectx.search import load_checkpoint

    ck = torch.load(ckpt_path, map_location="cpu")

    # Memory is built using the SAME real adversarial search (rounds=
    # adv_rounds) the deployed submission actually plays with, so the
    # stored trajectories are representative of the real deployed agent's
    # own play, not a different/weaker search's games.
    print(f"Building offline episodic memory ({n_memory_games} self-play games, mixed opponent, "
          f"real adversarial search rounds={adv_rounds})...")
    mem_ckpt = memory_ckpt_path or ckpt_path
    model, normalizer = load_checkpoint(mem_ckpt)
    env = ConnectXEnv(width=ck["board_width"], height=ck["board_height"], win_len=ck["win_len"])
    rng = random.Random(seed)
    # env.py's opponent_epsilon/opponent_strong_epsilon rolls read Python's
    # GLOBAL random module directly, not this `rng` object -- without this,
    # "same seed" memory-building runs are silently NOT reproducible.
    random.seed(seed)
    memory = build_episodic_memory(env, model, normalizer, rng, n_games=n_memory_games,
                                    opponent_epsilon=memory_opponent_epsilon,
                                    opponent_strong_epsilon=memory_opponent_strong_epsilon,
                                    adversarial_rounds=adv_rounds)
    memory_zs = [z.detach().cpu() for z in memory._zs]
    memory_outcomes = list(memory._outcomes)

    blob = _encode_tensor_blob(ck, memory_zs, memory_outcomes)
    width = 100
    chunks = [blob[i:i + width] for i in range(0, len(blob), width)]
    blob_literal = "\n".join(f'    "{c}"' for c in chunks)

    out = SUBMISSION_TEMPLATE.format(
        blob_literal=blob_literal, memory_weight=memory_weight, memory_k=memory_k,
        online_lr=online_lr, online_updates_per_episode=online_updates_per_episode,
        online_batch_size=online_batch_size, unsolved_penalty_mult=unsolved_penalty_mult,
        adv_rounds=adv_rounds, online_enabled=online_enabled,
        endgame_max_cols=endgame_max_cols, endgame_time_budget=endgame_time_budget,
        deeper_rounds=deeper_rounds, deeper_max_branching=deeper_max_branching,
        deeper_time_budget=deeper_time_budget,
    )
    with open(OUT_PATH, "w") as f:
        f.write(out)
    size_kb = len(out.encode("utf-8")) / 1024
    print(f"Wrote {OUT_PATH} ({len(memory_zs)} memory states, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
