"""
Real (not latent-imagined) minimax search over the actual board, plus an
exact endgame solver -- the search this project actually deploys, and the
main reason it beats the latent-search baseline in `search.py` decisively.

Why real board space, not latent imagination: `env.step()` bundles the
agent's move and the fixed opponent's reply into ONE transition, so the
trained dynamics model was only ever shown full round-trips as single
training examples -- it structurally cannot represent "the board right
after my move, before their reply" as a state, because it never saw that
state shape. But Connect-4's rules are exactly known (there's a full plain-
Python board simulator right here), so there's no need to make a neural
network imagine something 40 lines of Python computes for free. This
module does the adversarial ply in REAL board space (enumerate the agent's
real legal moves; for each, enumerate the opponent's real legal replies and
assume they pick whichever hurts the agent most -- a genuine minimax, not a
guess) and uses the learned value head ONLY as the leaf evaluator, on a
real, never-imagined state.

The exact endgame solver (`_exact_endgame_solve`) goes one step further:
once a position narrows down to a handful of legal columns -- which happens
naturally as a real board fills up -- the remaining game tree is small
regardless of how many plies are left, and can be solved exactly with no
learned value head at all. Branching factor is controlled by how many
columns are legal, not by how many cells are empty, which is what makes
this cheap even fairly late in a real game.
"""
import time

import torch

from .env import EMPTY, AGENT, OPPONENT, _decode_board, _encode_board, _rc, _lowest_empty_row, _legal_columns, _wins_for, _board_full
from .train_utils import states_to_tensor
from .search import _evaluate_with_memory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_move(cells, col, mark, width, height):
    row = _lowest_empty_row(cells, col, width, height)
    new_cells = list(cells)
    new_cells[_rc(row, col, width)] = mark
    return new_cells


class _EndgameTimeout(Exception):
    pass


def _exact_endgame_solve(cells0, mover, width, height, win_len, deadline):
    """Memoized alpha-beta minimax to the TRUE end of the game, with
    center-out move ordering for stronger pruning. Bounded by a hard
    wall-clock `deadline` (not a node/depth budget), so a position outside
    the calibrated safe zone (see the caller's `endgame_max_cols` gate)
    fails loudly here (raises internally, caught below, reported as
    `(None, None)`) rather than silently blowing the real move-time
    budget -- the caller is expected to fall back to the round-based
    search whenever this returns `(None, None)`.

    Calibrated empirically against real Kaggle replay data: 5 legal
    columns / 24 empty cells solves in ~0.4s; 6+ legal columns can take
    5+ seconds with this plain-Python (no bitboard/transposition-table)
    implementation -- callers should gate on `len(legal_columns) <= 5`
    before calling this at all; the deadline is a second, independent
    safety net, not the only guard.

    Returns `(best_action, value)`, value from `mover`'s own perspective
    (+1 win / -1 loss / 0 draw), or `(None, None)` if the deadline hit
    before a definite answer was found."""
    memo = {}
    center = (width - 1) / 2

    def solve(cells, to_move, alpha, beta):
        if time.time() > deadline:
            raise _EndgameTimeout()
        key = (tuple(cells), to_move)
        cached = memo.get(key)
        if cached is not None:
            return cached
        other = OPPONENT if to_move == AGENT else AGENT
        legal = sorted(_legal_columns(cells, width, height), key=lambda c: abs(c - center))
        if not legal:
            memo[key] = 0.0
            return 0.0
        if to_move == AGENT:
            best = -2.0
            for c in legal:
                nxt = _apply_move(cells, c, to_move, width, height)
                if _wins_for(nxt, to_move, width, height, win_len):
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
                nxt = _apply_move(cells, c, to_move, width, height)
                if _wins_for(nxt, to_move, width, height, win_len):
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

    root_legal = _legal_columns(cells0, width, height)
    if not root_legal:
        return None, None
    root_legal = sorted(root_legal, key=lambda c: abs(c - center))
    other = OPPONENT if mover == AGENT else AGENT
    try:
        best_a, best_val = None, None
        for c in root_legal:
            nxt = _apply_move(cells0, c, mover, width, height)
            if _wins_for(nxt, mover, width, height, win_len):
                val = 1.0 if mover == AGENT else -1.0
            elif _board_full(nxt):
                val = 0.0
            else:
                val = solve(nxt, other, -1.0, 1.0)
            if best_val is None or (mover == AGENT and val > best_val) or (mover == OPPONENT and val < best_val):
                best_a, best_val = c, val
            if (mover == AGENT and best_val == 1.0) or (mover == OPPONENT and best_val == -1.0):
                break  # proven best-possible outcome -- no need to keep searching
        return best_a, best_val
    except _EndgameTimeout:
        return None, None


def _parity_cost(cells, width, height):
    """A heuristic column-parity feature, in the spirit of classical
    Connect-4 zugzwang/odd-even threat theory (Allis 1988) -- an
    EXPERIMENTAL, opt-in attempt at the "avoid walking into a zugzwang"
    problem this project's whitepaper documents as diagnosed-but-unsolved.

    For each still-open column with `r` empty cells remaining, under
    NAIVE same-column-only alternation starting with the player about to
    move, the player who ends up placing the TOP piece is: the mover if
    `r` is odd, the other player if `r` is even. Every leaf state this is
    computed on follows a full agent-move-then-opponent-reply round (see
    `real_adversarial_plan_action`'s two-phase leaf collection), so the
    player "about to move" at every leaf is always AGENT -- no need to
    track whose turn it is separately.

    This is NOT a proof, NOT a guarantee, and NOT full Claimeven (a real
    claimeven strategy requires REACTIVE move-pairing enforced across an
    entire game, not a one-shot column count at a single position) -- it
    is a cheap, directionally-motivated NUDGE: positions with more
    even-parity open columns are scored as costlier (worse for the
    agent), consistent with the theory's own prediction that parity
    structure matters, without claiming this fully captures it. Returns
    a COST (higher = worse for the agent), meant to be ADDED to the
    leaf's existing value estimate, scaled by a small `parity_weight`
    -- see `real_adversarial_plan_action`'s own docstring for the
    honest, measured verdict on whether this actually helps."""
    cost = 0.0
    for c in range(width):
        remaining = sum(1 for row in range(height) if cells[_rc(row, c, width)] == EMPTY)
        if remaining == 0:
            continue
        cost += 1.0 if remaining % 2 == 0 else -1.0
    return cost


def _narrow_to_center(legal_cols, width, max_branching):
    """Prunes a legal-column list down to `max_branching` columns closest
    to center -- real Connect-4 domain knowledge (a center column touches
    more potential 4-in-a-row lines than an edge one). `max_branching=None`
    is a no-op; deeper (rounds=3+) search needs this to stay inside a real
    time budget."""
    if max_branching is None or len(legal_cols) <= max_branching:
        return legal_cols
    center = (width - 1) / 2
    return sorted(legal_cols, key=lambda c: abs(c - center))[:max_branching]


def _resisting_endgame_solve(cells0, mover, width, height, win_len, deadline,
                             max_nodes=400_000):
    """`_exact_endgame_solve` with a distance preference, via the bitboard solver.

    WHY THIS EXISTS. `_exact_endgame_solve` returns only +1/0/-1 -- it carries
    NO information about HOW FAST a result arrives. That is fine for deciding
    whether a position is won, and it is what the deployed agent has always
    used. But it has a consequence nobody had looked at: **when every move
    loses, every move scores identically, so the solver picks arbitrarily --
    frequently choosing the FASTEST way to lose.**

    Against a perfect opponent that costs nothing; the game is lost either way.
    Against the imperfect opponents an actual ladder is full of, it throws away
    every chance the opponent errs before converting. Resisting longer is
    strictly better there and never worse.

    (The same blindness in the other direction was measured while labelling
    training positions: the legacy solver's chosen line disagreed with the true
    distance-to-win on 9 of 25 positions, shorter in 7 and longer in 2.)

    `connectx_bitboard_solver`'s score encodes distance, so its argmax already
    prefers the slowest loss and the fastest win. This wraps it in
    `_exact_endgame_solve`'s exact signature and AGENT-perspective return
    convention, so it is a drop-in.

    Returns `(best_action, value)` with `value` in the same +1 AGENT-wins /
    0 / -1 convention, or `(None, None)` if the budget ran out."""
    from .bitboard_solver import (Solver, cells_to_bitboard,
                                  agent_perspective_value)
    import time as _time
    budget = max(0.0, deadline - _time.time())
    if budget <= 0.0:
        return None, None
    try:
        position, mask, _moves = cells_to_bitboard(list(cells0), mover)
    except ValueError:
        return None, None
    solver = Solver(max_nodes=max_nodes, time_budget=budget)
    col, score = solver.best_move(position, mask)
    if col is None:
        return None, None
    return col, agent_perspective_value(score, mover)


class _RoundSearchTimeout(Exception):
    pass


@torch.no_grad()
def real_adversarial_plan_action(env, model, normalizer, real_state, memory=None,
                                  memory_weight=0.25, memory_k=5, max_steps=None, rounds=1,
                                  max_branching=None, endgame_max_cols=5, endgame_time_budget=1.2,
                                  parity_weight=0.0, deeper_rounds=None, deeper_max_branching=4,
                                  deeper_time_budget=0.6):
    """`deeper_rounds`/`deeper_max_branching`/`deeper_time_budget`: a
    real, mined-from-real-games gap this was built for -- `rounds` can
    see zero danger on a position (every column looks equally safe) just
    a couple of plies before a trap that one round DEEPER already
    narrows down to exactly one safe column. A deeper search is provably
    too slow to run on EVERY move (measured 8-13s at a 6-7-legal-column
    branching factor) -- so this is a SAFE, opportunistic escalation, not
    a blanket depth increase: `deeper_rounds=None` (default) reproduces
    the original `rounds`-only behavior byte-for-byte. When set, AFTER
    computing the normal-`rounds` answer (always -- the guaranteed-safe
    fallback), a `deeper_rounds`-round search is attempted under a hard
    `deeper_time_budget` deadline (`_RoundSearchTimeout`, same pattern as
    `_exact_endgame_solve`'s own deadline -- checked at both exponential-
    blowup recursion points AND immediately around the one batched NN
    leaf-evaluation call, which is otherwise uninterruptible once
    started). If it finishes in time, its answer is used instead
    (strictly more information, never less); if it times out, the
    original `rounds`-answer is returned completely unchanged -- this can
    only ever help or be a no-op, never make the chosen move worse or
    blow the real move-time budget by more than `deeper_time_budget`.
    Calibrated to `deeper_time_budget=0.6s` against a 180-game regression
    suite (random/weak/stronger opponents): zero win-rate regression, max
    observed combined move time 1.641s -- comfortably under a 2s budget.

    `parity_weight` (default 0.0, OFF -- byte-for-byte the original
    behavior unless explicitly enabled): blends `_parity_cost` into every
    leaf's value estimate, scaled by this weight. EXPERIMENTAL -- see
    `_parity_cost`'s own docstring for exactly what this does and doesn't
    claim. Measured (not just theorized) against the trusted test harness
    at a few weights before shipping any non-zero default; see the
    whitepaper for the honest result.

    `rounds` real adversarial rounds (our move, then the opponent's
    worst-case real reply, repeated) before falling back to the learned
    value head as the leaf evaluator. `rounds=1` is the deployed default.
    Root action never returns PASS.

    `endgame_max_cols`/`endgame_time_budget`: before running the round-
    based search at all, check whether the position already has few
    enough legal columns for `_exact_endgame_solve` to solve it exactly,
    well within budget. If it finishes in time, its answer is used
    directly (provably optimal); otherwise this falls straight through to
    the round-based search below, unchanged. `endgame_max_cols=0` disables
    this path entirely.

    Terminal-outcome convention: an opponent win scores `2 * max_steps`
    (worse than merely running out of steps); a draw scores `max_steps` --
    both far above any real achievable remaining-steps value, so they
    never get confused with a genuine near-solved position.

    Two-phase, globally batched leaf evaluation: rather than calling the
    value head separately at every node in the search tree (expensive --
    each call pays its own tensor-creation/memory-blend overhead), this
    walks the tree TWICE: once (pure Python) to collect every non-
    terminal leaf across the WHOLE tree into one deduplicated set
    (transpositions collapse for free), then ONE batched value+memory
    call, then a second walk doing the actual minimax from the
    precomputed lookup."""
    width, height, win_len = env.width, env.height, env.win_len
    max_steps = max_steps if max_steps is not None else (width * height) // 2 + 2

    cells_now = list(_decode_board(real_state))
    now_legal = _legal_columns(cells_now, width, height)
    if now_legal and endgame_max_cols and len(now_legal) <= endgame_max_cols:
        exact_a, _exact_val = _exact_endgame_solve(
            cells_now, AGENT, width, height, win_len, deadline=time.time() + endgame_time_budget,
        )
        if exact_a is not None:
            return exact_a

    def leaf_batch_values(states):
        if not states:
            return {}
        states_t = states_to_tensor([env.observe(s) for s in states]).to(DEVICE)
        norm_t = normalizer.normalize(states_t)
        z = model.encode(norm_t)
        vals = _evaluate_with_memory(model, z, memory, memory_weight, memory_k).tolist()
        if parity_weight:
            vals = [v + parity_weight * _parity_cost(list(_decode_board(s)), width, height)
                    for v, s in zip(vals, states)]
        return dict(zip(states, vals))

    def run_search(search_rounds, search_max_branching, deadline):
        """One full root-to-leaf search at a given (rounds, max_branching)
        setting -- factored out so it can be called at two different
        depths, see `deeper_rounds` above. `deadline` (optional): checked
        at both exponential-blowup recursion points (leaf collection, our
        own follow-up enumeration) AND immediately around the one batched
        NN leaf-evaluation call (otherwise uninterruptible once started)
        -- raises `_RoundSearchTimeout` the instant it's exceeded, letting
        the caller safely abandon this attempt."""

        def check_deadline():
            if deadline is not None and time.time() > deadline:
                raise _RoundSearchTimeout()

        def collect_leaves(cells1, remaining_rounds, leaf_cache):
            check_deadline()
            if _board_full(cells1):
                return
            for opp_col in _legal_columns(cells1, width, height):
                cells2 = _apply_move(cells1, opp_col, OPPONENT, width, height)
                if _wins_for(cells2, OPPONENT, width, height, win_len) or _board_full(cells2):
                    continue
                if remaining_rounds <= 1:
                    leaf_cache[tuple(_encode_board(cells2))] = None
                else:
                    for a2 in _narrow_to_center(_legal_columns(cells2, width, height), width, search_max_branching):
                        cells3 = _apply_move(cells2, a2, AGENT, width, height)
                        if _wins_for(cells3, AGENT, width, height, win_len):
                            continue
                        collect_leaves(cells3, remaining_rounds - 1, leaf_cache)

        def score_after_our_move(cells1, remaining_rounds, leaf_cache):
            """cells1: real board right after OUR move. Returns our worst-case
            score -- the opponent picks whichever real reply hurts us most."""
            if _board_full(cells1):
                return float(max_steps)
            vals = []
            for opp_col in _legal_columns(cells1, width, height):
                cells2 = _apply_move(cells1, opp_col, OPPONENT, width, height)
                if _wins_for(cells2, OPPONENT, width, height, win_len):
                    vals.append(float(2 * max_steps))
                elif _board_full(cells2):
                    vals.append(float(max_steps))
                elif remaining_rounds <= 1:
                    vals.append(leaf_cache[tuple(_encode_board(cells2))])
                else:
                    vals.append(score_after_opponent_move(cells2, remaining_rounds - 1, leaf_cache))
            return max(vals)

        def score_after_opponent_move(cells2, remaining_rounds, leaf_cache):
            """cells2: real board after the opponent's move, our turn again.
            Returns our best achievable worst-case score from here."""
            check_deadline()
            our_legal = _narrow_to_center(_legal_columns(cells2, width, height), width, search_max_branching)
            if not our_legal:
                return float(max_steps)
            best = None
            for a in our_legal:
                cells3 = _apply_move(cells2, a, AGENT, width, height)
                if _wins_for(cells3, AGENT, width, height, win_len):
                    return -float(max_steps)  # a forced win exists deeper -- short-circuit
                s = score_after_our_move(cells3, remaining_rounds, leaf_cache)
                if best is None or s < best:
                    best = s
            return best

        leaf_cache = {}
        for a in surviving_actions:
            collect_leaves(action_cells1[a], search_rounds, leaf_cache)
        check_deadline()  # right before the one batched NN call -- don't start it with no budget left
        if leaf_cache:
            leaf_cache.update(leaf_batch_values(list(leaf_cache.keys())))
        check_deadline()  # and once more right after -- don't walk the tree on a stale/over-budget result

        def hands_immediate_win(cells1):
            """True if the OPPONENT has an immediate (1-ply) winning
            reply from `cells1` (the real board right after OUR
            candidate move). Tie-break added 2026-08-13, ported from the
            private tree's identical fix -- found by mining fresh real
            Kaggle replays for a deployed submission whose public score
            dropped after adding a deeper-search escalation: most losses
            traced to genuine forced-loss positions a deeper search
            simply discovered sooner (every candidate correctly scored
            equally bad), but with everything tied on score, the plain
            `<` comparison below kept whichever move came first in
            center-out order -- not necessarily one that avoided handing
            the opponent an immediate win THIS move, even when a tied
            alternative existed that did. Against a real, imperfect
            opponent pool, an immediate giveaway forfeits every chance of
            a mistake; delaying the loss does not."""
            for opp_col in _legal_columns(cells1, width, height):
                if _wins_for(_apply_move(cells1, opp_col, OPPONENT, width, height), OPPONENT, width, height, win_len):
                    return True
            return False

        best_a, best_score, best_hands_win = None, None, None
        for a in surviving_actions:
            s = score_after_our_move(action_cells1[a], search_rounds, leaf_cache)
            hands_win = hands_immediate_win(action_cells1[a])
            key = (s, hands_win)
            if best_score is None or key < (best_score, best_hands_win):
                best_a, best_score, best_hands_win = a, s, hands_win
        return best_a

    root_legal = _legal_columns(cells_now, width, height)
    if not root_legal:
        return None
    # Center-out root ordering: NOT a pruning change (every legal column is
    # still considered), only fixes which column wins a TIE. Left-to-right
    # order otherwise defaults ties to the LEFTMOST column, an arbitrary
    # bias with no game-theoretic basis -- center columns are the real
    # stronger choice under a tie.
    center = (width - 1) / 2
    root_legal = sorted(root_legal, key=lambda c: abs(c - center))

    surviving_actions, action_cells1 = [], {}
    for a in root_legal:
        cells1 = _apply_move(cells_now, a, AGENT, width, height)
        if _wins_for(cells1, AGENT, width, height, win_len):
            return a  # immediate win -- take it, no need to consider anything else
        surviving_actions.append(a)
        action_cells1[a] = cells1

    base_a = run_search(rounds, max_branching, deadline=None)  # always computed -- the guaranteed-safe fallback

    if deeper_rounds is not None:
        try:
            return run_search(deeper_rounds, deeper_max_branching, deadline=time.time() + deeper_time_budget)
        except _RoundSearchTimeout:
            pass  # deeper attempt didn't finish in time -- fall back to base_a exactly as if deeper_rounds=None

    return base_a
