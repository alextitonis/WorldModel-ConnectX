"""
An independent REFEREE for Connect-4 position values, written to settle
a disagreement rather than to be fast.

Context: `connectx_bitboard_solver` and the deployed
`connectx_adversarial_search._exact_endgame_solve` disagreed on the
game-theoretic value of 3 of 40 randomly-reached endgame positions.
Two solvers disagreeing tells you one is wrong; it does not tell you
which. Neither can adjudicate itself, and the older one being deployed
is not evidence -- it is only evidence that nothing has caught it yet.

So this is a third implementation with NO shared machinery and NO
optimisation that could be unsound:

* plain minimax, **no alpha-beta at all** -- so no value can ever be a
  bound mistaken for an exact score,
* memoisation keyed on the full `(cells, to_move)` position and storing
  only values computed under a full search of every child, which is
  safe precisely because there is no window to be relative to,
* win/legality via `connectx_env`'s own cell-scan helpers, not
  bitboards -- so a bit-layout error in the new solver cannot be
  reproduced here.

It returns AGENT-perspective values (+1 AGENT wins / 0 draw / -1 AGENT
loses), matching `_exact_endgame_solve`'s convention.

Run `python -m connectx.solver_referee` to reproduce the
adjudication.
"""
import random
import sys
import time

from .bitboard_solver import agent_perspective_value, solve_cells
from .adversarial_search import _exact_endgame_solve
from .env import (AGENT, OPPONENT, _board_full, _legal_columns,
                  _lowest_empty_row, _rc, _wins_for)

WIDTH, HEIGHT, WIN_LEN = 7, 6, 4


def _drop(cells, col, mark):
    row = _lowest_empty_row(cells, col, WIDTH, HEIGHT)
    out = list(cells)
    out[_rc(row, col, WIDTH)] = mark
    return out


def referee_value(cells, to_move, memo=None):
    """Exact AGENT-perspective value by exhaustive minimax, no pruning."""
    if memo is None:
        memo = {}
    key = (tuple(cells), to_move)
    if key in memo:
        return memo[key]
    legal = _legal_columns(cells, WIDTH, HEIGHT)
    if not legal:
        memo[key] = 0.0
        return 0.0
    other = OPPONENT if to_move == AGENT else AGENT
    vals = []
    for col in legal:
        nxt = _drop(cells, col, to_move)
        if _wins_for(nxt, to_move, WIDTH, HEIGHT, WIN_LEN):
            vals.append(1.0 if to_move == AGENT else -1.0)
        elif _board_full(nxt):
            vals.append(0.0)
        else:
            vals.append(referee_value(nxt, other, memo))
    val = max(vals) if to_move == AGENT else min(vals)
    memo[key] = val
    return val


def random_position(rng, n_plies, first=AGENT):
    cells = [0] * (WIDTH * HEIGHT)
    mover = first
    for _ in range(n_plies):
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal:
            break
        cells = _drop(cells, rng.choice(legal), mover)
        if _wins_for(cells, mover, WIDTH, HEIGHT, WIN_LEN):
            return None, None
        mover = OPPONENT if mover == AGENT else AGENT
    return cells, mover


def render(cells):
    sym = {0: ".", AGENT: "X", OPPONENT: "O"}
    return "\n".join("  " + " ".join(sym[cells[r * WIDTH + c]] for c in range(WIDTH))
                     for r in range(HEIGHT))


def main(n_target=40, seed=7):
    """Replays the exact comparison the self-test runs (same seed and
    sampling), and adjudicates every disagreement with the referee."""
    rng = random.Random(seed)
    compared = 0
    attempts = 0
    disagreements = []
    old_wrong = 0
    new_wrong = 0
    both_wrong = 0

    while compared < n_target and attempts < 4000:
        attempts += 1
        cells, mover = random_position(rng, rng.randint(24, 36))
        if cells is None:
            continue
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal or len(legal) > 5:
            continue
        if _wins_for(cells, AGENT, WIDTH, HEIGHT, WIN_LEN) or \
           _wins_for(cells, OPPONENT, WIDTH, HEIGHT, WIN_LEN):
            continue

        old_col, old_val = _exact_endgame_solve(
            list(cells), mover, WIDTH, HEIGHT, WIN_LEN, deadline=time.time() + 20.0)
        if old_col is None:
            continue
        new_col, new_score = solve_cells(cells, mover, max_nodes=3_000_000)
        if new_col is None:
            continue

        compared += 1
        new_val = agent_perspective_value(new_score, mover)
        if new_val == old_val:
            continue

        truth = referee_value(cells, mover)
        o_ok = (old_val == truth)
        n_ok = (new_val == truth)
        if o_ok and not n_ok:
            new_wrong += 1
        elif n_ok and not o_ok:
            old_wrong += 1
        else:
            both_wrong += 1
        disagreements.append((cells, mover, old_val, new_val, truth, o_ok, n_ok))

    print(f"compared {compared} positions ({attempts} draws)")
    print(f"disagreements: {len(disagreements)}")
    print(f"  referee says OLD (_exact_endgame_solve) was wrong: {old_wrong}")
    print(f"  referee says NEW (bitboard solver) was wrong:      {new_wrong}")
    print(f"  referee agrees with neither:                        {both_wrong}")

    for i, (cells, mover, ov, nv, truth, o_ok, n_ok) in enumerate(disagreements, 1):
        print(f"\n--- disagreement {i} --- to move: "
              f"{'AGENT(X)' if mover == AGENT else 'OPPONENT(O)'}"
              f"   legal cols {_legal_columns(cells, WIDTH, HEIGHT)}")
        print(render(cells))
        print(f"  old={ov:+.0f} ({'ok' if o_ok else 'WRONG'})   "
              f"new={nv:+.0f} ({'ok' if n_ok else 'WRONG'})   referee={truth:+.0f}")

    return 0 if new_wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
