"""
Does resisting longer in a LOST endgame actually convert losses into wins?

The claim under test: `_exact_endgame_solve` returns only +1/0/-1, so when every
move loses they all score the same and it picks arbitrarily -- often the
fastest loss. `_resisting_endgame_solve` prefers the slowest loss. Against a
perfect opponent that is worth exactly nothing. Against an imperfect one it
should be worth something, because a longer game is more chances for the
opponent to err.

That is a plausible story, which is precisely why it needs measuring rather
than asserting. The experiment:

* Sample real positions that are PROVEN LOST for us (verified with the bitboard
  solver, so "lost" is a fact and not an estimate).
* Play each one out twice from the identical position against the identical
  opponent with the identical seed -- once choosing our moves with the legacy
  solver, once with the resisting one.
* Count escapes: games we do NOT lose.

Controls that decide whether the number means anything:

1. **A perfect-opponent arm.** Against an opponent that never errs, both arms
   must escape 0% -- the positions are proven lost. If the resisting arm
   "escapes" there, the harness is broken, not the idea.
2. **Game length is reported**, since the mechanism is "survive longer". If
   resisting does not actually produce longer games, any escape difference is
   something else and the stated mechanism is wrong.
3. **Paired seeds.** The opponent's randomness is identical across arms, so a
   difference cannot come from one arm drawing easier opponents.

Run: python -m connectx.resisting_endgame_eval
"""
import argparse
import random
import time

from .adversarial_search import (_apply_move, _exact_endgame_solve,
                                 _resisting_endgame_solve)
from .bitboard_solver import Solver, cells_to_bitboard
from .env import (AGENT, OPPONENT, _board_full, _legal_columns,
                  _lowest_empty_row, _rc, _wins_for)
from tests.submission_test import (stronger_heuristic_opponent,
                                   weak_heuristic_opponent)

WIDTH, HEIGHT, WIN_LEN = 7, 6, 4


def drop(cells, col, mark):
    row = _lowest_empty_row(cells, col, WIDTH, HEIGHT)
    if row is None:
        return None
    out = list(cells)
    out[_rc(row, col, WIDTH)] = mark
    return out


def random_position(rng, n_plies):
    cells = [0] * (WIDTH * HEIGHT)
    mover = AGENT
    for _ in range(n_plies):
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal:
            break
        cells = drop(cells, rng.choice(legal), mover)
        if _wins_for(cells, mover, WIDTH, HEIGHT, WIN_LEN):
            return None
        mover = OPPONENT if mover == AGENT else AGENT
    return cells


def proven_lost_positions(rng, n_target, min_cols=4, max_cols=6, nodes=1_500_000):
    """AGENT-to-move positions the bitboard solver PROVES are losses."""
    out = []
    attempts = 0
    while len(out) < n_target and attempts < 40_000:
        attempts += 1
        cells = random_position(rng, rng.randint(20, 32))
        if cells is None:
            continue
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not (min_cols <= len(legal) <= max_cols):
            continue
        if _wins_for(cells, AGENT, WIDTH, HEIGHT, WIN_LEN) or \
           _wins_for(cells, OPPONENT, WIDTH, HEIGHT, WIN_LEN):
            continue
        s = Solver(max_nodes=nodes)
        pos, mask, _ = cells_to_bitboard(cells, AGENT)
        col, score = s.best_move(pos, mask)
        if col is None or score is None:
            continue
        if score < 0:                      # proven loss for AGENT
            out.append(cells)
    return out


def perfect_opponent_move(cells, rng, nodes=400_000):
    s = Solver(max_nodes=nodes)
    pos, mask, _ = cells_to_bitboard(cells, OPPONENT)
    col, _score = s.best_move(pos, mask)
    if col is None:
        return rng.choice(_legal_columns(cells, WIDTH, HEIGHT))
    return col


def play_out(cells0, our_solver, opponent, rng, budget_s=1.0):
    """Returns (outcome, plies) with outcome in {"loss", "escape"}.

    "escape" is any non-loss -- a win or a draw -- because from a proven-lost
    position both are the opponent failing to convert, which is exactly what
    the mechanism claims to buy."""
    cur = list(cells0)
    plies = 0
    for _ in range(WIDTH * HEIGHT + 1):
        legal = _legal_columns(cur, WIDTH, HEIGHT)
        if not legal:
            return "escape", plies                      # draw
        a, _v = our_solver(cur, AGENT, WIDTH, HEIGHT, WIN_LEN,
                           deadline=time.time() + budget_s)
        if a is None or a not in legal:
            a = rng.choice(legal)
        cur = _apply_move(cur, a, AGENT, WIDTH, HEIGHT)
        plies += 1
        if _wins_for(cur, AGENT, WIDTH, HEIGHT, WIN_LEN):
            return "escape", plies
        if _board_full(cur):
            return "escape", plies

        legal = _legal_columns(cur, WIDTH, HEIGHT)
        if not legal:
            return "escape", plies
        oc = opponent(cur, rng)
        if oc not in legal:
            oc = rng.choice(legal)
        cur = _apply_move(cur, oc, OPPONENT, WIDTH, HEIGHT)
        plies += 1
        if _wins_for(cur, OPPONENT, WIDTH, HEIGHT, WIN_LEN):
            return "loss", plies
        if _board_full(cur):
            return "escape", plies
    return "escape", plies


def run(positions, opponent_name, opponent, seed=0, budget_s=1.0):
    arms = {"legacy (value-only)": _exact_endgame_solve,
            "resisting (distance-aware)": _resisting_endgame_solve}
    results = {}
    for name, solver in arms.items():
        escapes = 0
        total_plies = 0
        for i, cells in enumerate(positions):
            rng = random.Random(seed * 7919 + i)        # paired across arms
            outcome, plies = play_out(cells, solver, opponent, rng, budget_s)
            escapes += (outcome == "escape")
            total_plies += plies
        results[name] = {
            "escape_rate": escapes / len(positions),
            "escapes": escapes,
            "mean_plies": total_plies / len(positions),
        }
    print(f"\nvs {opponent_name}   ({len(positions)} proven-lost positions)")
    print(f"  {'arm':<28}{'escaped':>10}{'rate':>9}{'mean plies':>13}")
    for name, r in results.items():
        print(f"  {name:<28}{r['escapes']:>4}/{len(positions):<5}"
              f"{r['escape_rate']:>9.3f}{r['mean_plies']:>13.1f}")
    a = results["legacy (value-only)"]
    b = results["resisting (distance-aware)"]
    print(f"  delta: escape {b['escape_rate'] - a['escape_rate']:+.3f}   "
          f"game length {b['mean_plies'] - a['mean_plies']:+.1f} plies")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-positions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"sampling {args.n_positions} PROVEN-LOST positions...", flush=True)
    positions = proven_lost_positions(rng, args.n_positions)
    print(f"  got {len(positions)}")
    if not positions:
        print("no positions sampled")
        return

    # CONTROL FIRST. Against perfect play the positions are proven lost, so
    # both arms must score exactly 0. Anything else means the harness is
    # measuring something other than what it claims.
    print("\n=== CONTROL: perfect opponent (both arms MUST escape 0.000) ===")
    ctrl = run(positions[:min(12, len(positions))], "perfect play",
               lambda c, r: perfect_opponent_move(c, r), args.seed, args.budget)
    ok = all(v["escape_rate"] == 0.0 for v in ctrl.values())
    print("  [PASS] control clean" if ok else
          "  [FAIL] a proven-lost position escaped perfect play -- harness bug")

    print("\n=== the real question: imperfect opponents ===")
    run(positions, "weak heuristic",
        lambda c, r: weak_heuristic_opponent(c, OPPONENT, r), args.seed, args.budget)
    run(positions, "stronger heuristic",
        lambda c, r: stronger_heuristic_opponent(c, OPPONENT, r), args.seed, args.budget)


if __name__ == "__main__":
    main()
