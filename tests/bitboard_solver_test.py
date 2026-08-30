"""
Self-test for `connectx_bitboard_solver`.

The load-bearing test here is CROSS-VALIDATION: the new bitboard solver
is checked move-for-move and value-for-value against the existing,
already-trusted `connectx_adversarial_search._exact_endgame_solve` on
real, randomly-reached positions. That solver is slow and narrow, but
it has been in the deployed submission since 2026-08-11 and its answers
are exact within its reach -- so disagreement means the NEW code is
wrong, and agreement over many independent positions is the only
evidence worth having that the bit-twiddling is right.

Everything else is a positive control in this project's usual sense: a
test that watches the detector FIRE on an input whose answer is known
independently, plus null controls that must come back clean.

Run: python -m tests.bitboard_solver_test
"""
import random
import sys
import time

from connectx.bitboard_solver import (
    HEIGHT, WIDTH, PerfectOpponent, Solver, agent_perspective_value,
    alignment, bitboard_to_cells, cells_to_bitboard, solve_cells)
from connectx.adversarial_search import _exact_endgame_solve
from connectx.env import (AGENT, OPPONENT, _legal_columns,
                          _lowest_empty_row, _rc, _wins_for)

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def empty_cells():
    return [0] * (WIDTH * HEIGHT)


def drop(cells, col, mark):
    row = _lowest_empty_row(cells, col, WIDTH, HEIGHT)
    assert row is not None, f"column {col} is full"
    out = list(cells)
    out[_rc(row, col, WIDTH)] = mark
    return out


def random_position(rng, n_plies, first=AGENT):
    """Play `n_plies` uniformly random legal moves, stopping early if
    somebody wins. Returns `(cells, to_move)`."""
    cells = empty_cells()
    mover = first
    for _ in range(n_plies):
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal:
            break
        cells = drop(cells, rng.choice(legal), mover)
        if _wins_for(cells, mover, WIDTH, HEIGHT, 4):
            return None, None            # terminal; caller resamples
        mover = OPPONENT if mover == AGENT else AGENT
    return cells, mover


# --------------------------------------------------------------------

def test_roundtrip():
    """Conversion between dialects must be lossless. A silent row-flip
    here would make every later result confidently wrong, so this runs
    first."""
    rng = random.Random(0)
    bad = 0
    for _ in range(300):
        cells, mover = random_position(rng, rng.randint(0, 30))
        if cells is None:
            continue
        pos, mask, moves = cells_to_bitboard(cells, mover)
        back = bitboard_to_cells(pos, mask, mover)
        if back != cells:
            bad += 1
        if moves != sum(1 for v in cells if v):
            bad += 1
    check("dialect round-trip is lossless", bad == 0, f"{bad} mismatches")


def test_gravity_guard():
    """A floating stone is impossible in a real game; accepting one
    would let the solver answer a position that cannot occur."""
    cells = empty_cells()
    cells[_rc(0, 3, WIDTH)] = AGENT          # top row, nothing beneath
    try:
        cells_to_bitboard(cells, AGENT)
        check("gravity guard rejects a floating stone", False, "no error raised")
    except ValueError:
        check("gravity guard rejects a floating stone", True)


def test_alignment_agrees_with_scan():
    """`alignment()` (4 shift-masks) must agree with the project's own
    full-board cell scan on every random position -- including the many
    where the answer is False, so this is not just a one-sided check."""
    rng = random.Random(1)
    bad = 0
    checked = 0
    hits = 0
    for _ in range(400):
        cells = empty_cells()
        mover = AGENT
        for _ in range(rng.randint(0, 40)):
            legal = _legal_columns(cells, WIDTH, HEIGHT)
            if not legal:
                break
            cells = drop(cells, rng.choice(legal), mover)
            for m in (AGENT, OPPONENT):
                pos, mask, _ = cells_to_bitboard(cells, m)
                got = alignment(pos)
                want = _wins_for(cells, m, WIDTH, HEIGHT, 4)
                checked += 1
                hits += bool(want)
                if got != want:
                    bad += 1
            if _wins_for(cells, mover, WIDTH, HEIGHT, 4):
                break
            mover = OPPONENT if mover == AGENT else AGENT
    check("alignment() matches the cell scan", bad == 0,
          f"{checked} positions, {hits} with a real four, {bad} mismatches")
    # Positive control: the comparison is worthless if no position ever
    # contained a four.
    check("  ...and the comparison actually saw fours", hits > 50,
          f"{hits} winning positions seen")


def test_immediate_win():
    """A position whose right answer is known without any solver."""
    # Three of ours in a row on the bottom, column 4 completes it.
    cells = empty_cells()
    for c in (1, 2, 3):
        cells = drop(cells, c, AGENT)
    for c in (0, 0, 5):
        cells = drop(cells, c, OPPONENT)
    col, score = solve_cells(cells, AGENT, max_nodes=200_000)
    check("takes an available immediate win", col == 4 and score is not None and score > 0,
          f"chose {col}, score {score}")


def test_never_hands_over_an_avoidable_win():
    """A far stronger property than any single blocking fixture: over
    many random positions, whenever at least one move exists that does
    NOT let the opponent win on the very next ply, the solver must
    choose one of them.

    Stated as a property rather than a fixture because the first
    version of this test used a hand-built 'must block column 4'
    position that was ALREADY a forced loss -- so every move scored
    equally and the solver's centre-column choice was correct while the
    test called it a failure. A fixture can only be as good as the
    tester's own reading of the position; this cannot be wrong that
    way."""
    rng = random.Random(23)
    tested = 0
    violations = 0
    had_choice = 0
    for _ in range(400):
        cells, mover = random_position(rng, rng.randint(6, 30))
        if cells is None:
            continue
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal:
            continue
        other = OPPONENT if mover == AGENT else AGENT
        # Skip positions where we can simply win now -- a different case.
        if any(_wins_for(drop(cells, c, mover), mover, WIDTH, HEIGHT, 4) for c in legal):
            continue
        safe = []
        for c in legal:
            nxt = drop(cells, c, mover)
            opp_legal = _legal_columns(nxt, WIDTH, HEIGHT)
            if not any(_wins_for(drop(nxt, c2, other), other, WIDTH, HEIGHT, 4)
                       for c2 in opp_legal):
                safe.append(c)
        if not safe:
            continue                    # every move loses; nothing to test
        had_choice += 1
        col, _ = solve_cells(cells, mover, max_nodes=300_000, time_budget=2.0)
        if col is None:
            continue
        tested += 1
        if col not in safe:
            violations += 1
    check("never hands the opponent an avoidable immediate win",
          violations == 0,
          f"{tested} positions with a safe move available, {violations} violations")
    # The bar is CALIBRATED to what this budget actually reaches, not
    # derived from anything: at 300k nodes / 2s roughly a third of these
    # positions are proved, because the sample deliberately includes
    # early positions, which are the expensive ones. Unsolved positions
    # report None and are excluded rather than silently counted as
    # passes -- so a budget miss can never inflate this number.
    check("  ...and the property was actually exercised", tested >= 50,
          f"{had_choice} positions had a choice, {tested} solved in budget, "
          f"{had_choice - tested} exceeded it (reported None, excluded)")


def _build_drawn_full_board():
    """Fill the board by backtracking so that neither side ever has a
    four. Generated rather than hand-written, because the hand-written
    pattern used first silently contained a diagonal four and the test
    was checking a fixture that was not a draw at all."""
    cells = empty_cells()
    counts = {AGENT: 0, OPPONENT: 0}
    order = [(r, c) for r in range(HEIGHT - 1, -1, -1) for c in range(WIDTH)]

    def ok(idx):
        return not (_wins_for(cells, AGENT, WIDTH, HEIGHT, 4)
                    or _wins_for(cells, OPPONENT, WIDTH, HEIGHT, 4))

    def rec(i):
        if i == len(order):
            return True
        r, c = order[i]
        # Try the currently-scarcer mark first, to keep the fill legal
        # as a real game (21 stones each).
        for mark in sorted((AGENT, OPPONENT), key=lambda m: counts[m]):
            if counts[mark] >= (WIDTH * HEIGHT) // 2:
                continue
            cells[_rc(r, c, WIDTH)] = mark
            counts[mark] += 1
            if ok(i) and rec(i + 1):
                return True
            counts[mark] -= 1
            cells[_rc(r, c, WIDTH)] = 0
        return False

    return cells if rec(0) else None


def test_full_board_is_a_draw():
    """Null control: a filled board with no four must score exactly 0
    and must not crash."""
    cells = _build_drawn_full_board()
    if cells is None:
        check("could construct a drawn full board", False, "backtracking failed")
        return
    no_four = not (_wins_for(cells, AGENT, WIDTH, HEIGHT, 4)
                   or _wins_for(cells, OPPONENT, WIDTH, HEIGHT, 4))
    full = all(v != 0 for v in cells)
    check("constructed a genuinely full, four-free board", no_four and full)
    pos, mask, _ = cells_to_bitboard(cells, AGENT)
    s = Solver(max_nodes=100_000)
    check("a full drawn board scores exactly 0", s.solve_position(pos, mask) == 0)


def test_budget_returns_none_not_a_guess():
    """A budget miss must be reported as None. If it ever returned a
    partial score instead, every downstream 'proven' claim would be
    unfounded -- this is the single most important failure convention
    in the module."""
    cells = empty_cells()          # the expensive case, by construction
    s = Solver(max_nodes=500)
    pos, mask, _ = cells_to_bitboard(cells, AGENT)
    check("tiny node budget on an empty board returns None",
          s.solve_position(pos, mask) is None)
    col, score = solve_cells(cells, AGENT, max_nodes=500)
    check("  ...and best_move reports (None, None) too",
          col is None and score is None)


def test_cross_validate_against_exact_endgame():
    """THE load-bearing test.

    On positions the old solver can actually reach (<= 5 legal columns,
    its own documented gate), the two must agree on the game's VALUE.
    Move choice is deliberately NOT required to match -- several moves
    can share the same optimal value, and the old solver stops at the
    first proven-best root option, so a move mismatch is not evidence of
    a bug. Values are compared in the old solver's AGENT-perspective
    convention via `agent_perspective_value`."""
    rng = random.Random(7)
    compared = 0
    agree = 0
    disagree = []
    decisive = 0
    old_total = 0.0
    new_total = 0.0

    attempts = 0
    while compared < 40 and attempts < 4000:
        attempts += 1
        cells, mover = random_position(rng, rng.randint(24, 36))
        if cells is None:
            continue
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal or len(legal) > 5:
            continue
        if _wins_for(cells, AGENT, WIDTH, HEIGHT, 4) or _wins_for(cells, OPPONENT, WIDTH, HEIGHT, 4):
            continue

        t0 = time.time()
        old_col, old_val = _exact_endgame_solve(
            list(cells), mover, WIDTH, HEIGHT, 4, deadline=time.time() + 20.0)
        old_total += time.time() - t0
        if old_col is None:
            continue

        t0 = time.time()
        new_col, new_score = solve_cells(cells, mover, max_nodes=3_000_000)
        new_total += time.time() - t0
        if new_col is None:
            continue

        compared += 1
        new_val = agent_perspective_value(new_score, mover)
        if new_val == old_val:
            agree += 1
        else:
            disagree.append((old_val, new_val, old_col, new_col))
        if old_val != 0.0:
            decisive += 1

    check("cross-validation actually ran on enough positions", compared >= 25,
          f"{compared} comparable positions from {attempts} draws")
    # A suite of only-draws would agree trivially and prove nothing.
    check("  ...and included decisive (non-draw) positions", decisive >= 5,
          f"{decisive} of {compared} were wins/losses")
    check("bitboard solver agrees with _exact_endgame_solve on VALUE",
          not disagree,
          f"{agree}/{compared} agree" + (f"; first mismatch {disagree[0]}" if disagree else ""))
    if compared:
        print(f"        timing on the same positions: old {old_total:.2f}s "
              f"vs new {new_total:.2f}s  ({old_total / max(new_total, 1e-9):.1f}x)")


def test_reaches_past_the_old_solver():
    """The whole justification for this module is reach. Take positions
    with SIX+ legal columns -- outside the old solver's documented gate
    -- and confirm the new one actually resolves a useful share of them
    within a budget the old one is known to miss."""
    rng = random.Random(11)
    solved = 0
    tried = 0
    t0 = time.time()
    while tried < 20 and time.time() - t0 < 90:
        cells, mover = random_position(rng, rng.randint(14, 20))
        if cells is None:
            continue
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if len(legal) < 6:
            continue
        if _wins_for(cells, AGENT, WIDTH, HEIGHT, 4) or _wins_for(cells, OPPONENT, WIDTH, HEIGHT, 4):
            continue
        tried += 1
        col, score = solve_cells(cells, mover, max_nodes=1_500_000, time_budget=4.0)
        if col is not None:
            solved += 1
    check("resolves positions beyond the old solver's <=5-column gate",
          tried > 0 and solved > 0,
          f"{solved}/{tried} midgame positions proved within 1.5M nodes / 4s")


def test_perfect_opponent_reports_honest_coverage():
    """`PerfectOpponent` must never claim proof it does not have."""
    def fallback(board, mark, rng):
        return _legal_columns(board, WIDTH, HEIGHT)[0]

    # A budget so small nothing can be proved -> coverage must be 0.0.
    # The positions must be real (alternating play) and must have no
    # immediate win available: `best_move`'s immediate-win shortcut is a
    # legitimate proof that costs no nodes, so counting it would make
    # this test fail for a correct reason. The first version of this
    # test fed the opponent a board of AGENT-only stones, which grew a
    # vertical four and was "proved" for free.
    starved = PerfectOpponent(fallback, min_moves=0, max_nodes=1, time_budget=0.001)
    rng = random.Random(3)
    n = 0
    while n < 6:
        cells, mover = random_position(rng, rng.randint(8, 20))
        if cells is None:
            continue
        legal = _legal_columns(cells, WIDTH, HEIGHT)
        if not legal:
            continue
        if any(_wins_for(drop(cells, c, mover), mover, WIDTH, HEIGHT, 4) for c in legal):
            continue
        starved(cells, mover, rng)
        n += 1
    check("starved PerfectOpponent reports 0.0 coverage, not silent fallback",
          starved.total == 6 and starved.coverage() == 0.0,
          f"proved {starved.proved}/{starved.total}")

    # With a real budget on a late position it should actually prove.
    rich = PerfectOpponent(fallback, min_moves=0, max_nodes=500_000, time_budget=5.0)
    cells, mover = None, None
    r2 = random.Random(5)
    while cells is None:
        cells, mover = random_position(r2, 30)
    col = rich(cells, mover, r2)
    check("  ...and proves a real move when given a real budget",
          rich.proved == 1 and col in _legal_columns(cells, WIDTH, HEIGHT),
          f"proved {rich.proved}/{rich.total}, chose {col}")


def main():
    print("connectx_bitboard_solver self-test\n")
    for fn in (test_roundtrip,
               test_gravity_guard,
               test_alignment_agrees_with_scan,
               test_immediate_win,
               test_never_hands_over_an_avoidable_win,
               test_full_board_is_a_draw,
               test_budget_returns_none_not_a_guess,
               test_perfect_opponent_reports_honest_coverage,
               test_cross_validate_against_exact_endgame,
               test_reaches_past_the_old_solver):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
