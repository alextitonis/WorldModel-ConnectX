"""
An exact, strong Connect-4 solver on bitboards with a transposition
table -- the piece named as the single blocker for this domain on
2026-08-30 ("ConnectX's evaluation is SATURATED ... a real bitboard +
transposition-table Connect-4 solver is the single missing piece").

WHY THIS EXISTS, precisely. Two separate threads converged on the same
gap, and neither can be advanced without it:

1. **The evaluation is saturated.** Every opponent this project can
   currently field is beaten by the deployed agent at or near ceiling
   (baseline 51-60/60; the 2026-08-30 held-out suite returned 1.000 on
   five of six opponents, including its strongest). A one-game
   difference was the entire signal the ensemble experiment could
   produce. No mechanism comparison can be validated against a ceiling,
   so a *discriminating* opponent is needed -- and the 2026-08-30
   held-out result narrowed what kind: the exposure is to STRONGER
   opponents, not different ones.
2. **The zugzwang thread is stalled.** The one real, diagnosed weakness
   on record (episode-91888742) was found to be a forced loss from at
   least step=13 -- 30 empty cells, still the midgame. The existing
   `connectx_adversarial_search._exact_endgame_solve` could not resolve
   step=11 even at a 900s budget; it is plain Python with a
   tuple-keyed memo and its own docstring gates callers at <= 5 legal
   columns. That is its practical reach limit, not a found boundary.

WHAT IS DIFFERENT HERE (mechanism, not a claim to take on faith):

* **Bitboards.** A position is two Python ints over a 7-bits-per-column
  layout (6 playable rows + 1 sentinel row that stops shifts from
  wrapping between columns). Win detection is four shift-and-mask
  operations instead of a full-board scan over every cell and
  direction.
* **A unique, cheap key.** `position + mask` is a bijective encoding of
  a Connect-4 position (Pons' observation: the carry from the addition
  marks each column's height unambiguously). The existing solver keys
  its memo on `(tuple(cells), to_move)` -- a 42-element tuple hashed at
  every node.
* **A real transposition table with bounds.** Entries store a LOWER or
  UPPER bound, not just a value, so a cutoff found under one window
  still prunes under another. The existing memo caches only values
  produced under whatever `(alpha, beta)` happened to be live, which is
  unsound to reuse across different windows in general -- which is why
  that solver builds a fresh `memo` on every call.
* **Losing-move pruning.** Moves handing the opponent an immediate win
  are removed before search, and a position with two distinct forced
  replies is resolved with no recursion at all.
* **Move ordering by threats created**, not merely centre-out.

CONVENTIONS. This module speaks two dialects and converts between them
explicitly, because getting that wrong is the obvious way to ship a
confidently-wrong solver:

* *Project dialect* -- a flat `cells` list of `WIDTH*HEIGHT` entries,
  `EMPTY/AGENT/OPPONENT = 0/1/2`, index `row*WIDTH + col`, **row 0 is
  the TOP row** and `row = HEIGHT-1` is the bottom (see
  `connectx_env._lowest_empty_row`). Kaggle's `observation.board` uses
  the same layout.
* *Solver dialect* -- `(position, mask)`, bit `col*(HEIGHT+1) + h`
  where `h` counts UP from the bottom; `position` holds the stones of
  the side to move.

SCORE CONVENTION (Pons'). A score is from the side to move's own
perspective and encodes *how fast*: a win leaving `n` of your stones on
the board scores `(WIDTH*HEIGHT + 1)//2 - n`, so winning sooner scores
higher; a loss is the negation; 0 is a draw. Only the SIGN is a claim
about the game's value -- the magnitude is a speed preference. This
deliberately does NOT match `_exact_endgame_solve`'s +1/0/-1
AGENT-perspective convention; `agent_perspective_value()` converts, and
the self-test cross-validates the two solvers on real positions rather
than assuming they agree.

HONESTY ABOUT REACH. This is still Python. It does not solve the empty
7x6 board in reasonable time and nothing here depends on it doing so.
Every entry point takes an explicit node and wall-clock budget and
returns `None` -- an honest "did not finish" -- rather than a guess,
the same failure convention `_exact_endgame_solve` already uses.
`PerfectOpponent` is built on that: it plays exactly when it can prove
a move and falls back to a named heuristic otherwise, and it REPORTS
the fraction of moves it actually proved, so an eval can never quietly
present a heuristic as perfect play.
"""
import time

WIDTH, HEIGHT, WIN_LEN = 7, 6, 4

# One sentinel bit above each column keeps a shifted pattern from
# wrapping out of its own column into the next one.
_H1 = HEIGHT + 1
_H2 = HEIGHT + 2


def _column_mask(col):
    return ((1 << HEIGHT) - 1) << (col * _H1)


def _bottom_mask_col(col):
    return 1 << (col * _H1)


BOARD_MASK = 0
BOTTOM_MASK = 0
for _c in range(WIDTH):
    BOARD_MASK |= _column_mask(_c)
    BOTTOM_MASK |= _bottom_mask_col(_c)

# Centre-out column order, used as the tie-break inside move ordering.
COLUMN_ORDER = sorted(range(WIDTH), key=lambda c: (abs(c - (WIDTH - 1) / 2), c))


class SolverBudgetExceeded(Exception):
    """Internal control flow only. Every public entry point catches this
    and reports `None`; it must never escape to a caller, so a budget
    miss can never be mistaken for a proven result."""


# --------------------------------------------------------------------
# Bit-level primitives
# --------------------------------------------------------------------

def alignment(pos):
    """Whether `pos` contains four in a row (any direction)."""
    m = pos & (pos >> _H1)          # horizontal
    if m & (m >> (2 * _H1)):
        return True
    m = pos & (pos >> HEIGHT)       # diagonal /
    if m & (m >> (2 * HEIGHT)):
        return True
    m = pos & (pos >> _H2)          # diagonal \
    if m & (m >> (2 * _H2)):
        return True
    m = pos & (pos >> 1)            # vertical
    if m & (m >> 2):
        return True
    return False


def _compute_winning_positions(position, mask):
    """Every empty square that would complete a four for `position`.

    Not the same as "moves that win": a square here may be unreachable
    because cells below it are still empty. `_possible(mask) & this` is
    what turns it into playable winning moves."""
    # vertical
    r = (position << 1) & (position << 2) & (position << 3)
    for shift in (_H1, HEIGHT, _H2):
        p = (position << shift) & (position << (2 * shift))
        r |= p & (position << (3 * shift))
        r |= p & (position >> shift)
        p = (position >> shift) & (position >> (2 * shift))
        r |= p & (position << shift)
        r |= p & (position >> (3 * shift))
    return r & (BOARD_MASK ^ mask)


def _possible(mask):
    return (mask + BOTTOM_MASK) & BOARD_MASK


def _can_win_next(position, mask):
    return bool(_compute_winning_positions(position, mask) & _possible(mask))


def _possible_non_losing_moves(position, mask):
    """Playable moves that do not hand the opponent an immediate win.

    Returns 0 when every move loses -- including the case where the
    opponent has two distinct forced threats, which is resolved here
    with no recursion."""
    possible_mask = _possible(mask)
    opponent_win = _compute_winning_positions(position ^ mask, mask)
    forced = possible_mask & opponent_win
    if forced:
        if forced & (forced - 1):
            # Two separate immediate threats -- cannot block both.
            return 0
        possible_mask = forced
    # Never play directly beneath a square the opponent wins on.
    return possible_mask & ~(opponent_win >> 1)


def _popcount(x):
    return bin(x).count("1")


# --------------------------------------------------------------------
# Conversion: project dialect <-> solver dialect
# --------------------------------------------------------------------

def cells_to_bitboard(cells, mark):
    """`(position, mask, n_moves)` for the player `mark` to move.

    `cells` is the project's flat row-major list with row 0 at the TOP.
    Raises ValueError on a board violating gravity (a stone with a hole
    beneath it) -- silently accepting one would produce a confidently
    wrong answer for a position that cannot occur."""
    position = 0
    mask = 0
    for col in range(WIDTH):
        seen_empty_below = False
        for row in range(HEIGHT - 1, -1, -1):      # bottom row first
            v = cells[row * WIDTH + col]
            if v == 0:
                seen_empty_below = True
                continue
            if seen_empty_below:
                raise ValueError(
                    f"floating stone at row={row} col={col}: board violates gravity")
            bit = 1 << (col * _H1 + (HEIGHT - 1 - row))
            mask |= bit
            if v == mark:
                position |= bit
    return position, mask, _popcount(mask)


def bitboard_to_cells(position, mask, mark):
    """Inverse of `cells_to_bitboard`, for round-trip testing."""
    other = 2 if mark == 1 else 1
    cells = [0] * (WIDTH * HEIGHT)
    for col in range(WIDTH):
        for h in range(HEIGHT):
            bit = 1 << (col * _H1 + h)
            if mask & bit:
                row = HEIGHT - 1 - h
                cells[row * WIDTH + col] = mark if (position & bit) else other
    return cells


def _play(position, mask, move_bit):
    """`move_bit` is a single playable square. Returns the OPPONENT's
    `(position, mask)` -- i.e. the view flips to the other side.

    The new stone belongs to the player who just moved, so it must NOT
    appear in the returned `position` (which holds the stones of the
    side now to move). `position ^ mask` is the opponent's stone set;
    writing `position ^ new_mask` instead silently hands our own new
    stone to the opponent on every ply. That exact error was in the
    first version of this module and survived 37 of 40 cross-validation
    positions before the independent referee caught it -- see
    `connectx_solver_referee.py`."""
    return position ^ mask, mask | move_bit


# --------------------------------------------------------------------
# Search
# --------------------------------------------------------------------

class Solver:
    """One solver instance = one transposition table.

    Reusing an instance across related positions (successive plies of a
    game, or the root moves of one decision) is the point: the table is
    where nearly all of the speed comes from."""

    def __init__(self, max_nodes=2_000_000, time_budget=None, tt_limit=3_000_000):
        self.max_nodes = max_nodes
        self.time_budget = time_budget
        self.tt = {}
        self.tt_limit = tt_limit
        self.nodes = 0
        self._deadline = None

    def _check_budget(self):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SolverBudgetExceeded()
        # Time is checked sparsely; calling `time.time()` at every node
        # would itself dominate the node cost in Python.
        if self._deadline is not None and (self.nodes & 0x3FF) == 0:
            if time.time() > self._deadline:
                raise SolverBudgetExceeded()

    def _negamax(self, position, mask, alpha, beta, moves):
        """Exact score for the side to move. Assumes that side has NO
        immediate winning move (every caller guarantees this)."""
        self._check_budget()

        non_losing = _possible_non_losing_moves(position, mask)
        if non_losing == 0:
            # Every move loses immediately.
            return -((WIDTH * HEIGHT - moves) // 2)
        if moves >= WIDTH * HEIGHT - 2:
            return 0

        # Nobody can win before this many further moves, so the window
        # can be tightened before the table is even touched.
        min_score = -((WIDTH * HEIGHT - 2 - moves) // 2)
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha
        max_score = (WIDTH * HEIGHT - 1 - moves) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta

        key = position + mask
        entry = self.tt.get(key)
        if entry is not None:
            lower, upper = entry
            if lower >= beta:
                return lower
            if upper <= alpha:
                return upper
            if lower > alpha:
                alpha = lower
            if upper < beta:
                beta = upper
            if alpha >= beta:
                return alpha

        alpha_orig, beta_orig = alpha, beta

        # Order by how many new threats each move creates; centre-out
        # only as the tie-break.
        candidates = []
        for col in COLUMN_ORDER:
            in_col = non_losing & _column_mask(col)
            if not in_col:
                continue
            move_bit = in_col & -in_col            # lowest playable square
            nxt_pos, nxt_mask = _play(position, mask, move_bit)
            # after _play, `nxt_pos` is the OPPONENT's stones; ours are
            # the complement within the mask.
            ours = nxt_pos ^ nxt_mask
            threats = _popcount(_compute_winning_positions(ours, nxt_mask))
            candidates.append((-threats, nxt_pos, nxt_mask))
        candidates.sort(key=lambda t: t[0])

        best = -1000
        for _, nxt_pos, nxt_mask in candidates:
            val = -self._negamax(nxt_pos, nxt_mask, -beta, -alpha, moves + 1)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if len(self.tt) < self.tt_limit:
            if best <= alpha_orig:
                self.tt[key] = (-1000, best)        # upper bound
            elif best >= beta_orig:
                self.tt[key] = (best, 1000)         # lower bound
            else:
                self.tt[key] = (best, best)         # exact
        return best

    # ----- public entry points -----

    def solve_position(self, position, mask):
        """Exact score for the side to move, or None if out of budget."""
        self._deadline = (time.time() + self.time_budget) if self.time_budget else None
        moves = _popcount(mask)
        try:
            if _can_win_next(position, mask):
                return (WIDTH * HEIGHT + 1 - moves) // 2
            return self._negamax(position, mask, -(WIDTH * HEIGHT) // 2,
                                 (WIDTH * HEIGHT) // 2, moves)
        except SolverBudgetExceeded:
            return None

    def best_move(self, position, mask):
        """`(col, score)` for the side to move, or `(None, None)`.

        Every root move is evaluated, so the returned column is a
        genuinely best move and the score is exact -- not the artifact
        of a first beta cutoff."""
        self._deadline = (time.time() + self.time_budget) if self.time_budget else None
        moves = _popcount(mask)
        possible_mask = _possible(mask)
        win_mask = _compute_winning_positions(position, mask) & possible_mask
        if win_mask:
            for col in COLUMN_ORDER:
                if win_mask & _column_mask(col):
                    return col, (WIDTH * HEIGHT + 1 - moves) // 2

        try:
            best_col, best_score = None, None
            for col in COLUMN_ORDER:
                in_col = possible_mask & _column_mask(col)
                if not in_col:
                    continue
                nxt_pos, nxt_mask = _play(position, mask, in_col & -in_col)
                if _can_win_next(nxt_pos, nxt_mask):
                    # Opponent wins immediately: worst possible, but it
                    # is still a legal move and must be scored, so that
                    # a position where EVERY move loses still returns a
                    # column rather than None.
                    val = -((WIDTH * HEIGHT + 1 - (moves + 1)) // 2)
                else:
                    val = -self._negamax(nxt_pos, nxt_mask,
                                         -(WIDTH * HEIGHT) // 2,
                                         (WIDTH * HEIGHT) // 2, moves + 1)
                if best_score is None or val > best_score:
                    best_col, best_score = col, val
            return best_col, best_score
        except SolverBudgetExceeded:
            return None, None


# --------------------------------------------------------------------
# Project-dialect wrappers
# --------------------------------------------------------------------

def solve_cells(cells, mark, max_nodes=2_000_000, time_budget=None, solver=None):
    """`(best_col, score)` for `mark` to move on a project-dialect board.

    `score` is from `mark`'s own perspective (see the module docstring's
    SCORE CONVENTION). `(None, None)` means the budget ran out."""
    s = solver or Solver(max_nodes=max_nodes, time_budget=time_budget)
    position, mask, _ = cells_to_bitboard(cells, mark)
    return s.best_move(position, mask)


def agent_perspective_value(score, mark):
    """Convert a side-to-move score to `_exact_endgame_solve`'s
    +1 AGENT-wins / 0 draw / -1 AGENT-loses convention, so the two
    solvers can be compared directly."""
    if score is None:
        return None
    sign = (score > 0) - (score < 0)
    return float(sign if mark == 1 else -sign)


# --------------------------------------------------------------------
# A discriminating opponent
# --------------------------------------------------------------------

class PerfectOpponent:
    """A `(board, mark, rng) -> col` opponent for the eval harnesses.

    Plays a PROVEN-optimal move whenever it can afford one, and a named
    fallback heuristic otherwise. `proved` / `total` are exposed and
    must be reported: an opponent that only proved a third of its moves
    is not "perfect play", and presenting it as such would manufacture
    exactly the false ceiling this module exists to remove.

    `min_moves` gates when proving is attempted at all -- early
    positions are the expensive ones, and a budget miss there burns
    real time for nothing."""

    def __init__(self, fallback, min_moves=14, max_nodes=400_000,
                 time_budget=2.0, share_tt=True, name="perfect"):
        self.fallback = fallback
        self.min_moves = min_moves
        self.max_nodes = max_nodes
        self.time_budget = time_budget
        self.share_tt = share_tt
        self.name = name
        self.solver = Solver(max_nodes=max_nodes, time_budget=time_budget)
        self.proved = 0
        self.total = 0

    def reset(self):
        if self.share_tt:
            self.solver = Solver(max_nodes=self.max_nodes,
                                 time_budget=self.time_budget)

    def __call__(self, board, mark, rng):
        self.total += 1
        try:
            position, mask, moves = cells_to_bitboard(board, mark)
        except ValueError:
            return self.fallback(board, mark, rng)
        if moves >= self.min_moves:
            s = self.solver if self.share_tt else Solver(
                max_nodes=self.max_nodes, time_budget=self.time_budget)
            s.max_nodes = s.nodes + self.max_nodes   # per-move node budget
            col, _score = s.best_move(position, mask)
            if col is not None:
                self.proved += 1
                return col
        return self.fallback(board, mark, rng)

    def coverage(self):
        return (self.proved / self.total) if self.total else 0.0
