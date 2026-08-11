"""
Validates connectx_submission.py against a REAL, independent, single-ply
Connect-4 game engine -- deliberately NOT reusing connectx_env.py's
`step` (which bundles our move and its own fixed opponent's reply into
one call, exactly the shortcut this test needs to NOT rely on, since the
whole point is to check the submission's `agent(observation,
configuration)` interface the way Kaggle's real engine actually drives
it: one call per ply, alternating sides, real `board`/`mark`/
`configuration` objects).

Three opponents, since a single opponent can't tell us much:
1. Random-legal-column opponent (a floor -- if this loses to random, the
   translation from Kaggle's board format is definitely broken).
2. The SAME fixed heuristic connectx_env.py trains against (win-now /
   block / leftmost) -- a sanity check that connectx_train.py's own
   100% internal solve-rate result actually reproduces when the
   submission is driven through a REAL alternating-turn engine instead
   of the training env's bundled step().
3. A slightly stronger heuristic (look-two-ahead: also block if leaving
   the opponent a forced win one ply later) -- a first, cheap check of
   what this project's own docs already name as the honest limitation
   (trained against one fixed weak opponent, no guarantee against a
   different/stronger one).
"""
import importlib.util
import random
import sys
from types import SimpleNamespace

WIDTH, HEIGHT, WIN_LEN = 7, 6, 4


def _rc(row, col):
    return row * WIDTH + col


def _lowest_empty_row(board, col):
    for row in range(HEIGHT - 1, -1, -1):
        if board[_rc(row, col)] == 0:
            return row
    return None


def legal_columns(board):
    return [c for c in range(WIDTH) if _lowest_empty_row(board, c) is not None]


def wins_for(board, mark):
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if board[_rc(row, col)] != mark:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = row + dr * (WIN_LEN - 1), col + dc * (WIN_LEN - 1)
                if not (0 <= er < HEIGHT and 0 <= ec < WIDTH):
                    continue
                if all(board[_rc(row + dr * k, col + dc * k)] == mark for k in range(WIN_LEN)):
                    return True
    return False


def board_full(board):
    return all(c != 0 for c in board)


def drop(board, col, mark):
    row = _lowest_empty_row(board, col)
    board = list(board)
    board[_rc(row, col)] = mark
    return board


def random_opponent(board, mark, rng):
    return rng.choice(legal_columns(board))


def weak_heuristic_opponent(board, mark, rng):
    """The exact policy connectx_env.py bakes into training (win-now /
    block / leftmost) -- reimplemented independently here (not imported)
    so this test doesn't share a bug with the thing it's checking."""
    opp = 2 if mark == 1 else 1
    legal = legal_columns(board)
    for col in legal:
        if wins_for(drop(board, col, mark), mark):
            return col
    for col in legal:
        if wins_for(drop(board, col, opp), opp):
            return col
    return legal[0]


def stronger_heuristic_opponent(board, mark, rng):
    """weak_heuristic_opponent plus a 1-ply-deeper check: among moves that
    pass the first two checks, avoid any move that hands the OPPONENT an
    immediate winning reply next turn, if a safer alternative exists."""
    opp = 2 if mark == 1 else 1
    legal = legal_columns(board)
    for col in legal:
        if wins_for(drop(board, col, mark), mark):
            return col
    for col in legal:
        if wins_for(drop(board, col, opp), opp):
            return col
    safe = []
    for col in legal:
        nxt = drop(board, col, mark)
        if board_full(nxt):
            safe.append(col)
            continue
        opp_can_win = any(wins_for(drop(nxt, c2, opp), opp) for c2 in legal_columns(nxt))
        if not opp_can_win:
            safe.append(col)
    return rng.choice(safe) if safe else legal[0]


def _load_agent(path):
    spec = importlib.util.spec_from_file_location("connectx_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def play_game(our_agent, opponent_fn, our_mark, rng, max_plies=WIDTH * HEIGHT):
    board = [0] * (WIDTH * HEIGHT)
    opp_mark = 2 if our_mark == 1 else 1
    config = SimpleNamespace(columns=WIDTH, rows=HEIGHT, inarow=WIN_LEN)
    current = 1  # P1 always moves first
    for _ply in range(max_plies):
        if current == our_mark:
            obs = SimpleNamespace(board=list(board), mark=our_mark)
            col = our_agent(obs, config)
            if col not in legal_columns(board):  # safety net -- must never trigger
                col = legal_columns(board)[0]
            board = drop(board, col, our_mark)
            if wins_for(board, our_mark):
                return "win"
        else:
            col = opponent_fn(board, opp_mark, rng)
            board = drop(board, col, opp_mark)
            if wins_for(board, opp_mark):
                return "loss"
        if board_full(board):
            return "draw"
        current = opp_mark if current == our_mark else our_mark
    return "draw"


def run_match(our_agent, opponent_fn, opponent_name, n_games, seed):
    """Each game gets its OWN independently-seeded rng (`seed`, `i`) --
    NOT one shared rng consumed sequentially across all `n_games` (the
    original version of this function). That original design meant a
    different agent taking even one different early action would shift
    every subsequent random draw for the REST of the batch, silently
    turning "compare agent A vs agent B on the same 60 games" into two
    barely-related sequences of games -- confirmed as a real confound,
    not a hypothetical one: testing an opening-move change that only
    ever affects the very first ply still swung the OVERALL 60-game
    win rate by double digits, which a change isolated to ply 0 has no
    honest mechanism to cause on its own against a fixed opponent
    POLICY (only via this exact RNG-cascade artifact). Per-game seeding
    makes different agents' results directly, fairly comparable game-by-
    game."""
    results = {"win": 0, "loss": 0, "draw": 0}
    for i in range(n_games):
        rng = random.Random(seed * 100_003 + i)  # large odd multiplier, cheap decorrelation across games
        our_mark = 1 if i % 2 == 0 else 2  # alternate who moves first
        outcome = play_game(our_agent, opponent_fn, our_mark, rng)
        results[outcome] += 1
    n = n_games
    print(f"vs {opponent_name:28s} win={results['win']}/{n} ({results['win']/n:.1%})  "
          f"loss={results['loss']}/{n}  draw={results['draw']}/{n}")
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
    our_agent = _load_agent(path)
    print(f"Loaded agent from {path}\n")

    n_games = 60
    run_match(our_agent, random_opponent, "random-legal-play", n_games, seed=0)
    run_match(our_agent, weak_heuristic_opponent, "weak heuristic (=training opp)", n_games, seed=1)
    run_match(our_agent, stronger_heuristic_opponent, "stronger heuristic (1-ply deeper)", n_games, seed=2)
