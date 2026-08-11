"""
ConnectX environment. Two board sizes, same code:
- Small (`ConnectXEnv()`, 4x4, win_len=3): scaled down so an exact BFS
  oracle stays tractable -- used for the domain's own self-test below.
- Real (`ConnectXEnv(width=7, height=6, win_len=4)`, Kaggle's actual board):
  `bfs_solve` returns None unconditionally -- the full game tree isn't
  exhaustively searchable at this scale. Trained via on-policy Monte Carlo
  value learning instead (see verifier.py / train.py), not oracle regression.

State: the WIDTH*HEIGHT board, each cell one-hot over {EMPTY, AGENT,
OPPONENT}. Row 0 = top; dropping into a column fills the lowest (highest
row index) empty cell, standard Connect-4 gravity.

Actions: DROP(column) for each column, plus one always-legal PASS action
(guarantees `always_legal_actions` is a real, non-empty, unconditional
subset). num_actions = WIDTH + 1.

The fixed training opponent (deterministic unless the epsilon knobs below
are set): after the agent's move, if the agent didn't already win or fill
the board, the opponent (1) takes an immediate win if one exists, (2) else
blocks the agent's immediate win if one exists, (3) else plays the leftmost
legal column. This is this domain's one honest, named limitation for real
Kaggle play: the model is trained against THIS specific opponent shape (plus
diversification, see below), not whatever real opponent Kaggle's matchmaking
actually pairs it against.

Reward: -1 per agent action. "Solved" = the AGENT has win_len in a row after
its own move. A loss (opponent wins) or a draw is terminal (`done=True`) but
NOT solved -- this distinction matters: see search.py's comment on why
trusting `done` alone as "solved" is a real bug for an adversarial domain.
"""
import random
from collections import deque

from .environment import Environment

DEFAULT_WIDTH = 4
DEFAULT_HEIGHT = 4
DEFAULT_WIN_LEN = 3

EMPTY, AGENT, OPPONENT = 0, 1, 2
CELL_WIDTH = 3

# Above this many cells, exhaustive BFS is not attempted. The small-board
# default (16 cells) stays well under this; the real board (42 cells) is
# always above it.
BFS_MAX_CELLS = 20


def _onehot(idx, n):
    v = [0] * n
    v[idx] = 1
    return tuple(v)


def _onehot_index(bits):
    """Robust to a search-time DECODED state whose slot isn't cleanly
    one-hot (real states, built via `_encode_board`, never hit the
    fallback)."""
    return bits.index(1) if 1 in bits else 0


def _encode_board(cells):
    return tuple(b for c in cells for b in _onehot(c, CELL_WIDTH))


def _decode_board(state):
    cells = []
    off = 0
    while off < len(state):
        cells.append(_onehot_index(state[off:off + CELL_WIDTH]))
        off += CELL_WIDTH
    return cells


def _rc(row, col, width):
    return row * width + col


def _lowest_empty_row(cells, col, width, height):
    """Gravity: the row closest to the bottom that's still empty in this
    column, or None if the column is full."""
    for row in range(height - 1, -1, -1):
        if cells[_rc(row, col, width)] == EMPTY:
            return row
    return None


def _legal_columns(cells, width, height):
    return [c for c in range(width) if _lowest_empty_row(cells, c, width, height) is not None]


def _wins_for(cells, value, width, height, win_len):
    """Whether `value` (AGENT or OPPONENT) has win_len in a row anywhere --
    horizontal, vertical, or either diagonal."""
    for row in range(height):
        for col in range(width):
            if cells[_rc(row, col, width)] != value:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row = row + dr * (win_len - 1)
                end_col = col + dc * (win_len - 1)
                if not (0 <= end_row < height and 0 <= end_col < width):
                    continue
                if all(cells[_rc(row + dr * k, col + dc * k, width)] == value for k in range(win_len)):
                    return True
    return False


def _board_full(cells):
    return all(c != EMPTY for c in cells)


def _stronger_opponent_move(cells, width, height, win_len):
    """A second, deliberately stronger deterministic opponent -- same
    win-now/block-immediate-win base as `_fixed_opponent_move`, plus one
    more ply: among moves that survive those two checks, avoid any that
    would hand the AGENT an immediate winning reply next turn, if a safer
    alternative exists. Mixing this into TRAINING (see
    `opponent_strong_epsilon` below) gives the value head real exposure to
    a harder-to-punish opponent, not just noise around the weak one."""
    legal = _legal_columns(cells, width, height)
    for col in legal:
        row = _lowest_empty_row(cells, col, width, height)
        trial = list(cells)
        trial[_rc(row, col, width)] = OPPONENT
        if _wins_for(trial, OPPONENT, width, height, win_len):
            return col
    for col in legal:
        row = _lowest_empty_row(cells, col, width, height)
        trial = list(cells)
        trial[_rc(row, col, width)] = AGENT
        if _wins_for(trial, AGENT, width, height, win_len):
            return col
    safe = []
    for col in legal:
        row = _lowest_empty_row(cells, col, width, height)
        nxt = list(cells)
        nxt[_rc(row, col, width)] = OPPONENT
        if _board_full(nxt):
            safe.append(col)
            continue
        agent_can_win = False
        for col2 in _legal_columns(nxt, width, height):
            row2 = _lowest_empty_row(nxt, col2, width, height)
            trial2 = list(nxt)
            trial2[_rc(row2, col2, width)] = AGENT
            if _wins_for(trial2, AGENT, width, height, win_len):
                agent_can_win = True
                break
        if not agent_can_win:
            safe.append(col)
    return random.choice(safe) if safe else legal[0]


def _fixed_opponent_move(cells, width, height, win_len, opponent_epsilon=0.0, opponent_strong_epsilon=0.0,
                          opponent_selfplay_epsilon=0.0, opponent_policy_fn=None):
    """Deterministic base heuristic: win now if possible, else block the
    agent's immediate win, else leftmost legal column.

    `opponent_epsilon`: with this probability, ignore the heuristic and
    play a uniformly random legal column instead -- diversifies training
    trajectories (a fully deterministic opponent means every training walk
    from a matching starting side is the SAME exact game).

    `opponent_strong_epsilon`: with this probability (checked after the
    roll above), delegate the whole move to `_stronger_opponent_move`
    instead -- direct training exposure to a harder opponent, not just
    noise around the weak one.

    `opponent_selfplay_epsilon` / `opponent_policy_fn`: with this
    probability (checked last), delegate to an arbitrary caller-supplied
    move function -- in practice, a frozen snapshot of this same model's
    own move choice, viewed from the opponent's side (see
    `train.make_selfplay_pool_opponent_fn`). This is the actual
    "self-play" mechanism: every opponent above is a fixed, non-learning
    heuristic the trained policy eventually plateaus against; self-play
    is what lets it face something that keeps getting better."""
    legal = _legal_columns(cells, width, height)
    if opponent_epsilon > 0.0 and random.random() < opponent_epsilon:
        return random.choice(legal)
    if opponent_strong_epsilon > 0.0 and random.random() < opponent_strong_epsilon:
        return _stronger_opponent_move(cells, width, height, win_len)
    if opponent_selfplay_epsilon > 0.0 and opponent_policy_fn is not None \
            and random.random() < opponent_selfplay_epsilon:
        col = opponent_policy_fn(cells)
        if col in legal:
            return col
    for col in legal:
        row = _lowest_empty_row(cells, col, width, height)
        trial = list(cells)
        trial[_rc(row, col, width)] = OPPONENT
        if _wins_for(trial, OPPONENT, width, height, win_len):
            return col
    for col in legal:
        row = _lowest_empty_row(cells, col, width, height)
        trial = list(cells)
        trial[_rc(row, col, width)] = AGENT
        if _wins_for(trial, AGENT, width, height, win_len):
            return col
    return legal[0]


def is_solved(state, width, height, win_len):
    return _wins_for(_decode_board(state), AGENT, width, height, win_len)


def is_legal(state, action_idx, width, height):
    pass_action = width
    if action_idx == pass_action:
        return True
    if not (0 <= action_idx < width):
        return False
    return _lowest_empty_row(_decode_board(state), action_idx, width, height) is not None


def step(state, action_idx, width, height, win_len, opponent_epsilon=0.0, opponent_strong_epsilon=0.0,
         opponent_selfplay_epsilon=0.0, opponent_policy_fn=None):
    pass_action = width
    cells = list(_decode_board(state))
    reward = -1.0

    if action_idx != pass_action:
        row = _lowest_empty_row(cells, action_idx, width, height)
        cells[_rc(row, action_idx, width)] = AGENT

    if _wins_for(cells, AGENT, width, height, win_len):
        return _encode_board(cells), reward, True
    if _board_full(cells):
        return _encode_board(cells), reward, True  # draw -- terminal, not solved

    opp_col = _fixed_opponent_move(cells, width, height, win_len, opponent_epsilon=opponent_epsilon,
                                    opponent_strong_epsilon=opponent_strong_epsilon,
                                    opponent_selfplay_epsilon=opponent_selfplay_epsilon,
                                    opponent_policy_fn=opponent_policy_fn)
    opp_row = _lowest_empty_row(cells, opp_col, width, height)
    cells[_rc(opp_row, opp_col, width)] = OPPONENT

    if _wins_for(cells, OPPONENT, width, height, win_len):
        return _encode_board(cells), reward, True  # loss -- terminal, not solved
    done = _board_full(cells)  # draw after opponent's move
    return _encode_board(cells), reward, done


def random_problem(rng, width, height):
    """Every game starts from an empty board."""
    return _encode_board([EMPTY] * (width * height)), None


def bfs_solve(state, width, height, win_len, max_depth=8):
    """Exact BFS for a forced win against the fixed opponent baked into
    `step` -- not a general Connect-4 solver. Returns None above
    BFS_MAX_CELLS (the real 7x6 board is never attempted)."""
    if width * height > BFS_MAX_CELLS:
        return None
    if is_solved(state, width, height, win_len):
        return []
    frontier = deque([state])
    parent = {state: None}
    action_taken = {}
    depth = {state: 0}
    num_actions = width + 1
    while frontier:
        cur = frontier.popleft()
        if depth[cur] >= max_depth:
            continue
        for a_idx in range(num_actions):
            if not is_legal(cur, a_idx, width, height):
                continue
            nxt, _reward, done = step(cur, a_idx, width, height, win_len)
            if nxt in parent:
                continue
            parent[nxt] = cur
            action_taken[nxt] = a_idx
            depth[nxt] = depth[cur] + 1
            if is_solved(nxt, width, height, win_len):
                path = []
                node = nxt
                while parent[node] is not None:
                    path.append(action_taken[node])
                    node = parent[node]
                path.reverse()
                return path
            if not done:  # loss/draw states are terminal dead ends, don't expand
                frontier.append(nxt)
    return None


_CELL_CHAR = {EMPTY: ".", AGENT: "A", OPPONENT: "O"}


def format_state(state, width, height):
    cells = _decode_board(state)
    rows = [" ".join(_CELL_CHAR[cells[_rc(row, col, width)]] for col in range(width)) for row in range(height)]
    return "\n" + "\n".join(rows)


def format_action(action_idx, width):
    return "PASS" if action_idx == width else f"DROP(col={action_idx})"


class ConnectXEnv(Environment):
    """width/height/win_len fixed per instance -- state_dim/num_actions
    depend on them. Default (4x4, win_len=3) is the small, BFS-checkable
    board this module's own self-test uses; pass width=7, height=6,
    win_len=4 for the real Kaggle board."""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, win_len=DEFAULT_WIN_LEN,
                 opponent_epsilon=0.0, opponent_strong_epsilon=0.0,
                 opponent_selfplay_epsilon=0.0, opponent_policy_fn=None):
        self.width = width
        self.height = height
        self.win_len = win_len
        # Default 0.0 keeps this instance's `step` fully deterministic
        # (required by the small-board self-test's BFS oracle). The real-
        # board TRAINING env sets these > 0; its EVAL env keeps them at
        # the default so it's still graded against the originally-defined
        # fixed opponent.
        self.opponent_epsilon = opponent_epsilon
        self.opponent_strong_epsilon = opponent_strong_epsilon
        self.opponent_selfplay_epsilon = opponent_selfplay_epsilon
        self.opponent_policy_fn = opponent_policy_fn

    @property
    def state_dim(self):
        return self.width * self.height * CELL_WIDTH

    @property
    def num_actions(self):
        return self.width + 1

    @property
    def always_legal_actions(self):
        return [self.width]  # PASS

    def is_solved(self, state):
        return is_solved(state, self.width, self.height, self.win_len)

    def is_legal(self, state, action_idx):
        return is_legal(state, action_idx, self.width, self.height)

    def step(self, state, action_idx):
        return step(state, action_idx, self.width, self.height, self.win_len,
                     opponent_epsilon=self.opponent_epsilon,
                     opponent_strong_epsilon=self.opponent_strong_epsilon,
                     opponent_selfplay_epsilon=self.opponent_selfplay_epsilon,
                     opponent_policy_fn=self.opponent_policy_fn)

    def random_problem(self, rng, **kwargs):
        return random_problem(rng, self.width, self.height)

    def bfs_solve(self, state, max_depth=8):
        return bfs_solve(state, self.width, self.height, self.win_len, max_depth=max_depth)

    def format_state(self, state):
        return format_state(state, self.width, self.height)

    def format_action(self, action_idx):
        return format_action(action_idx, self.width)


if __name__ == "__main__":
    env = ConnectXEnv()
    print(f"state_dim={env.state_dim}  num_actions={env.num_actions}  "
          f"board={env.width}x{env.height}  win_len={env.win_len}\n")

    start, _ = env.random_problem(random.Random(0))
    print(f"Empty board: {format_state(start, env.width, env.height)}")
    path = env.bfs_solve(start)
    assert path is not None, "no forced win found against the fixed opponent from an empty board"
    print(f"Oracle's forced-win path: {[env.format_action(a) for a in path]}  (len={len(path)})")
    cur = start
    for a in path:
        cur, r, done = env.step(cur, a)
        print(f"  after {env.format_action(a)} (reward={r:.0f}, done={done}): {env.format_state(cur)}")
    assert env.is_solved(cur), "oracle path did not reach a solved (agent-won) state"
    print("\nCONFIRMED: the exact oracle finds a genuine forced win against the fixed opponent.\n")

    print("=== Random-legal-play smoke test (30 games, no crashes, always terminates) ===")
    rng = random.Random(1)
    solved_count, loss_count, draw_count = 0, 0, 0
    for _i in range(30):
        state, _ = env.random_problem(rng)
        for _ in range(env.width * env.height + 1):
            legal = [a for a in range(env.num_actions) if env.is_legal(state, a)]
            assert legal, "always_legal_actions guarantee violated -- PASS should always be legal"
            a = rng.choice([a for a in legal if a != env.width] or legal)
            state, _r, done = env.step(state, a)
            if done:
                break
        else:
            raise AssertionError("game did not terminate within the move cap")
        if env.is_solved(state):
            solved_count += 1
        elif _board_full(_decode_board(state)):
            draw_count += 1
        else:
            loss_count += 1
    print(f"agent wins={solved_count}  losses={loss_count}  draws={draw_count}  (out of 30, random legal play)")
    print("\nAll games terminated cleanly, always_legal_actions held in every state, no crashes.")

    print("\n=== Real-board smoke test (7x6, win_len=4, no BFS oracle at this scale) ===")
    real_env = ConnectXEnv(width=7, height=6, win_len=4)
    print(f"state_dim={real_env.state_dim}  num_actions={real_env.num_actions}")
    assert real_env.bfs_solve(real_env.random_problem(random.Random(0))[0]) is None, \
        "bfs_solve should return None at real-board scale (no oracle by design)"
    rng = random.Random(2)
    state, _ = real_env.random_problem(rng)
    for _ in range(real_env.width * real_env.height + 1):
        legal = [a for a in range(real_env.num_actions) if real_env.is_legal(state, a)]
        assert legal
        a = rng.choice([a for a in legal if a != real_env.width] or legal)
        state, _r, done = real_env.step(state, a)
        if done:
            break
    print(real_env.format_state(state))
    print("Real-board game ran to completion with no crashes; bfs_solve correctly returns None.")
