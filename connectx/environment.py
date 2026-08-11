"""
Minimal common interface for a "domain" this codebase can train on. A state
is any fixed-length tuple of numbers; how those numbers are interpreted is
entirely up to the implementation below (`connectx_env.py`). Nothing in
`model.py` / `train_utils.py` / `search.py` needs to change to support a new
domain that implements this interface.
"""
from abc import ABC, abstractmethod


class Environment(ABC):
    # Every state built by this codebase is a discrete, exactly-hashable
    # tuple (a one-hot-encoded Connect-4 board) -- kept as a class attribute
    # rather than hardcoded into `search.py`'s cycle-detection logic so a
    # future continuous-state domain could override it there without
    # touching this interface.
    discrete_state = True

    def observe(self, state):
        """What the model is allowed to see, as a function of the true
        state. Default: full observability (identity). ConnectX is fully
        observable, so this is never overridden -- kept as an explicit
        extension point rather than removed, since every place that feeds
        a state to the model calls this first, not the raw state."""
        return state

    @property
    @abstractmethod
    def state_dim(self):
        """Length of the fixed-size numeric tuple representing a state."""

    @property
    @abstractmethod
    def num_actions(self):
        """Size of the fixed, discrete action space."""

    @property
    @abstractmethod
    def always_legal_actions(self):
        """Action indices legal from EVERY state, unconditionally -- used
        by search's neurosymbolic decode-gate to fall back on safely."""

    @abstractmethod
    def is_solved(self, state):
        ...

    @abstractmethod
    def is_legal(self, state, action_idx):
        ...

    @abstractmethod
    def step(self, state, action_idx):
        """Returns (next_state, reward, done). Assumes legality."""

    @abstractmethod
    def random_problem(self, rng, **kwargs):
        """Returns (state, answer) -- answer is domain-specific (unused for
        ConnectX, every game starts from the same empty board)."""

    @abstractmethod
    def bfs_solve(self, state, max_depth=8):
        """Exact oracle: shortest forced win, or None if not found within
        max_depth (or if the state space is too large to search -- see
        connectx_env.py's BFS_MAX_CELLS)."""

    def format_state(self, state):
        return str(state)

    def format_action(self, action_idx):
        return str(action_idx)
