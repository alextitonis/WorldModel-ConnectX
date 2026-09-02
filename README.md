# ConnectX-WorldModel

**By Alexandros Titonis**

A reinforcement-learning world model (encoder + latent dynamics + value head)
combined with real adversarial search and an exact endgame solver, applied to
Kaggle's **[ConnectX competition](https://kaggle.com/competitions/connectx)**
(ranked Connect-4, 7-wide x 6-tall board, evergreen/Knowledge-only, no
deadline).

Instead of a hand-written Connect-4 bot, this trains a small model to predict
what happens next in latent space, then searches over that prediction to pick
a move — MuZero/AlphaZero-style planning [1][2], applied to a real,
Kaggle-ranked adversarial game, and trained end to end on a single laptop (RTX
4060 laptop GPU, Ryzen AI 9 HX 370, 32GB RAM — no cluster, no cloud run). Full
write-up, architecture rationale, every bug found along the way, an attempted
fix for a diagnosed zugzwang weakness, and honest limitations:
**[`docs/WHITEPAPER.pdf`](docs/WHITEPAPER.pdf)**.

This is also not a Connect-4-only architecture — it's one instance of a
general latent-space-simulation pattern (the same encoder/dynamics/value/
decoder + search code has been applied to other domains — equation solving,
constraint-satisfaction logic puzzles, route planning — with zero core-code
changes, just a new implementation of the small interface in
`connectx/environment.py`). ConnectX is documented here because it's the
first genuinely *adversarial* two-player domain this architecture was applied
to — see the whitepaper's Section 1 for the full framing.

## How it works, briefly

1. **A learned world model** (`connectx/model.py`) — an encoder maps a board
   to a latent vector, a dynamics model predicts the next latent + reward
   given an action, a value head estimates cost-to-go, and a diagnostic
   decoder checks the latent space isn't collapsing. Trained self-supervised
   (`scripts/train.py`, `connectx/train_utils.py`) with **no oracle** — the
   real 7x6 board's game tree is too large to brute-force, so the value head
   trains via on-policy Monte Carlo returns (`connectx/verifier.py`) instead
   of regression against exact labels.
2. **Real adversarial search, not latent imagination**
   (`connectx/adversarial_search.py`) — the board's rules are exactly known,
   so rather than asking the trained model to *imagine* the opponent's reply,
   the search enumerates the agent's real legal moves and the opponent's real
   legal replies directly against the actual board simulator
   (`connectx/env.py`), and uses the learned value head only as the leaf
   evaluator. This is the single biggest lever: it alone took the agent from
   struggling against even a fixed heuristic to reliably beating one it never
   trained against.
3. **An exact endgame solver**, folded into the same search — once a
   position narrows to a handful of legal columns (which happens naturally
   as a real game fills up), the remaining game tree is small regardless of
   how many moves are left, and gets solved exactly instead of estimated.
4. **Episodic memory + LoRA self-play fine-tuning**
   (`connectx/episodic_memory.py`, `connectx/lora.py`,
   `scripts/lora_selfplay_finetune.py`) layered on top, each confirmed to
   help before being kept.
5. **A first, honestly-reported attempt at a real diagnosed weakness**
   (Connect-4 zugzwang/parity traps) — built, measured, and shipped as an
   opt-in experimental parameter rather than oversold as solved. See the
   whitepaper's Section 4.

## Results

Win rate against an independent, trusted test harness (`tests/submission_test.py`
— an alternating-turn engine that does **not** reuse the environment's own
bundled `step()`, deliberately avoiding the shortcut a real submission
validator needs to avoid). Three opponents: random-legal play, the fixed
weak heuristic trained against, and a stronger 1-ply-deeper heuristic never
seen during training.

| Configuration | vs random | vs weak heuristic | vs stronger heuristic |
|---|---|---|---|
| Latent beam search (original baseline)² | 63.3% | 0.0% | 6.7% |
| Real adversarial search (1 round) | 96.7% | 100.0% | 66.7% |
| + episodic memory + online learner | 98.3% | 78.3%¹ | 50.0% |
| + a real bug fixed (loss ≠ draw) | 100.0% | 100.0% | 80.0% |
| + LoRA self-play fine-tune | — | — | 81.7% |
| + curriculum self-play (best-ever) | — | — | **85.0%** |
| + exact endgame solver (current, deployed) | 100.0% | 100.0% | 83.3% |

¹ Explained, not a mystery — that harness ran all opponent blocks
sequentially in one process, so the online learner had already drifted from
60 preceding random-opponent games by the time it reached this block. See
the whitepaper for the full table, the diagnosis, and every real bug found
along the way (7 of them — a couple are worth knowing if you build on this).

² Named for the contrast it draws, but stated precisely: the rollout and the
value judgement in `connectx/search.py` are fully latent, while the *legality*
of each branch is not. The root ply's allowed set comes from `env.is_legal` on
the real state, and deeper plies decode each beam entry back to an estimated
real state and ask `env.is_legal` about that — always, at the default
`caution=1.0`. Only `depth=1` never decodes. So the row above is latent
dynamics and latent evaluation with environment-defined legality gating, not a
search that never touches the real rules; the contrast with the row beneath it
is *imagined vs. enumerated opponent replies*, which is the part that actually
moved the numbers.

### On Kaggle's own leaderboard (2026-09-02)

The deployed `submission.py` currently sits at **630 rating, rank 66 / 190**.

Kaggle's number is a TrueSkill-style score that starts near 600 and converges
over dozens of real games, so a freshly-uploaded rating is not a verdict in
either direction, and this one will keep moving. For context on the ceiling:
7x6 Connect-4 is a mathematically **solved** game, so the pool above this
rating includes near-perfect solvers, and Kaggle's per-move time cap is as much
the binding constraint as evaluation quality. Read 630 as evidence the
architecture transfers to a real adversarial ladder — not as a claim about how
near the top of one it is.

## The exact solver, and what it changed (2026-08-30)

The agent has always carried an exact endgame solver, but it was plain Python
with a tuple-keyed memo, and it was gated at **≤ 5 legal columns** because that
was its practical reach. Two things were blocked behind that gate: a strong
enough opponent to tell agent configurations apart, and any analysis of a real
loss that was decided before the endgame.

`connectx/bitboard_solver.py` removes it — bitboards over a 7-bits-per-column
layout, `position + mask` as a unique key, a transposition table storing
**bounds** rather than plain values, losing-move pruning, and move ordering by
threats created. It agrees with the older solver on **40/40** randomly reached
endgame positions and runs **~50–70× faster** on the same ones, resolving
midgame positions with 6+ open columns that the older one could not.

**How it was validated, and why that matters.** The first version of the new
solver agreed with the old one on **37 of 40** positions — the kind of number
that reads like edge-case noise in fresh code. It was not: an independent third
implementation (`connectx/solver_referee.py` — plain minimax, *no alpha-beta*,
cell-scan win detection, no shared machinery) judged all three disagreements and
the new solver was wrong in **3 of 3**. The cause was a single line in the
position update. A high agreement rate is not a pass; the residual is where the
bug lives.

### A measured consequence: resisting in a lost endgame

The older solver returns only +1 / 0 / −1 — it carries no information about
*how fast* a result arrives. So when every move loses they all score alike and
it picks arbitrarily, frequently choosing the fastest loss. Against perfect play
that costs nothing. Against the imperfect agents a real ladder is full of, it
discards every chance the opponent errs first.

`_resisting_endgame_solve` prefers the slowest loss (and the fastest win).
Measured from **proven-lost** positions — verified lost by the solver, so this
is a fact and not an estimate — playing each out twice from the identical
position against the identical opponent with paired seeds:

| opponent | legacy escapes | resisting escapes | Δ | game length |
|---|---|---|---|---|
| perfect play *(control)* | 0.000 | **0.000** | +0.000 | +4.3 plies |
| weak heuristic | 0.100 | **0.300** | **+0.200** | +3.5 plies |
| stronger heuristic | 0.150 | **0.225** | +0.075 | +3.8 plies |

The control is what makes the rest trustworthy: against perfect play both arms
escape exactly 0.000, because the positions really are lost — while resisting
still survives 4.3 plies longer. The mechanism works without manufacturing
false escapes. At n = 40 the +0.200 is solid and the +0.075 is directional.

### Deployed (2026-08-31)

Both are now **on by default** in the shipped `submission.py`. The gate is set
by calibration, not optimism — legal-column count controls branching, so it is
what the gate keys on:

| legal columns | solved within 1.0s | median | p90 | max |
|---|---|---|---|---|
| 4 | 30/30 | 0.000s | 0.001s | 0.001s |
| 5 | 30/30 | 0.000s | 0.003s | **0.017s** |
| 6 | 30/30 | 0.000s | 0.037s | **0.843s** |
| 7 | **23/30** | 0.068s | 1.008s | capped |

So `_ENDGAME_MAX_COLS` moved 5 → **6** and `_ENDGAME_TIME_BUDGET` moved
1.2s → **0.6s**. The budget went *down* while reach went *up*: at 5 columns the
older solver's worst case was 0.373s and this one's is 0.017s. Seven columns
resolves only 23/30 within a second, which is why the gate stops at six.

**The older solver is kept as a fallback, not replaced** — if the bitboard
solver misses its budget the call falls through to it, then to the round
search, so the failure path is exactly the previous behaviour.

Measured on the rebuilt agent before shipping: **103 timed moves, median
0.416s, p90 0.713s, max 0.967s** against Kaggle's 2.0s cap, with the whole-call
1.5s cap still behind it. Against a discriminating opponent over 6 seeds the
change scored **mean +0.036, interval [−0.001, +0.074]** versus the previous
build — which **straddles zero, so this is recorded as no regression rather
than as a confirmed improvement.** That is the expected shape: the endgame
solver only fires once a position narrows to ≤6 columns, and resisting only
pays in positions already lost.

## Running locally

Requires Python 3.10+ and PyTorch (CPU is fine — nothing here needs a GPU;
the reference results above were produced on a laptop GPU but nothing in the
code requires one).

```bash
pip install torch

# Self-test the environment (small board, exact BFS oracle) + a real-board smoke test
python -m connectx.env

# Verify the already-trained, shipped agent against the trusted harness
python -m tests.submission_test

# Train a fresh checkpoint from scratch (real 7x6 board, no oracle at this scale, ~1hr on a laptop)
python -m scripts.train

# LoRA self-play fine-tune an existing checkpoint (this is the step that produced the deployed one)
python -m scripts.lora_selfplay_finetune

# Package a checkpoint + offline self-play memory into a self-contained Kaggle submission.py
python -c "from scripts.build_submission import main; main()"

# Exact-solver self-test: cross-validates the bitboard solver against the older
# one on real positions, and checks it reaches past the older one's limits
python -m tests.bitboard_solver_test

# Adjudicate a disagreement between the two solvers with an independent referee
python -m connectx.solver_referee

# Does resisting longer in a proven-lost endgame actually convert losses?
python -m connectx.resisting_endgame_eval
```

Run everything from the repository root (not from inside `connectx/` or
`scripts/`) — every command above is `python -m <package>.<module>`, matching
the layout below.

`submission.py` (repo root) is the actual file to upload to Kaggle as-is —
it has zero dependencies beyond `torch`/`base64`/`io`/`time`, with the
trained weights and episodic memory embedded directly in the file.

## Training this for a different game

Nothing here is Connect-4-specific beyond `connectx/env.py`. Every other
file only depends on the small interface `connectx/environment.py` defines:

- `state_dim`, `num_actions`, `always_legal_actions` (properties)
- `is_solved(state)`, `is_legal(state, action_idx)`
- `step(state, action_idx) -> (next_state, reward, done)`
- `random_problem(rng) -> (state, answer)`
- `bfs_solve(state, max_depth=8) -> path or None` (an exact oracle, if one
  exists at a scale small enough to brute-force — return `None`
  unconditionally if it doesn't, the way `env.py` does above `BFS_MAX_CELLS`)

To point this at a new game or puzzle:

1. Implement `Environment` for your domain (see `connectx/env.py` for a full
   worked example, including the small-board-with-an-oracle / large-board-
   without-one pattern).
2. Run `connectx.train_utils.train_stage1` on your new environment to get a
   working encoder/dynamics/decoder — no oracle needed for this stage in any
   domain.
3. **If your domain is single-agent** (a puzzle, not a two-player game):
   the value head can train directly against `bfs_solve` labels if you have
   an oracle, or via `connectx.verifier.train_mc_value_onpolicy` (drop
   `unsolved_penalty`, since a single-agent domain never has a "loss," only
   "unsolved") if you don't.
4. **If your domain is adversarial** (two players, like this one): bake a
   fixed opponent into your `step()` first (the "easy path" — see `env.py`'s
   own module docstring) to get the pipeline working end to end, THEN set
   `train_mc_value_onpolicy(..., unsolved_penalty=<something>)` — this is
   the one setting that matters most for an adversarial domain and is
   exactly what bug #2 in the whitepaper was about: without it, the value
   head never sees a single example of "this leads to losing."
5. If your domain's rules are exactly known (not something that needs to be
   learned) and the state space is too large for `bfs_solve` to search
   exhaustively at decision time, real search over the real environment
   (`connectx/adversarial_search.py`'s pattern, or the plain single-agent
   search in `connectx/search.py`) will almost always beat asking the
   dynamics model to imagine ahead — that was this project's single biggest
   result.

## Layout

```
connectx-opensource/
  connectx/                        The importable package -- everything domain-agnostic + ConnectX itself
    environment.py                   The generic Environment interface everything else depends on
    env.py                           ConnectX itself: board, rules, the fixed training opponent
    model.py                         Encoder / dynamics / value head / decoder
    train_utils.py                   Self-supervised stage-1 training (no oracle, no labels)
    verifier.py                      On-policy Monte Carlo value-head training (no oracle)
    search.py                        Latent-space search (comparison baseline; decode-gated legality) + load_checkpoint
    adversarial_search.py            The REAL search actually deployed: minimax + exact endgame solver
                                      + the experimental parity-heuristic attempt (Section 4.1)
    episodic_memory.py               k-NN memory over real self-play trajectories (won + lost)
    memory_build.py                  Builds that memory offline, at packaging time
    lora.py                          Generic LoRA wrapper
    bitboard_solver.py               Exact strong solver: bitboards + transposition table (Section 4.2)
    solver_referee.py                Independent no-alpha-beta minimax, used to adjudicate solver disagreements
    resisting_endgame_eval.py        Measures whether resisting longer in a lost endgame converts losses
  scripts/                         Executable entry points (run as `python -m scripts.<name>`)
    train.py                         Full training pipeline (stage 1 + stage 2 + self-play)
    lora_selfplay_finetune.py        LoRA self-play fine-tune of an existing checkpoint
    build_submission.py              Packages a checkpoint + memory into one self-contained submission.py
  tests/
    submission_test.py               The trusted, independent test harness
    bitboard_solver_test.py          Solver self-test, incl. cross-validation against the older solver
  checkpoints/
    connectx_checkpoint.pt           Trained weights
  docs/
    WHITEPAPER.pdf                   Full write-up: architecture, every bug, all results, limitations
    build_whitepaper.py              Regenerates WHITEPAPER.pdf (requires `pip install fpdf2`)
  submission.py                    THE deployed file — upload this to Kaggle as-is
  LICENSE
  .gitignore
```

## Citations

Full reference list with page/venue detail is in `docs/WHITEPAPER.pdf`.
Headline credits: latent-space planning follows MuZero [1] / AlphaZero [2];
episodic memory follows Model-Free Episodic Control [3] and Neural Episodic
Control [4]; LoRA fine-tuning follows Hu et al. [5]; the zugzwang/parity
endgame theory referenced in the whitepaper's case study traces to Victor
Allis's 1988 solution of Connect-4 [6]; the board/config schema and
fixed-opponent-in-`step` convention are ported from Kaggle's own ConnectX
competition [8] and the `kaggle_environments` package [9], not from any
published agent's code.

## License

[MIT](LICENSE).
