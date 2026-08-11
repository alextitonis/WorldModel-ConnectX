# One-off build script for WHITEPAPER.pdf -- not part of the codebase
# itself, run once to (re)generate the PDF from this content. Requires
# `pip install fpdf2`. Run from the docs/ directory: `python build_whitepaper.py`.
from fpdf import FPDF

TITLE = "Latent-Space Simulation Meets a Real Adversary: A World-Model Approach to Kaggle ConnectX"
AUTHOR = "Alexandros Titonis"

pdf = FPDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=20)
pdf.set_margins(20, 20, 20)
pdf.add_page()
pdf.set_font("Helvetica", "B", 18)
pdf.multi_cell(0, 9, TITLE)
pdf.ln(2)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(90, 90, 90)
pdf.multi_cell(0, 6, "A reinforcement-learning world model, real adversarial search, and an exact "
                      "endgame solver applied to Kaggle's ConnectX competition.")
pdf.ln(1)
pdf.multi_cell(0, 6, f"{AUTHOR}  |  Competition: https://kaggle.com/competitions/connectx")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)


def h1(text):
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(4)
    pdf.multi_cell(0, 8, text)
    pdf.ln(1)


def h2(text):
    pdf.set_font("Helvetica", "B", 12)
    pdf.ln(2)
    pdf.multi_cell(0, 7, text)
    pdf.ln(1)


def p(text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 5.6, text)
    pdf.ln(1.5)


def bullet(text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_x(pdf.l_margin + 4)
    pdf.multi_cell(0, 5.6, "-  " + text)
    pdf.ln(0.5)


def table(headers, rows, widths, body_size=9):
    """Proper multi-line table: wraps each cell's text to its column width
    FIRST (via a dry-run multi_cell), takes the max line count in the row,
    then draws every cell's border/text at that shared row height -- avoids
    fpdf2's plain `cell()` silently misaligning columns when text embeds a
    newline or wraps."""
    line_h = 5.0
    pdf.set_font("Helvetica", "B", 9.5)
    header_h = 7.0
    x_start, y_start = pdf.get_x(), pdf.get_y()
    x = x_start
    for htext, w in zip(headers, widths):
        pdf.set_xy(x, y_start)
        pdf.cell(w, header_h, htext, border=1)
        x += w
    pdf.set_xy(x_start, y_start + header_h)

    pdf.set_font("Helvetica", "", body_size)
    for row in rows:
        cell_lines = []
        for text, w in zip(row, widths):
            lines = pdf.multi_cell(w - 2, line_h, text, dry_run=True, output="LINES")
            cell_lines.append(lines)
        n_lines = max(len(lns) for lns in cell_lines)
        row_h = n_lines * line_h
        x_start, y_start = pdf.get_x(), pdf.get_y()
        if y_start + row_h > pdf.page_break_trigger:
            pdf.add_page()
            x_start, y_start = pdf.get_x(), pdf.get_y()
        x = x_start
        for lines, w in zip(cell_lines, widths):
            pdf.rect(x, y_start, w, row_h)
            for i, line in enumerate(lines):
                pdf.set_xy(x + 1, y_start + i * line_h)
                pdf.cell(w - 2, line_h, line)
            x += w
        pdf.set_xy(x_start, y_start + row_h)
    pdf.ln(4)


# ---------------------------------------------------------------------------
h1("1. What this is, and what it isn't")
p("The core idea: instead of generating a move token-by-token or via a hand-written heuristic, "
  "a small learned model (encoder + latent dynamics + value head + a diagnostic decoder) rolls "
  "forward candidate moves in latent space, and a search procedure picks the move whose imagined "
  "future looks best -- MuZero-style planning [1].")
p("This is one instance of a GENERAL latent-space-simulation architecture, not a Connect-4-specific "
  "system: the same encoder/dynamics/value/decoder + search pattern used here has also been applied "
  "to algebraic equation solving, constraint-satisfaction-style logic puzzles, and route-planning "
  "problems, with zero domain-specific changes to the core training/search code -- swapping domains "
  "means implementing one small interface (see `environment.py`), not rewriting the architecture. "
  "ConnectX is the instance documented in this paper because it is the first genuinely ADVERSARIAL "
  "two-player domain the architecture was applied to: a single-agent puzzle solver only ever has to "
  "be right, while a Connect-4 agent has to be right against an opponent actively trying to make it "
  "wrong. That one change breaks assumptions the rest of the pipeline had never been asked to "
  "question, and is the actual subject of this write-up.")
p("This is NOT a hand-written Connect-4 bot. The board simulator and rules are exact (no need to "
  "make a neural network learn something 40 lines of Python computes for free), but every DECISION "
  "-- what a position is worth, which past games are relevant, when the value head should update "
  "online -- comes from a learned model trained without a Connect-4-specific oracle.")

h2("1.1 How this compares to typical RL game-playing systems")
p("Systems in the AlphaZero/MuZero lineage [1][2] are normally trained at large scale: thousands of "
  "self-play games generated in parallel across many machines (often TPU or GPU clusters), running "
  "for days. This project's result was trained on a single consumer laptop -- see Section 1.2 for "
  "the exact specs -- with a full training-pipeline run (stage 1 + stage 2 + self-play) confirmed at "
  "roughly one hour of wall-clock time, and on the order of tens of thousands of self-play games, not "
  "millions. This is presented honestly as evidence that the ARCHITECTURE (real search over exactly-"
  "known rules, substituting for an expensive learned-world-model rollout) is a genuinely efficient "
  "way to reach competent play in a domain with known rules -- not a claim that this matches "
  "AlphaZero-level strength; Section 6 states plainly where this still falls short of a fully-solved "
  "or superhuman opponent.")

h2("1.2 Training hardware")
p("Every result in this paper was produced on a single machine: an RTX 4060 laptop GPU, a Ryzen AI "
  "9 HX 370 CPU, and 32GB RAM. No cluster, no cloud training run, no multi-GPU parallelism -- "
  "everything in this repository, including a full from-scratch training run, was reproduced on that "
  "one laptop.")

h1("2. Architecture")

h2("2.1 The easy path: collapse the adversary into an ordinary MDP")
p("The cheapest way to test whether this works at all: bake a fixed, non-learning opponent policy "
  "directly into the environment's step() function. From the agent's perspective this is then "
  "indistinguishable from a single-agent puzzle -- it picks an action, the environment returns the "
  "resulting state (already reflecting the opponent's reply). Zero changes needed to the rest of "
  "the pipeline. At the real 7x6 board scale there is no exact oracle (the game tree is too large "
  "for brute-force search), so the value head is trained via on-policy Monte Carlo returns instead "
  "of regression against oracle labels.")

h2("2.2 Why the dynamics model can't imagine \"after my move, before theirs\"")
p("step() bundles the agent's move and the fixed opponent's reply into ONE transition, so the "
  "trained dynamics model only ever saw full round-trips as single training examples -- it "
  "structurally cannot represent \"the board immediately after my move, before their reply\" as a "
  "state, because it never saw that state shape. Asking it to imagine that intermediate state would "
  "be extrapolating outside its training distribution.")

h2("2.3 The fix: real search over the real board, learned value only at the leaf")
p("Since Connect-4's rules are exactly known, there's no need to make the network imagine anything "
  "mid-ply. The deployed search does the adversarial ply in REAL board space: enumerate the agent's "
  "real legal moves; for each, enumerate the opponent's real legal replies and assume they pick "
  "whichever hurts the agent most (a genuine minimax, not a single guess). The learned value head is "
  "used only as the LEAF evaluator, on a real, never-imagined state. This is structurally closer to "
  "AlphaZero's design (real search + a learned value net, viable because the rules are exactly "
  "known) [2] than MuZero's (search over a LEARNED model, needed specifically when true dynamics "
  "are not available) [1].")
p("Result of this one architectural change: 95-100% win rate vs. random play, 100% vs. the fixed "
  "weak training heuristic, up to 66.7% vs. a stronger 1-ply-deeper heuristic never seen during "
  "training -- beating every latent-search configuration tried on every metric simultaneously (the "
  "best latent-search result had been 75% / 50% / 11.7%).")

h2("2.4 The exact endgame solver")
p("Real search over real board states opens a door latent-space search never could: once a position "
  "narrows down to a handful of legal columns -- which happens naturally as the board fills -- the "
  "remaining game tree is small regardless of how many plies are left, and can be SOLVED EXACTLY, "
  "no learned value head, no guessing. The solver is a memoized alpha-beta minimax with center-out "
  "move ordering, triggered whenever 5 or fewer columns remain legal (calibrated empirically against "
  "real Kaggle replay data: 5 columns / 24 empty cells solves in under half a second; 6 columns did "
  "not finish inside a 5-second cap with this plain-Python implementation). It is bounded by a hard "
  "wall-clock deadline as an independent safety net -- a position outside the calibrated safe zone "
  "falls through to the round-based search rather than risking the real move-time budget.")
p("The insight that made this cheap: branching factor is controlled by how many columns are LEGAL, "
  "not by how many cells are EMPTY. A position can have twenty or more empty cells and still be "
  "trivial to solve exactly if only a few columns remain open -- exactly the shape a real Connect-4 "
  "endgame takes once most of the board has filled.")

h2("2.5 Episodic memory, LoRA fine-tuning, and online learning")
p("Three more general-purpose mechanisms were applied on top of the search:")
bullet("Episodic memory: a k-NN lookup over real self-play trajectories, both won and lost (a "
       "negative/repulsion signal, not just positive examples), blended into the value estimate at "
       "decision time [3][4]. Small but real positive effect vs. random play; no measurable effect "
       "vs. either fixed heuristic opponent.")
bullet("LoRA self-play fine-tuning: capacity-constrained low-rank deltas applied to the value head "
       "during self-play [5], rather than an unconstrained further gradient update -- 75.0% to "
       "81.7% win rate vs. the stronger heuristic.")
bullet("Curriculum self-play: fixing a real gap where the stronger heuristic was never actually "
       "mixed into self-play training at all -- 81.7% to 85.0%, the best-confirmed result for this "
       "domain.")
bullet("Best-effort online learning (value head only, updated live as real games are played): "
       "isolated via a clean A/B test (the same submission loaded twice, one copy's update call "
       "disabled) -- currently measures exactly 0.0% effect against the synthetic test-opponent set "
       "at evaluation scale. Not evidence it is useless in general -- Kaggle's real opponent pool is "
       "far wider than three fixed synthetic opponents -- just an honest, now-measured answer.")

h1("3. Every bug found and fixed")
p("Seven real, generic-class bugs were found and fixed during development. Each is recorded here "
  "because several are the kind of mistake that is easy to make again in a different adversarial "
  "domain.")
table(
    ["#", "Bug", "Found via"],
    [
        ["1", "Internal eval reported a false 100% win rate (done was assumed to imply a win)",
         "Tracing one \"solved\" game through an independent alternating-turn harness"],
        ["2", "The value head never saw a losing example (unsolved walks were silently discarded)",
         "Root-cause tracing after bug 1"],
        ["3", "Every prediction in the packaged submission was silently corrupted (a spurious extra ReLU)",
         "Diffing the packaged output directly against the real model, node by node"],
        ["4", "\"Same seed\" training runs were not actually reproducible (unseeded global RNG)",
         "Two same-seed runs produced different results"],
        ["5", "A whole session's A/B comparisons were confounded (one shared RNG across a batch)",
         "An opening-move-only change swung the overall win rate by double digits"],
        ["6", "A left-column tie-break bias with no game-theoretic basis",
         "Direct user-observed pattern in real play"],
        ["7", "The online learner taught the value head that losing and drawing are equally bad",
         "Mining real Kaggle replay data via the Kaggle API for missed blocks"],
    ],
    [10, 88, 72],
)
p("The generalizable lesson: bugs 2 and 7 are the same mistake, made twice, in two different places. "
  "\"Treat a loss and a draw identically\" is a natural default everywhere ELSE in a single-agent "
  "pipeline (every prior puzzle domain has exactly one failure mode -- unsolved -- with no "
  "distinction worth making), and it takes deliberate effort to remember that an adversarial domain "
  "has two qualitatively different bad outcomes.")

h1("4. Case study: diagnosing a real loss (zugzwang), and a first attempt at a fix")
p("A live, observed loss -- an opponent slowly building what looked, in hindsight, like an "
  "obviously winning diagonal, with the agent apparently ignoring it -- was root-caused not by "
  "guesswork but by pulling the actual Kaggle replay and solving the real endgame exhaustively, "
  "working backward through the game.")
p("Finding: the position was already a forced loss with 23 empty cells still on the board -- at "
  "least 15 real plies before the deployed 2-round search could possibly have seen it coming. The "
  "mechanism is a genuine, textbook Connect-4 ZUGZWANG / PARITY TRAP (classic odd/even threat "
  "theory [6]): once only two columns remained legal for an extended stretch, which player is "
  "forced to place the fatal piece is decided purely by the parity of total remaining cells across "
  "those two columns -- invisible to any bounded-depth positional search, and not addressed by the "
  "exact endgame solver above (its own calibrated safe zone starts well inside the window where "
  "this trap was already unavoidable).")

h2("4.1 A parity-heuristic attempt -- tried, measured, and honestly a mixed result")
p("A column-parity feature was built and tested directly, rather than left purely theoretical: for "
  "each still-open column with r empty cells, under naive same-column-only alternation the player "
  "who would place the TOP piece is the mover if r is odd, the other player if r is even. This gives "
  "a cheap, real-valued \"parity cost\" per board position -- more even-parity open columns scored as "
  "worse for the agent -- blended into the leaf value estimate at a tunable weight.")
p("This is explicitly NOT full Claimeven: a genuine claimeven strategy requires REACTIVE move-"
  "pairing enforced across an entire game, not a one-shot column count at a single position. It was "
  "built and measured as a testable nudge, not assumed to work because the theory motivates it.")
p("Measured against the trusted test harness at several weights: the feature produces a REAL, "
  "repeatable effect, but not a clean win. At weight 0.5, win rate vs. the stronger heuristic rose "
  "from 70.0% to 77.5% -- but win rate vs. random-legal play fell from 100.0% to 90.0% in the same "
  "run. Smaller weights (0.1-0.3) showed the same pattern at reduced magnitude in both directions. "
  "Net honest verdict: this specific heuristic trades performance against weaker opponents for a "
  "partial, not fully consistent gain against a stronger one -- a real, measured signal in the "
  "direction the theory predicts, but not yet a confirmed fix. It ships in this repository as an "
  "opt-in, OFF-by-default parameter (`parity_weight=0.0` in `adversarial_search.py`) rather than a "
  "new default, exactly so a reader can reproduce this exact finding rather than take it on faith.")
p("Two angles remain genuinely untried: a value head deliberately trained on zugzwang-rich self-play "
  "positions (so the pattern becomes an implicit learned feature rather than a hand-tuned scalar "
  "nudge), and a full, correctly reactive Claimeven implementation rather than a single-position "
  "heuristic proxy.")

h1("5. Results")
p("All win rates below are against an independent, trusted test harness (an alternating-turn engine "
  "that does NOT reuse the environment's own bundled step() -- exactly the shortcut a submission "
  "validator needs to avoid). Three opponents throughout: random-legal play, the fixed weak "
  "heuristic the model trained against, and a 1-ply-deeper \"stronger\" heuristic never seen during "
  "training.")
table(
    ["Configuration", "vs random", "vs weak heur.", "vs stronger heur."],
    [
        ["Latent beam search, depth=1 (original)", "63.3%", "0.0%", "6.7%"],
        ["Latent beam search, depth=3", "55.0%", "0.0%", "1.7%"],
        ["Real adversarial search (1 round)", "96.7%", "100.0%", "66.7%"],
        ["+ memory + online learner (bug 7 present)", "98.3%", "78.3%*", "50.0%"],
        ["+ bug 7 (loss-penalty) fixed", "100.0%", "100.0%", "80.0%"],
        ["+ LoRA self-play fine-tune", "-", "-", "81.7%"],
        ["+ curriculum self-play (best-ever)", "-", "-", "85.0%"],
        ["+ exact endgame solver (current, deployed)", "100.0%", "100.0%", "83.3%"],
    ],
    [95, 30, 32, 33],
)
p("* The weak-heuristic dip in that one row is explained, not a mystery: that harness ran all "
  "opponent blocks sequentially in one process, so the online learner had already drifted from 60 "
  "preceding random-opponent games by the time it reached this block.")
p("On Kaggle's own rating (a TrueSkill-style Gaussian score that starts uncertain and converges "
  "over dozens of real games against other real submissions): a freshly-uploaded submission's "
  "rating is not comparable to one that has had a day to settle. Confirmed directly during "
  "development -- a previously-deployed submission itself started near 300 right after upload and "
  "climbed past 450 only after 27 real games. 7x6 Connect-4 is a mathematically SOLVED game (the "
  "first player wins with perfect play) [6], so the real competitive pool likely includes near-"
  "perfect solvers -- the results above demonstrate the ARCHITECTURE works on this domain; they "
  "should not be read as a leaderboard-rating prediction.")

h1("6. Honest limitations")
bullet("The zugzwang-avoidance problem (Section 4) is diagnosed and a first fix attempt is measured, "
       "but not solved -- see Section 4.1's own honest verdict.")
bullet("Online learning's real-world value is unconfirmed -- it measures zero effect against three "
       "fixed synthetic opponents, which is not the same as zero effect against Kaggle's actual, "
       "much wider real pool.")
bullet("Deeper search beyond 2 rounds (without the endgame solver) was tried and found WORSE, not "
       "just unexplored -- unpruned 3-round search blows the time budget, and the only pruning "
       "width that stayed safe tested worse than plain 2-round search.")
bullet("The real Kaggle competitive pool likely includes near-perfect solvers, since 7x6 Connect-4 "
       "is a solved game -- see Section 5's closing note.")

h1("References")
refs = [
    "[1] Schrittwieser, J. et al. \"Mastering Atari, Go, Chess and Shogi by Planning with a Learned "
    "Model.\" Nature, 2020.",
    "[2] Silver, D. et al. \"A general reinforcement learning algorithm that masters chess, shogi, "
    "and Go through self-play.\" Science, 2018.",
    "[3] Blundell, C. et al. \"Model-Free Episodic Control.\" arXiv:1606.04460, 2016.",
    "[4] Pritzel, A. et al. \"Neural Episodic Control.\" ICML, 2017.",
    "[5] Hu, E. J. et al. \"LoRA: Low-Rank Adaptation of Large Language Models.\" "
    "arXiv:2106.09685, 2021.",
    "[6] Allis, L. V. \"A Knowledge-Based Approach of Connect-Four.\" M.Sc. thesis, Vrije "
    "Universiteit Amsterdam, 1988. (First complete game-theoretic solution of Connect-4; the "
    "classical source for odd/even threat and zugzwang theory referenced in Section 4.)",
    "[7] Mnih, V. et al. \"Human-level control through deep reinforcement learning.\" Nature, 2015. "
    "(Experience-replay precedent used by the on-policy Monte Carlo value-training loop.)",
    "[8] Adam, Addison Howard, and Bovard Doerschuk-Tiberi. \"Connect X.\" "
    "https://kaggle.com/competitions/connectx, 2020. Kaggle.",
    "[9] Kaggle. \"kaggle_environments.\" https://github.com/Kaggle/kaggle-environments. "
    "(Board/config schema and fixed-opponent-baked-into-step convention are ported from this "
    "package's own design, not from any published agent's code.)",
]
pdf.set_font("Helvetica", "", 9.5)
for r in refs:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5.2, r)
    pdf.ln(1.5)

pdf.output("WHITEPAPER.pdf")
print("Wrote WHITEPAPER.pdf")
