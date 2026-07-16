# The Tragedy of the Conventions — reproduction package

Code to check the simulation results and regenerate the figures in the paper
as it currently stands. Everything runs on **Python 3.9+** with only **numpy**
and **matplotlib** (no LaTeX, no other dependencies).

## Files

| file | what it is |
|------|------------|
| `tragedy_figures.py` | The model **and** all five figures in one self-contained file. The model layer (analytic thresholds, the coupled replicator, the conditional-cooperation and finite-population routines) is the single source of truth. |
| `check_results.py` | Prints every quantitative claim in the paper as `PAPER:` (stated) vs `CODE:` (computed). Imports the model from `tragedy_figures.py`, so the checks and the figures cannot drift apart. |
| `tragedy.py` | The full simulation engine of record, with the author's `result_R1`…`result_R7` cross-checks. Running it prints each result block and regenerates the conditional-cooperation CSVs. Layer (A) is a closed-form analytic backbone; layer (B) is one Monte-Carlo replicator engine reused everywhere; the two cross-check each other. |
| `cc_deterministic_sweep.csv`, `cc_finite_scaling.csv` | The conditional-cooperation data behind Fig. 6 (deterministic sweep at 10 000 trials; finite sweep at 2 500 trials). `tragedy_figures.py` reads these if present and regenerates them from the model if deleted. |

## Quick start

```bash
python check_results.py        # verify the numbers   (~3 min; TOC_QUICK=1 for ~30 s)
python tragedy_figures.py      # regenerate the figures as PDF+PNG
python tragedy.py              # run the full R1-R7 engine cross-checks (slow: R6/R7)
```

Environment knobs: `TOC_QUICK=1` (fewer Monte-Carlo trials in the checker),
`TOC_FIG_QUICK=1` (coarser/faster figures), `TOC_FIG_OUT=path` (figure output
directory; default is the current directory).

## What `check_results.py` covers

- **§2 / Fig. 2** — single-game basin boundary `beta/(alpha+beta)` and its leftward shift under correlation (0.48 → 0.29).
- **§3.2** — trigger threshold `1/(alpha+1)`, and the finite single-game `x0/theta` escape law.
- **§6 / Fig. 3** — the all-A basin shrinks under coupling (independent vs interdependent joint basin).
- **§6 / Fig. 4** — whole-ring all-A probability collapsing 42% → 1.4% as `N` grows 2 → 10.
- **§6** — the two regimes (weak `a=0.8`, strong `a=2`, both `c=0.3`) satisfy `a+c>1` and `c<1`.
- **§7.1 / Fig. 5** — correlation's effect **and its interaction with system size**: the r = 0 → 0.9 composition at `N=2` and `N=10` for both regimes, including the text's +28 pts (strong) / −14 pts (weak) at `N=2` and the polarisation to all-A (strong) / all-B (weak) at `N=10`. *This is the correlation-under-scale result that wasn't in the earlier code.*
- **§7.2** — the three thresholds (optimistic 0.30, `theta-hat` 0.45, solo 0.77), the deterministic escape at each end of the band, and the finite-population escape (13% vs 2% at `N=50`) with its decline as `N` grows.

## One thing the checker flags

Under §7.2 the checker prints `>>> CHECK` for the footnote claim that a
threshold near `theta ≈ 0.30` "succeeds in the large majority of runs." The
model gives roughly **14%**, not a majority: 0.30 lies below `theta-hat = 0.45`,
so a switch triggered there falls back. This reads like a leftover from the
draft that predated the `theta-hat` correction; worth reconciling in the text.

## Figure ↔ filename ↔ function

| paper figure | file written | function |
|---|---|---|
| Fig. 2 single-game basins | `fig2_single_game_basin.pdf` | `fig2()` |
| Fig. 3 interdependent basin | `fig3_interdependent_basin.pdf` | `fig3()` |
| Fig. 4 convention breakdown | `fig4_convention_breakdown.pdf` | `fig4()` |
| Fig. 5 correlation × scale | `fig_correlation_composition_grid.pdf` | `fig5()` |
| Fig. 6 conditional cooperation | `fig6_conditional_cooperation.pdf` | `fig6()` |

## Notes

- Monte-Carlo quantities vary slightly run-to-run; seeds are fixed so a given
  run is reproducible. Tolerances in the checker are set to accept normal
  sampling noise.
- Fig. 6's finite-population panel is the slowest thing here. With the shipped
  CSVs it is instant; if you delete them, the first `tragedy_figures.py` run
  recomputes them (a few minutes — use `TOC_FIG_QUICK=1` for a fast preview).
- The full simulation engine is included as `tragedy.py`, with the author's own
  cross-checks: `R1` independent basin (engine vs analytic), `R2` basin shrinkage,
  `R3` viability falling with ring size, `R4` correlation and its interaction with
  scale (the Fig. 5 result), `R5` the endogenous threshold band (θ̂ as a verified
  root), `R6` deterministic conditional cooperation, `R7` finite-population
  conditional cooperation (including the single-game `x0/theta` validation).
  `check_results.py` imports the leaner model from `tragedy_figures.py`; `tragedy.py`
  is the standalone engine of record. Running `tragedy.py` overwrites the two
  shipped CSVs with fresh sweeps (`R6`/`R7` are the slow steps).
