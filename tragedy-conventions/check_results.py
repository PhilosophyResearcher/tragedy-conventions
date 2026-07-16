"""
check_results.py
================
Reproduces every quantitative claim in "The Tragedy of the Conventions"
so the simulation results can be checked against the text.

It imports the model from tragedy_figures.py -- the SAME code that draws the
figures -- so the numbers here and the figures cannot drift apart.

Run:
    python check_results.py            # full check (~3 min)
    TOC_QUICK=1 python check_results.py # faster, noisier (~30 s)

For each claim the script prints
    PAPER : the value as stated in the manuscript (section/figure noted)
    CODE  : the value this run computes
Monte-Carlo quantities wobble run-to-run; seeds are fixed for reproducibility.
A line flagged  >>> CHECK  marks a place where code and text do not agree and
that is worth a human look.
"""

import os
import numpy as np
import tragedy_figures as M     # model + figures (import does not draw anything)

QUICK = bool(os.environ.get("TOC_QUICK"))
T_DIST   = 3000 if QUICK else 15000    # Monte-Carlo trials for _dist(...)
T_CC     = 1500 if QUICK else 8000     # deterministic conditional-coop sweep
T_FIN    = 500  if QUICK else 2000     # finite-population imitation runs
FIN_NS   = [25, 50] if QUICK else [25, 50, 75, 100]   # 150 also in paper; slow


def line(paper, code, ok=None, note=""):
    tag = "" if ok is None else ("   ok" if ok else "   >>> CHECK")
    print(f"    PAPER : {paper}")
    print(f"    CODE  : {code}{tag}")
    if note:
        print(f"            {note}")
    print()


def head(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


# ===========================================================================
head("Sec. 2 / Fig. 2  --  single-game basin boundary and correlation")
b0 = M.independent_threshold(1.1)
b8 = M.correlated_threshold(1.1, 0.8)
print("  Basin boundary x* = beta/(alpha+beta), alpha=1.1, beta=1:")
line("x* = 0.48 at r=0", f"x* = {b0:.4f}", abs(b0 - 0.48) < 0.01)
print("  With correlation r=0.8:")
line("x* = 0.29", f"x* = {b8:.4f}", abs(b8 - 0.29) < 0.01,
     "correlation widens the all-A basin (boundary moves left)")

# ===========================================================================
head("Sec. 3.2  --  conditional-cooperation threshold and the x0/theta escape")
print("  Single-game trigger threshold theta* = 1/(alpha+1):")
line("theta* = 1/3 at alpha=2", f"theta* = {M.independent_threshold(2.0):.4f}",
     abs(M.independent_threshold(2.0) - 1/3) < 1e-6)
print("  Finite single game: seed x0 reaches theta with prob ~ x0/theta.")
esc = M.finite_escape(50, 0.15, 0.40, 2.0, 0.3, coupled=False, domains=1,
                      trials=max(T_FIN, 3000))
line("x0/theta = 0.15/0.40 = 0.375", f"escape = {esc:.3f}", abs(esc - 0.375) < 0.05,
     "validates the unbiased-walk / gambler's-ruin escape law")

# ===========================================================================
head("Sec. 6 / Fig. 3  --  interdependence shrinks the all-A basin")
a, c = 2.0, 0.3
th_ind = M.independent_threshold(a + c)               # each domain, fixed alpha=a+c
basin_ind = (1 - th_ind) ** 2                          # joint independent all-A basin
basin_dep = M._dist(2, a, c, 0.0, T_DIST)[-1]          # coupled all-A basin (MC)
print("  Joint all-A basin fraction, independent vs interdependent (a=2, c=0.3):")
line("Fig. 3: coupled basin < independent basin",
     f"independent = {basin_ind:.3f}   coupled = {basin_dep:.3f}",
     basin_dep < basin_ind, "basin shrinks under coupling")

# ===========================================================================
head("Sec. 6 / Fig. 4  --  convention breakdown by ring size N (a=2, c=0.3)")
allA = {N: M._dist(N, a, c, 0.0, T_DIST)[-1] for N in (2, 4, 6, 8, 10)}
print("  Probability the whole ring reaches all-A:")
for N in (2, 4, 6, 8, 10):
    print(f"      N={N:2d} : all-A = {allA[N]:.3f}")
line("42% at N=2  ->  1.4% at N=10",
     f"{100*allA[2]:.0f}% at N=2  ->  {100*allA[10]:.1f}% at N=10",
     abs(allA[2] - 0.42) < 0.04 and allA[10] < 0.03)

# ===========================================================================
head("Sec. 6  --  parameter regimes (both keep A optimal but neighbour-dependent)")
for lab, aa in (("weak", 0.8), ("strong", 2.0)):
    print(f"  {lab} coupling a={aa}, c=0.3:  a+c={aa+0.3:.1f} (need >1),  c=0.3 (need <1)")
line("weak a=0.8, strong a=2, both c=0.3; a+c>1 and c<1", "both satisfied",
     (0.8 + 0.3 > 1) and (2.0 + 0.3 > 1) and (0.3 < 1))

# ===========================================================================
head("Sec. 7.1 / Fig. 5  --  correlation, and correlation UNDER SYSTEM SIZE")
print("  All-A rate at r=0 vs r=0.9, and how correlation moves the composition.")
print("  (split/mixed = outcomes that are neither all-A nor all-B)\n")
for lab, aa in (("strong (a=2)", 2.0), ("weak (a=0.8)", 0.8)):
    for N in (2, 10):
        d0 = M._dist(N, aa, 0.3, 0.0, T_DIST)
        d9 = M._dist(N, aa, 0.3, 0.9, T_DIST)
        A0, A9 = d0[-1], d9[-1]
        B0, B9 = d0[0], d9[0]
        mix0, mix9 = 1 - A0 - B0, 1 - A9 - B9
        print(f"  {lab}, N={N}:")
        print(f"      r=0.0 :  all-B={B0:.2f}  split/mix={mix0:.2f}  all-A={A0:.2f}")
        print(f"      r=0.9 :  all-B={B9:.2f}  split/mix={mix9:.2f}  all-A={A9:.2f}"
              f"   (delta all-A = {100*(A9-A0):+.0f} pts)")
        print()
print("  Key numbers quoted in the text (both at N=2):")
sN2 = M._dist(2, 2.0, 0.3, 0.9, T_DIST)[-1] - M._dist(2, 2.0, 0.3, 0.0, T_DIST)[-1]
wN2 = M._dist(2, 0.8, 0.3, 0.9, T_DIST)[-1] - M._dist(2, 0.8, 0.3, 0.0, T_DIST)[-1]
line("strong coupling: all-A rises ~ +28 pts (r: 0 -> 0.9)",
     f"delta all-A = {100*sN2:+.0f} pts", abs(100*sN2 - 28) < 8)
line("weak coupling: all-A falls ~ -14 pts (r: 0 -> 0.9)",
     f"delta all-A = {100*wN2:+.0f} pts", abs(100*wN2 + 14) < 8)
print("  Under system size (N=10): correlation drains the split and polarises")
print("  the whole ring -- to all-A under strong coupling, all-B under weak.")

# ===========================================================================
head("Sec. 7.2  --  the three thresholds (strong regime a=2, c=0.3)")
opt = M.optimistic_threshold(a, c)
hat = M.self_consistent_threshold(a, c)
solo = M.solo_threshold(c)
line("optimistic end ~ 0.30", f"{opt:.3f}", abs(opt - 0.30) < 0.01)
line("sophisticated threshold  theta-hat ~ 0.45", f"{hat:.3f}", abs(hat - 0.45) < 0.01)
line("solo threshold ~ 0.77", f"{solo:.3f}", abs(solo - 0.77) < 0.01)

head("Sec. 7.2  --  deterministic escape at the two ends of the band")
s_solo = M.cc_success(solo, a, c, trials=T_CC)
s_opt  = M.cc_success(opt,  a, c, trials=T_CC)
print("  Escape probability if the threshold is set at each end of the band:")
line("theta ~ 0.77 (calibrated to a stalled neighbour): ~3% success",
     f"cc_success(0.77) = {100*s_solo:.0f}%", abs(100*s_solo - 3) < 4)
line("theta ~ 0.30: footnote says 'large majority of runs'",
     f"cc_success(0.30) = {100*s_opt:.0f}%", ok=(s_opt > 0.5),
     note="the 0.30 end does better than 0.77, but the model puts it well below a "
          "majority (0.30 < theta-hat=0.45, so a switch there reverts). The "
          "'large majority' footnote looks like a leftover from the pre-theta-hat draft.")

# ===========================================================================
head("Sec. 7.2 / Fig. 6b  --  finite-population escape and its decline with N")
print("  theta=0.40, x0=0.15, a=2, c=0.3.  Escape probability per domain size:\n")
ind_ref = None
print(f"      {'N':>4}   {'independent':>12}   {'interdependent':>14}")
for N in FIN_NS:
    ei = M.finite_escape(N, 0.15, 0.40, a, c, coupled=False, trials=T_FIN)
    ed = M.finite_escape(N, 0.15, 0.40, a, c, coupled=True, trials=T_FIN)
    if N == 50:
        ind50, dep50 = ei, ed
    ind_ref = ei
    print(f"      {N:>4}   {ei:>12.3f}   {ed:>14.3f}")
print()
line("N=50: independent ~13%, interdependent ~2%",
     f"independent = {ind50*100:.0f}%   interdependent = {dep50*100:.0f}%",
     abs(ind50 - 0.13) < 0.06 and dep50 < 0.06)
print("  Paper also states the interdependent rate keeps falling with N while the")
print("  independent rate stays flat -- visible in the table above.\n")

print("=" * 74)
print("Done. Regenerate the figures with:  python tragedy_figures.py")
print("=" * 74)
