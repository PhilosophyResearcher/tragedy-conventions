"""
tragedy_figures.py
==================
All diagrams for "The Tragedy of the Conventions" in one self-contained file.
Depends only on numpy and matplotlib (no local modules, no usetex).

Run
    python tragedy_figures.py
to regenerate every figure as PDF + PNG into $TOC_FIG_OUT (default: current dir).

Figures produced:
    fig2_single_game_basin.pdf        single-game basins; correlation widens all-A
    fig3_interdependent_basin.pdf     interdependence shrinks the basin (saddle theta-hat)
    fig4_convention_breakdown.pdf     outcome distribution vs ring size N
    fig_correlation_composition_grid.pdf   correlation x scale (N=2 / N=10, strong/weak)
    fig6_conditional_cooperation.pdf  conditional cooperation: deterministic + finite

Fig 6 reads two CSVs (cc_deterministic_sweep.csv, cc_finite_scaling.csv) from
$TOC_FIG_OUT; if absent, it regenerates them from the model functions below.

Resolution knobs are collected at the top. Defaults are publication-grade and
Fig 5 / Fig 6 take a few minutes; set the environment variable TOC_FIG_QUICK=1
for a fast, noisy smoke run.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

OUT = os.environ.get("TOC_FIG_OUT", ".")

# ---- resolution knobs (lower => faster/noisier) ---------------------------
_QUICK = bool(os.environ.get("TOC_FIG_QUICK"))
GRID_N          = 80    if _QUICK else 240     # fig3 phase-portrait resolution
FIG4_TRIALS     = 3000  if _QUICK else 30000   # fig4 Monte-Carlo trials
FIG5_NR         = 6     if _QUICK else 15       # fig5 correlation samples per panel
FIG5_TRIALS_N2  = 3000  if _QUICK else 30000
FIG5_TRIALS_N10 = 3000  if _QUICK else 16000
CC_SWEEP_TRIALS = 1500  if _QUICK else 10000    # fig6(a) deterministic sweep
CC_FINITE_TRIALS = 400  if _QUICK else 2500     # fig6(b) finite scaling


# ===========================================================================
# Shared style
# ===========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
    "figure.dpi": 140, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

A_LINE, A_FILL = "#1f5c8b", "#cfe0ec"      # optimal convention A (blue)
B_LINE, B_FILL = "#c07a2b", "#f0e2cb"      # suboptimal convention B (ochre)
FRAG_FILL = "#dedede"
GREY, LANDMARK = "#5a5a5a", "#333333"
STABLE   = dict(marker="o", ms=8, mfc=A_LINE, mec=A_LINE, zorder=5)
UNSTABLE = dict(marker="o", ms=8, mfc="white", mec=LANDMARK, mew=1.4, zorder=5)
# diverging fill for composition figures: all-B (ochre) -> mixed -> all-A (blue)
CMAP = LinearSegmentedColormap.from_list("BA", [B_LINE, "#e8e8e8", A_LINE])


def landmark(ax, x, label, y=1.02, color=LANDMARK):
    ax.axvline(x, color=color, ls=":", lw=1.2, zorder=1)
    ax.text(x, y, label, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", color=color, fontsize=10)


def finish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def _save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# Model: analytic backbone + coupled replicator (beta = 1 throughout)
# ===========================================================================
BETA = 1.0

def independent_threshold(alpha, beta=BETA):
    return beta / (alpha + beta)

def correlated_threshold(alpha, r, beta=BETA):
    num = beta - alpha * r
    return 0.0 if num <= 0 else num / ((1.0 - r) * (alpha + beta))

def self_consistent_threshold(a, c, beta=BETA):
    return (-(c + beta) + np.sqrt((c + beta) ** 2 + 4.0 * a * beta)) / (2.0 * a)

def solo_threshold(c, beta=BETA):
    return beta / (c + beta)

def optimistic_threshold(a, c, beta=BETA):
    return beta / (a + c + beta)


def _alphas(x, a, c):
    neigh = 0.5 * (np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1))
    return a * neigh + c

def _deriv(x, a, c, r):
    alpha = _alphas(x, a, c)
    pi_A = alpha * (r + (1.0 - r) * x)
    pi_B = r + (1.0 - r) * (1.0 - x)
    return x * (1.0 - x) * (pi_A - pi_B)

def _integrate(x0, a, c, r=0.0, dt=0.1, steps=800):
    """RK4 forward integration of a batch of trajectories to (near-)corner rest."""
    x = x0.copy()
    for _ in range(steps):
        k1 = _deriv(x, a, c, r)
        k2 = _deriv(x + 0.5 * dt * k1, a, c, r)
        k3 = _deriv(x + 0.5 * dt * k2, a, c, r)
        k4 = _deriv(x + dt * k3, a, c, r)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        np.clip(x, 0.0, 1.0, out=x)
    return x

def _dist(N, a, c, r, trials, seed=0):
    """Distribution over the number of domains ending on A (fraction 0..N)."""
    rng = np.random.default_rng(seed)
    xf = _integrate(rng.random((trials, N)), a, c, r=r)
    k = (xf > 0.5).sum(axis=1)
    return np.bincount(k, minlength=N + 1) / len(k)


# ---- conditional cooperation (Fig. 6) -------------------------------------
def _trapped_B(k=2.0):
    return lambda rng, shape: rng.random(shape) ** k

def cc_success(theta, a, c, trials=20000, seed=0, steps=800):
    """Deterministic escape rate of threshold theta in two coupled domains."""
    rng = np.random.default_rng(seed)
    m = _trapped_B(2.0)(rng, (trials, 2))
    p0 = np.where(m >= theta, m, 0.0)
    pf = _integrate(p0, a, c, r=0.0, steps=steps)
    return (pf > 0.99).all(axis=1).mean()

def cc_success_independent(theta, a, c, trials=20000, seed=0):
    """Independent baseline: each domain a fixed game with alpha = a + c."""
    rng = np.random.default_rng(seed)
    m = _trapped_B(2.0)(rng, (trials, 2))
    boundary = 1.0 / ((a + c) + 1.0)
    return ((m >= theta) & (m > boundary)).all(axis=1).mean()

def finite_escape(N, x0, theta, a, c, coupled=True, domains=2, s=1.0,
                  trials=4000, max_steps=None, seed=0):
    """Finite-population escape under pairwise-comparison (Fermi) imitation."""
    rng = np.random.default_rng(seed)
    if max_steps is None:
        max_steps = 60 * N * N
    C = np.full((trials, domains), int(round(x0 * N)), dtype=np.int64)
    denom = max(N - 1, 1)
    for _ in range(max_steps):
        m = C / N
        trig = m >= theta
        p = np.where(trig, m, 0.0)
        if coupled and domains > 1:
            neigh = 0.5 * (np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))
            alpha = a * neigh + c
        else:
            alpha = np.full_like(p, a + c)
        piB = 1.0 - p
        piC = np.where(trig, alpha * p, piB)
        frac = C / N
        p_up = (1 - frac) * (C / denom) / (1.0 + np.exp(-s * (piC - piB)))
        p_dn = frac * ((N - C) / denom) / (1.0 + np.exp(-s * (piB - piC)))
        rr = rng.random((trials, domains))
        up = rr < p_up
        dn = (~up) & (rr < p_up + p_dn)
        C = C + up.astype(np.int64) - dn.astype(np.int64)
        np.clip(C, 0, N, out=C)
        if ((C == 0) | (C == N)).all():
            break
    return (C == N).all(axis=1).mean()


# ===========================================================================
# Figure 2 -- single-game basins; correlation widens the all-A basin
# ===========================================================================
def fig2():
    alpha = 1.1
    x0 = independent_threshold(alpha)          # 0.4762
    x8 = correlated_threshold(alpha, 0.8)      # 0.2857
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for y, xb, rlab in [(1.0, x0, "$r=0$"), (0.35, x8, "$r=0.8$")]:
        ax.add_patch(Rectangle((0, y - 0.11), xb, 0.22, fc=B_FILL, ec="none", zorder=0))
        ax.add_patch(Rectangle((xb, y - 0.11), 1 - xb, 0.22, fc=A_FILL, ec="none", zorder=0))
        ax.plot([0, 1], [y, y], color="#888", lw=0.8, zorder=1)
        for (xs, xe) in [(xb - 0.06, 0.10), (xb + 0.06, 0.90)]:
            ax.add_patch(FancyArrowPatch((xs, y), (xe, y), arrowstyle="-|>",
                         mutation_scale=12, color="#666", lw=1.3, zorder=2))
        ax.plot(0, y, **STABLE); ax.plot(1, y, **STABLE); ax.plot(xb, y, **UNSTABLE)
        ax.text(xb, y - 0.20, f"$x^*={xb:.2f}$", ha="center", va="top",
                color=LANDMARK, fontsize=10)
        ax.text(-0.02, y, rlab, ha="right", va="center", fontsize=11)
    ax.text(0.0, 1.30, "all-$B$", ha="center", fontsize=10, color=B_LINE)
    ax.text(1.0, 1.30, "all-$A$", ha="center", fontsize=10, color=A_LINE)
    ax.set_xlim(-0.16, 1.05); ax.set_ylim(0.05, 1.45)
    ax.set_yticks([]); ax.set_xlabel("frequency of $A$")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=3)
    _save(fig, "fig2_single_game_basin")


# ===========================================================================
# Figure 3 -- interdependence shrinks the basin
# ===========================================================================
def _fate_grid(a, c, n, coupled=True):
    xs = np.linspace(0.001, 0.999, n)
    X1, X2 = np.meshgrid(xs, xs)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    if coupled:
        xf = _integrate(pts.copy(), a, c, steps=700)
    else:
        alpha = a + c
        xf = pts.copy()
        for _ in range(700):
            xf = xf + 0.1 * xf * (1 - xf) * (alpha * xf - (1 - xf))
            np.clip(xf, 0, 1, out=xf)
    A = xf > 0.5
    code = np.where(A[:, 0] & A[:, 1], 0, np.where(~A[:, 0] & ~A[:, 1], 1, 2))
    return xs, code.reshape(n, n)

def fig3():
    a, c = 2.0, 0.3
    cmap = ListedColormap([A_FILL, B_FILL, FRAG_FILL])
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.5), sharey=True)
    for ax, coupled, title in [(axes[0], False, "independent"),
                               (axes[1], True, "interdependent")]:
        xs, code = _fate_grid(a, c, GRID_N, coupled=coupled)
        ax.pcolormesh(xs, xs, code, cmap=cmap, vmin=0, vmax=2, shading="auto", zorder=0)
        ax.contour(xs, xs, (code == 0).astype(float), levels=[0.5],
                   colors=A_LINE, linewidths=1.6, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$ (frequency of $A$, domain 1)")
        ax.set_title(title); ax.tick_params(length=3)
    axes[0].set_ylabel("$x_2$ (frequency of $A$, domain 2)")
    axes[1].text(0.82, 0.82, "all-$A$", ha="center", color=A_LINE, fontsize=10)
    axes[1].text(0.16, 0.16, "all-$B$", ha="center", color=B_LINE, fontsize=10)
    axes[1].text(0.85, 0.16, "frag.", ha="center", color="#777", fontsize=9)
    axes[1].text(0.16, 0.85, "frag.", ha="center", color="#777", fontsize=9)
    _save(fig, "fig3_interdependent_basin")


# ===========================================================================
# Figure 4 -- outcome breakdown by ring size N (strong coupling)
# ===========================================================================
def fig4():
    a, c = 2.0, 0.3
    Ns = [2, 4, 6, 8, 10]
    dists = {N: _dist(N, a, c, 0.0, FIG4_TRIALS) for N in Ns}
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.9), sharey=True)
    ymax = max(d.max() for d in dists.values())
    for ax, N in zip(axes, Ns):
        d = dists[N]; frac = np.arange(N + 1) / N
        ax.bar(frac, d, width=0.9 / N, color=[CMAP(f) for f in frac],
               edgecolor="white", linewidth=0.3)
        ax.set_title(f"$N={N}$", pad=4)
        ax.set_xlim(-0.09, 1.09); ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "", "1"]); ax.set_ylim(0, ymax * 1.08)
        finish(ax)
    axes[0].set_ylabel("probability")
    axes[0].text(0, -0.14, "all-$B$", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=8.5, color=B_LINE)
    axes[0].text(1, -0.14, "all-$A$", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", fontsize=8.5, color=A_LINE)
    fig.supxlabel("fraction of domains reaching the optimal convention $A$",
                  y=0.04, fontsize=12)
    fig.subplots_adjust(bottom=0.24, wspace=0.12, top=0.86)
    _save(fig, "fig4_convention_breakdown")


# ===========================================================================
# Figure 5 -- correlation x scale: composition vs r (N=2 top, N=10 bottom)
# ===========================================================================
def fig5():
    c = 0.3
    rows = [(2, FIG5_TRIALS_N2), (10, FIG5_TRIALS_N10)]
    cols = [("strong coupling ($a=2$)", 2.0), ("weak coupling ($a=0.8$)", 0.8)]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, (N, trials) in enumerate(rows):
        rs = np.linspace(0.0, 0.9, FIG5_NR)
        for j, (lab, a) in enumerate(cols):
            ax = axes[i, j]
            P = np.array([_dist(N, a, c, r, trials) for r in rs])
            cum = np.zeros(len(rs))
            for k in range(N + 1):
                ax.fill_between(rs, cum, cum + P[:, k], color=CMAP(k / N), lw=0)
                cum = cum + P[:, k]
            ax.set_xlim(0, 0.9); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
            finish(ax)
            if i == 0:
                ax.set_title(lab, pad=6)
            if i == len(rows) - 1:
                ax.set_xlabel("correlation $r$")
            if j == 0:
                ax.set_ylabel("outcome composition")
        axes[i, 0].annotate(f"$N={N}$", xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-52, 0), textcoords="offset points",
                            ha="center", va="center", rotation=90, fontsize=15)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.045, pad=0.015, ticks=[0, 0.5, 1.0])
    cbar.set_label("fraction of domains reaching $A$", fontsize=10)
    cbar.ax.set_yticklabels(["0\n(all-$B$)", "0.5", "1\n(all-$A$)"])
    _save(fig, "fig_correlation_composition_grid")


# ===========================================================================
# Figure 6 -- conditional cooperation: deterministic (a) + finite (b)
# ===========================================================================
def _load_or_build_cc_data():
    a, c = 2.0, 0.3
    det_path = f"{OUT}/cc_deterministic_sweep.csv"
    fin_path = f"{OUT}/cc_finite_scaling.csv"
    if not os.path.exists(det_path):
        thetas = np.linspace(0.02, 0.96, 32)
        dep = np.array([cc_success(t, a, c, trials=CC_SWEEP_TRIALS) for t in thetas])
        ind = np.array([cc_success_independent(t, a, c, trials=CC_SWEEP_TRIALS)
                        for t in thetas])
        np.savetxt(det_path, np.column_stack([thetas, dep, ind]), delimiter=",",
                   header="theta,interdependent,independent", comments="# ", fmt="%.6f")
    if not os.path.exists(fin_path):
        Ns = np.array([25, 50, 75, 100, 150])
        ind = np.array([finite_escape(N, 0.15, 0.40, a, c, coupled=False,
                                      trials=CC_FINITE_TRIALS) for N in Ns])
        dep = np.array([finite_escape(N, 0.15, 0.40, a, c, coupled=True,
                                      trials=CC_FINITE_TRIALS) for N in Ns])
        np.savetxt(fin_path, np.column_stack([Ns, ind, dep]), delimiter=",",
                   header="N,independent,interdependent", comments="# ",
                   fmt=["%d", "%.5f", "%.5f"])
    return (np.genfromtxt(det_path, delimiter=",", comments="#"),
            np.genfromtxt(fin_path, delimiter=",", comments="#"))

def fig6():
    a, c = 2.0, 0.3
    hat, solo = self_consistent_threshold(a, c), solo_threshold(c)
    det, fin = _load_or_build_cc_data()
    th, dep_det, ind_det = det[:, 0], det[:, 1], det[:, 2]
    N, ind_fin, dep_fin = fin[:, 0], fin[:, 1], fin[:, 2]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.9, 3.5))
    axA.plot(th, ind_det, color=GREY, ls="--", label="independent")
    axA.plot(th, dep_det, color=A_LINE, ls="-", label="interdependent")
    for x, lab in [(hat, r"$\hat{\theta}$"), (solo, r"$\theta_{\mathrm{solo}}$")]:
        landmark(axA, x, lab)
    axA.set_xlabel(r"conditional-cooperation threshold $\theta$")
    axA.set_ylabel("escape probability")
    axA.set_xlim(0, 1); axA.set_ylim(0, max(dep_det.max(), ind_det.max()) * 1.25)
    axA.set_title("(a) deterministic limit", pad=20)
    axA.legend(frameon=False, loc="lower left"); finish(axA)

    axB.semilogy(N, ind_fin, color=GREY, ls="--", marker="s", ms=4, label="independent")
    axB.semilogy(N, dep_fin, color=A_LINE, ls="-", marker="o", ms=4, label="interdependent")
    axB.set_xlabel("population size per domain $N$")
    axB.set_ylabel("system escape probability")
    axB.set_xticks([25, 50, 75, 100, 150])
    axB.set_title("(b) finite populations", pad=20)
    axB.legend(frameon=False, loc="lower left"); finish(axB)
    _save(fig, "fig6_conditional_cooperation")


# ===========================================================================
def main():
    fig2(); print("fig2 done")
    fig3(); print("fig3 done")
    fig4(); print("fig4 done")
    fig5(); print("fig5 done")
    fig6(); print("fig6 done")
    print(f"all figures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
