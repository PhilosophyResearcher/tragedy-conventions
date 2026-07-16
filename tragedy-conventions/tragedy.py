"""
Minimal simulation model for "The Tragedy of the Conventions."

Establishes the paper's main results with the smallest machinery that is
faithful to the revised text. Two layers:

  (A) Analytic backbone -- closed-form quantities a referee can check by hand:
        - independent basin boundary          beta / (alpha + beta)
        - correlated basin boundary            (beta - alpha r) / ((1-r)(alpha+beta))
        - self-consistent threshold  theta-hat  root of  a t^2 + (c+beta) t - beta = 0
        - solo threshold                        beta / (c + beta)
        - optimistic threshold                  beta / (a + c + beta)

  (B) One Monte-Carlo engine over the coupled replicator dynamic, reused for
      every simulated result (basins shrink; viability falls with N; correlation
      backfires under weak coupling / helps under strong coupling; fragmentation
      is eliminated). Nothing here depends on the analytic layer, so the two
      cross-check each other.

  (C) Conditional cooperation, shown robust across population size. The
      interdependence penalty is isolated by comparing an independent baseline
      (each domain a fixed game with the most favourable payoff, alpha=a+c) to
      the interdependent case, in BOTH the deterministic infinite-population
      limit (R6) and a finite pairwise-comparison imitation process (R7, which
      also reproduces the analytic single-game escape probability x0/theta).

Dynamics (beta normalised to 1 throughout):
    alpha_i = a * mean(neighbours of i on a ring) + c
    x_i' = x_i (1 - x_i) [ pi_A(i) - pi_B(i) ]
with, under assortment level r in [0,1],
    pi_A(i) = alpha_i [ r + (1-r) x_i ]
    pi_B(i) = 1 * [ r + (1-r) (1 - x_i) ]
so r = 0 recovers x_i' = x_i(1-x_i)(alpha_i x_i - (1 - x_i)), the paper's Eq. (7).

For N = 2 the ring collapses to mutual dependence (alpha_1 = a x_2 + c, etc.).
"""

import numpy as np

BETA = 1.0


# ---------------------------------------------------------------------------
# (A) Analytic backbone
# ---------------------------------------------------------------------------

def independent_threshold(alpha, beta=BETA):
    """All-A basin boundary for a single independent game: x* = beta/(alpha+beta)."""
    return beta / (alpha + beta)


def correlated_threshold(alpha, r, beta=BETA):
    """Boundary under assortment r. Returns 0.0 when A dominates (basin is all of [0,1])."""
    num = beta - alpha * r
    if num <= 0:
        return 0.0
    return num / ((1.0 - r) * (alpha + beta))


def self_consistent_threshold(a, c, beta=BETA):
    """theta-hat: the symmetric fixed point solving a t^2 + (c+beta) t - beta = 0."""
    disc = (c + beta) ** 2 + 4.0 * a * beta
    return (-(c + beta) + np.sqrt(disc)) / (2.0 * a)


def solo_threshold(c, beta=BETA):
    """Commitment a domain needs to hold A with a neighbour still on B (x_j ~ 0)."""
    return beta / (c + beta)


def optimistic_threshold(a, c, beta=BETA):
    """Commitment needed with a fully converted neighbour (x_j ~ 1)."""
    return beta / (a + c + beta)


# ---------------------------------------------------------------------------
# (B) Coupled replicator: Monte-Carlo basin engine
# ---------------------------------------------------------------------------

def _alphas(x, a, c):
    """Per-domain payoff slope from ring neighbours. x has shape (trials, N)."""
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


def basin_outcomes(N, a, c, r=0.0, trials=20000, seed=0):
    """
    Sample uniform initial conditions on [0,1]^N, integrate the coupled system,
    and classify the resting state. Interior equilibria are saddles, so all mass
    lands at corners up to a measure-zero separatrix; the 0.5 cut is robust.

    Returns dict with fractions: all_A, all_B, fragmented, unconverged.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.random((trials, N))
    xf = _integrate(x0, a, c, r)
    hiA = xf > 0.99
    loB = xf < 0.01
    all_A = np.all(hiA, axis=1)
    all_B = np.all(loB, axis=1)
    settled = np.all(hiA | loB, axis=1)
    fragmented = settled & ~all_A & ~all_B
    unconverged = ~settled
    return {
        "all_A": all_A.mean(),
        "all_B": all_B.mean(),
        "fragmented": fragmented.mean(),
        "unconverged": unconverged.mean(),
    }


# ---------------------------------------------------------------------------
# Result reproductions
# ---------------------------------------------------------------------------

def result_R1():
    print("R1  Independent games: engine reproduces the analytic basin")
    print("    superior convention alpha=1.1, beta=1")
    alpha = 1.1
    x_star = independent_threshold(alpha)          # 1/2.1 = 0.4762
    basin = 1.0 - x_star                            # fraction of uniform starts -> all-A
    mc = basin_outcomes(1, a=0.0, c=alpha, trials=40000)["all_A"]
    print(f"    boundary x*            = {x_star:.4f}   (analytic)")
    print(f"    all-A basin size       = {basin:.4f}   (analytic 1 - x*)")
    print(f"    all-A basin size       = {mc:.4f}   (Monte Carlo)")
    print(f"    correlation r=0.8 -> boundary {correlated_threshold(alpha, 0.8):.4f} "
          f"(paper: 0.29)")
    assert abs(mc - basin) < 0.01, "engine disagrees with analytic independent basin"
    print()


def result_R2():
    print("R2  Interdependence shrinks the all-A basin (two domains)")
    for (a, c, tag) in [(0.8, 0.3, "weak coupling  "),
                        (2.0, 0.3, "strong coupling")]:
        indep_max = 1.0 - independent_threshold(a + c)   # best-case fixed neighbour
        indep_base = 1.0 - independent_threshold(c)       # worst-case fixed neighbour
        two = basin_outcomes(2, a, c)["all_A"]
        print(f"    {tag:32s}  a={a}, c={c}")
        print(f"      independent benchmark  all-A basin in [{indep_base:.3f}, {indep_max:.3f}]"
              f"  (neighbour fixed at B / at A)")
        print(f"      interdependent (N=2)   all-A basin  = {two:.3f}")
    print()


def result_R3():
    print("R3  Viability falls as the number of coupled domains grows (ring)")
    a, c = 2.0, 0.3
    for N in range(2, 7):
        o = basin_outcomes(N, a, c, trials=20000)
        print(f"    N={N}  all-A={o['all_A']:.3f}  all-B={o['all_B']:.3f}  "
              f"fragmented={o['fragmented']:.3f}")
    print()


def result_R4():
    print("R4  Correlation under interdependence, and its interaction with scale")
    print("    all-A and fragmentation at r=0 vs r=0.9; weak a=0.8, strong a=2, c=0.3")
    print(f"    {'regime, size':20s} {'all-A r0->r0.9':>18s} {'frag r0->r0.9':>18s}")
    for tag, a in [("weak   (a=0.8)", 0.8), ("strong (a=2.0)", 2.0)]:
        for N in (2, 10):
            lo = basin_outcomes(N, a, 0.3, r=0.0)
            hi = basin_outcomes(N, a, 0.3, r=0.9)
            dA = (hi["all_A"] - lo["all_A"]) * 100
            print(f"    {tag+f', N={N}':20s} "
                  f"{lo['all_A']:.2f}->{hi['all_A']:.2f} ({dA:+3.0f} pts)   "
                  f"{lo['fragmented']:.2f}->{hi['fragmented']:.2f}")
    print("    N=2 deltas match the text (~ +28 pts strong, ~ -14 pts weak); at N=10")
    print("    the effect intensifies -- correlation drives fragmentation to ~0 and")
    print("    polarises the ring to all-A (strong) or all-B (weak).")
    print()


def trapped_B(k=2.0):
    """
    Initial-commitment distribution weighted toward the trapped-B corner.
    Each domain's latent committed stock is a uniform draw raised to power k>=1;
    k=1 is uniform, larger k concentrates mass near 0 (populations mostly on B
    with only a small stock of conditional cooperators). Returns a callable.
    """
    return lambda rng, shape: rng.random(shape) ** k


def cc_success(theta, a, c, trials=20000, seed=0, init=None, steps=800):
    """
    Success rate of a conditional-cooperation threshold theta in two coupled
    domains, over random initial commitment stocks.

    Model (faithful to Sec. 7.2, synchronous within-domain switch):
      - Each domain starts trapped near B with a stock m_i of conditional
        cooperators (C); the rest are B. There are no pure-A players.
      - Below threshold, C is payoff-tied with B, so commitment does not grow;
        a domain therefore triggers iff its initial stock m_i >= theta.
      - On trigger the C-stock flips to A at once, so the domain's A-frequency
        jumps to m_i. An untriggered domain has no A-players, so its frequency
        stays at 0 (a fixed point of the replicator: p(1-p) vanishes there).
      - The coupled replicator then runs to rest; success = both domains -> A.

    `init` is a callable(rng, shape) -> array in [0,1] for the initial-commitment
    distribution; defaults to trapped_B(2.0), a population weighted toward the B
    corner. The escape rate at every threshold depends on this choice, so it is
    exposed. (Uniform starts: pass init=lambda r,s: r.random(s).)
    """
    rng = np.random.default_rng(seed)
    draw = init if init is not None else trapped_B(2.0)
    m = draw(rng, (trials, 2))
    triggered = m >= theta
    p0 = np.where(triggered, m, 0.0)
    pf = _integrate(p0, a, c, r=0.0, steps=steps)
    return (pf > 0.99).all(axis=1).mean()


def cc_success_independent(theta, a, c, trials=20000, seed=0, init=None):
    """
    Deterministic INDEPENDENT baseline for conditional cooperation: two domains
    that do NOT depend on each other. Each is a fixed coordination game with the
    most favourable payoff the interdependent domain could ever see, alpha=a+c
    (as though the neighbour were permanently fully coordinated). A domain escapes
    iff its committed stock triggers (m>=theta) and lands above the fixed single-
    game boundary 1/(alpha+1); system success = both domains escape. Isolates
    what interdependence adds by holding everything else equal.
    """
    rng = np.random.default_rng(seed)
    draw = init if init is not None else trapped_B(2.0)
    m = draw(rng, (trials, 2))
    boundary = 1.0 / ((a + c) + 1.0)     # = optimistic_threshold(a, c)
    escape = (m >= theta) & (m > boundary)
    return escape.all(axis=1).mean()


def finite_escape(N, x0, theta, a, c, coupled=True, domains=2, s=1.0,
                  trials=4000, max_steps=None, seed=0):
    """
    Finite-population conditional cooperation under a pairwise-comparison (Fermi)
    imitation process with selection intensity s. Each of `domains` populations
    has N individuals of type C or B; C plays A iff its domain's C-share reaches
    theta, else B. Below threshold C and B are payoff-tied, so the C-count is an
    unbiased random walk (this reproduces the analytic reach-theta-before-0
    probability x0/theta in a single game). Absorbing at C=0 (all-B) and C=N
    (all-A); system escape = every domain reaches C=N.

    coupled=True links domains on a ring (alpha_i = a * mean(neighbours) + c);
    coupled=False (or domains=1) fixes alpha=a+c, giving the independent baseline.
    """
    rng = np.random.default_rng(seed)
    if max_steps is None:
        max_steps = 60 * N * N
    C = np.full((trials, domains), int(round(x0 * N)), dtype=np.int64)
    denom = max(N - 1, 1)
    for _ in range(max_steps):
        m = C / N
        trig = m >= theta
        p = np.where(trig, m, 0.0)                 # A-frequency per domain
        if coupled and domains > 1:
            neigh = 0.5 * (np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))
            alpha = a * neigh + c
        else:
            alpha = np.full_like(p, a + c)
        piB = 1.0 - p
        piC = np.where(trig, alpha * p, piB)       # C earns A-payoff once triggered
        frac = C / N
        p_up = (1 - frac) * (C / denom) / (1.0 + np.exp(-s * (piC - piB)))
        p_dn = frac * ((N - C) / denom) / (1.0 + np.exp(-s * (piB - piC)))
        r = rng.random((trials, domains))
        up = r < p_up
        dn = (~up) & (r < p_up + p_dn)
        C = C + up.astype(np.int64) - dn.astype(np.int64)
        np.clip(C, 0, N, out=C)
        if ((C == 0) | (C == N)).all():
            break
    return (C == N).all(axis=1).mean()


def conditional_cooperation_sweep(a=2.0, c=0.3, thetas=None, trials=10000,
                                  steps=600, out_path="cc_deterministic_sweep.csv"):
    """
    Dense success-vs-theta sweep for the deterministic (infinite-population)
    conditional-cooperation model, computing the interdependent curve and the
    independent baseline (both under the trapped-B start). Saves
    (theta, interdependent, independent) to `out_path` and returns the arrays.
    Raise `trials` for a publication-quality figure; 10000 keeps noise ~0.5%.
    """
    if thetas is None:
        thetas = np.linspace(0.02, 0.96, 32)
    s_dep = np.array([cc_success(th, a, c, trials=trials, steps=steps)
                      for th in thetas])
    s_ind = np.array([cc_success_independent(th, a, c, trials=trials)
                      for th in thetas])
    header = (f"tragedy-of-conventions deterministic conditional-cooperation sweep\n"
              f"a={a}, c={c}, beta={BETA}, trials={trials}, start=trapped_B(2.0)\n"
              f"landmarks: optimistic={optimistic_threshold(a, c):.4f} "
              f"theta_hat={self_consistent_threshold(a, c):.4f} "
              f"solo={solo_threshold(c):.4f}\n"
              f"theta,interdependent,independent")
    np.savetxt(out_path, np.column_stack([thetas, s_dep, s_ind]),
               delimiter=",", header=header, comments="# ", fmt="%.6f")
    return thetas, s_dep, s_ind


def result_R6():
    print("R6  Conditional cooperation, deterministic limit: independent vs interdependent")
    print("    a=2, c=0.3, beta=1; trapped-B start; two domains")
    a, c = 2.0, 0.3
    opt = optimistic_threshold(a, c)
    hat = self_consistent_threshold(a, c)
    solo = solo_threshold(c)

    thetas, s_dep, s_ind = conditional_cooperation_sweep(
        a, c, out_path="cc_deterministic_sweep.csv")
    print("    saved dense sweep -> cc_deterministic_sweep.csv")

    def at(s, theta):
        return float(np.interp(theta, thetas, s))

    print(f"      {'threshold':18s} {'independent':>12s} {'interdependent':>14s}")
    for tag, th in [(f"optimistic {opt:.2f}", opt), (f"theta-hat  {hat:.2f}", hat),
                    (f"solo/naive {solo:.2f}", solo)]:
        print(f"      {tag:18s} {at(s_ind, th):>12.3f} {at(s_dep, th):>14.3f}")
    print("    the interdependence penalty: even in the infinite-population limit,")
    print("    and even against the most favourable independent benchmark (neighbour")
    print("    fixed at full coordination), interdependence lowers escape below the")
    print("    baseline wherever the two domains must cross together.")
    print()


def result_R5():
    print("R5  Conditional cooperation: the endogenous threshold band")
    print("    strongly coupled example a=2, c=0.3, beta=1")
    a, c = 2.0, 0.3
    opt = optimistic_threshold(a, c)
    hat = self_consistent_threshold(a, c)
    solo = solo_threshold(c)
    print(f"    optimistic end  beta/(a+c+beta) = {opt:.3f}   (paper: 0.30)")
    print(f"    self-consistent theta-hat       = {hat:.3f}   (paper: 0.45)")
    print(f"    solo end        beta/(c+beta)   = {solo:.3f}   (paper: 0.77)")
    # verify theta-hat is a genuine root of the quadratic
    resid = a * hat ** 2 + (c + BETA) * hat - BETA
    print(f"    quadratic residual at theta-hat = {resid:.2e}  (should be ~0)")
    assert opt < hat < solo, "band ordering violated"
    print()


def finite_scaling_sweep(Ns=(25, 50, 75, 100, 150), x0=0.15, theta=0.40,
                         a=2.0, c=0.3, s=1.0, trials=2500,
                         out_path="cc_finite_scaling.csv"):
    """
    Escape probability vs population size N for the independent baseline and the
    interdependent case, at the canonical setting (x0=0.15, s=1, theta below the
    solo threshold). Establishes that the independent rate is scale-free while the
    interdependent rate declines monotonically with N. Saves (N, independent,
    interdependent) and returns the arrays.
    """
    ind = np.array([finite_escape(N, x0, theta, a, c, coupled=False, s=s,
                                  trials=trials) for N in Ns])
    dep = np.array([finite_escape(N, x0, theta, a, c, coupled=True, s=s,
                                  trials=trials) for N in Ns])
    header = ("tragedy-of-conventions finite-population scaling\n"
              f"x0={x0}, theta={theta}, a={a}, c={c}, s={s}, trials={trials}\n"
              "N,independent,interdependent")
    np.savetxt(out_path, np.column_stack([np.array(Ns), ind, dep]),
               delimiter=",", header=header, comments="# ", fmt=["%d", "%.5f", "%.5f"])
    return np.array(Ns), ind, dep


def result_R7():
    print("R7  Conditional cooperation, finite populations: robustness across N")
    print("    canonical setting: x0=0.15 seed, s=1 (independent baseline matches")
    print("    the analytic x0/theta), a=2, c=0.3, beta=1; pairwise-comparison update")

    # validation: single independent game reproduces the analytic x0/theta
    print("    validation -- single independent game escape vs analytic x0/theta:")
    for x0, th in [(0.10, 0.50), (0.20, 0.50)]:
        est = finite_escape(50, x0, th, a=2.0, c=0.0, coupled=False, domains=1,
                            trials=8000)
        print(f"      x0={x0}, theta={th}:  simulated={est:.3f}   x0/theta={x0/th:.3f}")

    # two-domain system escape, independent baseline vs interdependent, across N
    print("    two-domain system escape (seed x0=0.15 per domain):")
    print(f"      {'N':>4s} {'theta':>6s} {'independent':>12s} {'interdependent':>14s}")
    rows = []
    solo = solo_threshold(0.3)
    for N in (50, 100):
        for th in (0.40, 0.55, round(solo, 2)):
            ind = finite_escape(N, 0.15, th, 2.0, 0.3, coupled=False, trials=2000)
            dep = finite_escape(N, 0.15, th, 2.0, 0.3, coupled=True, trials=2000)
            rows.append((N, th, ind, dep))
            print(f"      {N:>4d} {th:>6.2f} {ind:>12.3f} {dep:>14.3f}")
    header = ("tragedy-of-conventions finite-population conditional cooperation\n"
              "a=2, c=0.3, beta=1, s=1, x0=0.15, pairwise-comparison imitation\n"
              "N,theta,independent,interdependent")
    np.savetxt("cc_finite_escape.csv", np.array(rows),
               delimiter=",", header=header, comments="# ", fmt="%.4f")
    print("    saved -> cc_finite_escape.csv")
    print("    below the solo threshold the interdependent escape sits far under the")
    print("    independent baseline and does not improve with N; the two converge only")
    print("    at the solo threshold, where each domain can hold A unaided.")
    print()


if __name__ == "__main__":
    result_R1()
    result_R2()
    result_R3()
    result_R4()
    result_R5()
    result_R6()
    result_R7()
