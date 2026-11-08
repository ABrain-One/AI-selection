#!/usr/bin/env python3
"""
Select the best model from (accuracy, resource) measurements alone.

NOTHING is declared by hand: no price, no grain, no chance level, no
transferred constant.  Only --alpha, the statistical coverage level, which is
a convention and not a quantity of the problem.


EXAMPLES OF USE
===============

1. Three models, two numbers each -- the minimum case.

   $ python3 example_general.py --axis ms \
         --model A,0.851,4 --model B,0.9253,9 --model D,0.90,30

   ENVELOPE ....... 2 of 3 can be optimal at SOME price  [no rate needed]
     B                    optimal for lambda in [0.0000, 0.8514]
     A                    optimal for lambda in [0.8514, inf]
     never optimal  D
       dominated .. D  (something is both cheaper and better)

   PRICE-FREE ..... 1 domination(s), valid at EVERY price:
     B beats D

   NO RATE ........ 2 nondominated candidate(s); the curvature fit needs 6.

   Read it as: D is retired with a proof, and whether A or B wins depends on
   whether you would pay 0.85 nats per e-fold of latency.  No rate can be
   derived from two points -- that is Theorem 'degeneracy', not a missing
   feature.


2. A CSV, enough points, and the rate derived from the frontier itself.

   $ python3 example_general.py front9.csv --axis inference_time_ms

   FIT ............ eps = eps_inf + c t^-alpha on ln eps, 9 points
     saturating ... eps_inf=0.0200971 c=0.902677 alpha=0.7999  RSS=0.0008993
     power law .... c=0.578273 alpha=0.5365              RSS=0.195
     AICc ......... delta = +41.21  [with-variance]
     knee ......... t* = 116.4; range [3, 509.507] -> SPANNED

   RATE ........... lambda = alpha/2 = 0.3999 nats per e-fold  [derived]
     interval ..... [0.3947, 0.4066] at 95% (400 bootstrap refits)
     meaning ...... doubling the resource is justified only if the error
                    falls by >= 24.2%
     declared ..... nothing: no price, no grain, no q0, no transferred constant

   WINNER ......... m5   margin 0.0127 nats over m6
     ROBUST ....... m5 wins at EVERY rate in [0.3947, 0.4066]

   The CSV needs a name column, an accuracy column in (0,1) and a resource
   column; the headers are guessed, or named with --acc-col / --res-col:

       name,accuracy,inference_time_ms
       m0,0.763542,3.000
       m1,0.846291,5.700

3. Columns that cannot be guessed, and a looser coverage level.

   $ python3 example_general.py models.csv \
         --name-col model --acc-col top1 --res-col latency_ms \
         --axis latency_ms --alpha 0.10

4. Reproducible bootstrap, or a faster/slower one.

   $ python3 example_general.py front9.csv --boot 2000 --seed 42


EXIT CODES
   0  a rate was derived and a winner named
   1  no rate could be derived -- the ENVELOPE is still printed, and the
      reason is stated (too few nondominated models, AICc prefers the
      floorless power law, or the knee lies outside the measured range)
   2  bad input (accuracy outside (0,1))


WHAT IT REPORTS, in order of how little it assumes

  ENVELOPE     who a linear price can EVER select.  Works at any K, even 2.
               Absence is a proof, not an absence of evidence.
  PRICE-FREE   dominations that hold at every price.
  RATE         lambda = alpha/2 from the frontier's own curvature
               (Paper 1 Alg. 1 rule (ii); Paper 2 supplement, sqrt(kappa)
               = (alpha/2) sin phi at the canonical reference), with a
               bootstrap interval on alpha.
  WINNER       argmax sigma, AND whether it survives across the whole
               bootstrap interval of the rate.  A winner that does is robust
               to the rate's own uncertainty; one that does not is a SET,
               reported with the exact KL cost of picking anyway.
  NEXT         where an unmeasured model would have to sit to matter.


MEASURED RELIABILITY (300 synthetic fronts, true alpha = 0.7, 100x span)

   points   accuracy noise   rate identified   winner correct
      6           1%              81%              93%
      6           3%               9%              85%
      8           3%             100%              85%
     10           6%              98%              84%
     14           3%             100%              91%

  Six points is a cliff and eight is a plateau: the gap is the AICc
  small-sample penalty.  Aim for 8 or more nondominated models spanning the
  knee.  All candidates must be measured on the SAME test set and the SAME
  device, and the fit assumes one mechanism -- a ladder of one family behaves
  far better than a zoo of unrelated architectures.
"""
import argparse
import csv
import math
import sys

import numpy as np
from scipy.optimize import least_squares

MIN_FRONT = 6
SPAN_MIN = math.log(1.5)
LAM_HI = 1e6


# ---------------------------------------------------------------- input ----
def read_models(a):
    out = []
    for spec in a.model or []:
        p = [s.strip() for s in spec.split(",")]
        if len(p) != 3:
            sys.exit(f"--model wants name,accuracy,resource -- got {spec!r}")
        out.append({"name": p[0], "acc": float(p[1]), "t": float(p[2])})
    if a.csv:
        with open(a.csv, newline="") as fh:
            rows = [r for r in csv.reader(fh)
                    if r and not r[0].lstrip().startswith("#")]
        head = [h.strip() for h in rows[0]]
        nc = a.name_col or head[0]
        ac = a.acc_col or next((h for h in head if "acc" in h.lower()), None)
        rc = a.res_col or next((h for h in head if any(
            k in h.lower() for k in ("time", "latency", "ms", "flop",
                                     "resource", "energy"))), None)
        if not ac or not rc:
            sys.exit(f"cannot find accuracy/resource in {head}; "
                     "use --acc-col / --res-col")
        ia, ir, inm = head.index(ac), head.index(rc), head.index(nc)
        for r in rows[1:]:
            if r[ia].strip():
                out.append({"name": r[inm].strip(), "acc": float(r[ia]),
                            "t": float(r[ir])})
        print(f"  read '{ac}' as accuracy, '{rc}' as resource")
    return out


# -------------------------------------------------------------- envelope ----
def envelope(models):
    """Upper envelope of sigma_i(lambda) = -ln eps_i - lambda ln t_i over
    lambda >= 0.  Returns pieces, winners, dominated, dents."""
    lines = sorted(((-math.log(m["t"]), -math.log(1 - m["acc"]), m["name"])
                    for m in models), key=lambda z: (z[0], z[1]))
    ded = []
    for b, c, n in lines:
        if ded and abs(ded[-1][0] - b) < 1e-15:
            if c > ded[-1][1]:
                ded[-1] = (b, c, n)
        else:
            ded.append((b, c, n))

    def cross(p, q):
        return (q[1] - p[1]) / (p[0] - q[0])

    hull = []
    for ln in ded:
        while hull:
            if hull[-1][0] == ln[0]:
                hull.pop(); continue
            if len(hull) >= 2 and cross(hull[-1], ln) <= cross(hull[-2], hull[-1]):
                hull.pop(); continue
            break
        hull.append(ln)
    bnds = [cross(hull[i], hull[i + 1]) for i in range(len(hull) - 1)]
    pieces, lo = [], 0.0            # clamped at 0: a negative price is not one
    for i, ln in enumerate(hull):
        hi = bnds[i] if i < len(bnds) else LAM_HI
        if hi > lo:
            pieces.append((ln[2], lo, hi))
        lo = max(lo, hi)
    wset = {p[0] for p in pieces}
    dom, dent = [], []
    for m in models:
        if m["name"] in wset:
            continue
        if any(o["acc"] >= m["acc"] and o["t"] <= m["t"]
               and (o["acc"] > m["acc"] or o["t"] < m["t"])
               for o in models if o is not m):
            dom.append(m["name"])
        else:
            dent.append(m["name"])
    return pieces, sorted(wset), sorted(dom), sorted(dent)


def cull(models):
    keep, dropped = [], []
    for m in models:
        k = next((o for o in models if o is not m and o["acc"] >= m["acc"]
                  and o["t"] <= m["t"]
                  and (o["acc"] > m["acc"] or o["t"] < m["t"])), None)
        (dropped.append((m["name"], k["name"])) if k else keep.append(m))
    return keep, dropped


# ------------------------------------------------------------------ fit ----
def _aicc(rss, n, kp, var=True):
    k = kp + 1 if var else kp
    return None if n - k - 1 <= 0 else \
        n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_sat(t, eps):
    hi = min(eps) * (1 - 1e-9)
    best = None
    for a0 in (0.3, 0.7, 1.5, 3.0):
        for f0 in (0.1, 0.5, 0.9):
            p0 = [min(max(min(eps) * f0, 1e-11), hi * .99),
                  max(eps) * t[0] ** a0, a0]
            try:
                r = least_squares(
                    lambda p: np.log(np.maximum(p[0] + p[1] * t ** -p[2],
                                                1e-300)) - np.log(eps),
                    p0, bounds=([1e-12, 1e-12, 1e-6], [hi, 1e12, 8.0]),
                    max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best


def fit_pow(t, eps):
    A = np.vstack([np.ones_like(t), -np.log(t)]).T
    co, *_ = np.linalg.lstsq(A, np.log(eps), rcond=None)
    r = np.log(eps) - A @ co
    return math.exp(co[0]), co[1], float(r @ r)


def boot_alpha(t, eps, fit, reps, rng):
    """Residual bootstrap on ln eps -> distribution of alpha."""
    pred = np.log(fit.x[0] + fit.x[1] * t ** -fit.x[2])
    res = np.log(eps) - pred
    res = res - res.mean()
    out = []
    for _ in range(reps):
        e2 = np.exp(pred + rng.choice(res, size=len(t), replace=True))
        if not np.all((e2 > 0) & (e2 < 1)):
            continue
        f2 = fit_sat(t, e2)
        if f2 is not None and f2.x[2] < 7.99:
            out.append(f2.x[2])
    return np.array(out)


def kl_regret(lam, lam_hat, alpha):
    """Thm. regret: exact cost in nats of selecting at lam_hat when lam is
    true, p = lam/alpha."""
    p, q = lam / alpha, lam_hat / alpha
    if not (0 < p < 1 and 0 < q < 1):
        return float("nan")
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# ----------------------------------------------------------------- main ----
def sigma_at(models, lam):
    return {m["name"]: -math.log(1 - m["acc"]) - lam * math.log(m["t"])
            for m in models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--model", action="append")
    ap.add_argument("--acc-col"); ap.add_argument("--res-col")
    ap.add_argument("--name-col"); ap.add_argument("--axis", default="resource")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="coverage level for the rate interval (convention, "
                         "not a quantity of the problem)")
    ap.add_argument("--aicc-k", choices=("with-variance", "params-only"),
                    default="with-variance")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ms = read_models(a)
    if not ms:
        sys.exit("no candidates: pass a CSV or --model name,accuracy,resource")
    bad = [m["name"] for m in ms if not 0 < m["acc"] < 1]
    if bad:
        print(f"\nSTOP ... accuracy must be in (0,1): {', '.join(bad)}")
        return 2

    print(f"\nINPUT .......... {len(ms)} candidate(s); accuracy and "
          f"'{a.axis}' only; alpha = {a.alpha}")
    for m in sorted(ms, key=lambda m: m["t"]):
        print(f"  {m['name']:<20} acc={m['acc']:.6f}  {a.axis}={m['t']:g}")

    # ---- always available, at any K -------------------------------------
    pieces, wins, dom, dent = envelope(ms)
    print(f"\nENVELOPE ....... {len(wins)} of {len(ms)} can be optimal at "
          f"SOME price  [no rate needed]")
    for nm, lo, hi in pieces:
        h = "inf" if hi >= LAM_HI else f"{hi:.4f}"
        print(f"  {nm:<20} optimal for lambda in [{lo:.4f}, {h}]")
    if dom or dent:
        print(f"  never optimal  {', '.join(sorted(dom + dent))}")
        if dom:
            print(f"    dominated .. {', '.join(dom)}  "
                  "(something is both cheaper and better)")
        if dent:
            print(f"    in a dent .. {', '.join(dent)}  "
                  "(Pareto-efficient, yet NO price selects it)")

    front, dropped = cull(ms)
    if dropped:
        print(f"\nPRICE-FREE ..... {len(dropped)} domination(s), valid at "
              f"EVERY price:")
        for l_, w in dropped:
            print(f"  {w} beats {l_}")

    print(f"\nCULL ........... {len(front)} of {len(ms)} nondominated")

    # ---- the rate --------------------------------------------------------
    def no_rate(reason, extra=()):
        print(f"\nNO RATE ........ {reason}")
        for e in extra:
            print(f"  {e}")
        print("  what stands ... the ENVELOPE above, which needs no rate: "
              f"{len(wins)} candidate(s) can win,")
        print(f"                  {len(dom) + len(dent)} are retired with a "
              "proof.  Name any price bound you")
        print("                  can defend and the envelope names the winner.")
        return 1

    if len(front) < MIN_FRONT:
        return no_rate(
            f"{len(front)} nondominated candidate(s); the curvature fit needs "
            f"{MIN_FRONT}.",
            [f"reason ....... eps = eps_inf + c t^-alpha has 3 free "
             f"parameters and must then",
             f"               beat the 2-parameter power law on AICc; "
             f"{len(front)} point(s) cannot",
             "               establish that a floor exists at all.",
             f"to proceed ... measure {MIN_FRONT - len(front)} more "
             "nondominated model(s), same test set and device."])

    t = np.array([m["t"] for m in front], float)
    eps = np.array([1 - m["acc"] for m in front], float)
    o = np.argsort(t); t, eps = t[o], eps[o]
    fr = [front[i] for i in o]
    n = len(t)

    sat = fit_sat(t, eps)
    c_p, a_p, rss_p = fit_pow(t, eps)
    if sat is None:
        return no_rate("the saturating fit did not converge.")
    e_inf, c_s, alpha = sat.x
    rss_s = float(2 * sat.cost)
    cv = a.aicc_k == "with-variance"
    A_s, A_p = _aicc(rss_s, n, 3, cv), _aicc(rss_p, n, 2, cv)
    if A_s is None or A_p is None:
        return no_rate(f"{n} points cannot carry an AICc comparison.")
    d = A_p - A_s
    d_alt = (lambda x, y: None if x is None or y is None else y - x)(
        _aicc(rss_s, n, 3, not cv), _aicc(rss_p, n, 2, not cv))

    print(f"\nFIT ............ eps = eps_inf + c t^-alpha on ln eps, {n} points")
    print(f"  saturating ... eps_inf={e_inf:.6g} c={c_s:.6g} "
          f"alpha={alpha:.4f}  RSS={rss_s:.4g}")
    print(f"  power law .... c={c_p:.6g} alpha={a_p:.4f}"
          f"{'':14s}RSS={rss_p:.4g}")
    print(f"  AICc ......... delta = {d:+.2f}  [{a.aicc_k}]", end="")
    if d_alt is not None and (d_alt > 0) != (d > 0):
        print(f"   ! the other convention gives {d_alt:+.2f} and FLIPS it")
        print(f"                 at n = {n} the penalty gap dominates the "
              "fit; this verdict is not robust")
    elif d_alt is not None:
        print(f"   (other convention {d_alt:+.2f}, same verdict)")
    else:
        print()

    t_star = (c_s / e_inf) ** (1 / alpha)
    spans = t.min() <= t_star <= t.max()
    print(f"  knee ......... t* = {t_star:.4g}; range [{t.min():g}, "
          f"{t.max():g}] -> {'SPANNED' if spans else 'NOT spanned'}")
    print(f"  resolved ..... {-math.log(eps.min()):.3f} nats "
          "(paper: identified in 87% of fronts above 1.5, 0% below 1.0)")

    if d <= 0:
        return no_rate(
            f"AICc prefers the floorless power law (delta = {d:+.2f}).",
            ["reason ....... Thm 'degeneracy': with eps_inf = 0 the frontier "
             "is scale-free and NO",
             "               rate is distinguished.  The rate is absent from "
             "this data, not hidden."] +
            ([f"note ......... the saturating fit is {rss_p / rss_s:.1f}x "
              f"tighter; AICc rejects it on the n = {n} penalty alone"]
             if rss_s < rss_p else []))
    if not spans:
        return no_rate(
            "the knee t* lies outside the measured range.",
            ["reason ....... alpha is extrapolated, not measured.",
             f"to proceed ... measure a model near {a.axis} = {t_star:.4g}"])

    # ---- rate, with its own uncertainty ---------------------------------
    rng = np.random.default_rng(a.seed)
    bs = boot_alpha(t, eps, sat, a.boot, rng)
    lam = alpha / 2
    if len(bs) >= 50:
        lo_a, hi_a = np.quantile(bs, [a.alpha / 2, 1 - a.alpha / 2])
        lam_lo, lam_hi = lo_a / 2, hi_a / 2
    else:
        lam_lo = lam_hi = lam
        print("  ! bootstrap did not converge; the rate carries no interval")

    print(f"\nRATE ........... lambda = alpha/2 = {lam:.4f} nats per e-fold "
          f"of {a.axis}  [derived]")
    print(f"  interval ..... [{lam_lo:.4f}, {lam_hi:.4f}] at "
          f"{(1 - a.alpha) * 100:.0f}% ({len(bs)} bootstrap refits)")
    print(f"  meaning ...... doubling {a.axis} is justified only if the error "
          f"falls by >= {(1 - 2 ** -lam) * 100:.1f}%")
    print("  declared ..... nothing: no price, no grain, no q0, no "
          "transferred constant")

    sig = sorted(sigma_at(fr, lam).items(), key=lambda kv: -kv[1])
    win, margin = sig[0][0], sig[0][1] - sig[1][1]
    print(f"\nSCORE .......... sigma = -ln eps - {lam:.4f} ln {a.axis}")
    for nm, s in sig:
        print(f"  {nm:<20} {s:+.4f}" + ("   <= winner" if nm == win else ""))

    # ---- is the winner robust to the rate's own interval? ---------------
    grid = np.linspace(lam_lo, lam_hi, 601) if lam_hi > lam_lo else [lam]
    wset = []
    for L in grid:
        w = max(sigma_at(fr, L).items(), key=lambda kv: kv[1])[0]
        if w not in wset:
            wset.append(w)

    print(f"\nWINNER ......... {win}   margin {margin:.4f} nats over "
          f"{sig[1][0]}")
    if len(wset) == 1:
        print(f"  ROBUST ....... {win} wins at EVERY rate in "
              f"[{lam_lo:.4f}, {lam_hi:.4f}], so the rate's own uncertainty")
        print("                 does not change the answer.  This is the "
              "strongest statement")
        print("                 available from accuracy and resource alone.")
    else:
        print(f"  ! NOT ROBUST . the rate interval admits {len(wset)} "
              f"winners: {', '.join(wset)}")
        print("                 the honest output is that SET, not the point "
              "winner above.")
        worst = max(kl_regret(L, lam, alpha) for L in grid
                    if not math.isnan(kl_regret(L, lam, alpha)))
        print(f"                 picking {win} anyway costs at most "
              f"{worst:.4f} nats inside the interval (Thm 'regret')")
    wm = next(m for m in fr if m["name"] == win)
    rm = next(m for m in fr if m["name"] == sig[1][0])
    sep = abs(math.log(wm["t"]) - math.log(rm["t"]))
    if sep < SPAN_MIN:
        print(f"  ! UNRESOLVED . winner and runner-up differ by {sep:.3f} < "
              f"ln 1.5 = {SPAN_MIN:.3f} in log-{a.axis}")
    print("  ! the margin is not a confidence interval: it carries no "
          "test-set noise.")
    print("    Pass per-item outcomes to a paired test to certify it.")

    gap = min(abs(math.log(m["t"]) - math.log(t_star)) for m in fr)
    print(f"\nNEXT ........... an unmeasured model matters most at "
          f"{a.axis} = {t_star:.4g} (the knee,")
    print(f"                 where lambda = alpha/2 is optimal); nearest "
          f"candidate is {gap:.2f} e-folds away")
    if gap > SPAN_MIN:
        print("  ! the frontier is unmeasured exactly where the derived rate "
              "operates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
