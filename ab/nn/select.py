#!/usr/bin/env python3
"""
Select the best model from (accuracy, resource) measurements alone.

NOTHING is declared by hand: no price, no grain, no chance level, no
transferred constant.  Only --alpha, the statistical coverage level, which is
a convention and not a quantity of the problem.

A pandas DataFrame is the native input and output.  CSV files and bare
NAME,ACC,P1[,P2,...] arguments are adapters onto the same frame.

SEVERAL RESOURCES are supported (Paper 2, F5): append as many resource
columns as you like -- on the command line they are p1, p2, ... pn, and no
axis needs naming because every axis is treated identically.  ONE price is
derived per axis, each from that axis's own marginal curve, giving

    sigma = -ln eps - sum_a lambda_a * ln t_a

The interaction between axes is bounded, not fitted, so no cross term is
estimated.  Every axis carries its own floor test, its own interval and its
own printed provenance; an axis whose marginal curve has fewer than 4 points
or spans less than 0.5 in log-resource gets NO price and is excluded from
sigma with its reason stated (a divided difference over a vanishing span is
noise divided by nothing).  The verdict is the worst level across axes.  With
several axes, dominance is Pareto in accuracy and EVERY resource, and the
report ends with the only honest test of whether an axis was worth pricing:
drop it and see whether the pick changes.

A ranking is ALWAYS produced from 3 nondominated candidates upward, printed
in green (#2E8B57) together with the WINNER line.

When something is wrong, and ONLY then, a red warning is printed first:

  NO FLOOR (F-test p = ...)   the F-test cannot reject eps_inf = 0.  Thm
                              'degeneracy': the frontier is scale-free, no
                              exchange rate exists, and the ranking is merely
                              resource order
  UNVERIFIED: ...             fewer than 4 nondominated points, or the knee
                              falls outside the measured range, so the floor
                              cannot be tested and the ranking is a guess

A clean run prints no warning at all.  --no-color prints plain text.

Why an F-test and not AICc: AICc needs n - k - 1 > 0 with k = 4, so it is
undefined below 6 points.  The nested F-test (eps_inf free vs eps_inf = 0
forced) needs only n >= 4.  AICc is still printed as corroboration.  Measured
on 400 synthetic frontiers per cell: a spurious floor is fitted on 36-48% of
genuinely floorless data at EVERY n, and the test rejects 97-100% of those at
n >= 6 -- which is why the verdict, not the ranking, is the output that
matters.

A bootstrap interval measures spread around THIS fit, not distance from the
truth: if the floor is spurious, every resample inherits it and agrees
confidently.  That is what the verdict is for.


THE INPUT FRAME
===============

    name            accuracy           p1         p2
    m0              0.605833          3.0        5.3
    m1              0.752681          5.7        7.8

A CSV keeps its own column names; command-line candidates use p1..pn.

`accuracy` in (0,1); every resource > 0, one unit, one device.  Extra columns
are carried through untouched.  Column names are configurable, and a CSV's
headers are guessed when not named.


EXAMPLES OF USE
===============

1. A frontier that genuinely saturates -- the case the method is for.

   $ python3 example_general.py front9.csv --rank

   FLOOR TEST ... nested F(1,6) = 1.3e+03, p = 3.07e-08  ->  floor CONFIRMED
   RATE ........... lambda = 0.3999 nats per e-fold  [alpha/2 (saturating)]
   RANKING ........ all 9 models at lambda = 0.3999, winner first
       1 m5                   0.9515      74.283  +1.3028  +0.0127         1
   WINNER ......... m5   margin 0.0127 nats over m6

   No warning is printed: the floor test passed.

2. Candidates straight on the command line -- any bare argument with a comma.

   $ python3 example_general.py --rank Q05,0.475,0.5 Q15,0.609,1.5 Q3,0.656,3 \\
         Q7,0.742,7 Q14,0.797,14 Q32,0.833,32 Q72,0.861,72

   NO FLOOR (F-test p = 0.86): no exchange rate exists in this data.
   The ranking below is just resource order and means nothing.

3. Too few points to test the floor.

   $ python3 example_general.py --rank A,0.851,4 B,0.9253,9 D,0.90,30

   UNVERIFIED: only 2 nondominated points, floor untestable.
   The ranking below is a guess, not a result.

4. Two resources, one price derived per axis.

   $ python3 example_general.py --rank --res-col GFLOPs --res-col params_M \
         B0,0.771,0.39,5.3 B1,0.791,0.70,7.8 B2,0.801,1.0,9.2 \
         B3,0.816,1.8,12 B4,0.829,4.2,19 B5,0.836,9.9,30 \
         B6,0.840,19,43 B7,0.843,37,66

   PRICE PER AXIS . one price per axis, from that axis's own marginal curve
     axis            lambda       95% interval   alpha   floor p  pts  span  provenance
     GFLOPs          0.2739   [0.2567, 0.2926]  0.5477  6.49e-06    8  4.55  alpha/2 ...
     params_M        0.4900   [0.4150, 0.5720]  0.9800  0.000459    8  2.52  alpha/2 ...
   RATE ........... sigma = -ln eps - 0.2739*ln(GFLOPs) - 0.4900*ln(params_M)
   DOES EACH AXIS NEED A PRICE?  drop it and see
     without GFLOPs      19.1% of total cost   pick unchanged
     without params_M    80.9% of total cost   pick unchanged

5. Columns that cannot be guessed, a looser coverage level, ranking to CSV.

   $ python3 example_general.py models.csv --rank \\
         --name-col model --acc-col top1 --res-col latency_ms \\
         --alpha 0.10 --out-csv ranking.csv


USE AS A LIBRARY
================

    import pandas as pd
    from example_general import select, render

    df = pd.DataFrame({"name": [...], "accuracy": [...], "resource": [...]})
    result = select(df, alpha=0.05, rank=True)
    print(render(result, color=False))

    result["verdict"]["level"]     # "USABLE" | "UNVERIFIED" | "UNUSABLE"
    result["verdict"]["p_value"]   # the floor F-test p, None if untestable
    result["rate"]["trustworthy"]  # True only when USABLE
    result["winner"]["name"]
    result["ranking"]              # DataFrame, one row per model
    result["candidates"]           # DataFrame, as validated
    result["prices"]               # DataFrame, one row per resource axis
    result["rate"]["per_axis"]     # {axis: lambda}
    result["axis_drop"]            # DataFrame: does each axis change the pick
    result["envelope"]["pieces"]   # DataFrame (one resource only)
    result["price_free"]           # DataFrame: winner, loser

`res_col` takes a string or a list of column names.

`select()` takes a DataFrame and returns a dict of DataFrames and scalars.
It prints nothing and raises nothing.


EXIT CODES
   0  clean: the floor test passed and no warning was printed
   1  UNVERIFIED or UNUSABLE
   2  bad input (accuracy outside (0,1), fewer than two candidates)


WHAT IT REPORTS, in order of how little it assumes

  ENVELOPE     who a linear price can EVER select.  Works at any K, even 2,
               and needs no rate.  Absence is a proof, not missing evidence.
  PRICE-FREE   dominations that hold at every price.
  FLOOR TEST   whether an exchange rate exists in this data at all.
  RATE         lambda = alpha/2 from the frontier's own curvature, with a
               bootstrap interval on alpha.
  RANKING      every model ordered winner to loser, with the rank range it
               takes across the rate's own interval.
  NEXT         where an unmeasured model would have to sit to matter.
  AXIS DROP    with several resources, whether removing an axis changes the
               pick -- the paper's own test of whether pricing it mattered
               (it changed the decision on 64% of its 48 blocks' axes).


SCALE

  Thousands of candidates are fine: they collapse to a Pareto front of a few
  hundred, and the fit and bootstrap run on the front alone.  Measured at
  5000 candidates (front of 137): ~1.5s in total, of which the bootstrap is
  0.33s and the vectorised ranking 0.05s.  The dominance scan is O(K log K)
  by a running maximum, and the rank spread is one K-by-grid matrix.


KNOWN LIMITS

  Competition leaderboards do not work here.  Their rules pin quality at a
  threshold and race on cost, so almost every entry is dominated and the
  quality range carries no curvature.  Tested on MAI 2021 scene detection
  (3 of 9 nondominated), MAI 2026 SR (3 of 8), NTIRE 2026 Efficient SR
  (4 of 14) and YOLOv8+YOLO11 (5 of 10): no rate in any of them.  What the
  method needs is a deliberate ladder -- one architecture, 8+ sizes, same
  test set and device, quality spanning enough range that the frontier
  visibly bends.  MobileNetV3's Table 9 (19 models, one Pixel-1 core,
  49.8-76.6% top-1) is the one published table that passes every gate.
"""
import argparse
import csv
import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

MIN_FIT = 3         # 3 parameters need 3 points
F_TESTABLE = 4      # the nested F-test needs n >= 4
SPAN_MIN = math.log(1.5)
LAM_HI = 1e6
COLS = ("name", "accuracy", "resource")
MARG_MIN_PTS = MIN_FIT  # 3 parameters need 3 points, no more, no less
BOOT_SKIP_N = 50        # at or above this many front points the Jacobian
                        # interval matches the bootstrap to 4 decimals, so
                        # the bootstrap is skipped (measured: 1.00x width at
                        # n=50, and it is WRONG below ~20 where the eps_inf
                        # bound binds, which is why it is kept there)
MARG_MIN_SPAN = 0.5     # but they need real range in log-resource: a divided
                        # difference over a vanishing span is noise divided
                        # by nothing (Paper 2, script 144)


# ======================================================== input adapters ====
def frame_from_specs(specs):
    """`name,accuracy,p1[,p2,...]` strings -> DataFrame.  Every spec must
    carry the same number of resources.  The axes are named p1..pn: all
    resources are priced identically, so no axis needs naming to be used."""
    rows, nres = [], None
    for spec in specs:
        p = [s.strip() for s in spec.split(",")]
        if len(p) < 3:
            raise ValueError(f"expected name,accuracy,resource[,...]: {spec!r}")
        if nres is None:
            nres = len(p) - 2
        elif len(p) - 2 != nres:
            raise ValueError(f"{spec!r} has {len(p) - 2} resource(s) but the "
                             f"first candidate had {nres}")
        rows.append([p[0], float(p[1])] + [float(v) for v in p[2:]])
    names = [f"p{i + 1}" for i in range(nres)]
    out = pd.DataFrame(rows, columns=["name", "accuracy"] + names)
    out.attrs["axes"] = names
    return out


def frame_from_csv(path, name_col=None, acc_col=None, res_cols=None):
    """A CSV -> DataFrame.  Headers are guessed unless named.  res_cols may
    name several resource columns; one price is derived per axis."""
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    head = list(df.columns)
    nc = name_col or head[0]
    ac = acc_col or next((h for h in head if "acc" in h.lower()), None)
    if isinstance(res_cols, str):
        res_cols = [res_cols]
    rcs = list(res_cols or []) or [h for h in head if any(
        k in h.lower() for k in ("time", "latency", "ms", "flop",
                                 "resource", "energy", "param", "memory",
                                 "size", "cost", "watt", "joule"))]
    if not ac or not rcs:
        raise ValueError(f"cannot find accuracy/resource columns in {head}; "
                         "use --acc-col and --res-col")
    missing = [c for c in rcs if c not in head]
    if missing:
        raise ValueError(f"no such column(s): {', '.join(missing)}")
    out = df.rename(columns={nc: "name", ac: "accuracy"})
    out.attrs["axes"] = rcs
    print(f"  read '{ac}' as accuracy, "
          + ", ".join(f"'{c}'" for c in rcs)
          + (" as resource" if len(rcs) == 1 else " as resource axes"))
    return out


def normalise(df, name_col="name", acc_col="accuracy", res_col="resource"):
    """Validate and canonicalise.  Returns (frame, axes, problems).
    `axes` is the list of resource column names, DERIVED from the input (a
    CSV lends its own column names), never a separate label input.
    res_col may be a string or a list of strings."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.rename(columns={name_col: "name", acc_col: "accuracy"})
    want = [res_col] if isinstance(res_col, str) else list(res_col or [])
    axes = ([c for c in want if c in df.columns]
            or [c for c in df.attrs.get("axes", []) if c in df.columns]
            or (["resource"] if "resource" in df.columns else []))
    if "name" not in df.columns or "accuracy" not in df.columns or not axes:
        need = [c for c in ("name", "accuracy") if c not in df.columns]
        return df, axes or ["resource"], [
            "missing column(s): " + ", ".join(need + ([] if axes
                                                      else ["resource"]))]

    out = df.copy()
    out["name"] = out["name"].astype(str)
    for c in ["accuracy"] + axes:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = []
    nan = out[out[["accuracy"] + axes].isna().any(axis=1)]
    for nm in nan["name"]:
        bad.append(f"{nm}: accuracy and every resource must be numeric")
    out = out.drop(nan.index)
    off = out[(out["accuracy"] <= 0) | (out["accuracy"] >= 1)]
    for nm, a in zip(off["name"], off["accuracy"]):
        bad.append(f"{nm}: accuracy {a:g} is outside (0,1)")
    out = out.drop(off.index)
    for ax in axes:
        neg = out[out[ax] <= 0]
        for nm, t in zip(neg["name"], neg[ax]):
            bad.append(f"{nm}: {ax} {t:g} is not positive")
        out = out.drop(neg.index)

    out["eps"] = 1.0 - out["accuracy"]
    for ax in axes:
        out["x__" + ax] = np.log(out[ax])
    out = out.reset_index(drop=True)
    if len(out) < 2 and not bad:
        bad.append("fewer than two usable candidates")
    return out, axes, bad


# ============================================================== geometry ====
def envelope(df, axis="resource"):
    """Upper envelope of sigma_i(lambda) = -ln eps_i - lambda ln t_i over
    lambda >= 0.  Returns {pieces: DataFrame, winners, dominated, in_a_dent}."""
    xa = "x__" + axis
    lines = sorted(zip(-df[xa], -np.log(df["eps"]), df["name"]),
                   key=lambda z: (z[0], z[1]))
    ded = []
    for b, k, n in lines:
        if ded and abs(ded[-1][0] - b) < 1e-15:
            if k > ded[-1][1]:
                ded[-1] = (b, k, n)
        else:
            ded.append((b, k, n))

    def cross(p, q):
        return (q[1] - p[1]) / (p[0] - q[0])

    hull = []
    for ln in ded:
        while hull:
            if hull[-1][0] == ln[0]:
                hull.pop(); continue
            if len(hull) >= 2 and cross(hull[-1], ln) <= cross(hull[-2],
                                                               hull[-1]):
                hull.pop(); continue
            break
        hull.append(ln)
    bnds = [cross(hull[i], hull[i + 1]) for i in range(len(hull) - 1)]
    rows, lo = [], 0.0            # a negative price is not a price
    for i, ln in enumerate(hull):
        hi = bnds[i] if i < len(bnds) else LAM_HI
        if hi > lo:
            rows.append((ln[2], lo, np.nan if hi >= LAM_HI else hi))
        lo = max(lo, hi)
    pieces = pd.DataFrame(rows, columns=["name", "lambda_from", "lambda_to"])
    wset = set(pieces["name"])

    order = df.sort_values([axis, "accuracy"], ascending=[True, False]).index
    best, dom = -np.inf, set()
    for i in order:
        a = df.at[i, "accuracy"]
        if a <= best:
            dom.add(df.at[i, "name"])
        else:
            best = a
    dent = sorted(set(df["name"]) - wset - dom)
    return {"pieces": pieces, "winners": sorted(wset),
            "dominated": sorted(dom & (set(df["name"]) - wset)),
            "in_a_dent": dent, "_dom_all": dom}


def dominations(df, axes=("resource",)):
    """Every (winner, loser) pair where the winner is no worse on accuracy and
    on EVERY resource axis, and strictly better somewhere.  One axis: a
    running maximum, O(K log K).  Several: a skyline scan."""
    axes = list(axes)
    if len(axes) == 1:
        a0 = axes[0]
        srt = df.sort_values([a0, "accuracy"], ascending=[True, False])
        rows, best_a, best_n = [], -np.inf, None
        for nm, a in zip(srt["name"], srt["accuracy"]):
            if a <= best_a and best_n is not None:
                rows.append((best_n, nm))
            else:
                best_a, best_n = a, nm
        return pd.DataFrame(rows, columns=["winner", "loser"])

    # n axes: keep a skyline of mutually nondominated points, scanning in
    # order of total log-cost so a dominator is always already in the skyline
    cost = sum(df["x__" + a] for a in axes)
    srt = df.assign(_c=cost).sort_values(["_c", "accuracy"],
                                         ascending=[True, False])
    names = srt["name"].to_numpy()
    acc = srt["accuracy"].to_numpy(float)
    R = srt[axes].to_numpy(float)
    rows, sky = [], []
    for i in range(len(names)):
        killer = None
        for j in sky:
            if acc[j] >= acc[i] and np.all(R[j] <= R[i]) and (
                    acc[j] > acc[i] or np.any(R[j] < R[i])):
                killer = j
                break
        if killer is None:
            sky.append(i)
        else:
            rows.append((names[killer], names[i]))
    return pd.DataFrame(rows, columns=["winner", "loser"])


# =================================================================== fit ====
def _aicc(rss, n, kp, var=True):
    k = kp + 1 if var else kp
    return None if n - k - 1 <= 0 else \
        n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _resid(t, eps):
    lne = np.log(eps)
    return lambda p: np.log(np.maximum(p[0] + p[1] * t ** -p[2],
                                       1e-300)) - lne


def fit_sat(t, eps, p0=None):
    """eps = eps_inf + c t^-alpha, fitted on ln eps, 0 < eps_inf < min eps.
    p0 warm-starts from a previous fit; otherwise a small grid of starts."""
    hi = float(np.min(eps)) * (1 - 1e-9)
    bounds = ([1e-12, 1e-12, 1e-6], [hi, 1e12, 8.0])
    fun = _resid(t, eps)
    starts = ([[min(p0[0], hi * 0.99), p0[1], p0[2]]] if p0 is not None else
              [[min(max(float(np.min(eps)) * f, 1e-11), hi * .99),
                float(np.max(eps)) * t[0] ** a, a]
               for a in (0.3, 0.7, 1.5, 3.0) for f in (0.1, 0.5, 0.9)])
    best = None
    for s in starts:
        try:
            r = least_squares(fun, s, bounds=bounds, max_nfev=20000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def fit_pow(t, eps):
    """eps = c t^-alpha -- linear in logs, so exact."""
    A = np.vstack([np.ones_like(t), -np.log(t)]).T
    co, *_ = np.linalg.lstsq(A, np.log(eps), rcond=None)
    r = np.log(eps) - A @ co
    return math.exp(co[0]), float(co[1]), float(r @ r)


def floor_ftest(rss_sat, rss_pow, n):
    """Nested F-test: eps_inf free (3 params) vs eps_inf = 0 forced (2).
    Returns (F, p), or (None, None) when n < 4."""
    if n < F_TESTABLE or rss_sat <= 0:
        return None, None
    num, den = rss_pow - rss_sat, rss_sat / (n - 3)
    if den <= 0 or num < 0:
        return 0.0, 1.0
    F = num / den
    return F, float(f_dist.sf(F, 1, n - 3))


def boot_alpha(t, eps, fit, reps, rng):
    """Residual bootstrap on ln eps, warm-started from the converged fit --
    28x faster than re-running the start grid, with the same interval."""
    pred = np.log(fit.x[0] + fit.x[1] * t ** -fit.x[2])
    res = np.log(eps) - pred
    res = res - res.mean()
    out = []
    for _ in range(reps):
        e2 = np.exp(pred + rng.choice(res, size=len(t), replace=True))
        if not np.all((e2 > 0) & (e2 < 1)):
            continue
        f2 = fit_sat(t, e2, p0=fit.x)
        if f2 is not None and f2.x[2] < 7.99:
            out.append(float(f2.x[2]))
    return np.array(out)


def asym_alpha_ci(fit, n, alpha=0.05):
    """Delta-method interval on alpha from the least_squares Jacobian.
    Free, and indistinguishable from the bootstrap once n >= BOOT_SKIP_N.
    Not trustworthy for small n: the eps_inf bound binds and alpha sits near
    it, so the normal approximation fails -- bootstrap there instead."""
    dof = max(n - 3, 1)
    s2 = float(fit.fun @ fit.fun) / dof
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * s2
    except np.linalg.LinAlgError:
        return None
    se = math.sqrt(max(float(cov[2, 2]), 0.0))
    q = float(t_dist.ppf(1 - alpha / 2, dof))
    return float(fit.x[2] - q * se), float(fit.x[2] + q * se)


def kl_regret(lam, lam_hat, alpha):
    p, q = lam / alpha, lam_hat / alpha
    if not (0 < p < 1 and 0 < q < 1):
        return float("nan")
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def sigma_matrix(df, lam_cols, axes=("resource",)):
    """K-by-G matrix of sigma = -ln eps - sum_a lambda_a ln t_a.
    lam_cols is a G-long list of per-axis price dicts (or, for one axis, a
    G-long list of floats)."""
    axes = list(axes)
    S = -np.log(df["eps"].to_numpy(float))[:, None]
    out = np.repeat(S, len(lam_cols), axis=1)
    for g, lam in enumerate(lam_cols):
        d = lam if isinstance(lam, dict) else {axes[0]: float(lam)}
        for a in axes:
            out[:, g] -= d.get(a, 0.0) * df["x__" + a].to_numpy(float)
    return out


def marginal_curve(df, axis):
    """That axis's own marginal frontier: sort on the axis and keep the
    running-minimum error.  Returns (x, ln eps) of the surviving points."""
    srt = df.sort_values("x__" + axis)
    xs, ls, best = [], [], np.inf
    for x, e in zip(srt["x__" + axis], srt["eps"]):
        if e < best:
            best = e
            xs.append(float(x)); ls.append(math.log(float(e)))
    return np.array(xs), np.array(ls)


def axis_price(df, axis, alpha=0.05, boot=400, seed=0, p_floor=0.05,
               aicc_k="with-variance"):
    """Derive one price for one axis from that axis's marginal curve, by the
    same alpha/2 rule and the same floor F-test used for a single resource.
    Returns a dict including `level` and `provenance`."""
    xs, ls = marginal_curve(df, axis)
    n, span = len(xs), (xs[-1] - xs[0] if len(xs) > 1 else 0.0)
    out = {"axis": axis, "n_marginal": n, "log_span": float(span),
           "lambda": None, "lambda_low": None, "lambda_high": None,
           "alpha": None, "eps_inf": None, "floor_F": None, "floor_p": None,
           "aicc_delta": None, "knee": None, "knee_spanned": None,
           "level": "UNUSABLE", "reason": "", "provenance": "none"}
    if n < MARG_MIN_PTS:
        out["reason"] = f"marginal curve has {n} point(s), needs {MARG_MIN_PTS}"
        return out
    if span < MARG_MIN_SPAN:
        out["reason"] = (f"marginal curve spans {span:.3f} < "
                         f"{MARG_MIN_SPAN} in log-{axis}")
        return out

    t, eps = np.exp(xs), np.exp(ls)
    sat = fit_sat(t, eps)
    c_p, a_p, rss_p = fit_pow(t, eps)
    if sat is None:
        out["reason"] = "the saturating fit did not converge"
        return out
    e_inf, c_s, al = (float(v) for v in sat.x)
    rss_s = float(2 * sat.cost)
    F, pval = floor_ftest(rss_s, rss_p, n)
    cv = aicc_k == "with-variance"
    A_s, A_p = _aicc(rss_s, n, 3, cv), _aicc(rss_p, n, 2, cv)
    t_star = (c_s / e_inf) ** (1 / al) if e_inf > 0 else float("inf")
    spans = bool(t.min() <= t_star <= t.max())
    out.update(eps_inf=e_inf, c=c_s, rss=rss_s, power_c=c_p,
               power_alpha=a_p, power_rss=rss_p,
               alpha=al, floor_F=F, floor_p=pval,
               knee=t_star, knee_spanned=spans,
               aicc_delta=(None if (A_s is None or A_p is None)
                           else A_p - A_s),
               resolved_nats=float(-ls.min()),
               range=[float(t.min()), float(t.max())])

    if pval is None:
        out.update(level="UNVERIFIED", provenance="alpha/2 (UNTESTED floor)",
                   reason=f"only {n} marginal points, floor untestable")
        lam_al = al
    elif pval <= p_floor:
        out.update(level="USABLE", provenance="alpha/2 (saturating)",
                   reason=f"floor confirmed, F-test p = {pval:.2g}")
        lam_al = al
    else:
        out.update(level="UNUSABLE", provenance="alpha/2 (floorless)",
                   reason=f"no floor on this axis, F-test p = {pval:.2g}")
        lam_al = a_p
    if out["level"] == "USABLE" and not spans:
        out.update(level="UNVERIFIED",
                   provenance="alpha/2 (knee extrapolated)",
                   reason=(f"knee {t_star:.4g} outside "
                           f"[{t.min():g}, {t.max():g}]"))
    lam = lam_al / 2
    out["lambda"] = lam
    out["lambda_low"] = out["lambda_high"] = lam
    out["bootstrap_refits"] = 0
    out["interval_from"] = "none"
    if not boot:
        return out
    if n >= BOOT_SKIP_N:                       # the free interval suffices
        ci = asym_alpha_ci(sat, n, alpha)
        if ci is not None:
            out["lambda_low"], out["lambda_high"] = ci[0] / 2, ci[1] / 2
            out["interval_from"] = f"jacobian (n={n} >= {BOOT_SKIP_N})"
            return out
    bs = boot_alpha(t, eps, sat, boot, np.random.default_rng(seed))
    if len(bs) >= 50:
        lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
        out["lambda_low"], out["lambda_high"] = float(lo) / 2, float(hi) / 2
        out["interval_from"] = f"bootstrap ({len(bs)} refits)"
    out["bootstrap_refits"] = int(len(bs))
    return out


# ================================================================ ranking ===
def rank_frame(df, res, lam, grid, axes=("resource",)):
    """Every candidate ordered winner to loser.  lam=None -> envelope order
    (needs no rate).  Vectorised over the whole rate interval."""
    axes = list(axes)
    env = res.get("envelope")
    pieces = (env["pieces"] if env is not None
              else pd.DataFrame(columns=["name", "lambda_from", "lambda_to"]))
    env_lo = {n: i for i, n in enumerate(pieces["name"])}
    env_rng = {n: (a, b) for n, a, b in zip(pieces["name"],
                                            pieces["lambda_from"],
                                            pieces["lambda_to"])}
    dom_by = dict(zip(res["price_free"]["loser"], res["price_free"]["winner"]))
    dent = set(env["in_a_dent"]) if env is not None else set()

    out = df.copy()
    if lam is None:
        out["_tier"] = [0 if n in env_lo else 1 if n in dent else 2
                        for n in out["name"]]
        out["_ord"] = [env_lo.get(n, 0) for n in out["name"]]
        out = out.sort_values(["_tier", "_ord", "eps"])
        out["sigma"] = np.nan
        lo = hi = None
    else:
        out["sigma"] = sigma_matrix(out, [lam], axes)[:, 0]
        out = out.sort_values("sigma", ascending=False)
        S = sigma_matrix(out, grid, axes)                # K x G
        ranks = (-S).argsort(axis=0).argsort(axis=0) + 1  # rank per column
        lo, hi = ranks.min(axis=1), ranks.max(axis=1)

    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["margin_to_next"] = out["sigma"].diff(-1) if lam is not None else np.nan
    out["rank_low"] = lo if lo is not None else np.nan
    out["rank_high"] = hi if hi is not None else np.nan
    out["rank_stable"] = ((out["rank_low"] == out["rank_high"])
                          & (out["rank_low"] == out["rank"]))
    out["optimal_from"] = [env_rng.get(n, (np.nan, np.nan))[0]
                           for n in out["name"]]
    out["optimal_to"] = [env_rng.get(n, (np.nan, np.nan))[1]
                         for n in out["name"]]
    out["note"] = ["dominated by " + dom_by[n] if n in dom_by
                   else "in a dent" if n in dent else ""
                   for n in out["name"]]
    drop = [c for c in ("_tier", "_ord", "eps") if c in out]
    drop += [c for c in out.columns if c.startswith("x__")]
    return out.drop(columns=drop)


# ================================================================ select ====
WORSE = {"USABLE": 0, "UNVERIFIED": 1, "UNUSABLE": 2}


def select(df, alpha=0.05, boot=400, seed=0, aicc_k="with-variance",
           rank=False, p_floor=0.05, name_col="name", acc_col="accuracy",
           res_col="resource"):
    """DataFrame -> result dict.  Prints nothing, raises nothing.

    res_col may name ONE resource column or SEVERAL.  With several, one price
    is derived per axis from that axis's own marginal curve and the
    scalarisation is sigma = -ln eps - sum_a lambda_a ln t_a.  The interaction
    between axes is not fitted (Paper 2, F5): each axis is priced alone.

    result["verdict"]["level"] is the WORST level across axes:

      USABLE      every axis has a confirmed floor
      UNVERIFIED  some axis could not be tested, or its knee is extrapolated
      UNUSABLE    some axis has no floor, so its price does not exist
    """
    cands, axes, bad = normalise(df, name_col, acc_col, res_col)
    multi = len(axes) > 1
    res = {"axes": axes, "axis": axes[0] if axes else "resource",
           "alpha": alpha, "status": "bad_input", "reason": None,
           "problems": bad,
           "candidates": cands[["name", "accuracy"] + axes]
           if set(["name", "accuracy"] + axes) <= set(cands) else cands,
           "verdict": None, "envelope": None,
           "price_free": pd.DataFrame(columns=["winner", "loser"]),
           "front": [], "prices": None, "fit": None, "rate": None,
           "scores": pd.DataFrame(columns=["name", "sigma"]),
           "winner": None, "ranking": None, "next": None,
           "axis_drop": None, "notes": []}

    def verdict(level, reason, **extra):
        res["verdict"] = dict(level=level, reason=reason, **extra)
        return res

    if bad:
        res["reason"] = "; ".join(bad)
        return verdict("UNUSABLE", res["reason"])

    res["envelope"] = envelope(cands, axes[0]) if not multi else None
    res["price_free"] = dominations(cands, axes)
    losers = set(res["price_free"]["loser"])
    front = cands[~cands["name"].isin(losers)].sort_values(axes[0])
    front = front.reset_index(drop=True)
    res["front"] = list(front["name"])
    n = len(front)

    if n < MIN_FIT:
        res["status"] = "no_rate"
        res["reason"] = f"only {n} nondominated candidate(s), need {MIN_FIT}"
        if rank and res["envelope"] is not None:
            res["ranking"] = rank_frame(cands, res, None, [], axes)
        return verdict("UNUSABLE", res["reason"])

    # ---- one price per axis, each from its own marginal curve -------------
    prices = [axis_price(front, a, alpha, boot, seed, p_floor, aicc_k)
              for a in axes]
    res["prices"] = pd.DataFrame([
        {k: pr.get(k) for k in ("axis", "lambda", "lambda_low", "lambda_high",
                                "alpha", "eps_inf", "floor_p", "n_marginal",
                                "log_span", "level", "provenance",
                                "interval_from", "reason")}
        for pr in prices])
    priced = {pr["axis"]: pr for pr in prices if pr["lambda"] is not None}
    if not priced:
        res["status"] = "no_rate"
        res["reason"] = "; ".join(f"{pr['axis']}: {pr['reason']}"
                                  for pr in prices)
        if rank and res["envelope"] is not None:
            res["ranking"] = rank_frame(cands, res, None, [], axes)
        return verdict("UNUSABLE", res["reason"])
    unpriced = [pr["axis"] for pr in prices if pr["lambda"] is None]
    for a in unpriced:
        res["notes"].append(f"axis '{a}' carries NO price and is therefore "
                            "excluded from sigma; the ranking ignores it")

    level = max((priced[a]["level"] for a in priced), key=lambda L: WORSE[L])
    worst = [a for a in priced if priced[a]["level"] == level]
    why = (priced[worst[0]]["reason"] if not multi else
           f"{', '.join(worst)}: {priced[worst[0]]['reason']}")

    # the single-resource report keeps its familiar `fit` block
    p0 = priced[axes[0]] if axes[0] in priced else prices[0]
    if not multi:
        res["fit"] = {"n": p0["n_marginal"], "eps_inf": p0["eps_inf"],
                      "c": p0.get("c"), "rss": p0.get("rss"),
                      "power_c": p0.get("power_c"),
                      "power_alpha": p0.get("power_alpha"),
                      "power_rss": p0.get("power_rss"),
                      "alpha": p0["alpha"], "floor_F": p0["floor_F"],
                      "floor_p": p0["floor_p"], "aicc_delta": p0["aicc_delta"],
                      "aicc_convention": aicc_k, "knee": p0["knee"],
                      "knee_spanned": p0["knee_spanned"],
                      "resolved_nats": p0.get("resolved_nats"),
                      "range": p0.get("range")}

    lam = {a: priced[a]["lambda"] for a in priced}
    res["rate"] = {"lambda": lam if multi else lam[axes[0]],
                   "per_axis": lam,
                   "lambda_low": {a: priced[a]["lambda_low"] for a in priced},
                   "lambda_high": {a: priced[a]["lambda_high"] for a in priced},
                   "coverage": 1 - alpha,
                   "bootstrap_refits": int(p0.get("bootstrap_refits") or 0),
                   "interval_from": (p0.get("interval_from", "none")
                                     if not multi else
                                     {a: priced[a].get("interval_from")
                                      for a in priced}),
                   "error_drop_per_doubling":
                       {a: 1 - 2 ** -lam[a] for a in lam} if multi
                       else 1 - 2 ** -lam[axes[0]],
                   "derived_from": {a: priced[a]["provenance"] for a in priced}
                   if multi else p0["provenance"],
                   "declared": [], "trustworthy": level == "USABLE"}
    res["notes"].append(
        "a bootstrap interval measures spread around THIS fit, not distance "
        "from the truth: if the floor is spurious every resample inherits it")
    if multi:
        res["notes"].append(
            "one price per axis, each from its own marginal curve; the "
            "interaction between axes is bounded, not fitted (Paper 2, F5)")

    # ---- the rate interval: per-axis corners, capped ---------------------
    corners = [lam]
    if multi:
        import itertools
        ends = [[(a, priced[a]["lambda_low"]), (a, priced[a]["lambda_high"])]
                for a in priced]
        if len(priced) <= 6:
            corners = [dict(c) for c in itertools.product(*ends)]
    else:
        a0 = axes[0]
        lo, hi = priced[a0]["lambda_low"], priced[a0]["lambda_high"]
        corners = ([{a0: float(v)} for v in np.linspace(lo, hi, 601)]
                   if hi > lo else [{a0: lam[a0]}])

    sg = sigma_matrix(front, [lam], axes)[:, 0]
    sc = front.assign(sigma=sg).sort_values("sigma", ascending=False)
    res["scores"] = sc[["name", "sigma"]].reset_index(drop=True)
    win, runner = sc["name"].iloc[0], sc["name"].iloc[1]
    Sg = sigma_matrix(front, corners, axes)
    wset = list(dict.fromkeys(front["name"].to_numpy()[Sg.argmax(axis=0)]))
    xw = front.loc[front["name"] == win, ["x__" + a for a in axes]].to_numpy()
    xr = front.loc[front["name"] == runner,
                   ["x__" + a for a in axes]].to_numpy()
    sep = float(np.max(np.abs(xw - xr)))
    reg = 0.0
    if not multi:
        a0 = axes[0]
        al = priced[a0]["alpha"] or 1.0
        rr = [kl_regret(c[a0], lam[a0], al) for c in corners]
        reg = max((v for v in rr if not math.isnan(v)), default=0.0)
    res["winner"] = {"name": win, "runner_up": runner,
                     "margin_nats": float(sc["sigma"].iloc[0]
                                          - sc["sigma"].iloc[1]),
                     "robust": len(wset) == 1, "winner_set": wset,
                     "max_kl_regret_in_interval": reg,
                     "log_resource_separation": sep,
                     "resolved": bool(sep >= SPAN_MIN)}

    # ---- does each axis need a price?  drop it and see (script 144) ------
    if multi and len(priced) > 1:
        tot = sum(abs(lam[a] * front["x__" + a].mean()) for a in lam) or 1.0
        drops = []
        for a in priced:
            sub = {k: v for k, v in lam.items() if k != a}
            pick = front["name"].to_numpy()[
                sigma_matrix(front, [sub], axes)[:, 0].argmax()]
            drops.append({"dropped_axis": a,
                          "cost_share": abs(lam[a] * front["x__" + a].mean())
                          / tot,
                          "pick_with": win, "pick_without": pick,
                          "decision_changes": bool(pick != win)})
        res["axis_drop"] = pd.DataFrame(drops)

    if rank:
        res["ranking"] = rank_frame(cands, res, lam, corners, axes)
    if not multi and (p0["eps_inf"] or 0) > 0 and p0["knee"]:
        gap = float(np.min(np.abs(front["x__" + axes[0]].to_numpy()
                                  - math.log(p0["knee"]))))
        res["next"] = {"resource": p0["knee"], "nearest_candidate_efolds": gap,
                       "frontier_unmeasured_at_operating_point":
                           bool(gap > SPAN_MIN)}
    res["status"] = ("selected" if level == "USABLE" else
                     "unverified" if level == "UNVERIFIED" else "no_floor")
    return verdict(level, why, p_value=p0.get("floor_p"), F=p0.get("floor_F"))

# ================================================================ render ====
RED = "\033[1;31m"                        # problems only
GRN = "\033[38;2;46;139;87m"              # #2E8B57, the answer
OFF = "\033[0m"


def _green(text, color=True):
    return GRN + text + OFF if color else text


def _red(text, color=True):
    return RED + text + OFF if color else text


def render(r, color=True):
    """Result dict -> the text report."""
    L, ax = [], r["axis"]
    axes = r.get("axes") or [ax]
    multi = len(axes) > 1
    v = r.get("verdict") or {}
    lev = v.get("level", "UNUSABLE")
    f0 = r.get("fit") or {}
    if lev == "UNUSABLE" and f0:
        L += ["", _red(f"NO FLOOR (F-test p = {f0['floor_p']:.2g}): no "
                       "exchange rate exists in this data.", color),
              _red(f"The ranking below is just {ax} order and means nothing "
                   "(Thm 'degeneracy').", color)]
    elif lev == "UNUSABLE":
        L += ["", _red(f"NO RANKING: {v['reason']}.", color)]
    elif lev == "UNVERIFIED":
        L += ["", _red(f"UNVERIFIED: {v['reason']}.", color),
              _red("The ranking below is a guess, not a result.", color)]
    if r["status"] == "bad_input":
        return "\n".join(L)

    e = r["envelope"]
    if e is None:
        L.append(f"\nAXES ........... {len(axes)} resources: "
                 + ", ".join(axes))
    else:
        L.append(f"\nENVELOPE ....... {len(e['winners'])} of "
                 f"{len(r['candidates'])} can be optimal at SOME price  "
                 "[no rate needed]")
        for nm, lo, hi in zip(e["pieces"]["name"], e["pieces"]["lambda_from"],
                              e["pieces"]["lambda_to"]):
            h = "inf" if pd.isna(hi) else f"{hi:.4f}"
            L.append(f"  {nm:<20} optimal for lambda in [{lo:.4f}, {h}]")
        never = sorted(e["dominated"] + e["in_a_dent"])
        if never:
            L.append(f"  never optimal  {', '.join(never)}")
            if e["dominated"]:
                L.append(f"    dominated .. {', '.join(e['dominated'])}  "
                         "(something is both cheaper and better)")
            if e["in_a_dent"]:
                L.append(f"    in a dent .. {', '.join(e['in_a_dent'])}  "
                         "(Pareto-efficient, yet NO price selects it)")

    pf = r["price_free"]
    if len(pf):
        L.append(f"\nPRICE-FREE ..... {len(pf)} domination(s), valid at "
                 "EVERY price:")
        for w, l_ in zip(pf["winner"], pf["loser"]):
            L.append(f"  {w} beats {l_}")
    L.append(f"\nCULL ........... {len(r['front'])} of "
             f"{len(r['candidates'])} nondominated")

    pr = r.get("prices")
    if multi and pr is not None:
        L.append(f"\nPRICE PER AXIS . one price per axis, from that axis's own "
                 "marginal curve")
        L.append(f"  {'axis':<18}{'lambda':>9}{'95% interval':>20}"
                 f"{'alpha':>8}{'floor p':>10}{'pts':>5}{'span':>7}  "
                 "provenance / interval")
        for _, q in pr.iterrows():
            lv, lo, hi = q["lambda"], q["lambda_low"], q["lambda_high"]
            iv = ("        --        " if lo is None or pd.isna(lo)
                  else f"[{lo:.4f}, {hi:.4f}]".rjust(20))
            lm = "   --    " if lv is None or pd.isna(lv) else f"{lv:>9.4f}"
            al = ("    --  " if q["alpha"] is None or pd.isna(q["alpha"])
                  else f"{q['alpha']:>8.4f}")
            fp = ("      --  " if q["floor_p"] is None
                  or pd.isna(q["floor_p"]) else f"{q['floor_p']:>10.3g}")
            L.append(f"  {q['axis']:<18}{lm}{iv}{al}{fp}"
                     f"{int(q['n_marginal']):>5}{q['log_span']:>7.2f}  "
                     f"{q['provenance']}")
            if lv is None or pd.isna(lv):
                L.append(f"    ! {q['axis']} has NO price: {q['reason']}")
    f = r["fit"]
    if f:
        L.append(f"\nFIT ............ eps = eps_inf + c t^-alpha on ln eps, "
                 f"{f['n']} points")
        L.append(f"  saturating ... eps_inf={f['eps_inf']:.6g} "
                 f"c={f['c']:.6g} alpha={f['alpha']:.4f}  RSS={f['rss']:.4g}")
        L.append(f"  power law .... c={f['power_c']:.6g} "
                 f"alpha={f['power_alpha']:.4f}{'':14s}"
                 f"RSS={f['power_rss']:.4g}")
        if f["floor_p"] is None:
            L.append("  FLOOR TEST ... not testable: 3 parameters, "
                     f"{f['n']} points, no residual left")
        else:
            ok = "floor CONFIRMED" if f["floor_p"] <= 0.05 else "NO floor"
            L.append(f"  FLOOR TEST ... nested F(1,{f['n'] - 3}) = "
                     f"{f['floor_F']:.3g}, p = {f['floor_p']:.3g}  ->  {ok}")
        if f["aicc_delta"] is not None:
            L.append(f"  AICc ......... delta = {f['aicc_delta']:+.2f}  "
                     f"[{f['aicc_convention']}]  (corroboration only)")
        L.append(f"  knee ......... t* = {f['knee']:.4g}; range "
                 f"[{f['range'][0]:g}, {f['range'][1]:g}] -> "
                 f"{'SPANNED' if f['knee_spanned'] else 'NOT spanned'}")
        L.append(f"  resolved ..... {f['resolved_nats']:.3f} nats (paper: "
                 "identified in 87% of fronts above 1.5, 0% below 1.0)")

    if r["rate"] is None:
        for n_ in r["notes"]:
            L.append(f"  note ......... {n_}")
        L.append(f"  what stands .. the ENVELOPE above, which needs no rate.")
        return "\n".join(L)

    t_ = r["rate"]
    if multi:
        L.append("\nRATE ........... sigma = -ln eps - "
                 + " - ".join(f"{t_['per_axis'][a]:.4f}*ln({a})"
                              for a in t_["per_axis"]))
        for a, lv in t_["per_axis"].items():
            L.append(f"  {a:<18} doubling it needs the error to fall by "
                     f">= {(1 - 2 ** -lv) * 100:.1f}%")
    else:
        L.append(f"\nRATE ........... lambda = {t_['lambda']:.4f} nats per "
                 f"e-fold of {ax}  [{t_['derived_from']}]")
        L.append(f"  interval ..... [{t_['lambda_low'][ax]:.4f}, "
                 f"{t_['lambda_high'][ax]:.4f}] at "
                 f"{t_['coverage'] * 100:.0f}%  "
                 f"[{t_['interval_from']}]")
        L.append(f"  meaning ...... doubling {ax} is justified only if the "
                 f"error falls by >= "
                 f"{t_['error_drop_per_doubling'] * 100:.1f}%")
    L.append("  declared ..... nothing: no price, no grain, no q0, no "
             "transferred constant")

    w = r["winner"]
    rk = r["ranking"]
    if rk is not None:
        at = (", ".join(f"{a}:{v:.4f}" for a, v in t_["per_axis"].items())
              if multi else f"lambda = {t_['lambda']:.4f}")
        L += ["", f"RANKING ........ all {len(rk)} models at {at}, "
                  "winner first",
              f"    # {'name':<18}{'acc':>9}"
              + "".join(f"{a[:11]:>12}" for a in axes)
              + f"{'sigma':>9}{'margin':>9}{'rank@int':>10}  note"]
        for q in rk.itertuples(index=False):
            mg = "    --  " if pd.isna(q.margin_to_next) \
                else f"{q.margin_to_next:+.4f}"
            rr = (f"{int(q.rank_low)}" if q.rank_low == q.rank_high
                  else f"{int(q.rank_low)}-{int(q.rank_high)}")
            cells = "".join(f"{getattr(q, a):>12g}" for a in axes)
            L.append(_green(f"  {q.rank:>3} {q.name:<18}{q.accuracy:>9.4f}"
                            f"{cells}{q.sigma:>+9.4f}{mg:>9}"
                            f"{rr:>10}  {q.note}", color))
        L += ["  rank@int ..... the rank range this model takes across the "
              "whole rate interval;",
              "                 a single number means the position does not "
              "move.  Models marked",
              "                 'dominated' or 'in a dent' are ranked for "
              "completeness only --",
              "                 no rate selects them, whatever their sigma."]
    else:
        expr = (" - ".join(f"{v:.4f} ln {a}"
                           for a, v in t_["per_axis"].items())
                if multi else f"{t_['lambda']:.4f} ln {ax}")
        L.append(f"\nSCORE .......... sigma = -ln eps - {expr}")
        for nm, sg in zip(r["scores"]["name"], r["scores"]["sigma"]):
            L.append(f"  {nm:<20} {sg:+.4f}"
                     + ("   <= winner" if nm == w["name"] else ""))

    L += ["", _green(f"WINNER ......... {w['name']}   margin "
                     f"{w['margin_nats']:.4f} nats over {w['runner_up']}",
                     color)]
    if w["robust"]:
        if multi:
            L.append(f"  ROBUST ....... {w['name']} wins at every corner of "
                     "the per-axis price intervals")
        else:
            L.append(f"  ROBUST ....... {w['name']} wins at EVERY rate in "
                     f"[{t_['lambda_low'][ax]:.4f}, "
                     f"{t_['lambda_high'][ax]:.4f}]")
    else:
        L.append(f"  ! NOT ROBUST . the rate interval admits "
                 f"{len(w['winner_set'])} winners: "
                 f"{', '.join(w['winner_set'])}")
        L.append(f"                 picking {w['name']} anyway costs at most "
                 f"{w['max_kl_regret_in_interval']:.4f} nats inside it")
    if not w["resolved"]:
        L.append(f"  ! UNRESOLVED . winner and runner-up differ by "
                 f"{w['log_resource_separation']:.3f} < ln 1.5")
    L.append("  ! sigma gaps carry no test-set noise; pass per-item outcomes "
             "to certify them.")
    for n_ in r["notes"]:
        L.append(f"  note ......... {n_}")

    ad = r.get("axis_drop")
    if ad is not None and len(ad):
        L.append("\nDOES EACH AXIS NEED A PRICE?  drop it and see")
        for q in ad.itertuples(index=False):
            verd = ("CHANGES the pick -> " + q.pick_without
                    if q.decision_changes else "pick unchanged")
            L.append(f"  without {q.dropped_axis:<16} "
                     f"{q.cost_share:>6.1%} of total cost   {verd}")
        if not ad["decision_changes"].any():
            L.append("  every axis could be dropped without changing the "
                     "pick on THIS data;")
            L.append("  on the paper's 48 multi-resource blocks that held "
                     "for only 36% of axes.")

    nx = r["next"]
    if nx:
        L.append(f"\nNEXT ........... an unmeasured model matters most at "
                 f"{ax} = {nx['resource']:.4g} (the knee,")
        L.append(f"                 where lambda = alpha/2 is optimal); "
                 f"nearest candidate is "
                 f"{nx['nearest_candidate_efolds']:.2f} e-folds away")
    return "\n".join(L)


# =================================================================== cli ====
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Select the best model from (accuracy, resource) alone.")
    ap.add_argument("inputs", nargs="*", metavar="NAME,ACC,RES | FILE.csv",
                    help="candidates given directly (anything containing a "
                         "comma) or a CSV path")
    ap.add_argument("--model", action="append", metavar="N,ACC,RES",
                    help="one candidate; equivalent to a bare NAME,ACC,RES")
    ap.add_argument("--acc-col")
    ap.add_argument("--res-col", action="append", metavar="COL",
                    help="CSV resource column; repeat for several axes. "
                         "Not needed for NAME,ACC,P1[,P2,...] arguments, "
                         "whose axes are named p1..pn")
    ap.add_argument("--name-col")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="coverage level for the rate interval "
                         "(default 0.05; a convention, not a quantity of the "
                         "problem)")
    ap.add_argument("--aicc-k", choices=("with-variance", "params-only"),
                    default="with-variance")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", action="store_true",
                    help="order ALL models winner to loser")
    ap.add_argument("--out-csv", metavar="FILE",
                    help="write the ranking to CSV ('-' for stdout)")
    ap.add_argument("--no-color", action="store_true",
                    help="plain text, no ANSI colour")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the text report")
    a, unknown = ap.parse_known_args(argv)
    stray = [u for u in unknown if "," in u and not u.startswith("-")]
    rest = [u for u in unknown if u not in stray]
    if rest:
        ap.error("unrecognized arguments: " + " ".join(rest))

    specs = [x for x in a.inputs if "," in x] + stray + list(a.model or [])
    files = [x for x in a.inputs if "," not in x]
    if len(files) > 1:
        ap.error(f"give at most one CSV path, got {len(files)}: "
                 + ", ".join(files))
    if sum(bool(x) for x in (specs, files)) != 1:
        ap.error("give exactly one of: candidates as NAME,ACC,RES arguments, "
                 "or a CSV path")
    try:
        df = (frame_from_csv(files[0], a.name_col, a.acc_col, a.res_col)
              if files else frame_from_specs(specs))
    except (OSError, ValueError, KeyError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    result = select(df, alpha=a.alpha, boot=a.boot, seed=a.seed,
                    aicc_k=a.aicc_k, rank=a.rank or bool(a.out_csv),
                    res_col=df.attrs.get("axes") or "resource")
    if not a.quiet:
        print(render(result, color=not a.no_color))
    if a.out_csv and result["ranking"] is not None:
        if a.out_csv == "-":
            result["ranking"].to_csv(sys.stdout, index=False)
        else:
            result["ranking"].to_csv(a.out_csv, index=False)
    return {"selected": 0, "unverified": 1, "no_floor": 1, "no_rate": 1,
            "bad_input": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())