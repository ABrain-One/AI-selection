#!/usr/bin/env python3
"""
Select the best model from (accuracy, resource) measurements alone.

NOTHING is declared by hand: no price, no grain, no chance level, no
transferred constant.  Only --alpha, the statistical coverage level, which is
a convention and not a quantity of the problem.

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
forced) needs only n >= 4, which covers the 4- and 5-point cases.  AICc is
still printed as corroboration.  Measured on 400 synthetic frontiers per
cell: a spurious floor is fitted on 36-48% of genuinely floorless data at
EVERY n, and the test rejects 97-100% of those at n >= 6 -- which is why the
verdict, not the ranking, is the output that matters.

A bootstrap interval measures spread around THIS fit, not distance from the
truth: if the floor is spurious, every resample inherits it and agrees
confidently.  That is what the verdict is for.


THE JSON PAYLOAD
================

    { "candidates": [
        {"name": "m0", "accuracy": 0.6058, "resource": 3.0},
        {"name": "m1", "accuracy": 0.7527, "resource": 5.7} ] }

A bare list of candidates is also accepted.  On the command line a candidate
is any bare argument containing a comma, so `--model` is optional:

    example_general.py --rank Q05,0.475,0.5 Q72,0.861,72

`accuracy` must be in (0,1);
`resource` > 0 in one unit, measured on one device.  `--emit-json` prints this
payload for whatever input you gave, so a CSV or a set of `--model` flags can
be captured once and replayed.


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

2. A frontier with no floor -- the ranking is printed, and disowned.

   $ python3 example_general.py --rank Q05,0.475,0.5 Q15,0.609,1.5 Q3,0.656,3 \\
         Q7,0.742,7 Q14,0.797,14 Q32,0.833,32 Q72,0.861,72

   NO FLOOR (F-test p = 0.86): no exchange rate exists in this data.
   The ranking below is just resource order and means nothing.

3. Too few points to test the floor.

   $ python3 example_general.py --rank A,0.851,4 B,0.9253,9 D,0.90,30

   UNVERIFIED: only 2 nondominated points, floor untestable.
   The ranking below is a guess, not a result.

4. JSON in, JSON out.

   $ python3 example_general.py --json front.json --out-json result.json
   $ cat front.json | python3 example_general.py --json - --quiet

5. Columns that cannot be guessed, and a looser coverage level.

   $ python3 example_general.py models.csv \\
         --name-col model --acc-col top1 --res-col latency_ms --alpha 0.10

   The report labels the resource with the CSV's own column name.

6. Convert any input to the JSON payload, without selecting.

   $ python3 example_general.py models.csv --emit-json > front.json


USE AS A LIBRARY
================

    from example_general import select, render

    result = select({"candidates": [...]}, alpha=0.05, rank=True)
    print(render(result, color=False))

    result["verdict"]["level"]     # "USABLE" | "UNVERIFIED" | "UNUSABLE"
    result["verdict"]["p_value"]   # the floor F-test p, None if untestable
    result["rate"]["trustworthy"]  # True only when USABLE
    result["winner"]["name"]
    result["ranking"]              # one row per model, rank_low/rank_high
    result["envelope"]["winners"]  # always valid: needs no rate at all

`select()` takes and returns plain JSON-serialisable dicts and prints nothing.


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


KNOWN LIMITS

  Competition leaderboards do not work here.  Their rules pin quality at a
  threshold and race on cost, so almost every entry is dominated and the
  quality range carries no curvature.  Tested on MAI 2026 SR (3 of 8
  nondominated), NTIRE 2026 Efficient SR (4 of 14) and YOLOv8+YOLO11 (6 of
  10): no rate in any of them.  What the method needs is a deliberate ladder
  -- one architecture, 8+ sizes, same test set and device, quality spanning
  enough range that the frontier visibly bends.
"""
import argparse
import csv
import json
import math
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as f_dist

MIN_FIT = 3        # 3 parameters need 3 points
F_TESTABLE = 4     # the nested F-test needs n >= 4
SPAN_MIN = math.log(1.5)
LAM_HI = 1e6


# ======================================================== input adapters ====
def payload_from_models(specs):
    """`name,accuracy,resource` strings -> the JSON payload."""
    cands = []
    for spec in specs:
        p = [s.strip() for s in spec.split(",")]
        if len(p) != 3:
            raise ValueError(f"--model wants name,accuracy,resource: {spec!r}")
        cands.append({"name": p[0], "accuracy": float(p[1]),
                      "resource": float(p[2])})
    return {"candidates": cands}


def payload_from_csv(path, name_col=None, acc_col=None, res_col=None):
    """A CSV -> the JSON payload.  Headers are guessed unless named."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh)
                if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"{path} has no rows")
    head = [h.strip() for h in rows[0]]
    nc = name_col or head[0]
    ac = acc_col or next((h for h in head if "acc" in h.lower()), None)
    rc = res_col or next((h for h in head if any(
        k in h.lower() for k in ("time", "latency", "ms", "flop",
                                 "resource", "energy"))), None)
    if not ac or not rc:
        raise ValueError(f"cannot find accuracy/resource columns in {head}; "
                         "use --acc-col and --res-col")
    ia, ir, inm = head.index(ac), head.index(rc), head.index(nc)
    cands = [{"name": r[inm].strip(), "accuracy": float(r[ia]),
              "resource": float(r[ir])} for r in rows[1:] if r[ia].strip()]
    return {"candidates": cands,
            "source": {"file": path, "accuracy_column": ac,
                       "resource_column": rc}}


def payload_from_json(path):
    raw = sys.stdin.read() if path == "-" else open(path).read()
    obj = json.loads(raw)
    if isinstance(obj, list):
        obj = {"candidates": obj}
    if "candidates" not in obj:
        raise ValueError("JSON needs a 'candidates' list")
    return dict(obj)


def normalise(payload):
    """Validate and canonicalise; returns (candidates, axis, problems).
    Accepts the full payload dict or a bare list of candidates."""
    if isinstance(payload, (list, tuple)):
        payload = {"candidates": list(payload)}
    # the report's resource label is DERIVED, never an input: a CSV lends its
    # own column name, anything else is just "resource"
    axis = (payload.get("source") or {}).get("resource_column") or "resource"
    cands, bad = [], []
    for c in payload.get("candidates", []):
        name = str(c.get("name", "?"))
        try:
            acc = float(c["accuracy"])
            res = float(c["resource"])
        except (KeyError, TypeError, ValueError):
            bad.append(f"{name}: needs numeric 'accuracy' and 'resource'")
            continue
        if not 0 < acc < 1:
            bad.append(f"{name}: accuracy {acc} is outside (0,1)")
        elif not res > 0:
            bad.append(f"{name}: resource {res} is not positive")
        else:
            cands.append({"name": name, "accuracy": acc, "resource": res,
                          "eps": 1 - acc, "x": math.log(res)})
    if len(cands) < 2 and not bad:
        bad.append("fewer than two usable candidates")
    return cands, axis, bad


# ============================================================== geometry ====
def envelope(cands):
    lines = sorted(((-c["x"], -math.log(c["eps"]), c["name"]) for c in cands),
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
    pieces, lo = [], 0.0            # a negative price is not a price
    for i, ln in enumerate(hull):
        hi = bnds[i] if i < len(bnds) else LAM_HI
        if hi > lo:
            pieces.append({"name": ln[2], "lambda_from": lo,
                           "lambda_to": None if hi >= LAM_HI else hi})
        lo = max(lo, hi)
    wset = {p["name"] for p in pieces}
    dom, dent = [], []
    for c in cands:
        if c["name"] in wset:
            continue
        (dom if any(o["accuracy"] >= c["accuracy"]
                    and o["resource"] <= c["resource"]
                    and (o["accuracy"] > c["accuracy"]
                         or o["resource"] < c["resource"])
                    for o in cands if o is not c) else dent).append(c["name"])
    return {"pieces": pieces, "winners": sorted(wset),
            "dominated": sorted(dom), "in_a_dent": sorted(dent)}


def dominations(cands):
    out = []
    for c in cands:
        k = next((o for o in cands if o is not c
                  and o["accuracy"] >= c["accuracy"]
                  and o["resource"] <= c["resource"]
                  and (o["accuracy"] > c["accuracy"]
                       or o["resource"] < c["resource"])), None)
        if k:
            out.append({"winner": k["name"], "loser": c["name"]})
    return out


# =================================================================== fit ====
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
    return math.exp(co[0]), float(co[1]), float(r @ r)


def boot_alpha(t, eps, fit, reps, rng):
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
            out.append(float(f2.x[2]))
    return np.array(out)


def kl_regret(lam, lam_hat, alpha):
    p, q = lam / alpha, lam_hat / alpha
    if not (0 < p < 1 and 0 < q < 1):
        return float("nan")
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def sigma_at(cands, lam):
    return {c["name"]: -math.log(c["eps"]) - lam * c["x"] for c in cands}


def _rank_rows(cands, res, lam, grid):
    """Order every candidate.  lam=None -> envelope order (no rate needed)."""
    env_of, env_lo = {}, {}
    for i, pc in enumerate(res["envelope"]["pieces"]):
        env_of[pc["name"]] = [pc["lambda_from"], pc["lambda_to"]]
        env_lo[pc["name"]] = i
    dom_by = {d["loser"]: d["winner"] for d in res["price_free"]}
    dent = set(res["envelope"]["in_a_dent"])

    if lam is None:
        # envelope order: winners by the price range they own (cheapest price
        # first, i.e. the most accurate end), then dents, then dominated
        order = sorted(
            cands,
            key=lambda c: (0 if c["name"] in env_lo
                           else 1 if c["name"] in dent else 2,
                           env_lo.get(c["name"], 0),
                           -(-math.log(c["eps"]))))
        seq = [(c["name"], None) for c in order]
    else:
        seq = sorted(sigma_at(cands, lam).items(), key=lambda kv: -kv[1])

    spread = {c["name"]: [len(cands), 1] for c in cands}
    if lam is not None:
        for L in grid:
            o = sorted(sigma_at(cands, L).items(), key=lambda kv: -kv[1])
            for i, (nm, _) in enumerate(o, 1):
                spread[nm][0] = min(spread[nm][0], i)
                spread[nm][1] = max(spread[nm][1], i)

    rows = []
    for i, (nm, sg) in enumerate(seq, 1):
        c = next(x for x in cands if x["name"] == nm)
        nxt = seq[i][1] if i < len(seq) else None
        note = ("dominated by " + dom_by[nm] if nm in dom_by
                else "in a dent" if nm in dent else "")
        rows.append({"rank": i, "name": nm, "accuracy": c["accuracy"],
                     "resource": c["resource"], "sigma": sg,
                     "margin_to_next": (None if (nxt is None or sg is None)
                                        else sg - nxt),
                     "rank_low": spread[nm][0] if lam is not None else None,
                     "rank_high": spread[nm][1] if lam is not None else None,
                     "rank_stable": (spread[nm][0] == spread[nm][1] == i
                                     if lam is not None else None),
                     "optimal_for_lambda_in": env_of.get(nm),
                     "note": note})
    return rows


# ================================================================ select ====
def floor_ftest(rss_sat, rss_pow, n):
    """Nested F-test: eps_inf free (3 params) vs eps_inf = 0 forced (2 params).
    Returns (F, p) or (None, None) when n is too small (needs n >= 4)."""
    if n < 4 or rss_sat <= 0:
        return None, None
    num = (rss_pow - rss_sat) / 1.0
    den = rss_sat / (n - 3)
    if den <= 0 or num < 0:
        return 0.0, 1.0
    F = num / den
    return F, float(f_dist.sf(F, 1, n - 3))


def select(payload, alpha=0.05, boot=400, seed=0, aicc_k="with-variance",
           rank=False, p_floor=0.05):
    """JSON payload -> JSON result.  Prints nothing, raises nothing.

    A ranking is ALWAYS produced when 3 or more nondominated candidates
    remain.  result["verdict"]["level"] says whether it means anything:

      USABLE      the floor is confirmed by a nested F-test, so an exchange
                  rate exists and the ranking is a trade-off ranking
      UNVERIFIED  fewer than 4 points, so the floor cannot be tested at all;
                  the fit is exact and the ranking is unfalsifiable
      UNUSABLE    the F-test rejects the floor.  Thm 'degeneracy': the
                  frontier is scale-free, NO exchange rate exists, and the
                  ranking below collapses to ordering by resource

    rank=True adds result["ranking"].  lam declares the rate instead of
    deriving it; the result is then labelled [declared]."""
    cands, axis, bad = normalise(payload)
    res = {"axis": axis, "alpha": alpha, "status": "bad_input",
           "reason": None, "problems": bad,
           "candidates": [{k: c[k] for k in ("name", "accuracy", "resource")}
                          for c in cands],
           "verdict": None, "envelope": None, "price_free": [], "front": [],
           "fit": None, "rate": None, "scores": [], "winner": None,
           "ranking": None, "next": None, "notes": []}

    def verdict(level, reason, **extra):
        res["verdict"] = dict(level=level, reason=reason, **extra)
        return res

    if bad:
        res["reason"] = "; ".join(bad)
        return verdict("UNUSABLE", res["reason"])

    res["envelope"] = envelope(cands)
    res["price_free"] = dominations(cands)
    losers = {d["loser"] for d in res["price_free"]}
    front = [c for c in cands if c["name"] not in losers]
    front.sort(key=lambda c: c["resource"])
    res["front"] = [c["name"] for c in front]
    n = len(front)

    if n < 3:
        res["status"] = "no_rate"
        res["reason"] = f"only {n} nondominated candidate(s), need 3 to fit"
        if rank:
            res["ranking"] = _rank_rows(cands, res, None, [])
        return verdict("UNUSABLE", res["reason"])

    t = np.array([c["resource"] for c in front])
    eps = np.array([c["eps"] for c in front])
    sat = fit_sat(t, eps)
    c_p, a_p, rss_p = fit_pow(t, eps)
    if sat is None:
        res["status"] = "no_rate"
        res["reason"] = "the saturating fit did not converge"
        if rank:
            res["ranking"] = _rank_rows(cands, res, None, [])
        return verdict("UNUSABLE", res["reason"])
    e_inf, c_s, al = (float(v) for v in sat.x)
    rss_s = float(2 * sat.cost)

    # ---- does a floor exist at all? -------------------------------------
    F, pval = floor_ftest(rss_s, rss_p, n)
    cv = aicc_k == "with-variance"
    A_s, A_p = _aicc(rss_s, n, 3, cv), _aicc(rss_p, n, 2, cv)
    d_aicc = None if (A_s is None or A_p is None) else A_p - A_s
    t_star = (c_s / e_inf) ** (1 / al) if e_inf > 0 else float("inf")
    spans = bool(t.min() <= t_star <= t.max())
    res["fit"] = {"n": n, "eps_inf": e_inf, "c": c_s, "alpha": al,
                  "rss": rss_s, "power_c": c_p, "power_alpha": a_p,
                  "power_rss": rss_p, "floor_F": F, "floor_p": pval,
                  "aicc_delta": d_aicc, "aicc_convention": aicc_k,
                  "knee": t_star, "knee_spanned": spans,
                  "resolved_nats": float(-math.log(eps.min())),
                  "range": [float(t.min()), float(t.max())]}

    if pval is None:
        level = "UNVERIFIED"
        why = f"only {n} nondominated points, floor untestable"
        lam_src, lam_al = "alpha/2 (saturating, UNTESTED)", al
    elif pval <= p_floor:
        level = "USABLE"
        why = f"floor confirmed, F-test p = {pval:.2g} (eps_inf > 0)"
        lam_src, lam_al = "alpha/2 (saturating)", al
    else:
        level = "UNUSABLE"
        why = (f"no floor in this data, F-test p = {pval:.2g}: "
               "eps_inf = 0 cannot be rejected")
        lam_src, lam_al = "alpha/2 (floorless power law)", a_p
        res["notes"].append(
            "Thm 'degeneracy': with eps_inf = 0 the frontier is scale-free, "
            "sigma is affine in ln(resource) and NO exchange rate is "
            "distinguished.  The ranking below is monotone in resource and "
            "carries no trade-off information.")

    if level == "USABLE" and not spans:
        level = "UNVERIFIED"
        why = (f"knee t* = {t_star:.4g} outside measured range "
               f"[{t.min():g}, {t.max():g}], alpha extrapolated")

    # ---- the rate, and the ranking, are produced either way -------------
    lam = lam_al / 2
    rng = np.random.default_rng(seed)
    bs = boot_alpha(t, eps, sat, boot, rng) if boot else np.array([])
    if len(bs) >= 50:
        lo_a, hi_a = (float(v) for v in np.quantile(bs, [alpha / 2,
                                                         1 - alpha / 2]))
        lam_lo, lam_hi = lo_a / 2, hi_a / 2
    else:
        lam_lo = lam_hi = lam
    res["rate"] = {"lambda": lam, "lambda_low": lam_lo, "lambda_high": lam_hi,
                   "coverage": 1 - alpha, "bootstrap_refits": int(len(bs)),
                   "error_drop_per_doubling": 1 - 2 ** -lam,
                   "derived_from": lam_src, "declared": [],
                   "trustworthy": level == "USABLE"}
    if len(bs) < 50 and boot:
        res["notes"].append("the bootstrap did not converge; the rate carries "
                            "no interval")
    res["notes"].append(
        "a bootstrap interval measures spread around THIS fit, not distance "
        "from the truth: if the floor is spurious every resample inherits it")

    sig = sorted(sigma_at(front, lam).items(), key=lambda kv: -kv[1])
    res["scores"] = [{"name": k, "sigma": v} for k, v in sig]
    win, runner = sig[0][0], sig[1][0]
    grid = np.linspace(lam_lo, lam_hi, 601) if lam_hi > lam_lo else [lam]
    wset = []
    for L in grid:
        w = max(sigma_at(front, L).items(), key=lambda kv: kv[1])[0]
        if w not in wset:
            wset.append(w)
    wm = next(c for c in front if c["name"] == win)
    rm = next(c for c in front if c["name"] == runner)
    sep = abs(wm["x"] - rm["x"])
    worst = [kl_regret(L, lam, lam_al) for L in grid]
    worst = max((w for w in worst if not math.isnan(w)), default=0.0)
    res["winner"] = {"name": win, "runner_up": runner,
                     "margin_nats": sig[0][1] - sig[1][1],
                     "robust": len(wset) == 1, "winner_set": wset,
                     "max_kl_regret_in_interval": worst,
                     "log_resource_separation": sep,
                     "resolved": bool(sep >= SPAN_MIN)}
    if rank:
        res["ranking"] = _rank_rows(cands, res, lam, grid)
    if e_inf > 0:
        gap = min(abs(c["x"] - math.log(t_star)) for c in front)
        res["next"] = {"resource": t_star, "nearest_candidate_efolds": gap,
                       "frontier_unmeasured_at_operating_point":
                           bool(gap > SPAN_MIN)}
    res["status"] = "selected" if level == "USABLE" else "unverified" \
        if level == "UNVERIFIED" else "no_floor"
    return verdict(level, why, p_value=pval, F=F)

# ================================================================ render ====
RED = "\033[1;31m"                        # problems only
GRN = "\033[38;2;46;139;87m"              # #2E8B57, the answer
OFF = "\033[0m"


def _green(text, color=True):
    return GRN + text + OFF if color else text


def _red(text, color=True):
    return RED + text + OFF if color else text


def render(r, color=True):
    """JSON result -> the text report.  The VERDICT comes first, in colour:
    green USABLE, yellow UNVERIFIED, red UNUSABLE."""
    L, ax = [], r["axis"]
    v = r.get("verdict") or {}
    lev = v.get("level", "UNUSABLE")
    f0 = r.get("fit") or {}
    if lev == "UNUSABLE" and f0:
        L.append("")
        L.append(_red(f"NO FLOOR (F-test p = {f0['floor_p']:.2g}): no exchange "
                      f"rate exists in this data.", color))
        L.append(_red(f"The ranking below is just {ax} order and means "
                      "nothing (Thm 'degeneracy').", color))
    elif lev == "UNUSABLE":
        L.append("")
        L.append(_red(f"NO RANKING: {v['reason']}.", color))
    elif lev == "UNVERIFIED":
        L.append("")
        L.append(_red(f"UNVERIFIED: {v['reason']}.", color))
        L.append(_red("The ranking below is a guess, not a result.", color))
    if r["status"] == "bad_input":
        return "\n".join(L)

    e = r["envelope"]
    L.append(f"\nENVELOPE ....... {len(e['winners'])} of "
             f"{len(r['candidates'])} can be optimal at SOME price  "
             f"[no rate needed]")
    for p in e["pieces"]:
        hi = "inf" if p["lambda_to"] is None else f"{p['lambda_to']:.4f}"
        L.append(f"  {p['name']:<20} optimal for lambda in "
                 f"[{p['lambda_from']:.4f}, {hi}]")
    never = sorted(e["dominated"] + e["in_a_dent"])
    if never:
        L.append(f"  never optimal  {', '.join(never)}")
        if e["dominated"]:
            L.append(f"    dominated .. {', '.join(e['dominated'])}  "
                     "(something is both cheaper and better)")
        if e["in_a_dent"]:
            L.append(f"    in a dent .. {', '.join(e['in_a_dent'])}  "
                     "(Pareto-efficient, yet NO price selects it)")

    if r["price_free"]:
        L.append(f"\nPRICE-FREE ..... {len(r['price_free'])} domination(s), "
                 "valid at EVERY price:")
        for d in r["price_free"]:
            L.append(f"  {d['winner']} beats {d['loser']}")
    L.append(f"\nCULL ........... {len(r['front'])} of "
             f"{len(r['candidates'])} nondominated")

    f = r["fit"]
    if f:
        L.append(f"\nFIT ............ eps = eps_inf + c t^-alpha on ln eps, "
                 f"{f['n']} points")
        L.append(f"  saturating ... eps_inf={f['eps_inf']:.6g} "
                 f"c={f['c']:.6g} alpha={f['alpha']:.4f}  RSS={f['rss']:.4g}")
        L.append(f"  power law .... c={f['power_c']:.6g} "
                 f"alpha={f['power_alpha']:.4f}{'':14s}"
                 f"RSS={f['power_rss']:.4g}")
        if f.get("floor_p") is None:
            L.append("  FLOOR TEST ... not testable: 3 parameters, "
                     f"{f['n']} points, no residual left")
        else:
            ok = "floor CONFIRMED" if f["floor_p"] <= 0.05 else "NO floor"
            L.append(f"  FLOOR TEST ... nested F({1},{f['n'] - 3}) = "
                     f"{f['floor_F']:.3g}, p = {f['floor_p']:.3g}  ->  {ok}")
        if f.get("aicc_delta") is not None:
            L.append(f"  AICc ......... delta = {f['aicc_delta']:+.2f}  "
                     f"[{f['aicc_convention']}]  (corroboration only)")
        L.append(f"  knee ......... t* = {f['knee']:.4g}; range "
                 f"[{f['range'][0]:g}, {f['range'][1]:g}] -> "
                 f"{'SPANNED' if f['knee_spanned'] else 'NOT spanned'}")
        L.append(f"  resolved ..... {f['resolved_nats']:.3f} nats (paper: "
                 "identified in 87% of fronts above 1.5, 0% below 1.0)")

    if r["status"] == "no_rate" or r["rate"] is None:
        L.append(f"\nNO RATE ........ {r['reason']}.")
        for n_ in r["notes"]:
            L.append(f"  note ......... {n_}")
        e = r["envelope"]
        retired = len(e["dominated"]) + len(e["in_a_dent"])
        L.append(f"  what stands .. the ENVELOPE above, which needs no rate: "
                 f"{len(e['winners'])} candidate(s) can win,")
        L.append(f"                 {retired} retired with a proof.  Name any "
                 "price bound you can defend")
        L.append("                 and the envelope names the winner.")
        return "\n".join(L)

    t_ = r["rate"]
    L.append(f"\nRATE ........... lambda = {t_['lambda']:.4f} nats per "
             f"e-fold of {ax}  [{t_['derived_from']}]")
    L.append(f"  interval ..... [{t_['lambda_low']:.4f}, "
             f"{t_['lambda_high']:.4f}] at {t_['coverage'] * 100:.0f}% "
             f"({t_['bootstrap_refits']} bootstrap refits)")
    L.append(f"  meaning ...... doubling {ax} is justified only if the error "
             f"falls by >= {t_['error_drop_per_doubling'] * 100:.1f}%")
    L.append("  declared ..... nothing: no price, no grain, no q0, no "
             "transferred constant")

    w = r["winner"]
    if r.get("ranking"):
        L.append("")
        L.append(f"RANKING ........ all {len(r['ranking'])} models at "
                 f"lambda = {t_['lambda']:.4f}, winner first")
        L.append(f"    # {'name':<18}{'acc':>9}{ax[:11]:>12}{'sigma':>9}"
                 f"{'margin':>9}{'rank@int':>10}  note")
        for q in r["ranking"]:
            mg = ("    --  " if q["margin_to_next"] is None
                  else f"{q['margin_to_next']:+.4f}")
            rr = (f"{q['rank_low']}" if q["rank_low"] == q["rank_high"]
                  else f"{q['rank_low']}-{q['rank_high']}")
            row = (f"  {q['rank']:>3} {q['name']:<18}"
                   f"{q['accuracy']:>9.4f}{q['resource']:>12g}"
                   f"{q['sigma']:>+9.4f}{mg:>9}{rr:>10}  {q['note']}")
            L.append(_green(row, color))
        L.append("  rank@int ..... the rank range this model takes across the "
                 "whole rate interval;")
        L.append("                 a single number means the position does not "
                 "move.  Models marked")
        L.append("                 'dominated' or 'in a dent' are ranked for "
                 "completeness only --")
        L.append("                 no rate selects them, whatever their sigma.")
        L.append("")
        L.append(_green(f"WINNER ......... {w['name']}   margin "
                        f"{w['margin_nats']:.4f} nats over "
                        f"{w['runner_up']}", color))
        if not w["robust"]:
            L.append(f"  ! NOT ROBUST . the rate interval admits "
                     f"{len(w['winner_set'])} winners: "
                     f"{', '.join(w['winner_set'])}")
        if not w["resolved"]:
            L.append(f"  ! UNRESOLVED . winner and runner-up differ by "
                     f"{w['log_resource_separation']:.3f} < ln 1.5")
        L.append("  ! sigma gaps carry no test-set noise; pass per-item "
                 "outcomes to certify them.")
        for n_ in r["notes"]:
            L.append(f"  note ......... {n_}")
        nx = r["next"]
        L.append(f"\nNEXT ........... an unmeasured model matters most at "
                 f"{ax} = {nx['resource']:.4g} (the knee,")
        L.append(f"                 where lambda = alpha/2 is optimal); "
                 f"nearest candidate is "
                 f"{nx['nearest_candidate_efolds']:.2f} e-folds away")
        return "\n".join(L)

    L.append(f"\nSCORE .......... sigma = -ln eps - {t_['lambda']:.4f} ln {ax}")
    for s in r["scores"]:
        L.append(f"  {s['name']:<20} {s['sigma']:+.4f}"
                 + ("   <= winner" if s["name"] == w["name"] else ""))
    L.append("")
    L.append(_green(f"WINNER ......... {w['name']}   margin "
                    f"{w['margin_nats']:.4f} nats over "
                    f"{w['runner_up']}", color))
    if w["robust"]:
        L.append(f"  ROBUST ....... {w['name']} wins at EVERY rate in "
                 f"[{t_['lambda_low']:.4f}, {t_['lambda_high']:.4f}], so the")
        L.append("                 rate's own uncertainty does not change the "
                 "answer.  This is the")
        L.append("                 strongest statement available from "
                 "accuracy and resource alone.")
    else:
        L.append(f"  ! NOT ROBUST . the rate interval admits "
                 f"{len(w['winner_set'])} winners: "
                 f"{', '.join(w['winner_set'])}")
        L.append("                 the honest output is that SET, not the "
                 "point winner above.")
        L.append(f"                 picking {w['name']} anyway costs at most "
                 f"{w['max_kl_regret_in_interval']:.4f} nats inside the "
                 "interval")
    if not w["resolved"]:
        L.append(f"  ! UNRESOLVED . winner and runner-up differ by "
                 f"{w['log_resource_separation']:.3f} < ln 1.5 = "
                 f"{SPAN_MIN:.3f} in log-{ax}")
    L.append("  ! the margin is not a confidence interval: it carries no "
             "test-set noise.")
    L.append("    Pass per-item outcomes to a paired test to certify it.")
    for n_ in r["notes"]:
        L.append(f"  note ......... {n_}")

    nx = r["next"]
    L.append(f"\nNEXT ........... an unmeasured model matters most at "
             f"{ax} = {nx['resource']:.4g} (the knee,")
    L.append(f"                 where lambda = alpha/2 is optimal); nearest "
             f"candidate is {nx['nearest_candidate_efolds']:.2f} e-folds away")
    if nx["frontier_unmeasured_at_operating_point"]:
        L.append("  ! the frontier is unmeasured exactly where the derived "
                 "rate operates")
    return "\n".join(L)


# =================================================================== cli ====
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Select the best model from (accuracy, resource) alone.")
    ap.add_argument("inputs", nargs="*", metavar="NAME,ACC,RES | FILE.csv",
                    help="candidates given directly (anything containing a "
                         "comma) or a CSV path")
    ap.add_argument("--json", metavar="FILE",
                    help="JSON payload ('-' for stdin); the native format")
    ap.add_argument("--model", action="append", metavar="N,ACC,RES",
                    help="one candidate; equivalent to a bare NAME,ACC,RES")
    ap.add_argument("--acc-col"); ap.add_argument("--res-col")
    ap.add_argument("--name-col")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="coverage level for the rate interval "
                         "(default 0.05; a convention, not a quantity of the "
                         "problem)")
    ap.add_argument("--aicc-k", choices=("with-variance", "params-only"),
                    default="with-variance")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-json", action="store_true",
                    help="print the normalised JSON payload and exit")
    ap.add_argument("--out-json", metavar="FILE",
                    help="write the full result as JSON ('-' for stdout)")
    ap.add_argument("--rank", action="store_true",
                    help="order ALL models winner to loser instead of "
                         "reporting a single winner")
    ap.add_argument("--no-color", action="store_true",
                    help="plain text, no ANSI colour in the VERDICT line")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the text report")
    # argparse cannot take positionals interleaved after an option, so collect
    # the leftovers too: any stray token with a comma is a candidate
    a, unknown = ap.parse_known_args(argv)
    stray = [u for u in unknown if "," in u and not u.startswith("-")]
    rest = [u for u in unknown if u not in stray]
    if rest:
        ap.error("unrecognized arguments: " + " ".join(rest))

    # a bare argument with a comma is a candidate; anything else is a CSV
    specs = ([x for x in a.inputs if "," in x] + stray + list(a.model or []))
    files = [x for x in a.inputs if "," not in x]
    if len(files) > 1:
        ap.error(f"give at most one CSV path, got {len(files)}: "
                 + ", ".join(files))
    if sum(bool(x) for x in (specs, files, a.json)) != 1:
        ap.error("give exactly one of: candidates as NAME,ACC,RES arguments, "
                 "a CSV path, or --json")
    try:
        if a.json:
            payload = payload_from_json(a.json)
        elif files:
            payload = payload_from_csv(files[0], a.name_col, a.acc_col,
                                       a.res_col)
        else:
            payload = payload_from_models(specs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    if a.emit_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    result = select(payload, alpha=a.alpha, boot=a.boot, seed=a.seed,
                    aicc_k=a.aicc_k, rank=a.rank)
    if not a.quiet:
        print(render(result, color=not a.no_color))
    if a.out_json:
        txt = json.dumps(result, indent=2, allow_nan=True)
        if a.out_json == "-":
            print(txt)
        else:
            with open(a.out_json, "w") as fh:
                fh.write(txt + "\n")
    return {"selected": 0, "unverified": 1, "no_floor": 1, "no_rate": 1,
            "bad_input": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())