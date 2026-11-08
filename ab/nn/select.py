#!/usr/bin/env python3
"""
Select the best model from (accuracy, resource) measurements alone.

NOTHING is declared by hand: no price, no grain, no chance level, no
transferred constant.  Only --alpha, the statistical coverage level, which is
a convention and not a quantity of the problem.

JSON is the native input and output format.  `--model` arguments and CSV files
are adapters: both are normalised to the same JSON payload, which is what
`select()` consumes.


THE JSON PAYLOAD
================

    {
      "candidates": [
        {"name": "m0", "accuracy": 0.6058, "resource": 3.0},
        {"name": "m1", "accuracy": 0.7527, "resource": 5.7}
      ]
    }

A bare list of candidates is also accepted.  `accuracy` must be in (0,1);
`resource` > 0 in one unit, measured on one device.  `--emit-json` prints
exactly this payload for whatever input you gave, so a CSV or a set of
`--model` flags can be captured once and replayed.


EXAMPLES OF USE
===============

1. The working case: 9 nondominated models, rate derived from the frontier.

   $ python3 example_general.py front9.csv

   FIT ............ eps = eps_inf + c t^-alpha on ln eps, 9 points
     saturating ... eps_inf=0.0200971 c=0.902677 alpha=0.7999  RSS=0.0008993
     power law .... c=0.578273 alpha=0.5365              RSS=0.195
     AICc ......... delta = +41.21  [with-variance]
     knee ......... t* = 116.4; range [3, 509.507] -> SPANNED

   RATE ........... lambda = alpha/2 = 0.3999 nats per e-fold  [derived]
     interval ..... [0.3947, 0.4066] at 95% (400 bootstrap refits)
     declared ..... nothing: no price, no grain, no q0, no transferred constant

   WINNER ......... m5   margin 0.0127 nats over m6
     ROBUST ....... m5 wins at EVERY rate in [0.3947, 0.4066]

   Eight or more nondominated models is the working range -- see MEASURED
   RELIABILITY below; at six the rate is identified only 9% of the time at
   realistic noise.

2. Rank every model, winner to loser, instead of naming one winner.

   $ python3 example_general.py front9.csv --rank

   RANKING ........ all 9 models at lambda = 0.3999, winner first
       # name                    acc inference_t    sigma   margin  rank@int  note
       1 m5                   0.9515      74.283  +1.3028  +0.0127         1
       2 m6                   0.9620     141.138  +1.2901  +0.0307         2
       3 m7                   0.9697     268.162  +1.2595  +0.0331         3
       ...
       8 m1                   0.7527         5.7  +0.7010  +0.2094         8  in a dent
       9 m0                   0.6058           3  +0.4916     --           9

   `rank@int` is the rank range the model takes across the WHOLE bootstrap
   interval of the rate; a single number means its position does not move,
   `3-5` means the data cannot pin it down.  Models noted `dominated` or
   `in a dent` are listed for completeness only -- no rate selects them,
   whatever their sigma.

3. JSON in, JSON out.

   $ python3 example_general.py --json front.json --out-json result.json
   $ cat front.json | python3 example_general.py --json - --quiet

4. All the models on the command line.

   $ python3 example_general.py \\
         --model m0,0.6058,3 --model m1,0.7527,5.7 --model m2,0.8479,10.8 \\
         --model m3,0.9025,20.6 --model m4,0.9276,39.1 \\
         --model m5,0.9458,74.3 --model m6,0.9570,141 \\
         --model m7,0.9628,268 --model m8,0.9647,509

5. Convert any input to the JSON payload, without selecting.

   $ python3 example_general.py models.csv --emit-json > front.json

6. Columns that cannot be guessed, and a looser coverage level.

   $ python3 example_general.py models.csv \\
         --name-col model --acc-col top1 --res-col latency_ms --alpha 0.10

   The report labels the resource with the CSV's own column name.

7. Too few models -- what you still get, and why there is no winner.

   Three models cannot yield a rate, so this is NOT the intended use; it is
   shown because it is the common situation and the output is still useful.

   $ python3 example_general.py \\
         --model A,0.851,4 --model B,0.9253,9 --model D,0.90,30

   ENVELOPE ....... 2 of 3 can be optimal at SOME price  [no rate needed]
     B                    optimal for lambda in [0.0000, 0.8514]
     A                    optimal for lambda in [0.8514, inf]
     never optimal  D
       dominated .. D  (something is both cheaper and better)

   PRICE-FREE ..... 1 domination(s), valid at EVERY price:
     B beats D

   NO RATE ........ 2 nondominated candidate(s); the curvature fit needs 6.

   D is retired with a proof, and whether A or B wins depends on whether you
   would pay 0.85 nats per e-fold of latency.  No rate can be derived from two
   points -- that is Theorem 'degeneracy', not a missing feature.


USE AS A LIBRARY
================

    from example_general import select, render

    result = select({"candidates": [
        {"name": "m0", "accuracy": 0.6058, "resource": 3.0},
        ...
    ]}, alpha=0.05, rank=True)

    print(render(result))          # the text report
    result["status"]               # "selected" | "no_rate" | "bad_input"
    result["winner"]["name"]       # None unless status == "selected"
    result["winner"]["robust"]     # True if it wins across the rate interval
    result["envelope"]["winners"]  # always present, needs no rate
    result["ranking"]              # None unless rank=True; else one row per
                                   # model with rank, sigma, margin_to_next,
                                   # rank_low / rank_high / rank_stable

`select()` takes and returns plain JSON-serialisable dicts and prints nothing.


EXIT CODES
   0  a rate was derived and a winner named
   1  no rate could be derived -- the ENVELOPE is still printed, and the
      reason is stated (too few nondominated models, AICc prefers the
      floorless power law, or the knee lies outside the measured range)
   2  bad input (accuracy outside (0,1), fewer than two candidates)


WHAT IT REPORTS, in order of how little it assumes

  ENVELOPE     who a linear price can EVER select.  Works at any K, even 2.
               Absence is a proof, not an absence of evidence.
  PRICE-FREE   dominations that hold at every price.
  RATE         lambda = alpha/2 from the frontier's own curvature
               (Paper 1 Alg. 1 rule (ii); Paper 2 supplement, sqrt(kappa)
               = (alpha/2) sin phi at the canonical reference), with a
               bootstrap interval on alpha.
  RANKING      with rank=True / --rank, every model ordered winner to loser
               at the derived rate, each with the rank range it takes across
               the rate's own interval.  Replaces WINNER's score table.
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
import json
import math
import sys

import numpy as np
from scipy.optimize import least_squares

MIN_FRONT = 6
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


# ================================================================ select ====
def select(payload, alpha=0.05, boot=400, seed=0, aicc_k="with-variance",
           rank=False):
    """JSON payload -> JSON result.  Prints nothing, raises nothing.

    rank=True adds result["ranking"]: every candidate ordered winner to
    loser at the derived rate, with the rank range each one takes across the
    rate's own bootstrap interval."""
    cands, axis, bad = normalise(payload)
    res = {"axis": axis, "alpha": alpha, "status": "bad_input",
           "reason": None, "problems": bad,
           "candidates": [{k: c[k] for k in ("name", "accuracy", "resource")}
                          for c in cands],
           "envelope": None, "price_free": [], "front": [], "fit": None,
           "rate": None, "scores": [], "winner": None, "ranking": None,
           "next": None, "notes": []}
    if bad:
        res["reason"] = "; ".join(bad)
        return res

    res["envelope"] = envelope(cands)
    res["price_free"] = dominations(cands)
    losers = {d["loser"] for d in res["price_free"]}
    front = [c for c in cands if c["name"] not in losers]
    front.sort(key=lambda c: c["resource"])
    res["front"] = [c["name"] for c in front]
    res["status"] = "no_rate"

    n = len(front)
    if n < MIN_FRONT:
        res["reason"] = (f"{n} nondominated candidate(s); the curvature fit "
                         f"needs {MIN_FRONT}")
        res["notes"].append(
            f"eps = eps_inf + c t^-alpha has 3 free parameters and must then "
            f"beat the 2-parameter power law on AICc; {n} point(s) cannot "
            f"establish that a floor exists at all")
        res["notes"].append(
            f"to proceed: measure {MIN_FRONT - n} more nondominated model(s), "
            "same test set and device")
        return res

    t = np.array([c["resource"] for c in front])
    eps = np.array([c["eps"] for c in front])
    sat = fit_sat(t, eps)
    c_p, a_p, rss_p = fit_pow(t, eps)
    if sat is None:
        res["reason"] = "the saturating fit did not converge"
        return res
    e_inf, c_s, al = (float(v) for v in sat.x)
    rss_s = float(2 * sat.cost)
    cv = aicc_k == "with-variance"
    A_s, A_p = _aicc(rss_s, n, 3, cv), _aicc(rss_p, n, 2, cv)
    if A_s is None or A_p is None:
        res["reason"] = f"{n} points cannot carry an AICc comparison"
        return res
    d = A_p - A_s
    alt = (lambda x, y: None if x is None or y is None else y - x)(
        _aicc(rss_s, n, 3, not cv), _aicc(rss_p, n, 2, not cv))
    t_star = (c_s / e_inf) ** (1 / al)
    spans = bool(t.min() <= t_star <= t.max())
    res["fit"] = {"n": n, "eps_inf": e_inf, "c": c_s, "alpha": al,
                  "rss": rss_s, "power_c": c_p, "power_alpha": a_p,
                  "power_rss": rss_p, "aicc_delta": d,
                  "aicc_convention": aicc_k, "aicc_delta_other": alt,
                  "knee": t_star, "knee_spanned": spans,
                  "resolved_nats": float(-math.log(eps.min())),
                  "range": [float(t.min()), float(t.max())]}
    if alt is not None and (alt > 0) != (d > 0):
        res["notes"].append(
            f"the other AICc convention gives delta = {alt:+.2f} and flips "
            f"the verdict; at n = {n} the penalty gap dominates the fit")

    if d <= 0:
        res["reason"] = (f"AICc prefers the floorless power law "
                         f"(delta = {d:+.2f})")
        res["notes"].append(
            "Thm 'degeneracy': with eps_inf = 0 the frontier is scale-free "
            "and no rate is distinguished -- the rate is absent from this "
            "data, not hidden")
        if rss_s < rss_p:
            res["notes"].append(
                f"the saturating fit is {rss_p / rss_s:.1f}x tighter; AICc "
                f"rejects it on the n = {n} penalty alone")
        return res
    if not spans:
        res["reason"] = "the knee lies outside the measured range"
        res["notes"].append("alpha is extrapolated, not measured")
        res["notes"].append(f"to proceed: measure a model near "
                            f"{axis} = {t_star:.4g}")
        return res

    rng = np.random.default_rng(seed)
    bs = boot_alpha(t, eps, sat, boot, rng)
    lam = al / 2
    if len(bs) >= 50:
        lo_a, hi_a = (float(v) for v in np.quantile(bs, [alpha / 2,
                                                         1 - alpha / 2]))
        lam_lo, lam_hi = lo_a / 2, hi_a / 2
    else:
        lam_lo = lam_hi = lam
        res["notes"].append("the bootstrap did not converge; the rate "
                            "carries no interval")
    res["rate"] = {"lambda": lam, "lambda_low": lam_lo, "lambda_high": lam_hi,
                   "coverage": 1 - alpha, "bootstrap_refits": int(len(bs)),
                   "error_drop_per_doubling": 1 - 2 ** -lam,
                   "derived_from": "alpha/2", "declared": []}

    sig = sorted(sigma_at(front, lam).items(), key=lambda kv: -kv[1])
    res["scores"] = [{"name": k, "sigma": v} for k, v in sig]
    win, runner = sig[0][0], sig[1][0]
    margin = sig[0][1] - sig[1][1]
    grid = np.linspace(lam_lo, lam_hi, 601) if lam_hi > lam_lo else [lam]
    wset = []
    for L in grid:
        w = max(sigma_at(front, L).items(), key=lambda kv: kv[1])[0]
        if w not in wset:
            wset.append(w)
    wm = next(c for c in front if c["name"] == win)
    rm = next(c for c in front if c["name"] == runner)
    sep = abs(wm["x"] - rm["x"])
    worst = [kl_regret(L, lam, al) for L in grid]
    worst = max((w for w in worst if not math.isnan(w)), default=0.0)
    res["winner"] = {
        "name": win, "runner_up": runner, "margin_nats": margin,
        "robust": len(wset) == 1, "winner_set": wset,
        "max_kl_regret_in_interval": worst,
        "log_resource_separation": sep,
        "resolved": bool(sep >= SPAN_MIN)}
    if rank:
        env_of = {}
        for pc in res["envelope"]["pieces"]:
            env_of[pc["name"]] = [pc["lambda_from"], pc["lambda_to"]]
        dom_by = {d["loser"]: d["winner"] for d in res["price_free"]}
        dent = set(res["envelope"]["in_a_dent"])
        all_sig = sorted(sigma_at(cands, lam).items(), key=lambda kv: -kv[1])
        # rank each candidate at every rate in the interval
        spread = {c["name"]: [len(cands), 1] for c in cands}
        for L in grid:
            order = sorted(sigma_at(cands, L).items(), key=lambda kv: -kv[1])
            for i, (nm, _) in enumerate(order, 1):
                spread[nm][0] = min(spread[nm][0], i)
                spread[nm][1] = max(spread[nm][1], i)
        rows = []
        for i, (nm, sg) in enumerate(all_sig, 1):
            c = next(x for x in cands if x["name"] == nm)
            nxt = all_sig[i][1] if i < len(all_sig) else None
            note = ("dominated by " + dom_by[nm] if nm in dom_by
                    else "in a dent" if nm in dent else "")
            rows.append({"rank": i, "name": nm, "accuracy": c["accuracy"],
                         "resource": c["resource"], "sigma": sg,
                         "margin_to_next": None if nxt is None else sg - nxt,
                         "rank_low": spread[nm][0], "rank_high": spread[nm][1],
                         "rank_stable": spread[nm][0] == spread[nm][1] == i,
                         "optimal_for_lambda_in": env_of.get(nm),
                         "note": note})
        res["ranking"] = rows

    gap = min(abs(c["x"] - math.log(t_star)) for c in front)
    res["next"] = {"resource": t_star, "nearest_candidate_efolds": gap,
                   "frontier_unmeasured_at_operating_point": bool(gap >
                                                                  SPAN_MIN)}
    res["status"] = "selected"
    return res


# ================================================================ render ====
def render(r):
    """JSON result -> the text report."""
    L, ax = [], r["axis"]
    if r["status"] == "bad_input":
        L.append(f"\nSTOP ........... {r['reason']}")
        return "\n".join(L)

    L.append(f"\nINPUT .......... {len(r['candidates'])} candidate(s); "
             f"accuracy and '{ax}' only; alpha = {r['alpha']}")
    for c in sorted(r["candidates"], key=lambda c: c["resource"]):
        L.append(f"  {c['name']:<20} acc={c['accuracy']:.6f}  "
                 f"{ax}={c['resource']:g}")

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
        other = ("" if f["aicc_delta_other"] is None
                 else f"   (other convention {f['aicc_delta_other']:+.2f})")
        L.append(f"  AICc ......... delta = {f['aicc_delta']:+.2f}  "
                 f"[{f['aicc_convention']}]{other}")
        L.append(f"  knee ......... t* = {f['knee']:.4g}; range "
                 f"[{f['range'][0]:g}, {f['range'][1]:g}] -> "
                 f"{'SPANNED' if f['knee_spanned'] else 'NOT spanned'}")
        L.append(f"  resolved ..... {f['resolved_nats']:.3f} nats (paper: "
                 "identified in 87% of fronts above 1.5, 0% below 1.0)")

    if r["status"] == "no_rate":
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
    L.append(f"\nRATE ........... lambda = alpha/2 = {t_['lambda']:.4f} nats "
             f"per e-fold of {ax}  [derived]")
    L.append(f"  interval ..... [{t_['lambda_low']:.4f}, "
             f"{t_['lambda_high']:.4f}] at {t_['coverage'] * 100:.0f}% "
             f"({t_['bootstrap_refits']} bootstrap refits)")
    L.append(f"  meaning ...... doubling {ax} is justified only if the error "
             f"falls by >= {t_['error_drop_per_doubling'] * 100:.1f}%")
    L.append("  declared ..... nothing: no price, no grain, no q0, no "
             "transferred constant")

    w = r["winner"]
    if r.get("ranking"):
        L.append(f"\nRANKING ........ all {len(r['ranking'])} models at "
                 f"lambda = {t_['lambda']:.4f}, winner first")
        L.append(f"    # {'name':<18}{'acc':>9}{ax[:11]:>12}{'sigma':>9}"
                 f"{'margin':>9}{'rank@int':>10}  note")
        for q in r["ranking"]:
            mg = ("    --  " if q["margin_to_next"] is None
                  else f"{q['margin_to_next']:+.4f}")
            rr = (f"{q['rank_low']}" if q["rank_low"] == q["rank_high"]
                  else f"{q['rank_low']}-{q['rank_high']}")
            L.append(f"  {q['rank']:>3} {q['name']:<18}"
                     f"{q['accuracy']:>9.4f}{q['resource']:>12g}"
                     f"{q['sigma']:>+9.4f}{mg:>9}{rr:>10}  {q['note']}")
        L.append("  rank@int ..... the rank range this model takes across the "
                 "whole rate interval;")
        L.append("                 a single number means the position does not "
                 "move.  Models marked")
        L.append("                 'dominated' or 'in a dent' are ranked for "
                 "completeness only --")
        L.append("                 no rate selects them, whatever their sigma.")
        L.append(f"\nWINNER ......... {w['name']}   margin "
                 f"{w['margin_nats']:.4f} nats over {w['runner_up']}")
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
    L.append(f"\nWINNER ......... {w['name']}   margin "
             f"{w['margin_nats']:.4f} nats over {w['runner_up']}")
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
    ap.add_argument("csv", nargs="?", help="CSV of candidates")
    ap.add_argument("--json", metavar="FILE",
                    help="JSON payload ('-' for stdin); the native format")
    ap.add_argument("--model", action="append", metavar="N,ACC,RES",
                    help="one candidate; repeat for each")
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
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the text report")
    a = ap.parse_args(argv)

    srcs = sum(bool(x) for x in (a.csv, a.json, a.model))
    if srcs != 1:
        ap.error("give exactly one of: a CSV path, --json, or --model flags")
    try:
        if a.json:
            payload = payload_from_json(a.json)
        elif a.csv:
            payload = payload_from_csv(a.csv, a.name_col, a.acc_col,
                                       a.res_col)
        else:
            payload = payload_from_models(a.model)
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
        print(render(result))
    if a.out_json:
        txt = json.dumps(result, indent=2, allow_nan=True)
        if a.out_json == "-":
            print(txt)
        else:
            with open(a.out_json, "w") as fh:
                fh.write(txt + "\n")
    return {"selected": 0, "no_rate": 1, "bad_input": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
