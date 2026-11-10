#!/usr/bin/env python3
"""Regression tests for ab.nn.select.

Run from the repository root:

    python -m unittest discover -s tests -v
    python -m unittest tests.test_select.TestPublishedCases -v

The suite is organised by how little each group assumes, the same order the
report prints in, so a failure localises:

  TestInput            validation and the CSV / command-line adapters
  TestEnvelope         who a price can ever select; needs no rate
  TestPriceFree        dominations that hold at every price
  TestJet              the curve from three accuracies; the equal-spacing
                       identity is exact and is asserted as such
  TestRung             the repaired third-order test from four accuracies
  TestVerdict          USABLE / UNVERIFIED / UNUSABLE and the status map
  TestRate             the derived price, and which reading supplies it
  TestScore            the score's invariance to the candidate set
  TestBandMinimax      the band, the worst-case regret and the tie-break
  TestPublishedCases   FROZEN winners for the README's own examples
  TestRobustness       render() and select() never raise, on anything

TestPublishedCases is the one that guards the user-visible answer: its
expected values are what BOTH this implementation and the pre-band one
return, so a change there is a change in what the tool tells people and has
to be deliberate.
"""
import io
import math
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ab.nn.select import (                                   # noqa: E402
    JET_MIN_NODES, MIN_FIT, SPAN_MIN, _dd, dominations, envelope,
    fit_pow, frame_from_csv, frame_from_specs, jet_at, marginal_curve,
    minimax_band, normalise, read_jet, read_rung, render, select,
    sigma_matrix,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------- fixtures ----
def frontier(K=9, floor=0.05, c=0.9, alpha=0.8, lo=0.0, hi=5.5, noise=0.0,
             seed=0, name="m"):
    """A saturating frontier eps = floor + c t^-alpha on a geometric grid."""
    t = np.exp(np.linspace(lo, hi, K))
    eps = floor + c * t ** -alpha
    if noise:
        eps = eps * (1 + np.random.default_rng(seed).normal(0, noise, K))
    eps = np.clip(eps, 1e-9, 1 - 1e-9)
    return pd.DataFrame({"name": [f"{name}{i}" for i in range(K)],
                         "accuracy": 1 - eps, "resource": t})


def power_law(K=7, c=0.8, alpha=0.5, hi=5.0):
    """No floor at all: eps = c t^-alpha, straight in log-log."""
    t = np.exp(np.linspace(0, hi, K))
    return pd.DataFrame({"name": [f"q{i}" for i in range(K)],
                         "accuracy": 1 - c * t ** -alpha, "resource": t})


SPECS = {
    "mai2026_ssim": "koreatechV2,0.8499,4.35 Godzilla,0.8831,215160 "
                    "AntSR,0.8731,4.52 LoongSR,0.8586,5729 IN2GM,0.8701,46.5 "
                    "z6,0.8699,5.49 AIO_MAI,0.8634,41.1 AntYSP,0.8729,4.33",
    "mai2026_psnr3": "koreatechV2,0.998758348,4.35,464,77.1 "
                     "Godzilla,0.999085887,215160,215160,16253 "
                     "AntSR,0.998995384,4.52,404,65.3 "
                     "LoongSR,0.998943182,5729,5729,708 "
                     "IN2GM,0.998964858,46.5,526,64.3 "
                     "z6,0.998983751,5.49,330,67.4 "
                     "AIO_MAI,0.998950458,41.1,610,96.1 "
                     "AntYSP,0.998990747,4.33,338,54.9",
    "effnet": "B7,0.843,37,66 B0,0.771,0.39,5.3 B4,0.829,4.2,19 "
              "B6,0.840,19,43 B2,0.801,1,9.2 B1,0.791,0.7,7.8 "
              "B3,0.816,1.8,12 B5,0.836,9.9,30",
    "yolo11": "YOLO11l,0.534,6.2,238.6,25.3,87.2 "
              "YOLO11m,0.515,4.7,183.2,20.1,68.1 "
              "YOLO11x,0.547,11.3,462.8,56.9,195.3 "
              "YOLO11n,0.395,1.5,56.1,2.6,6.5 YOLO11s,0.470,2.5,90.0,9.4,21.6",
    "mai2021": "Airia-Det,0.9300,17.51 Sidiki,0.7800,1.74 "
               "ByteScene,0.9500,4.44 MobileNet-V2,0.9417,16.38 "
               "PyImageSearch,0.8967,45.88 ALONG,0.9467,64.45 "
               "neptuneai,0.8367,4.17 EVAI,0.9300,3.35 DataArt,0.9150,54.13 "
               "Team-Horizon,0.9233,7.7",
    "powerlaw": "Q05,0.475,0.5 Q15,0.609,1.5 Q3,0.656,3 Q7,0.742,7 "
                "Q14,0.797,14 Q32,0.833,32 Q72,0.861,72",
}


def run_spec(key, **kw):
    f = frame_from_specs(SPECS[key].split())
    return select(f, rank=True, res_col=f.attrs["axes"], **kw)


# ----------------------------------------------------------------- input ---
class TestInput(unittest.TestCase):
    def test_accuracy_outside_unit_interval_is_rejected(self):
        df = pd.DataFrame({"name": list("abc"), "accuracy": [1.5, .6, .7],
                           "resource": [1., 2., 3.]})
        r = select(df, boot=0)
        self.assertIn("accuracy 1.5 is outside (0,1)", " ".join(r["problems"]))

    def test_zero_and_one_are_outside_too(self):
        for bad in (0.0, 1.0):
            df = pd.DataFrame({"name": list("abc"),
                               "accuracy": [bad, .6, .7],
                               "resource": [1., 2., 3.]})
            self.assertTrue(select(df, boot=0)["problems"],
                            f"accuracy {bad} should be rejected")

    def test_non_positive_resource_is_rejected(self):
        df = pd.DataFrame({"name": list("abc"), "accuracy": [.5, .6, .7],
                           "resource": [0., 2., 3.]})
        self.assertIn("is not positive", " ".join(select(df, boot=0)["problems"]))

    def test_fewer_than_two_candidates(self):
        df = pd.DataFrame({"name": ["a"], "accuracy": [.5], "resource": [1.]})
        r = select(df, boot=0)
        self.assertEqual(r["status"], "bad_input")

    def test_missing_column(self):
        df = pd.DataFrame({"name": list("ab"), "accuracy": [.5, .6]})
        self.assertEqual(select(df, boot=0)["status"], "bad_input")

    def test_non_numeric_cells_are_dropped_not_crashed(self):
        df = pd.DataFrame({"name": list("abcd"),
                           "accuracy": [".5", "oops", ".7", ".8"],
                           "resource": ["1", "2", "3", "4"]})
        r = select(df, boot=0)
        self.assertTrue(any("numeric" in p for p in r["problems"]))

    def test_specs_adapter_names_axes_p1_pn(self):
        f = frame_from_specs(["a,0.5,1,2", "b,0.6,2,3"])
        self.assertEqual(f.attrs["axes"], ["p1", "p2"])

    def test_specs_adapter_rejects_ragged_rows(self):
        with self.assertRaises(ValueError):
            frame_from_specs(["a,0.5,1,2", "b,0.6,2"])

    def test_specs_adapter_rejects_short_rows(self):
        with self.assertRaises(ValueError):
            frame_from_specs(["a,0.5"])

    def test_csv_adapter_guesses_headers(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.csv")
            pd.DataFrame({"model": list("abc"), "top1_acc": [.5, .6, .7],
                          "latency_ms": [1., 2., 3.]}).to_csv(p, index=False)
            buf, keep = io.StringIO(), sys.stdout
            sys.stdout = buf
            try:
                f = frame_from_csv(p)
            finally:
                sys.stdout = keep
            self.assertIn("accuracy", f.columns)
            self.assertEqual(f.attrs["axes"], ["latency_ms"])

    def test_csv_adapter_rejects_unknown_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.csv")
            pd.DataFrame({"model": list("ab"), "acc": [.5, .6],
                          "ms": [1., 2.]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                frame_from_csv(p, res_cols=["nope"])

    def test_normalise_derives_axes_and_logs(self):
        out, axes, bad = normalise(frontier(4))
        self.assertEqual((axes, bad), (["resource"], []))
        self.assertTrue(np.allclose(out["x__resource"],
                                    np.log(out["resource"])))
        self.assertTrue(np.allclose(out["eps"], 1 - out["accuracy"]))

    def test_res_col_accepts_string_or_list(self):
        df = frontier(5).rename(columns={"resource": "ms"})
        for spec in ("ms", ["ms"]):
            self.assertEqual(select(df, res_col=spec, boot=0)["axes"], ["ms"])


# -------------------------------------------------------------- envelope ---
class TestEnvelope(unittest.TestCase):
    def setUp(self):
        self.df, _, _ = normalise(frontier(9))

    def test_pieces_tile_the_positive_price_axis(self):
        p = envelope(self.df)["pieces"]
        self.assertAlmostEqual(float(p["lambda_from"].iloc[0]), 0.0)
        self.assertTrue(pd.isna(p["lambda_to"].iloc[-1]))
        for i in range(len(p) - 1):
            self.assertAlmostEqual(float(p["lambda_to"].iloc[i]),
                                   float(p["lambda_from"].iloc[i + 1]))

    def test_breakpoints_are_the_pairwise_break_evens(self):
        p = envelope(self.df)["pieces"]
        by = self.df.set_index("name")
        for i in range(len(p) - 1):
            a, b = p["name"].iloc[i], p["name"].iloc[i + 1]
            lam = float(p["lambda_to"].iloc[i])
            # sigma_a(lam) == sigma_b(lam) at a breakpoint
            sa = -math.log(by.at[a, "eps"]) - lam * by.at[a, "x__resource"]
            sb = -math.log(by.at[b, "eps"]) - lam * by.at[b, "x__resource"]
            self.assertAlmostEqual(sa, sb, places=9)

    def test_envelope_is_cheapest_first(self):
        p = envelope(self.df)["pieces"]
        res = self.df.set_index("name")["resource"]
        got = [res[n] for n in p["name"]]
        self.assertEqual(got, sorted(got, reverse=True),
                         "a rising price must move to cheaper models")

    def test_a_dent_is_pareto_yet_selected_by_no_price(self):
        # three collinear points plus one pushed just below the chord
        t = np.array([1., 2., 4., 8.])
        eps = np.array([.5, .3, .28, .1])      # the third sits in a dent
        df, _, _ = normalise(pd.DataFrame(
            {"name": list("abcd"), "accuracy": 1 - eps, "resource": t}))
        e = envelope(df)
        self.assertIn("c", e["in_a_dent"])
        self.assertNotIn("c", e["winners"])
        self.assertTrue(dominations(df)["loser"].eq("c").sum() == 0,
                        "a dent point is not dominated by anyone")

    def test_absence_from_the_envelope_is_reported_on_the_ranking(self):
        r = select(frontier(9), rank=True, boot=0)
        rk = r["ranking"].set_index("name")
        for n in r["envelope"]["in_a_dent"] + r["envelope"]["dominated"]:
            self.assertTrue(pd.isna(rk.at[n, "optimal_from"]))

    def test_works_with_only_two_candidates(self):
        df, _, _ = normalise(pd.DataFrame(
            {"name": list("ab"), "accuracy": [.5, .6], "resource": [1., 2.]}))
        self.assertEqual(len(envelope(df)["pieces"]), 2)


# ------------------------------------------------------------ price-free ---
class TestPriceFree(unittest.TestCase):
    def test_cheaper_and_better_dominates(self):
        df, axes, _ = normalise(pd.DataFrame(
            {"name": list("ab"), "accuracy": [.9, .5], "resource": [1., 2.]}))
        d = dominations(df, axes)
        self.assertEqual((list(d["winner"]), list(d["loser"])), (["a"], ["b"]))

    def test_domination_needs_every_axis(self):
        df = pd.DataFrame({"name": list("ab"), "accuracy": [.9, .5],
                           "r0": [1., 2.], "r1": [9., 2.]})
        n, axes, _ = normalise(df, res_col=["r0", "r1"])
        self.assertEqual(len(dominations(n, axes)), 0,
                         "a is better and cheaper on r0 but dearer on r1")

    def test_price_free_losers_are_culled(self):
        df = frontier(6)
        df = pd.concat([df, pd.DataFrame(
            {"name": ["dud"], "accuracy": [df["accuracy"].min() - .01],
             "resource": [df["resource"].max() * 2]})], ignore_index=True)
        r = select(df, rank=True, boot=0)
        self.assertIn("dud", set(r["price_free"]["loser"]))
        self.assertNotIn("dud", r["front"])
        self.assertEqual(r["ranking"].set_index("name").at["dud", "note"][:12],
                         "dominated by")

    def test_one_axis_and_many_axes_agree_on_one_axis(self):
        df, axes, _ = normalise(frontier(12, noise=.05, seed=4))
        many = dominations(df.assign(x__dup=df["x__resource"],
                                     dup=df["resource"]), ["resource", "dup"])
        one = dominations(df, axes)
        self.assertEqual(sorted(map(tuple, one.values)),
                         sorted(map(tuple, many.values)))


# ------------------------------------------------------------------- jet ---
class TestJet(unittest.TestCase):
    def test_equal_spacing_reduces_to_the_papers_formula(self):
        rng = np.random.default_rng(1)
        checked = 0
        for _ in range(3000):
            h, x0 = rng.uniform(.05, 3), rng.uniform(-3, 6)
            lp, l0, lm = sorted(rng.uniform(-7, -.01, 3))
            if (lm - 2 * l0 + lp) <= 0:
                continue
            j = jet_at([x0 - h, x0, x0 + h], [lm, l0, lp])
            if j is None:
                continue
            paper = (lp - lm) ** 2 / (4 * (lm - 2 * l0 + lp))
            self.assertAlmostEqual(j["n_eff"] / paper, 1.0, places=9)
            self.assertAlmostEqual(j["s"], (lm - lp) / (2 * h), places=9)
            self.assertAlmostEqual(j["kappa"], (lm - 2 * l0 + lp) / h ** 2,
                                   places=9)
            checked += 1
        self.assertGreater(checked, 500)

    def test_n_eff_does_not_depend_on_the_step(self):
        l3 = [-1.0, -1.9, -2.5]
        vals = [jet_at([-h, 0., h], l3)["n_eff"] for h in (.25, 1., 4.)]
        for v in vals[1:]:
            self.assertAlmostEqual(v, vals[0], places=9)

    def test_lambda_star_is_the_slope(self):
        j = jet_at([-1., 0., 1.], [-1.0, -1.9, -2.5])
        self.assertAlmostEqual(j["lambda_star"], j["s"], places=12)

    def test_unreadable_when_not_improving_or_not_log_convex(self):
        self.assertIsNone(jet_at([0., 1., 2.], [-2., -1.5, -1.]))  # worsening
        self.assertIsNone(jet_at([0., 1., 2.], [-1., -1.9, -3.]))  # concave
        self.assertIsNone(jet_at([0., 0., 1.], [-1., -2., -3.]))   # repeated

    def test_divided_differences_match_a_known_polynomial(self):
        x = [0., 1., 3., 4.]
        f = [2 + 3 * v - v ** 2 + 0.5 * v ** 3 for v in x]
        d1, d2, d3 = _dd(x, f)
        self.assertAlmostEqual(d3, 0.5, places=9)   # leading coefficient
        self.assertAlmostEqual(d2, -1 + 0.5 * (0 + 1 + 3), places=9)

    def test_reads_every_interior_triple(self):
        xs, ls = marginal_curve(*[normalise(frontier(9))[0]], "resource")
        j = read_jet(xs, ls)
        self.assertEqual(j["n_nodes"], len(xs) - 2)
        self.assertTrue(all(n["n_eff"] > 0 for n in j["nodes"]))

    def test_a_single_triple_is_reported_but_never_prices(self):
        r = run_spec("mai2026_ssim")
        self.assertEqual(len(r["front"]), 3)
        jet = r["jet"]
        self.assertEqual(jet["n_nodes"], 1)
        self.assertIsNotNone(jet["lambda"], "the quote is still reported")
        self.assertLess(jet["n_nodes"], JET_MIN_NODES)
        self.assertIn("alpha/2", r["rate"]["derived_from"],
                      "one triple must not displace the fitted price")

    def test_the_jet_prices_only_where_the_floor_was_not_confirmed(self):
        r = select(frontier(12), rank=True)
        if r["verdict"]["level"] == "USABLE":
            self.assertIn("alpha/2", r["rate"]["derived_from"])
            self.assertAlmostEqual(r["prices"]["lambda"].iloc[0],
                                   r["prices"]["fitted_lambda"].iloc[0])

    def test_jet_price_switch_restores_alpha_over_two(self):
        a = run_spec("effnet")
        b = run_spec("effnet", jet_price=False)
        for i in range(len(b["prices"])):
            self.assertAlmostEqual(b["prices"]["lambda"].iloc[i],
                                   b["prices"]["fitted_lambda"].iloc[i])
        self.assertEqual(a["winner"]["name"], b["winner"]["name"])

    def test_no_floor_keeps_the_fitted_price(self):
        r = select(power_law(), rank=True, boot=0)
        self.assertEqual(r["verdict"]["level"], "UNUSABLE")
        self.assertIn("alpha/2", r["rate"]["derived_from"])


# ------------------------------------------------------------------ rung ---
class TestRung(unittest.TestCase):
    def test_unreadable_on_a_straight_log_log_frontier(self):
        df, _, _ = normalise(power_law())
        g = read_rung(*marginal_curve(df, "resource"))
        self.assertFalse(g["readable"])
        self.assertIn("kappa > 0", g["reason"])

    def test_needs_four_points(self):
        df, _, _ = normalise(frontier(3))
        g = read_rung(*marginal_curve(df, "resource"))
        self.assertFalse(g["readable"])
        self.assertIn("needs 4", g["reason"])

    def test_floor_sign_detects_a_floor_without_any_fit(self):
        df, _, _ = normalise(frontier(9, floor=.05))
        g = read_rung(*marginal_curve(df, "resource"))
        self.assertTrue(g["readable"])
        self.assertTrue(g["floor_sign"], "D1 < 1 <=> eps_inf > 0")
        self.assertLess(g["D1"], 1.0)

    def test_nodes_are_the_widest_quadruple_at_index_thirds(self):
        df, _, _ = normalise(frontier(10))
        xs, ls = marginal_curve(df, "resource")
        g = read_rung(xs, ls)
        self.assertAlmostEqual(g["nodes"][0], math.exp(xs[0]), places=9)
        self.assertAlmostEqual(g["nodes"][-1], math.exp(xs[-1]), places=9)
        self.assertGreaterEqual(g["spacing_ratio"], 1.0)

    def test_the_strict_rung_is_the_plug_in_invariant(self):
        df, _, _ = normalise(frontier(9))
        g = read_rung(*marginal_curve(df, "resource"))
        self.assertAlmostEqual(
            g["D1"], 1.5 * g["dd1"] * g["dd3"] / g["dd2"] ** 2 - 1, places=12)
        self.assertEqual(g["rung_strict"], g["D1"] >= 0)
        self.assertEqual(g["rung_loose"], g["dd3"] >= 0)

    def test_the_verdict_carries_the_rung(self):
        v = select(frontier(9), rank=True, boot=0)["verdict"]
        self.assertIn("rung", v)
        self.assertEqual(set(v["rung"]) >= {"D1", "readable", "floor_sign"},
                         True)


# --------------------------------------------------------------- verdict ---
class TestVerdict(unittest.TestCase):
    def test_saturating_frontier_is_usable(self):
        r = select(frontier(12, floor=.05, hi=6.5), rank=True)
        self.assertEqual(r["verdict"]["level"], "USABLE")
        self.assertEqual(r["status"], "selected")
        self.assertTrue(r["rate"]["trustworthy"])

    def test_floorless_frontier_is_unusable(self):
        r = select(power_law(), rank=True, boot=0)
        self.assertEqual((r["verdict"]["level"], r["status"]),
                         ("UNUSABLE", "no_floor"))
        self.assertFalse(r["rate"]["trustworthy"])

    def test_too_few_points_is_unverified(self):
        r = run_spec("mai2021")
        self.assertEqual((r["verdict"]["level"], r["status"]),
                         ("UNVERIFIED", "unverified"))

    def test_two_nondominated_candidates_give_no_rate(self):
        df = pd.DataFrame({"name": list("abc"), "accuracy": [.5, .6, .55],
                           "resource": [1., 2., 3.]})
        r = select(df, rank=True, boot=0)
        self.assertEqual(r["status"], "no_rate")
        self.assertLess(len(r["front"]), MIN_FIT)
        self.assertIsNotNone(r["envelope"], "the envelope still stands")

    def test_status_values_are_the_documented_set(self):
        seen = {run_spec(k)["status"] for k in SPECS}
        self.assertTrue(seen <= {"selected", "unverified", "no_floor",
                                 "no_rate", "bad_input"}, seen)


# ------------------------------------------------------------------ rate ---
class TestRate(unittest.TestCase):
    def test_error_drop_per_doubling_matches_lambda(self):
        r = select(frontier(12), rank=True)
        lam = r["rate"]["lambda"]
        self.assertAlmostEqual(r["rate"]["error_drop_per_doubling"],
                               1 - 2 ** -lam, places=12)

    def test_one_price_per_axis(self):
        r = run_spec("yolo11")
        self.assertEqual(len(r["prices"]), 4)
        self.assertEqual(set(r["rate"]["per_axis"]) <= set(r["axes"]), True)

    def test_an_axis_without_range_gets_no_price_and_is_excluded(self):
        df = frontier(8)
        df["flat"] = np.linspace(1.0, 1.05, len(df))   # spans < MARG_MIN_SPAN
        r = select(df, res_col=["resource", "flat"], rank=True, boot=0)
        row = r["prices"].set_index("axis").loc["flat"]
        self.assertTrue(pd.isna(row["lambda"]))
        self.assertNotIn("flat", r["rate"]["per_axis"])
        self.assertTrue(any("flat" in n for n in r["notes"]))

    def test_interval_brackets_the_price(self):
        r = select(frontier(12), rank=True)
        ax = r["axis"]
        self.assertLessEqual(r["rate"]["lambda_low"][ax], r["rate"]["lambda"])
        self.assertLessEqual(r["rate"]["lambda"], r["rate"]["lambda_high"][ax])

    def test_axis_drop_reported_with_several_axes(self):
        r = run_spec("effnet")
        self.assertIsNotNone(r["axis_drop"])
        self.assertAlmostEqual(float(r["axis_drop"]["cost_share"].sum()), 1.0,
                               places=6)


# ----------------------------------------------------------------- score ---
class TestScore(unittest.TestCase):
    def test_eps_eff_is_exp_minus_sigma_and_score_is_monotone(self):
        rk = select(frontier(9), rank=True, boot=0)["ranking"]
        self.assertTrue(np.allclose(rk["eps_eff"], np.exp(-rk["sigma"])))
        self.assertTrue(np.allclose(rk["score"], 1 / (1 + rk["eps_eff"])))
        self.assertTrue((rk["score"].diff().dropna() <= 1e-12).all(),
                        "the ranking must be sorted by score")

    def test_score_does_not_move_when_other_candidates_are_removed(self):
        df = frontier(12)
        full = select(df, rank=True, boot=0)
        lam = full["rate"]["lambda"]
        kept = df[df["name"].isin(df["name"].iloc[:8])]
        part = select(kept, rank=True, boot=0)
        # re-score by hand at the SAME price, which is what invariance means
        for frame in (full, part):
            rk = frame["ranking"].set_index("name")
            for n in rk.index:
                eps = 1 - rk.at[n, "accuracy"]
                t = rk.at[n, "resource"]
                self.assertAlmostEqual(rk.at[n, "score"],
                                       1 / (1 + eps * t ** frame["rate"]["lambda"]),
                                       places=9)
        self.assertGreater(lam, 0)

    def test_score_is_in_the_unit_interval(self):
        rk = select(frontier(9), rank=True, boot=0)["ranking"]
        self.assertTrue(((rk["score"] > 0) & (rk["score"] < 1)).all())

    def test_ranking_columns_and_order(self):
        rk = select(frontier(9), rank=True, boot=0)["ranking"]
        lead = ["name", "score", "rank", "accuracy", "resource"]
        self.assertEqual(list(rk.columns)[:5], lead)
        for c in ("sigma", "eps_eff", "margin_to_next", "rank_low",
                  "rank_high", "rank_stable", "optimal_from", "optimal_to",
                  "regret_sup", "in_band", "note"):
            self.assertIn(c, rk.columns)
        self.assertEqual(list(rk["rank"]), list(range(1, len(rk) + 1)))


# ---------------------------------------------------------- band, minimax --
class TestBandMinimax(unittest.TestCase):
    def test_regret_is_zero_for_the_envelope_optimum_at_its_own_price(self):
        df, axes, _ = normalise(frontier(9))
        lam = {"resource": 0.4}
        mm = minimax_band(df, axes, [lam], dict(zip(df["name"], [0] * len(df))))
        self.assertEqual(len(mm["band"]), 1)
        self.assertAlmostEqual(mm["regret"][mm["band"][0]], 0.0, places=12)

    def test_regret_is_non_negative_and_zero_somewhere(self):
        r = select(frontier(12), rank=True)
        reg = r["minimax"]["regret"]
        self.assertTrue(all(v >= -1e-12 for v in reg.values()))
        self.assertTrue(any(abs(reg[n]) < 1e-9 for n in r["band"]["members"]))

    def test_minimax_is_never_worse_than_the_point_pick(self):
        for K in (5, 8, 12, 20):
            r = select(frontier(K, noise=.02, seed=K), rank=True)
            m = r["minimax"]
            self.assertLessEqual(m["regret_nats"],
                                 m["point_regret_nats"] + 1e-12,
                                 f"K={K}: the derived price must lie in Lambda")

    def test_the_pick_comes_from_the_band(self):
        for K in (5, 9, 15):
            r = select(frontier(K, noise=.03, seed=K), rank=True)
            self.assertIn(r["minimax"]["pick"], r["band"]["members"])
            self.assertTrue(set(r["band"]["members"]) <= set(r["front"]))

    def test_a_band_of_one_leaves_the_winner_at_the_point_pick(self):
        for K in (6, 9, 12):
            r = select(frontier(K), rank=True)
            if r["band"]["size"] == 1:
                self.assertEqual(r["winner"]["name"],
                                 r["minimax"]["point_pick"])
                self.assertEqual(r["winner"]["tiebreak"], "point")

    def test_tiebreak_point_is_the_sigma_argmax(self):
        for K in (5, 9, 14):
            r = select(frontier(K, noise=.03, seed=K + 1), rank=True,
                       tiebreak="point")
            self.assertEqual(r["winner"]["name"], r["scores"]["name"].iloc[0])
            self.assertEqual(r["winner"]["name"], r["minimax"]["point_pick"])
            self.assertFalse(r["minimax"]["applied"])

    def test_support_mode_widens_lambda(self):
        r = select(frontier(9), rank=True, tiebreak="support")
        i = select(frontier(9), rank=True, tiebreak="interval")
        self.assertGreaterEqual(r["band"]["size"], i["band"]["size"])
        self.assertLessEqual(r["band"]["lambda_from"], i["band"]["lambda_from"])
        self.assertGreaterEqual(r["band"]["lambda_to"], i["band"]["lambda_to"])

    def test_minimax_is_an_alias_for_support(self):
        a = select(frontier(9), rank=True, tiebreak="support")
        b = select(frontier(9), rank=True, tiebreak="minimax")
        self.assertEqual(a["winner"]["name"], b["winner"]["name"])
        self.assertEqual(a["band"]["members"], b["band"]["members"])

    def test_no_tie_is_broken_when_the_floor_is_rejected(self):
        r = select(power_law(), rank=True, boot=0)
        self.assertEqual(r["verdict"]["level"], "UNUSABLE")
        self.assertFalse(r["minimax"]["applied"])
        self.assertEqual(r["winner"]["name"], r["minimax"]["point_pick"])

    def test_supported_span_is_reported_whatever_lambda_is(self):
        r = select(frontier(9), rank=True, tiebreak="interval")
        span = r["band"]["supported_span"]
        self.assertIsNotNone(span)
        self.assertLess(span[0], span[1])

    def test_regret_sup_column_agrees_with_the_minimax_block(self):
        r = select(frontier(12, noise=.02, seed=2), rank=True)
        rk = r["ranking"].set_index("name")
        for n, v in r["minimax"]["regret"].items():
            self.assertAlmostEqual(rk.at[n, "regret_sup"], v, places=12)
            self.assertEqual(bool(rk.at[n, "in_band"]),
                             n in set(r["band"]["members"]))

    def test_winner_fields_are_consistent(self):
        r = select(frontier(12, noise=.03, seed=7), rank=True)
        w = r["winner"]
        self.assertEqual(w["robust"],
                         len(w["winner_set"]) == 1
                         and w["winner_set"][0] == w["name"])
        self.assertAlmostEqual(w["score"], 1 / (1 + math.exp(-w["sigma"])),
                               places=12)
        self.assertEqual(w["resolved"],
                         w["log_resource_separation"] >= SPAN_MIN)


# ------------------------------------------------------- published cases ---
class TestPublishedCases(unittest.TestCase):
    """FROZEN. These are the README's own examples, and the values are what
    the pre-band implementation returned too.  A diff here changes the answer
    users see: update it only on purpose, and say so in the release notes."""

    EXPECTED = {
        "mai2026_ssim":  ("AntYSP",    "unverified", "UNVERIFIED", 3),
        "mai2026_psnr3": ("AntYSP",    "no_floor",   "UNUSABLE",   4),
        "effnet":        ("B0",        "unverified", "UNVERIFIED", 8),
        "yolo11":        ("YOLO11n",   "no_floor",   "UNUSABLE",   5),
        "mai2021":       ("ByteScene", "unverified", "UNVERIFIED", 3),
        "powerlaw":      ("Q72",       "no_floor",   "UNUSABLE",   7),
    }

    def test_frozen_winners(self):
        for key, (win, status, level, front) in self.EXPECTED.items():
            with self.subTest(case=key):
                r = run_spec(key)
                self.assertEqual(r["winner"]["name"], win)
                self.assertEqual(r["status"], status)
                self.assertEqual(r["verdict"]["level"], level)
                self.assertEqual(len(r["front"]), front)

    def test_mai2021_keeps_its_documented_cull(self):
        r = run_spec("mai2021")
        self.assertEqual(len(r["front"]), 3)
        self.assertEqual(len(r["price_free"]), 7)

    def test_results_are_deterministic(self):
        for key in ("effnet", "mai2021"):
            with self.subTest(case=key):
                a, b = run_spec(key), run_spec(key)
                self.assertEqual(a["winner"], b["winner"])
                self.assertTrue(a["ranking"].equals(b["ranking"]))

    def test_tiebreak_point_reproduces_the_frozen_winners(self):
        for key, (win, _, _, _) in self.EXPECTED.items():
            with self.subTest(case=key):
                self.assertEqual(
                    run_spec(key, tiebreak="point")["winner"]["name"], win,
                    "the default must agree with the pre-band behaviour here")


# ------------------------------------------------------------ robustness ---
class TestRobustness(unittest.TestCase):
    def test_render_never_raises(self):
        frames = [frontier(12), frontier(3), power_law(),
                  pd.DataFrame({"name": list("ab"), "accuracy": [.5, .6],
                                "resource": [1., 2.]}),
                  pd.DataFrame({"name": list("ab"), "accuracy": [5., .6],
                                "resource": [1., 2.]})]
        for i, df in enumerate(frames):
            with self.subTest(frame=i):
                for color in (True, False):
                    out = render(select(df, rank=True, boot=0), color=color)
                    self.assertIsInstance(out, str)

    def test_select_never_raises_on_random_input(self):
        rng = np.random.default_rng(99)
        for trial in range(40):
            K = int(rng.integers(1, 12))
            df = pd.DataFrame({
                "name": [f"m{i}" for i in range(K)],
                "accuracy": rng.uniform(-.1, 1.1, K),
                "resource": rng.uniform(-1, 50, K)})
            if rng.random() < .3:
                df["r2"] = rng.uniform(.1, 9, K)
            with self.subTest(trial=trial):
                r = select(df, rank=True, boot=0,
                           res_col=["resource", "r2"][:1 + ("r2" in df)])
                render(r, color=False)
                self.assertIn(r["status"], {"selected", "unverified",
                                            "no_floor", "no_rate",
                                            "bad_input"})

    def test_many_axes_do_not_break_the_band(self):
        df = frontier(10)
        for j in range(7):
            df[f"a{j}"] = df["resource"] * (1 + .1 * j)
        r = select(df, res_col=[f"a{j}" for j in range(7)], rank=True, boot=0)
        self.assertIsNotNone(r["band"])
        self.assertIn(r["minimax"]["pick"], r["band"]["members"])

    def test_duplicate_names_are_tolerated(self):
        df = frontier(6)
        df["name"] = ["same"] * len(df)
        r = select(df, rank=True, boot=0)
        self.assertIn(r["status"], {"selected", "unverified", "no_floor",
                                    "no_rate"})

    def test_boot_zero_gives_a_degenerate_interval_and_the_point_pick(self):
        r = select(frontier(12), rank=True, boot=0)
        ax = r["axis"]
        self.assertAlmostEqual(r["rate"]["lambda_low"][ax],
                               r["rate"]["lambda_high"][ax])
        self.assertEqual(r["winner"]["name"], r["minimax"]["point_pick"])

    def test_resource_unit_changes_scores_but_not_the_order(self):
        df = frontier(10)
        a = select(df, rank=True, boot=0)
        b = select(df.assign(resource=df["resource"] * 1000), rank=True,
                   boot=0)
        self.assertEqual(list(a["ranking"]["name"]), list(b["ranking"]["name"]))
        self.assertAlmostEqual(a["rate"]["lambda"], b["rate"]["lambda"],
                               places=6)

    def test_sigma_matrix_matches_the_definition(self):
        df, axes, _ = normalise(frontier(5))
        lam = {"resource": .37}
        S = sigma_matrix(df, [lam], axes)[:, 0]
        want = -np.log(df["eps"]) - .37 * df["x__resource"]
        self.assertTrue(np.allclose(S, want))

    def test_fit_pow_is_exact_on_a_power_law(self):
        t = np.exp(np.linspace(0, 4, 9))
        c, a, rss = fit_pow(t, 0.7 * t ** -0.6)
        self.assertAlmostEqual(c, 0.7, places=9)
        self.assertAlmostEqual(a, 0.6, places=9)
        self.assertLess(rss, 1e-20)


if __name__ == "__main__":
    unittest.main(verbosity=2)