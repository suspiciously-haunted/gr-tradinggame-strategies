"""Estimate which distribution the rewards are being drawn from.

The leaderboard page only shows rewards on turns where somebody accepted, so the
raw payouts are biased towards high values.  This script corrects for that:

  * Value-blind teams (they accept on every free turn, e.g. bot-greedy) accept
    regardless of the reward, so the rewards they take are unbiased draws.
  * Threshold teams only accept rewards above their threshold.  On turns where no
    value-blind team was free, the reward is only visible because it beat the
    lowest threshold among the free threshold teams, so it is fitted as a
    left-truncated sample (density f(x) / P(X >= tau)).

Every candidate family is fitted by maximum likelihood on that combined sample
and ranked by BIC.  The BIC weights are approximate probabilities that each
family is the one generating the rewards.  A KS test on the probability
integral transform (which handles the truncated samples) checks absolute fit.

Usage:
  uv run --with scipy python analyze_distribution.py                       # live leaderboard
  uv run --with scipy python analyze_distribution.py --file snapshots/x.html
  uv run --with scipy python analyze_distribution.py --snapshots           # pool saved runs of the newest round
  uv run --with scipy python analyze_distribution.py --rewards r.txt       # plain list of unbiased rewards
  uv run --with scipy python analyze_distribution.py --unbiased bot-random # treat extra teams as value-blind
"""
import argparse
import glob
import math
import os
import re
import statistics
from collections import defaultdict

import numpy as np

try:
	from scipy import optimize, stats
except ImportError:
	raise SystemExit("scipy is required: uv run --with scipy python analyze_distribution.py")

from analyze_players import SNAP_DIR, URL, accepts_of, fetch, parse

GREEDY_WAIT = 0.5     # mean idle turns per accept below which a team counts as value-blind
ENDGAME_GAPS = 3      # threshold teams drop their threshold in the last ENDGAME_GAPS * gap turns

# (label, scipy distribution, fixed parameters, needs strictly positive data)
CANDIDATES = [
	("Exponential", stats.expon, {"loc": 0.0}, True),
	("Shifted exponential", stats.expon, {}, False),
	("Normal", stats.norm, {}, False),
	("Uniform", stats.uniform, {}, False),
	("Log-normal", stats.lognorm, {"loc": 0.0}, True),
	("Gamma", stats.gamma, {"loc": 0.0}, True),
	("Weibull", stats.weibull_min, {"loc": 0.0}, True),
	("Half-normal", stats.halfnorm, {"loc": 0.0}, True),
	("Rayleigh", stats.rayleigh, {"loc": 0.0}, True),
	("Pareto", stats.pareto, {"loc": 0.0}, True),
	("Laplace", stats.laplace, {}, False),
	("Logistic", stats.logistic, {}, False),
	("Student t", stats.t, {}, False),
	("Gumbel", stats.gumbel_r, {}, False),
]


# ---------- sample extraction ----------

def extract_samples(traces, force_unbiased=()):
	"""Split the visible rewards into exact draws and left-truncated draws.

	Returns (exact, truncated, info) where truncated is a list of (reward, tau).
	The reward on a turn is the largest payout anyone got (robust to sharing).
	"""
	T = max(max(x) for x, _ in traces.values())
	acc = {name: accepts_of(x, y) for name, (x, y) in traces.items()}
	gaps_all = [b[0] - a[0] for a_list in acc.values() for a, b in zip(a_list, a_list[1:])]
	gap = min(gaps_all) if gaps_all else 16
	endgame = T - ENDGAME_GAPS * gap

	blind, tau = set(), {}
	for name, a in acc.items():
		if len(a) < 2:
			continue
		waits = [b[0] - p[0] - gap for p, b in zip(a, a[1:])]
		if name in force_unbiased or statistics.mean(waits) < GREEDY_WAIT:
			blind.add(name)
		else:
			mid = [d for t, d in a if t <= endgame]
			if mid:
				tau[name] = min(mid)  # every accept is >= its threshold

	by_turn = defaultdict(list)
	for name, a in acc.items():
		for t, d in a:
			by_turn[t].append((name, d))

	exact, truncated = [], []
	last = {name: -10**9 for name in acc}
	for t in range(1, T + 1):
		v = by_turn.get(t)
		if v:
			names = {n for n, _ in v}
			r = max(d for _, d in v)
			if names & blind:
				exact.append(r)
			elif t <= endgame:
				free_taus = [tau[n] for n in tau if t - last[n] >= gap]
				if free_taus:
					truncated.append((r, min(free_taus)))
			for n in names:
				last[n] = t
	info = {"T": T, "gap": gap, "blind": sorted(blind), "tau": tau}
	return exact, truncated, info


def load_pages(args):
	"""Return a list of (meta, traces) to analyse."""
	if args.file:
		return [parse(open(args.file).read())[:2]]
	if args.snapshots:
		files = sorted(glob.glob(os.path.join(SNAP_DIR, "*.html")), key=os.path.getmtime)
		pages, seen = [], set()
		for fn in files:
			meta, traces, _ = parse(open(fn).read())
			if traces and meta["last_run"] not in seen:
				seen.add(meta["last_run"])
				pages.append((meta, traces))
		if not pages:
			raise SystemExit(f"No usable snapshots in {SNAP_DIR}")
		newest_round = pages[-1][0]["round"]
		return [p for p in pages if p[0]["round"] == newest_round]
	return [parse(fetch(URL))[:2]]


# ---------- fitting ----------

def unpack(dist, fixed, theta):
	k = dist.numargs
	shapes = [math.exp(v) for v in theta[:k]]
	i = k
	if "loc" in fixed:
		loc = fixed["loc"]
	else:
		loc = theta[i]
		i += 1
	return shapes, loc, math.exp(theta[i])


def neg_loglik(theta, dist, fixed, ex, tx, tt):
	shapes, loc, scale = unpack(dist, fixed, theta)
	with np.errstate(all="ignore"):
		ll = dist.logpdf(ex, *shapes, loc=loc, scale=scale).sum()
		if len(tx):
			ll += (dist.logpdf(tx, *shapes, loc=loc, scale=scale)
				   - dist.logsf(tt, *shapes, loc=loc, scale=scale)).sum()
	return -ll if np.isfinite(ll) else 1e300


def start_points(dist, fixed, data_sets):
	fkw = {"f" + k: v for k, v in fixed.items()}
	starts = []
	for data in data_sets:
		if len(data) < 3:
			continue
		try:
			p = dist.fit(data, **fkw)
		except Exception:
			continue
		shapes, loc, scale = p[:-2], p[-2], p[-1]
		if scale <= 0 or any(s <= 0 for s in shapes):
			continue
		theta = [math.log(s) for s in shapes]
		if "loc" not in fixed:
			theta.append(loc)
		theta.append(math.log(scale))
		starts.append(np.array(theta))
	return starts


def fit_candidate(dist, fixed, ex, tx, tt):
	best = None
	for x0 in start_points(dist, fixed, [ex, np.concatenate([ex, tx])]):
		for _ in range(2):  # restart once from the optimum to escape a stalled simplex
			res = optimize.minimize(neg_loglik, x0, args=(dist, fixed, ex, tx, tt), method="Nelder-Mead",
									options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-9})
			x0 = res.x
		if res.fun < 1e299 and (best is None or res.fun < best.fun):
			best = res
	return best


def pit_values(dist, shapes, loc, scale, ex, tx, tt):
	"""Probability integral transform; uniform on [0, 1] if the model is right."""
	u = list(dist.cdf(ex, *shapes, loc=loc, scale=scale))
	if len(tx):
		ft = dist.cdf(tt, *shapes, loc=loc, scale=scale)
		fx = dist.cdf(tx, *shapes, loc=loc, scale=scale)
		with np.errstate(all="ignore"):
			u += list((fx - ft) / (1 - ft))
	return np.clip(np.nan_to_num(np.array(u), nan=0.5), 0, 1)


def fmt_params(dist, fixed, shapes, loc, scale):
	names = [s.strip() for s in dist.shapes.split(",")] if dist.shapes else []
	parts = [f"{n}={v:.4g}" for n, v in zip(names, shapes)]
	parts.append(f"loc={loc:.4g}" + ("(fixed)" if "loc" in fixed else ""))
	parts.append(f"scale={scale:.4g}")
	return " ".join(parts)


def analyse(exact, truncated):
	ex = np.array(exact, dtype=float)
	tx = np.array([r for r, _ in truncated], dtype=float)
	tt = np.array([t for _, t in truncated], dtype=float)
	n = len(ex) + len(tx)
	if n < 5:
		print("Not enough samples to fit anything.")
		return [], np.array([])
	all_vals = np.concatenate([ex, tx])
	positive = all_vals.min() > 0

	print(f"Samples: {len(ex)} exact, {len(tx)} truncated   "
		  f"(exact mean {ex.mean():.3f}, sd {ex.std(ddof=1):.3f}, min {ex.min():.3f}, max {ex.max():.3f})"
		  if len(ex) > 1 else f"Samples: {len(ex)} exact, {len(tx)} truncated")
	if np.mean(np.abs(all_vals - np.round(all_vals)) < 1e-9) > 0.9:
		print("Note: over 90% of rewards are whole numbers -- the distribution may be discrete.")

	results = []
	for label, dist, fixed, pos_only in CANDIDATES:
		if pos_only and not positive:
			continue
		res = fit_candidate(dist, fixed, ex, tx, tt)
		if res is None:
			continue
		shapes, loc, scale = unpack(dist, fixed, res.x)
		k = len(res.x)
		with np.errstate(all="ignore"):
			mean, sd = (float(v) for v in dist.stats(*shapes, loc=loc, scale=scale, moments="mv"))
		ks_p = stats.kstest(pit_values(dist, shapes, loc, scale, ex, tx, tt), "uniform").pvalue
		results.append({
			"label": label, "dist": dist, "fixed": fixed, "shapes": shapes, "loc": loc, "scale": scale,
			"nll": res.fun, "k": k, "aic": 2 * k + 2 * res.fun, "bic": k * math.log(n) + 2 * res.fun,
			"mean": mean, "sd": math.sqrt(sd) if sd >= 0 else float("nan"), "ks_p": ks_p,
		})

	results.sort(key=lambda r: r["bic"])
	b0 = results[0]["bic"]
	w = np.array([math.exp(-(r["bic"] - b0) / 2) for r in results])
	w /= w.sum()

	print(f"\n{'distribution':<21}{'k':>2}{'logL':>11}{'dBIC':>8}{'P(best)':>9}{'KS p':>8}{'mean':>9}{'sd':>9}  parameters")
	for r, wi in zip(results, w):
		print(f"{r['label']:<21}{r['k']:>2}{-r['nll']:11.2f}{r['bic'] - b0:8.2f}{wi:9.3f}{r['ks_p']:8.3f}"
			  f"{r['mean']:9.3f}{r['sd']:9.3f}  {fmt_params(r['dist'], r['fixed'], r['shapes'], r['loc'], r['scale'])}")

	best = results[0]
	print(f"\nMost likely: {best['label']}  ({fmt_params(best['dist'], best['fixed'], best['shapes'], best['loc'], best['scale'])})"
		  f"  mean {best['mean']:.3f}, sd {best['sd']:.3f}, P(best) {w[0]:.2f}")
	if best["ks_p"] < 0.05:
		print("Warning: KS p < 0.05, even the best candidate fits poorly -- the true distribution may not be in the list.")
	print("(P(best) uses BIC weights; KS p-values are optimistic because parameters were fitted on the same data.)")

	if len(ex) >= 10:
		print(f"\nQuantiles of exact samples vs {best['label']} fit:")
		print(f"{'q':>6}{'data':>10}{'model':>10}")
		for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
			model = best["dist"].ppf(p, *best["shapes"], loc=best["loc"], scale=best["scale"])
			print(f"{p:6.2f}{np.quantile(ex, p):10.3f}{model:10.3f}")
	return results, w


def main():
	ap = argparse.ArgumentParser()
	src = ap.add_mutually_exclusive_group()
	src.add_argument("--file", help="saved leaderboard page")
	src.add_argument("--snapshots", action="store_true", help="pool every saved snapshot of the newest round")
	src.add_argument("--rewards", help="text file of rewards (any separator), all treated as unbiased")
	ap.add_argument("--unbiased", nargs="*", default=[], help="extra teams to treat as value-blind")
	args = ap.parse_args()

	if args.rewards:
		vals = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", open(args.rewards).read())]
		print(f"Source: {args.rewards}")
		analyse(vals, [])
		return

	exact, truncated = [], []
	for meta, traces in load_pages(args):
		if not traces:
			print(f"Run {meta['last_run']}: no score curves on the page, skipped")
			continue
		e, tr, info = extract_samples(traces, set(args.unbiased))
		exact += e
		truncated += tr
		print(f"Run {meta['last_run']}   {meta['round']}   (server label, if shown, is the ground truth)")
		print(f"  value-blind teams: {', '.join(info['blind']) or 'none'}")
		if info["tau"]:
			print("  threshold teams (lowest accept before endgame): "
				  + ", ".join(f"{n}={v:.2f}" for n, v in sorted(info["tau"].items(), key=lambda kv: kv[1])))
		print(f"  {len(e)} exact + {len(tr)} truncated samples, lockout gap {info['gap']}")
	if not exact:
		print("\nWarning: no value-blind team found, so every sample is truncated and the fit is much weaker. "
			  "Use --unbiased TEAM if you know a team accepts regardless of the reward.")
	print()
	analyse(exact, truncated)


if __name__ == "__main__":
	main()
