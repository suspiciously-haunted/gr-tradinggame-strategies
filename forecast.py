"""Forecast future runs of a round with an unknown reward distribution.

Steps:
  1. Fit the reward distribution from the leaderboard curves (analyze_distribution.py).
  2. Model every team's strategy from its accepts and skips:
       blind      - accepts with probability p on every free turn, ignoring the reward
       threshold  - accepts rewards >= tau, and anything from turn `endgame` onwards
       optimal    - follows the optimal-stopping thresholds for the fitted distribution
     The model with the best likelihood on the team's observed decisions is used.
  3. Compute the optimal threshold schedule for you under the fitted distribution.
  4. Simulate many full games (same reward sequence for every team, as on the
     server) and report each team's final-score range, mean rank and win chance.
     By default each simulated game draws its distribution family by BIC weight,
     so uncertainty about which distribution it is shows up in the forecast.

Payouts are not shared: every team that accepts on a turn receives the full
reward, so other teams only affect your rank, not your score.

Usage:
  uv run --with scipy python forecast.py                        # live leaderboard
  uv run --with scipy python forecast.py --snapshots            # pool saved runs of the newest round
  uv run --with scipy python forecast.py --file snapshots/x.html
  uv run --with scipy python forecast.py --games 5000 --model best
"""
import argparse
import math
from collections import defaultdict

import numpy as np

from analyze_distribution import GREEDY_WAIT, analyse, extract_samples, load_pages
from analyze_players import accepts_of

try:
	from config import TEAM_NAME
except Exception:
	TEAM_NAME = ""

DP_SAMPLES = 200_000


# ---------- optimal stopping ----------

def dp_thresholds(xs, T, gap):
	"""thr[t] = minimum reward worth accepting on turn t (1-based), from
	V(s) = E[max(X + V(s - gap), V(s - 1))] with s turns left."""
	xs = np.sort(np.asarray(xs, dtype=float))
	n = len(xs)
	suf = np.concatenate([np.cumsum(xs[::-1])[::-1], [0.0]])
	V = np.zeros(T + 1)
	c = np.zeros(T + 1)
	for s in range(1, T + 1):
		after = V[s - gap] if s - gap >= 0 else 0.0
		cs = V[s - 1] - after
		i = np.searchsorted(xs, cs, side="right")
		V[s] = V[s - 1] + (suf[i] - cs * (n - i)) / n
		c[s] = cs
	thr = np.full(T + 1, np.inf)
	thr[1:] = c[T:0:-1]  # turn t has s = T - t + 1 turns left
	return thr, V[T]


# ---------- team modelling ----------

def team_observations(pages, gap, T):
	"""Per team: free-turn counts, decisions on free turns with a known reward, final scores."""
	obs = defaultdict(lambda: {"free": 0, "acc": 0, "t": [], "r": [], "a": [], "final": [], "waits": [], "free_t": []})
	for _, traces in pages:
		acc = {name: dict(accepts_of(x, y)) for name, (x, y) in traces.items()}
		known = {}  # reward on every turn somebody accepted
		for a in acc.values():
			for t, d in a.items():
				known[t] = max(known.get(t, d), d)
		for name, a in acc.items():
			o = obs[name]
			o["final"].append(traces[name][1][-1])
			last, prev = -10**9, None
			for t in range(1, T + 1):
				if t - last < gap:
					continue
				o["free"] += 1
				o["free_t"].append(t)
				if t in a:
					o["acc"] += 1
					o["t"].append(t); o["r"].append(a[t]); o["a"].append(True)
					if prev is not None:
						o["waits"].append(t - prev - gap)
					last = prev = t
				elif t in known:
					o["t"].append(t); o["r"].append(known[t]); o["a"].append(False)
	return obs


def best_threshold(r, a):
	"""Single threshold that misclassifies the fewest (reward, accepted) pairs."""
	if len(r) == 0:
		return np.inf
	o = np.argsort(r)
	r, a = r[o], a[o].astype(int)
	n = len(r)
	acc_below = np.concatenate([[0], np.cumsum(a)])
	skip_above = (n - np.arange(n + 1)) - (a.sum() - acc_below)
	i = int(np.argmin(acc_below + skip_above))
	if i == 0:
		return r[0] - 1e-9
	if i == n:
		return r[-1] + 1e-9
	return 0.5 * (r[i - 1] + r[i])


def noisy_loglik(mismatches, n):
	if n == 0:
		return 0.0
	eps = min(max(mismatches / n, 0.005), 0.5)
	return mismatches * math.log(eps) + (n - mismatches) * math.log(1 - eps)


def take_rate(o, thr, sf):
	"""P(accept | free and reward >= threshold).  Skips are only visible when another
	team accepted, so count the free turns above the threshold from the fitted
	distribution instead of from the visible decisions."""
	expected = float(np.sum(sf(thr[np.array(o["free_t"], dtype=int)]))) if o["free_t"] else 0.0
	return min(o["acc"] / expected, 1.0) if expected > 0 else 1.0


def fit_team(name, o, T, gap, opt_thr, force_blind, sf):
	t, r, a = np.array(o["t"], dtype=int), np.array(o["r"], dtype=float), np.array(o["a"], dtype=bool)
	n = len(t)
	p = o["acc"] / o["free"] if o["free"] else 0.0
	models = []

	# blind: Bernoulli(p) on every free turn
	pc = min(max(p, 1e-3), 1 - 1e-3)
	ll_blind = float(np.sum(np.where(a, math.log(pc), math.log(1 - pc))))
	models.append((ll_blind - 1, {"kind": "blind", "thr": np.full(T + 1, -np.inf), "p": p, "err": float("nan")}))

	# threshold + endgame cutoff
	early = t < T - 4 * gap
	tau = best_threshold(r[early], a[early]) if early.any() else best_threshold(r, a)
	best_e, best_m = T + 1, None
	for e in range(T + 1, T - 4 * gap - 1, -1):
		thr_t = np.where(t >= e, -np.inf, tau)
		m = int(np.sum((r >= thr_t) != a))
		if best_m is None or m < best_m:
			best_e, best_m = e, m
	thr = np.full(T + 1, tau)
	thr[best_e:] = -np.inf
	models.append((noisy_loglik(best_m, n) - 2, {"kind": "threshold", "thr": thr, "p": take_rate(o, thr, sf), "tau": tau,
												 "endgame": best_e, "err": best_m / n if n else float("nan")}))

	# optimal stopping for the fitted distribution
	m = int(np.sum((r >= opt_thr[t]) != a)) if n else 0
	models.append((noisy_loglik(m, n), {"kind": "optimal", "thr": opt_thr, "p": take_rate(o, opt_thr, sf), "err": m / n if n else float("nan")}))

	waits = o["waits"]
	if force_blind or (waits and np.mean(waits) < GREEDY_WAIT):
		model = models[0][1]
	else:
		model = max(models, key=lambda mm: mm[0])[1]
	model["name"] = name
	model["observed"] = float(np.mean(o["final"]))
	return model


def describe(m):
	if m["kind"] == "blind":
		return f"blind, accepts {m['p']:.0%} of free turns"
	if m["kind"] == "threshold":
		eg = "none" if m["endgame"] > len(m["thr"]) - 1 else f"from turn {m['endgame']}"
		return f"threshold {m['tau']:.2f}, takes {m['p']:.0%} above it, endgame {eg}, misfit {m['err']:.1%}"
	return f"optimal for fitted distribution, takes {m['p']:.0%} above it, misfit {m['err']:.1%}"


# ---------- simulation ----------

def make_sampler(results, weights, mode):
	if mode == "best":
		results, weights = results[:1], np.array([1.0])

	def sample(N, T, rng):
		counts = rng.multinomial(N, weights)
		blocks = [r["dist"].rvs(*r["shapes"], loc=r["loc"], scale=r["scale"], size=(c, T), random_state=rng)
				  for r, c in zip(results, counts) if c > 0]
		R = np.concatenate(blocks)
		rng.shuffle(R)
		return R
	return sample


def simulate(policies, sampler, T, gap, N, rng):
	R = sampler(N, T, rng)
	K = len(policies)
	thr = np.array([p["thr"] for p in policies])  # (K, T + 1)
	pa = np.array([p["p"] for p in policies])
	score = np.zeros((N, K))
	next_free = np.ones((N, K), dtype=int)
	for t in range(1, T + 1):
		r = R[:, t - 1][:, None]
		take = (next_free <= t) & (r >= thr[:, t]) & (rng.random((N, K)) < pa)
		score += np.where(take, r, 0.0)
		next_free = np.where(take, t + gap, next_free)
	return score


def main():
	ap = argparse.ArgumentParser()
	src = ap.add_mutually_exclusive_group()
	src.add_argument("--file", help="saved leaderboard page")
	src.add_argument("--snapshots", action="store_true", help="pool every saved snapshot of the newest round")
	ap.add_argument("--unbiased", nargs="*", default=[], help="extra teams to treat as value-blind")
	ap.add_argument("--games", type=int, default=2000, help="number of simulated games")
	ap.add_argument("--model", choices=["mixture", "best"], default="mixture",
					help="mixture: draw the family per game by BIC weight; best: only the top fit")
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()
	rng = np.random.default_rng(args.seed)

	# 1. distribution
	pages = [(m, tr) for m, tr in load_pages(args) if tr]
	if not pages:
		raise SystemExit("No score curves found.")
	exact, truncated, info = [], [], None
	for meta, traces in pages:
		e, tr, info = extract_samples(traces, set(args.unbiased))
		exact += e
		truncated += tr
		print(f"Run {meta['last_run']}   {meta['round']}")
	T, gap = info["T"], info["gap"]
	print(f"{len(pages)} run(s), {T} turns, lockout gap {gap}\n")
	results, weights = analyse(exact, truncated)
	if not results:
		raise SystemExit("Cannot forecast without a fitted distribution.")
	best = results[0]

	# 2. optimal schedule for you
	xs = best["dist"].rvs(*best["shapes"], loc=best["loc"], scale=best["scale"], size=DP_SAMPLES, random_state=rng)
	opt_thr, opt_value = dp_thresholds(xs, T, gap)
	print(f"\n=== Optimal thresholds under {best['label']} (expected score {opt_value:.1f}) ===")
	print("accept on turn t if reward >= threshold and you are free")
	print(f"{'turn':>7}{'turns left':>12}{'threshold':>11}")
	for left in (T, T // 2, 500, 200, 100, 50, 3 * gap, 2 * gap, gap, gap // 2, 1):
		if 1 <= left <= T:
			print(f"{T - left + 1:7d}{left:12d}{opt_thr[T - left + 1]:11.3f}")

	# 3. team models
	obs = team_observations(pages, gap, T)
	def sf(x):
		return best["dist"].sf(x, *best["shapes"], loc=best["loc"], scale=best["scale"])
	policies = [fit_team(n, o, T, gap, opt_thr, n in args.unbiased, sf) for n, o in obs.items() if o["acc"] > 0]
	print("\n=== Team strategy models ===")
	for p in sorted(policies, key=lambda p: -p["observed"]):
		print(f"  {p['name'][:20]:<21}observed {p['observed']:9.1f}   {describe(p)}")
	me = f"{TEAM_NAME or 'You'} (optimal)"
	policies.append({"name": me, "kind": "optimal", "thr": opt_thr, "p": 1.0, "observed": float("nan")})

	# 4. simulate
	scores = simulate(policies, make_sampler(results, weights, args.model), T, gap, args.games, rng)
	ranks = 1 + (scores[:, None, :] > scores[:, :, None]).sum(axis=2)  # 1 + teams strictly ahead
	K = len(policies)
	label = "BIC-weighted mix of families" if args.model == "mixture" else best["label"]
	print(f"\n=== Forecast over {args.games} simulated games ({label}) ===")
	print(f"{'team':<26}{'observed':>9}{'mean':>9}{'sd':>8}{'5%':>9}{'50%':>9}{'95%':>9}{'rank':>7}{'P(win)':>8}{'P(beat you)':>12}")
	me_i = K - 1
	for i in np.argsort(-scores.mean(axis=0)):
		s = scores[:, i]
		beat = np.mean(s > scores[:, me_i]) if i != me_i else float("nan")
		obs_s = "" if math.isnan(policies[i]["observed"]) else f"{policies[i]['observed']:.1f}"
		print(f"{policies[i]['name'][:25]:<26}{obs_s:>9}{s.mean():9.1f}{s.std():8.1f}{np.quantile(s, .05):9.1f}"
			  f"{np.median(s):9.1f}{np.quantile(s, .95):9.1f}{ranks[:, i].mean():7.2f}{np.mean(ranks[:, i] == 1):8.1%}"
			  + (f"{beat:12.1%}" if i != me_i else f"{'-':>12}"))
	print("rank = mean finishing position (ties share the better rank); P(win) counts ties as wins.")
	print("Compare 'observed' with 'mean' to check each team model is calibrated.")


if __name__ == "__main__":
	main()
