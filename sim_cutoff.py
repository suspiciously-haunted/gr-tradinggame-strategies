"""Cutoff study: which acceptance bar gives the best results against the current field?

Steps:
  1. simulator that copies the server (2000 turns, lockout 15, shared reward
     sequence, every accepter gets the full reward).  Rewards come from
     --dist normal (Round 2 fit, default), --dist exp (Round 1) or a CSV of
     observed rewards (--rewards-file).
  2. opponents seen on the leaderboard (--field round1 / round2)
  3. two candidate families, searched automatically:
       DP+d    the optimal-stopping bar solution.py converges to, shifted by d
       flat c  one bar all game, grab anything positive on the last turn
     A coarse sweep over a wide range is followed by a fine sweep around the
     best value of every objective.
  4. the same reward sequences for every strategy (common random numbers, so
     differences between cutoffs are not drowned by luck)
  5. average score, head-to-head vs the strongest opponent, best-of-N
  6. pick per objective

Speed: with numba installed (uv run --with numba python sim_cutoff.py) the
simulator is a compiled loop run over games in parallel that skips the lockout
after every accept.  Without numba it falls back to a numpy loop over turns.

`--real` additionally runs the actual solution.py on fewer sequences to measure
what learning the distribution on the fly costs.
"""
import argparse
import importlib
import math
import time

import numpy as np

T = 2000
L = 15
GAP = L + 1
EPS = 1e-12  # bars are floored here so no candidate accepts a reward <= 0


# ---------- step 1: simulator ----------
def dp_bars_fn(gain):
	"""bar[s] with s turns left (incl. current); gain(c) = E[max(X - c, 0)]."""
	V = np.zeros(T + 1)
	c = np.zeros(T + 1)
	for s in range(1, T + 1):
		after = V[s - 1 - L] if s - 1 - L >= 0 else 0.0
		cs = V[s - 1] - after
		c[s] = cs
		V[s] = V[s - 1] + gain(cs)
	return c, V[T]


def gain_exp(mean):
	return lambda c: mean * math.exp(-max(c, 0.0) / mean) + max(-c, 0.0)


def gain_normal(mu, sd):
	def g(c):
		z = (mu - c) / sd
		return (mu - c) * 0.5 * (1 + math.erf(z / math.sqrt(2))) + sd * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
	return g


def gain_samples(xs):
	xs = np.sort(np.asarray(xs, dtype=float))
	n = len(xs)
	suf = np.concatenate([np.cumsum(xs[::-1])[::-1], [0.0]])

	def g(c):
		i = np.searchsorted(xs, c, side="right")
		return (suf[i] - c * (n - i)) / n
	return g


try:
	from numba import njit, prange

	@njit(parallel=True, cache=True)
	def _simulate_jit(by_turn, rewards, gap):
		P, S, n = by_turn.shape[0], rewards.shape[0], rewards.shape[1]
		out = np.zeros((P, S))
		for i in prange(S):  # games in parallel
			r = rewards[i]
			for p in range(P):
				b = by_turn[p]
				score = 0.0
				t = 0
				while t < n:
					if r[t] >= b[t]:
						score += r[t]
						t += gap  # skip the lockout
					else:
						t += 1
				out[p, i] = score
		return out
	@njit(parallel=True, cache=True)
	def _simulate_rows_jit(by_turn_rows, rewards, gap):
		S, n = rewards.shape
		out = np.zeros(S)
		for i in prange(S):
			r = rewards[i]
			b = by_turn_rows[i]
			score = 0.0
			t = 0
			while t < n:
				if r[t] >= b[t]:
					score += r[t]
					t += gap
				else:
					t += 1
			out[i] = score
		return out
	ENGINE = "numba (compiled, parallel)"
except ImportError:
	_simulate_jit = _simulate_rows_jit = None
	ENGINE = "numpy (install numba for a large speed-up: uv run --with numba ...)"


def simulate(bars, rewards):
	"""bars: (P, T+1) bar by turns-left; rewards: (S, T). Returns scores (P, S)."""
	by_turn = np.ascontiguousarray(bars[:, T:0:-1])  # column j is turn j + 1, which has T - j turns left
	if _simulate_jit is not None:
		return _simulate_jit(by_turn, np.ascontiguousarray(rewards), GAP)
	# numpy fallback: step through the turns, vectorised over strategies and games
	P, S = bars.shape[0], rewards.shape[0]
	score = np.zeros((P, S))
	next_free = np.zeros((P, S), dtype=np.int64)
	for j in range(T):
		r = rewards[:, j][None, :]
		take = (next_free <= j) & (r >= by_turn[:, j][:, None])
		score += np.where(take, r, 0.0)
		next_free = np.where(take, j + GAP, next_free)
	return score


def simulate_rows(bars_rows, rewards):
	"""Like simulate, but every sequence has its own bars: bars_rows (S, T+1). Returns (S,)."""
	by_turn = np.ascontiguousarray(bars_rows[:, T:0:-1])
	if _simulate_rows_jit is not None:
		return _simulate_rows_jit(by_turn, np.ascontiguousarray(rewards), GAP)
	S = rewards.shape[0]
	score = np.zeros(S)
	next_free = np.zeros(S, dtype=np.int64)
	for j in range(T):
		take = (next_free <= j) & (rewards[:, j] >= by_turn[:, j])
		score += np.where(take, rewards[:, j], 0.0)
		next_free = np.where(take, j + GAP, next_free)
	return score


_Z_BARS = {}


def oracle_bars(mus, sds):
	"""Optimal bars for sequences whose Normal(mu, sd) is known: sd * bars of Normal(mu / sd, 1).
	z is rounded to 0.01 so the tables can be cached."""
	out = np.empty((len(mus), T + 1))
	for i, (m, sd) in enumerate(zip(mus, sds)):
		z = round(m / sd, 2)
		if z not in _Z_BARS:
			_Z_BARS[z] = dp_bars_fn(gain_normal(z, 1.0))[0]
		out[i] = np.maximum(sd * _Z_BARS[z], EPS)
		out[i, 1] = EPS
	return out


def flat(c, last_turn_grab=False):
	b = np.full(T + 1, float(c))
	if last_turn_grab:
		b[1] = EPS
	return b


# ---------- step 2: opponents ----------
FIELDS = {
	"round1": {
		"fussball": flat(20.4, last_turn_grab=True),
		"Quant Star": flat(20.4),
		"JMRC": flat(19.5),
		"bot-pro": flat(14.0),
		"bot-greedy": flat(-np.inf),
	},
	# thresholds fitted by forecast.py on the 17:51 Round 2 run
	"round2": {
		"Trial1": flat(0.91),
		"Quant Star": flat(0.91),
		"JMRC": flat(0.82),
		"bot-pro": flat(0.57),
		"bot-novice": flat(EPS),
		"bot-greedy": flat(-np.inf),
	},
}


# ---------- step 3: candidates ----------
def dp_shift(base, d):
	b = np.maximum(base + d, EPS)
	b[1] = EPS  # always grab a positive reward on the last turn
	return b


def make_cands(base, family, values):
	values = [round(float(v), 6) + 0.0 for v in values]  # + 0.0 turns -0.0 into 0.0
	if family == "DP":
		return {f"DP{v:+.3f}": dp_shift(base, v) for v in values}
	return {f"flat {v:.3f}": flat(max(v, EPS), last_turn_grab=True) for v in values}


def best_of_n(mine, theirs, n):
	k = len(mine) // n
	a = mine[: k * n].reshape(k, n).max(axis=1)
	b = theirs[: k * n].reshape(k, n).max(axis=1)
	return (a > b).mean() + 0.5 * (a == b).mean(), k


def metrics(s, ref, n):
	diff = s - ref
	bo, _ = best_of_n(s, ref, n)
	return {"mean": s.mean(), "sd": s.std(), "beat": (diff > 1e-9).mean(), "tie": (np.abs(diff) <= 1e-9).mean(),
			"diff": diff.mean(), "bo": bo}


OBJECTIVES = [("mean", "maximise average (latest-score ranking)"),
			  ("beat", "single run vs {ref}"),
			  ("bo", "best-of-{n} vs {ref} (best-score ranking)")]


def run_real(rewards, field, module="solution", reuse=False):
	"""Run a strategy file, unmodified, against the opponents on the given sequences.
	reuse=True keeps one instance across games (lets it learn across games)."""
	sol = importlib.import_module(module)
	opp_bars = np.stack(list(field.values()))
	scores = []
	shared = sol.Solution().play
	for seq in rewards:
		play = shared if reuse else sol.Solution().play
		me, me_free = 0.0, 1
		o_score = [0.0] * len(opp_bars)
		o_free = [1] * len(opp_bars)
		others = []
		for t in range(1, T + 1):
			r = float(seq[t - 1])
			locked = me_free > t
			if play(r, L, t, T, me, others, locked) and not locked:
				me += r
				me_free = t + GAP
			for i in range(len(opp_bars)):
				if o_free[i] <= t and r >= opp_bars[i, T - t + 1]:
					o_score[i] += r
					o_free[i] = t + GAP
			others = [(o_score[i], o_free[i] > t + 1) for i in range(len(opp_bars))]
		scores.append(me)
	return np.array(scores)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dist", choices=["normal", "exp"], default="normal")
	ap.add_argument("--mean", type=float, help="reward mean (default -0.15 normal, 10 exp)")
	ap.add_argument("--sd", type=float, default=1.03, help="reward sd for --dist normal")
	ap.add_argument("--rewards-file", help="CSV from analyze_players.py --dump-rewards; bootstrap rewards from it")
	ap.add_argument("--round", help="only use samples whose round label contains this text, e.g. 'Round 2'")
	ap.add_argument("--field", choices=list(FIELDS), default="round2")
	ap.add_argument("--ref", help="opponent for head-to-head and best-of-N (default: strongest on average)")
	ap.add_argument("--seqs", type=int, default=5000)
	ap.add_argument("--n", type=int, default=24, help="runs per round for best-of-N")
	ap.add_argument("--coarse", type=int, default=21, help="cutoffs per family in the coarse sweep")
	ap.add_argument("--fine", type=int, default=11, help="cutoffs per objective in the fine sweep")
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--hyper", type=float, nargs=4, metavar=("MU_MEAN", "MU_SD", "SD_LO", "SD_HI"),
					help="every game draws mu ~ Normal(MU_MEAN, MU_SD) and sd ~ Uniform(SD_LO, SD_HI)")
	ap.add_argument("--real", type=int, default=0, help="also run the strategy files, unmodified, on this many sequences")
	ap.add_argument("--solution", nargs="+", default=["solution"], help="strategy modules for --real")
	ap.add_argument("--reuse", action="store_true", help="--real: one instance per module across all games")
	args = ap.parse_args()

	rng = np.random.default_rng(args.seed)
	t0 = time.time()

	# step 1
	mus = sds = None
	if args.hyper:
		mm, ms, lo, hi = args.hyper
		mus = rng.normal(mm, ms, args.seqs)
		sds = rng.uniform(lo, hi, args.seqs)
		pool = rng.normal(mm, ms, 200_000) + rng.uniform(lo, hi, 200_000) * rng.normal(size=200_000)
		desc = f"per game mu ~ Normal({mm}, {ms}), sd ~ Uniform({lo}, {hi}); fixed cutoffs tuned to the pooled rewards"
		gain, spread = gain_samples(pool), pool.std()
		draw = lambda size: mus[:, None] + sds[:, None] * rng.normal(size=size)
	elif args.rewards_file:
		rows = [line.rstrip("\n").rsplit(",", 1) for line in open(args.rewards_file) if line.strip()]
		samples = np.array([float(v) for lab, v in rows if not args.round or args.round in lab])
		desc = f"rewards resampled from {len(samples)} observed values (mean {samples.mean():.3f}, sd {samples.std():.3f})"
		gain, spread = gain_samples(samples), samples.std()
		draw = lambda size: rng.choice(samples, size=size)
	elif args.dist == "exp":
		mean = 10.0 if args.mean is None else args.mean
		desc = f"Exp(mean {mean})"
		gain, spread = gain_exp(mean), mean
		draw = lambda size: rng.exponential(mean, size=size)
	else:
		mean = -0.15 if args.mean is None else args.mean
		desc = f"Normal(mean {mean}, sd {args.sd})"
		gain, spread = gain_normal(mean, args.sd), args.sd
		draw = lambda size: rng.normal(mean, args.sd, size=size)
	base, ev = dp_bars_fn(gain)
	print(f"ENGINE  {ENGINE}")
	print(f"STEP 1  simulator: {T} turns, {desc}, lockout {L}, shared rewards, full payout")
	print(f"        optimal bar (DP+0): mid-round {base[1000]:.3f}, 100 left {base[100]:.3f}, 50 left {base[50]:.3f}, "
		  f"16 left {base[16]:.3f}, 5 left {base[5]:.3f}, last turn {max(base[1], 0):.3f}   expected score {ev:.1f}")

	# step 2
	field = FIELDS[args.field]
	print(f"\nSTEP 2  opponents ({args.field}): " + ", ".join(field))

	# step 4 (sequences are needed before the search)
	rewards = draw((args.seqs, T))
	opp_scores = dict(zip(field, simulate(np.stack(list(field.values())), rewards)))
	ref_name = args.ref or max(opp_scores, key=lambda k: opp_scores[k].mean())
	ref = opp_scores[ref_name]
	print(f"\nSTEP 4  {args.seqs} shared reward sequences (seed {args.seed}); reference opponent: {ref_name}")

	# step 3: coarse sweep, then fine sweep around each objective's winner
	mid = base[1000]
	half = 0.6 * spread
	coarse_step = 2 * half / (args.coarse - 1)
	sweeps = {"DP": np.linspace(-half, half, args.coarse), "flat": np.linspace(mid - half, mid + half, args.coarse)}
	print(f"\nSTEP 3  coarse sweep: DP shift {-half:+.3f}..{half:+.3f}, flat {mid - half:.3f}..{mid + half:.3f} "
		  f"({args.coarse} values each, step {coarse_step:.3f})")
	results = {}

	def evaluate(cands):
		new = {k: v for k, v in cands.items() if k not in results}
		if new:
			for name, s in zip(new, simulate(np.stack(list(new.values())), rewards)):
				results[name] = (s, metrics(s, ref, args.n))

	for fam, vals in sweeps.items():
		evaluate(make_cands(base, fam, vals))
	t_coarse = time.time()
	for fam, vals in sweeps.items():
		fam_names = list(make_cands(base, fam, vals))
		for obj, _ in OBJECTIVES:
			i = int(np.argmax([results[n][1][obj] for n in fam_names]))
			fine = np.linspace(vals[i] - coarse_step, vals[i] + coarse_step, args.fine)
			evaluate(make_cands(base, fam, fine))
	n_strats = len(results) + len(field)
	print(f"        fine sweep: {args.fine} values within one coarse step of every objective's winner")
	print(f"        simulated {n_strats} strategies x {args.seqs} games in {time.time() - t0:.1f}s "
		  f"(coarse {t_coarse - t0:.1f}s)")

	# step 5
	base_s = results.get("DP+0.000", (None,))[0]
	if base_s is None:
		base_s = simulate(dp_shift(base, 0.0)[None, :], rewards)[0]
	print(f"\nSTEP 5  results (head-to-head and best-of-{args.n} vs {ref_name} on the same sequences)")
	print(f"        {'strategy':<16}{'mean':>9}{'sd':>8}{'vs DP+0':>9}{'+-':>6}{'beat ref':>10}{'tie':>6}{'avg diff':>10}"
		  f"{'best-of-' + str(args.n):>12}")

	def row(name, s, m):
		d = s - base_s
		se = d.std() / math.sqrt(len(d))
		print(f"        {name[:15]:<16}{m['mean']:9.2f}{m['sd']:8.2f}{d.mean():+9.2f}{se:6.2f}{m['beat']:10.1%}{m['tie']:6.0%}"
			  f"{m['diff']:+10.2f}{m['bo']:12.1%}")

	for name, s in opp_scores.items():
		row(name, s, metrics(s, ref, args.n))
	print("        " + "-" * 86)
	for fam in sweeps:
		names = sorted((n for n in results if n.startswith(fam)), key=lambda n: float(n.split()[-1].replace("DP", "")))
		for name in names:
			row(name, *results[name])
		print("        " + "-" * 86)
	k = args.seqs // args.n
	print(f"        ('vs DP+0' = paired mean difference and its standard error; best-of-{args.n}: {k} rounds, "
		  f"50% = no edge, standard error about {100 * math.sqrt(0.25 / k):.1f} points)")

	# step 6
	print("\nSTEP 6  pick")
	for obj, label in OBJECTIVES:
		name = max(results, key=lambda n: results[n][1][obj])
		m = results[name][1]
		print(f"        {label.format(ref=ref_name, n=args.n):<44} {name:<14} mean {m['mean']:.2f}, beats ref {m['beat']:.1%}, "
			  f"best-of-{args.n} {m['bo']:.1%}")
	print("        Differences smaller than about two standard errors are noise; rerun with more --seqs or another --seed.")

	oracle = None
	if mus is not None:
		oracle = simulate_rows(oracle_bars(mus, sds), rewards)
		best_fixed = max(results, key=lambda n: results[n][1]["mean"])
		d = results[best_fixed][0] - oracle
		print(f"\nORACLE  knows each game's mu and sd: mean {oracle.mean():.2f}; best fixed cutoff ({best_fixed}) "
			  f"{d.mean():+.2f} +- {d.std() / math.sqrt(len(d)):.2f} vs oracle")

	if args.real:
		sub = rewards[: args.real]
		k = args.real
		known = oracle[:k] if oracle is not None else simulate(dp_shift(base, 0.0)[None, :], sub)[0]
		known_label = "oracle" if oracle is not None else "DP+0"
		print(f"\nREAL    strategy files, unmodified, on {k} sequences{' (one instance across games)' if args.reuse else ''}; "
			  f"{known_label} on these games: {known.mean():.2f}")
		for mod in args.solution:
			t1 = time.time()
			real = run_real(sub, field, mod, args.reuse)
			d = real - ref[:k]
			dk = real - known
			print(f"        {mod:<16} mean {real.mean():8.2f}   vs {known_label} {dk.mean():+.2f} +- {dk.std() / math.sqrt(k):.2f}   "
				  f"beats {ref_name} {(d > 1e-9).mean():.0%}   ({time.time() - t1:.0f}s, "
				  f"{1000 * (time.time() - t1) / (k * T):.2f} ms per call)")


if __name__ == "__main__":
	main()
