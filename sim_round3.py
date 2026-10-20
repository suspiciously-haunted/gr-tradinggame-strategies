"""Round 3 tournament: 15 teams see the same Exponential(mean 10) reward each turn;
if two or more accept on the same turn, all of them score zero (and all are locked
out for 15 turns).  Ranking is by latest score, so the average score is what counts.

Every preset in solution.PRESETS takes the ME seat in each field:
  live      the 15 other teams of the 18:26 Round 3 run, rules guessed from the
            rewards each banked (see LIVE)
  r1        the 14 other Round 1 teams resubmitted unchanged.  Rules are guessed from
            the Round 1 leaderboard: five teams scored exactly 2718.88 (modelled as
            the optimal no-competition cutoff), the 20.4 group, and flat cutoffs for
            the rest.
  adapted   the r1 field with four of the identical teams switched to Round 3 presets
  selfplay  all 15 seats run the candidate (what happens if everybody copies it)
The same reward sequences are used for every candidate (common random numbers).

  uv run python sim_round3.py --games 100 --selfplay-games 20
  uv run python sim_round3.py --presets hybrid combo blind --fields r1 --lag 1
"""
import argparse
import math
import random
import time
from multiprocessing import Pool

import solution_live as solution  # the presets live here; solution.py is the other agent's design

T = 2000
L = 15
GAP = L + 1


class Flat:
	def __init__(self, c, grab_last=False):
		self.c, self.grab_last = c, grab_last

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		if is_locked:
			return False
		return (self.grab_last and t == T) or reward >= self.c


class Blind:
	"""Optimal cutoff when nobody else plays (the Round 1 rule most teams converged on)."""
	def __init__(self, mult=1.0):
		self.cut = solution._unit_cut(GAP)
		self.mult = mult

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		if is_locked:
			return False
		return reward >= self.mult * solution.MEAN * self.cut[min(T - t + 1, solution.H_MAX)]


R1 = [
	("CIA", lambda: Blind()), ("Trial1", lambda: Blind()), ("Beans on Toast", lambda: Blind()),
	("GeneralRelativity", lambda: Blind()), ("SIG_PROC", lambda: Blind()),
	("Test", lambda: Flat(20.4)), ("mass", lambda: Flat(20.4)), ("Quant Star", lambda: Flat(20.4)),
	("fussball", lambda: Flat(20.4, grab_last=True)), ("GR GANG", lambda: Flat(21.0)),
	("eduroam", lambda: Flat(18.0)), ("Max's Team", lambda: Flat(17.0)), ("SPCV", lambda: Flat(14.0)),
	("GGS", lambda: Flat(30.0)),
]
class Band:
	def __init__(self, lo, hi):
		self.lo, self.hi = lo, hi

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		return not is_locked and self.lo <= reward < self.hi


# 18:26 Round 3 run, rules guessed from the rewards each team banked (min / median / max)
LIVE = [
	("GeneralRelativity", lambda: Flat(22.0)), ("GR GANG", lambda: Band(10.0, 20.0)),
	("Test", lambda: Flat(10.4)), ("GGS", lambda: Flat(12.0)), ("Max's Team", lambda: Flat(0.1)),
	("eduroam", lambda: Flat(0.1)), ("Beans on Toast", lambda: Flat(20.0)), ("Trial1", lambda: Flat(18.5)),
	("MASS", lambda: Flat(20.0)), ("Brendan Burgess", lambda: Flat(28.0)), ("Quant Star", lambda: Flat(28.0)),
	("SPCV", lambda: Flat(10.0)), ("CIA", lambda: Blind()), ("SKQuant", lambda: Blind()),
	("swirlG", lambda: solution.Solution("A_sole")),
]
ADAPTED_SWAPS = {"CIA": "A_sole", "Trial1": "combo", "Beans on Toast": "hybrid_bands", "GeneralRelativity": "hybrid"}


def make_field(field, preset, seed):
	if field == "selfplay":
		return [solution.Solution(preset, seed=seed * 31 + i) for i in range(len(R1) + 1)]
	players = [solution.Solution(preset, seed=seed * 31)]
	for i, (name, factory) in enumerate(LIVE if field == "live" else R1):
		if field == "adapted" and name in ADAPTED_SWAPS:
			players.append(solution.Solution(ADAPTED_SWAPS[name], seed=seed * 31 + i + 1))
		else:
			players.append(factory())
	return players


def run_game(players, seq, lag):
	P = len(players)
	score = [0.0] * P
	free_at = [1] * P
	last_acc = [-10**9] * P
	plays = [p.play for p in players]
	for t in range(1, T + 1):
		r = seq[t - 1]
		if t == 1:
			report = None
		elif lag == 0:
			report = [(score[k], free_at[k] > t) for k in range(P)]
		else:
			report = [(score[k], free_at[k] > t - 1 and last_acc[k] < t - 1) for k in range(P)]
		acc = []
		for j in range(P):
			others = [] if report is None else report[:j] + report[j + 1:]
			locked = free_at[j] > t
			if plays[j](r, L, t, T, score[j], others, locked) and not locked:
				acc.append(j)
		if len(acc) == 1:
			score[acc[0]] += r
		for j in acc:
			free_at[j] = t + GAP
			last_acc[j] = t
	return score


def task(args):
	field, preset, g, seed, lag = args
	rng = random.Random(seed * 1_000_003 + g)
	seq = [rng.expovariate(1.0 / solution.MEAN) for _ in range(T)]
	return field, preset, run_game(make_field(field, preset, seed * 7919 + g), seq, lag)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--presets", nargs="+", default=list(solution.PRESETS))
	ap.add_argument("--fields", nargs="+", default=["live", "r1", "adapted", "selfplay"], choices=["live", "r1", "adapted", "selfplay"])
	ap.add_argument("--games", type=int, default=100, help="games per preset in r1 and adapted")
	ap.add_argument("--selfplay-games", type=int, default=20)
	ap.add_argument("--lag", type=int, choices=[0, 1], default=0,
					help="0: an opponent's lock flag shows on the same report as its score change; 1: one report later")
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--workers", type=int, default=None)
	args = ap.parse_args()

	tasks = [(f, p, g, args.seed, args.lag) for f in args.fields for p in args.presets
			 for g in range(args.selfplay_games if f == "selfplay" else args.games)]
	t0 = time.time()
	res = {}
	with Pool(args.workers) as pool:
		for f, p, sc in pool.imap_unordered(task, tasks, chunksize=2):
			res.setdefault((f, p), []).append(sc)
	print(f"{len(tasks)} games in {time.time() - t0:.0f}s, lag {args.lag}\n")

	summary = {}
	for f in args.fields:
		print(f"FIELD {f}")
		print(f"  {'preset':<12}{'mean':>9}{'+-':>7}{'avg rank':>10}{'best other':>12}{'field avg':>11}")
		for p in args.presets:
			games = res[(f, p)]
			if f == "selfplay":
				mine = [sum(sc) / len(sc) for sc in games]       # every seat is the candidate
				ranks = [float("nan")]
			else:
				mine = [sc[0] for sc in games]
				ranks = [1 + sum(o > sc[0] for o in sc[1:]) + 0.5 * sum(o == sc[0] for o in sc[1:]) for sc in games]
			n = len(mine)
			m = sum(mine) / n
			se = math.sqrt(sum((x - m) ** 2 for x in mine) / max(n - 1, 1) / n)
			best = sum(max(sc[1:]) for sc in games) / n
			favg = sum(sum(sc[1:]) / (len(sc) - 1) for sc in games) / n
			summary[(f, p)] = m
			print(f"  {p:<12}{m:9.1f}{se:7.1f}{sum(ranks) / len(ranks):10.2f}{best:12.1f}{favg:11.1f}")
		print()
	head = [f for f in ("live", "r1", "adapted") if f in args.fields]
	if head:
		pick = max(args.presets, key=lambda p: sum(summary[(f, p)] for f in head))
		print(f"PICK  highest average over {' + '.join(head)}: {pick}")
		print("      Differences under about two standard errors are noise; check selfplay before switching ACTIVE.")


if __name__ == "__main__":
	main()
