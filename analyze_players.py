"""Analyse what other players (especially bot-pro) are doing on the leaderboard.

The leaderboard page embeds every team's cumulative score per turn of the last
run.  Each step up in a team's score is an accepted reward, so from the curves
we can recover:
  * when each team accepted and how much it received,
  * the real lockout gap (smallest spacing between two accepts),
  * the rewards that were on offer on turns where *anyone* accepted,
  * for a target team: rewards it accepted (lower bound on its threshold) and
    rewards it saw while free but skipped (upper bound), per phase of the round,
  * whether simultaneous accepters got the same amount (shared vs. full payout).

Usage:
  python3 analyze_players.py                      # fetch live page, analyse bot-pro
  python3 analyze_players.py --team JMRC          # analyse another team
  python3 analyze_players.py --file page.html     # analyse a saved page
  python3 analyze_players.py --watch              # poll every 30s, save each new run
"""
import argparse
import html
import os
import re
import statistics
import time
import urllib.request
from collections import defaultdict

URL = "http://ec2-54-217-141-255.eu-west-1.compute.amazonaws.com:8080/"
SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
EPS = 1e-6

TRACE_RE = re.compile(r"\{x:\[([^\]]*)\],y:\[([^\]]*)\],mode:'lines',name:\"((?:[^\"\\]|\\.)*)\"")
LAST_RUN_RE = re.compile(r"Last run:\s*([^<]+?)\s*<span[^>]*>([^<]*)</span>")
ROW_RE = re.compile(r"<tr[^>]*><td>(\d+)</td><td>(.*?)</td><td[^>]*>(.*?)</td><td>(.*?)</td>")


def fetch(url):
	with urllib.request.urlopen(url, timeout=20) as r:
		return r.read().decode("utf-8", "replace")


def parse(page):
	m = LAST_RUN_RE.search(page)
	meta = {"last_run": m.group(1).strip() if m else "?", "round": html.unescape(m.group(2)) if m else "?"}
	traces = {}
	for xs, ys, name in TRACE_RE.findall(page):
		name = html.unescape(name.encode().decode("unicode_escape"))
		x = [int(float(v)) for v in xs.split(",") if v.strip()]
		y = [float(v) for v in ys.split(",") if v.strip()]
		traces[name] = (x, y)
	table = []
	for rank, team, best, latest in ROW_RE.findall(page):
		table.append((int(rank), html.unescape(team), html.unescape(re.sub("<[^>]+>", "", best)),
					  html.unescape(re.sub("<[^>]+>", "", latest))))
	return meta, traces, table


def accepts_of(x, y):
	"""List of (turn, amount) where the score changed."""
	out = []
	for i in range(1, len(y)):
		d = y[i] - y[i - 1]
		if abs(d) > EPS:
			out.append((x[i], d))
	return out


def q(vals, p):
	if not vals:
		return float("nan")
	s = sorted(vals)
	return s[min(len(s) - 1, int(p * len(s)))]


def fmt(v):
	return "   -  " if v != v else f"{v:6.2f}"


def analyse(meta, traces, target):
	print(f"Run: {meta['last_run']}   {meta['round']}")
	if not traces:
		print("No score curves found on the page.")
		return
	T = max(max(x) for x, _ in traces.values())
	acc = {name: accepts_of(x, y) for name, (x, y) in traces.items()}

	# Lockout gap: smallest spacing between consecutive accepts of any team.
	gaps_all = [b[0] - a[0] for a_list in acc.values() for a, b in zip(a_list, a_list[1:])]
	gap = min(gaps_all) if gaps_all else 16
	print(f"Turns: {T}   minimum gap between accepts (lockout + 1): {gap}\n")

	# ---- summary per team ----
	print(f"{'team':<18}{'score':>9}{'accepts':>9}{'mean':>8}{'min':>8}{'med':>8}{'max':>8}{'wait/acc':>10}")
	for name, (x, y) in sorted(traces.items(), key=lambda kv: -kv[1][1][-1]):
		a = acc[name]
		amts = [d for _, d in a]
		waits = [b[0] - p[0] - gap for p, b in zip(a, a[1:])]
		print(f"{name[:17]:<18}{y[-1]:9.1f}{len(a):9d}{fmt(statistics.mean(amts)) if amts else fmt(float('nan')):>8}"
			  f"{fmt(min(amts) if amts else float('nan')):>8}{fmt(q(amts, .5)):>8}{fmt(max(amts) if amts else float('nan')):>8}"
			  f"{(statistics.mean(waits) if waits else float('nan')):10.1f}")

	# ---- simultaneous accepts: shared or full payout? ----
	by_turn = defaultdict(list)
	for name, a in acc.items():
		for t, d in a:
			by_turn[t].append((name, d))
	multi = {t: v for t, v in by_turn.items() if len(v) > 1}
	print(f"\nTurns with simultaneous accepts: {len(multi)}")
	if multi:
		same = sum(1 for v in multi.values() if max(d for _, d in v) - min(d for _, d in v) < 1e-3)
		print(f"  identical amounts on {same}/{len(multi)} of them "
			  f"({'everyone seems to get the same payout' if same == len(multi) else 'payouts differ -> sharing/random winner?'})")
		for t in sorted(multi)[:8]:
			print(f"  turn {t:5d}: " + ", ".join(f"{n}={d:.2f}" for n, d in multi[t]))
		solo = [d for v in by_turn.values() if len(v) == 1 for _, d in v]
		shared = [d for v in multi.values() for _, d in v]
		if solo and shared:
			print(f"  mean payout solo {statistics.mean(solo):.2f} vs shared turns {statistics.mean(shared):.2f}")

	# ---- target team: threshold reconstruction ----
	if target not in traces:
		print(f"\nTeam '{target}' not on the chart. Teams: {', '.join(traces)}")
		return
	print(f"\n=== {target} ===")
	# Rewards known on a turn = largest payout anyone received that turn
	# (if payouts are shared this underestimates, which still gives a valid upper bound on skipped values).
	known = {t: max(d for _, d in v) for t, v in by_turn.items()}
	mine = dict(acc[target])
	last = -10**9
	took, skipped = [], []  # (turn, reward)
	for t in range(1, T + 1):
		free = t - last >= gap
		if t in mine:
			took.append((t, mine[t]))
			last = t
		elif free and t in known:
			skipped.append((t, known[t]))
	gaps = [b[0] - a[0] for a, b in zip(acc[target], acc[target][1:])]
	if gaps:
		print(f"gaps between accepts: min {min(gaps)}  median {q(gaps, .5)}  max {max(gaps)}")
	print("Threshold by phase (accepted min = lower bound, free-but-skipped max = upper bound):")
	print(f"{'turns':>13}{'n_acc':>7}{'acc_min':>9}{'acc_med':>9}{'n_skip':>8}{'skip_max':>10}{'est_thr':>9}")
	edges = [1, 200, 500, 1000, 1500, T - 100, T - 50, T - 16, T + 1]
	edges = sorted(set(e for e in edges if 1 <= e <= T + 1))
	for lo, hi in zip(edges, edges[1:]):
		a = [r for t, r in took if lo <= t < hi]
		s = [r for t, r in skipped if lo <= t < hi]
		amin = min(a) if a else float("nan")
		smax = max(s) if s else float("nan")
		# best estimate: highest skip below the lowest accept, midpoint with that accept
		below = [r for r in s if a and r < amin]
		est = (amin + max(below)) / 2 if below else amin
		print(f"{lo:6d}-{hi - 1:<6d}{len(a):7d}{fmt(amin):>9}{fmt(q(a, .5)):>9}{len(s):8d}{fmt(smax):>10}{fmt(est):>9}")
	viol = [(t, r) for t, r in skipped if took and r > min(x for _, x in took)]
	print(f"Free-but-skipped rewards above its lowest accept: {len(viol)} "
		  f"(non-zero means the threshold moves over time or it is not a fixed threshold)")
	tail = [t for t, _ in took if t > T - 3 * gap]
	print(f"Accepts in the last {3 * gap} turns: {len(tail)}  {tail}")


def reward_samples(traces, team="bot-greedy"):
	"""bot-greedy accepts whenever free regardless of value, so its payouts are unbiased reward samples."""
	if team not in traces:
		return []
	return [d for _, d in accepts_of(*traces[team])]


def describe(xs):
	if len(xs) < 2:
		print("Not enough reward samples.")
		return
	xs = sorted(xs)
	n = len(xs)
	mean = statistics.mean(xs)
	sd = statistics.stdev(xs)
	print(f"Reward samples: {n}   mean {mean:.2f}   sd {sd:.2f}   sd/mean {sd / mean if mean else float('nan'):.2f}"
		  f"   min {xs[0]:.2f}   max {xs[-1]:.2f}   negative {sum(1 for x in xs if x < 0) / n:.1%}")
	print("Quantiles: " + "  ".join(f"{int(p * 100)}%={q(xs, p):.2f}" for p in (.1, .25, .5, .75, .9, .95, .99)))
	print("(Exponential would show sd/mean about 1.0, median about 0.69 x mean, 90% about 2.3 x mean)")


def dump(path, meta, xs):
	with open(path, "a") as f:
		for x in xs:
			f.write(f"{meta['last_run']},{meta['round']},{x}\n")


def watch(target, interval, dump_path=None):
	os.makedirs(SNAP_DIR, exist_ok=True)
	seen = None
	while True:
		try:
			page = fetch(URL)
			meta, traces, _ = parse(page)
			if meta["last_run"] != seen:
				seen = meta["last_run"]
				fn = os.path.join(SNAP_DIR, re.sub(r"[^0-9A-Za-z]+", "_", seen) + ".html")
				with open(fn, "w") as f:
					f.write(page)
				print("\n" + "=" * 70 + f"\nsaved {fn}")
				analyse(meta, traces, target)
				if dump_path:
					dump(dump_path, meta, reward_samples(traces))
					rnd = meta["round"]
					allx = [float(line.rsplit(",", 1)[1]) for line in open(dump_path) if f",{rnd}," in line]
					print(f"\nAll runs so far for {rnd}:")
					describe(allx)
		except Exception as e:
			print("fetch failed:", e)
		time.sleep(interval)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--team", default="bot-pro")
	ap.add_argument("--file")
	ap.add_argument("--watch", action="store_true")
	ap.add_argument("--interval", type=int, default=30)
	ap.add_argument("--dump-rewards", help="append reward samples (from bot-greedy) to this CSV file")
	args = ap.parse_args()
	if args.watch:
		watch(args.team, args.interval, args.dump_rewards)
		return
	page = open(args.file).read() if args.file else fetch(URL)
	meta, traces, table = parse(page)
	if table:
		print("Leaderboard:")
		for rank, team, best, latest in table:
			print(f"  {rank:2d}. {team:<20} best {best:>9}  latest {latest:>9}")
		print()
	analyse(meta, traces, args.team)
	print()
	describe(reward_samples(traces))
	if args.dump_rewards:
		dump(args.dump_rewards, meta, reward_samples(traces))


if __name__ == "__main__":
	main()
