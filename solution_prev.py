import math
from bisect import bisect_right, insort
from collections import deque
from statistics import NormalDist

# Strategy overview
# -----------------
# 1. Optimal stopping with lockout: V(s) = E[max(X + V(s-1-L), V(s-1))], where s is
#    the number of turns left and X is the reward *we would actually receive*.
#    Accept iff that received reward >= V(s-1) - V(s-1-L).  The threshold falls to 0
#    near the end of the round automatically.
# 2. The distribution of X is learned online from every reward we see (we see one
#    every turn, even while locked).  Until enough rewards are in, the samples are
#    padded with a normal prior fitted to the rewards seen so far; nothing is
#    accepted before MIN_SAMPLES rewards.  Round 2 sims: this learned threshold
#    scores the same as the best hand-tuned flat cutoff, and shifting it up or down
#    does not help, so it is used unchanged for the whole game.
# 3. Competition: "the reward may be added".  We learn how payouts depend on how many
#    players accept the same turn by watching everybody's score changes, phi(n) =
#    expected fraction of the reward each of n simultaneous accepters receives.
#    Prior is phi(n) = 1/n (shared / random winner); if the data shows everyone gets
#    the full reward, phi -> 1 and competition stops mattering.
# 4. We model each opponent as a threshold player, track when they are locked, and
#    predict how many of them will grab the current reward.  The counterfactual
#    "what would I have got" history feeds back into step 1.

MIN_SAMPLES = 10          # rewards to see before accepting anything
PRIOR_SAMPLES = 60
PRIOR_FADE = 150          # real samples after which the prior is dropped
DP_EVERY = 10             # recompute the DP table every N turns
DP_MAX_H = 400            # threshold is stationary beyond this horizon (flat from ~100 turns left)
CUTOFF_MULT = 1.0         # Round 2 sim: shifting the EV-optimal threshold did not help any objective
PAYOUT_PRIOR_SHARED = False  # leaderboard shows simultaneous accepters all get the full reward
EXP_FIT = True            # if rewards look exponential, fit the mean instead of using raw samples
DRIFT_WINDOW = 300        # if the recent mean shifts, learn only from this many recent rewards
OPP_WINDOW = 300          # recent observations used per opponent
OPP_REFIT = 25
MIN_ACCEPT = None         # optional manual flat cutoff (Round 2 best was 1.00); None = use the learned threshold
END_TURNS = 50            # with MIN_ACCEPT set, use the learned threshold only in the last this many turns


def _exp_quantiles(mean, n):
	return [-mean * math.log(1.0 - (i + 0.5) / n) for i in range(n)]


class Opponent:
	__slots__ = ("prev_score", "prev_locked", "last_accept", "obs", "theta", "p_hi", "p_lo", "n_fit")

	def __init__(self):
		self.prev_score = None
		self.prev_locked = False
		self.last_accept = None
		self.obs = deque(maxlen=OPP_WINDOW)
		self.theta = None
		self.p_hi = 0.5
		self.p_lo = 0.05
		self.n_fit = 0


def _fit_threshold(obs):
	"""Best single threshold for (reward, accepted) pairs; returns (theta, p_hi, p_lo)."""
	pts = sorted(obs)
	n = len(pts)
	total_acc = sum(1 for _, a in pts if a)
	if total_acc == 0:
		return float("inf"), 0.0, 0.0
	# errors(theta at index i) = accepts below i + skips at/above i
	best_err, best_i = None, 0
	acc_below = 0
	for i in range(n + 1):
		skips_above = (n - i) - (total_acc - acc_below)
		err = acc_below + skips_above
		if best_err is None or err < best_err:
			best_err, best_i = err, i
		if i < n and pts[i][1]:
			acc_below += 1
	if best_i == 0:
		theta = pts[0][0] - 1e-9
	elif best_i == n:
		theta = pts[-1][0] + 1e-9
	else:
		theta = 0.5 * (pts[best_i - 1][0] + pts[best_i][0])
	above = pts[best_i:]
	below = pts[:best_i]
	p_hi = (sum(1 for _, a in above if a) + 0.5) / (len(above) + 1)
	p_lo = (sum(1 for _, a in below if a) + 0.1) / (len(below) + 2)
	return theta, p_hi, p_lo


class Solution:
	def __init__(self):
		self.reset(None)

	def reset(self, T):
		self.T = T
		self.last_t = 0
		self.L = 15
		self.free_gap = None          # turns from my accept to my first free turn
		self.raw = []                 # sorted raw rewards
		self.raw_sum = 0.0
		self.eff = []                 # sorted counterfactual received rewards
		self.eff_seq = []             # same, in arrival order (for drift checks)
		self.eff_n = 0
		self.dp_c = None              # dp_c[s] = threshold with s turns left
		self.dp_at = -10**9
		self.prev_reward = None
		self.prev_accepted = False
		self.my_accept_t = None
		self.prev_my_score = None
		self.prev_my_locked = False
		self.opps = []
		self.pool = deque(maxlen=2000)  # pooled opponent observations
		self.pool_fit = (None, 0.5, 0.05)
		self.phi_sum = {}
		self.phi_cnt = {}
		self.last_thr = None          # last threshold computed, used if something fails

	# ---------- payout model ----------
	def phi(self, n):
		w = 2.0
		p1 = (self.phi_sum.get(1, 0.0) + w) / (self.phi_cnt.get(1, 0) + w)
		if n <= 1:
			return p1
		prior_beta = 1.0 if PAYOUT_PRIOR_SHARED else 0.0
		if self.phi_cnt.get(n, 0) > 0:
			return (self.phi_sum[n] + w * p1 * n ** (-prior_beta)) / (self.phi_cnt[n] + w)
		# extrapolate with phi(n) = phi(1) * n^-beta
		num, den = prior_beta * w, w
		for k, c in self.phi_cnt.items():
			if k < 2 or c == 0:
				continue
			ratio = max(self.phi_sum[k] / c / max(p1, 1e-9), 1e-6)
			num += c * (-math.log(ratio) / math.log(k))
			den += c
		beta = min(max(num / den, 0.0), 2.0)
		return p1 * n ** (-beta)

	# ---------- learning from the previous turn ----------
	def observe(self, t, your_score, other_scores, is_locked):
		r = self.prev_reward
		if r is None:
			return
		# my own lock calibration
		if self.my_accept_t is not None and self.free_gap is None and not is_locked and self.prev_my_locked:
			self.free_gap = t - self.my_accept_t
		self.prev_my_locked = is_locked

		if len(other_scores) != len(self.opps):
			self.opps = [Opponent() for _ in other_scores]
		pt = t - 1  # the turn the reports refer to
		accepters = []
		gap = self.free_gap or (self.L + 1)
		for op, (score, locked) in zip(self.opps, other_scores):
			if op.prev_score is None:
				op.prev_score, op.prev_locked = score, locked
				continue
			d = score - op.prev_score
			was_free = op.last_accept is None or pt - op.last_accept >= gap
			acc = (abs(d) > 1e-12) or (locked and not op.prev_locked)
			if acc and was_free:
				op.last_accept = pt
				accepters.append(d)
			if was_free:
				op.obs.append((r, acc))
				self.pool.append((r, acc))
			op.prev_score, op.prev_locked = score, locked

		k_others = len(accepters)
		n = k_others + (1 if self.prev_accepted else 0)
		if abs(r) > 1e-9 and n > 0:
			fracs = list(d / r for d in accepters)
			if self.prev_accepted and self.prev_my_score is not None:
				fracs.append((your_score - self.prev_my_score) / r)
			for f in fracs:
				f = min(max(f, -1.0), 2.0)
				self.phi_sum[n] = self.phi_sum.get(n, 0.0) + f
				self.phi_cnt[n] = self.phi_cnt.get(n, 0) + 1

		# counterfactual: what I would have received had I accepted
		insort(self.eff, (r, k_others))
		self.eff_seq.append((r, k_others))
		self.eff_n += 1

	# ---------- opponent prediction ----------
	def accept_prob(self, op, r):
		if len(op.obs) >= 20:
			if op.n_fit != len(op.obs) and (len(op.obs) - op.n_fit >= OPP_REFIT or op.theta is None):
				op.theta, op.p_hi, op.p_lo = _fit_threshold(op.obs)
				op.n_fit = len(op.obs)
			if op.theta is not None:
				return op.p_hi if r >= op.theta else op.p_lo
		theta, hi, lo = self.pool_fit
		if theta is None:
			return 0.3
		return hi if r >= theta else lo

	def expected_share(self, t, r, other_scores):
		gap = self.free_gap or (self.L + 1)
		probs = []
		for op, (_, locked) in zip(self.opps, other_scores):
			if op.last_accept is not None:
				free = t - op.last_accept >= gap
			else:
				free = not locked
			if free:
				probs.append(self.accept_prob(op, r))
		# Poisson-binomial distribution of the number of other accepters
		dist = [1.0]
		for p in probs:
			nd = [0.0] * (len(dist) + 1)
			for k, q in enumerate(dist):
				nd[k] += q * (1 - p)
				nd[k + 1] += q * p
			dist = nd
		return sum(q * self.phi(k + 1) for k, q in enumerate(dist))

	# ---------- distribution checks ----------
	@staticmethod
	def looks_exponential(xs):
		"""Non-negative, sd ~ mean, median ~ 0.69 mean, ~13.5% above 2x mean."""
		n = len(xs)
		if n < 50 or min(xs) < 0:
			return False
		mean = sum(xs) / n
		if mean <= 0:
			return False
		sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
		med = sorted(xs)[n // 2]
		tail = sum(1 for x in xs if x > 2 * mean) / n
		return 0.85 <= sd / mean <= 1.15 and 0.58 <= med / mean <= 0.80 and 0.08 <= tail <= 0.19

	def drifted(self):
		"""True if recent rewards have a clearly different mean from earlier ones."""
		n = len(self.eff_seq)
		if n < 2 * DRIFT_WINDOW:
			return False
		old = [r for r, _ in self.eff_seq[:-DRIFT_WINDOW]]
		new = [r for r, _ in self.eff_seq[-DRIFT_WINDOW:]]
		mo, mn = sum(old) / len(old), sum(new) / len(new)
		var = sum((x - mo) ** 2 for x in old) / (len(old) - 1)
		se = math.sqrt(var / len(new) + var / len(old))
		return se > 0 and abs(mn - mo) > 4 * se

	# ---------- dynamic programme ----------
	def samples(self):
		"""Sorted list of counterfactual received rewards (with prior padding)."""
		phis = {}
		vals = []
		seq = self.eff_seq
		if self.drifted():
			seq = seq[-DRIFT_WINDOW:]
		for r, k in seq:
			if k not in phis:
				phis[k] = self.phi(k + 1)
			vals.append(r * phis[k])
		n_real = len(vals)
		if EXP_FIT and n_real >= 50 and self.looks_exponential([r for r, _ in seq]):
			# smooth fit, scaled by the average payout share
			return _exp_quantiles(sum(vals) / n_real, 1000)
		if n_real < PRIOR_FADE and len(self.raw) >= 3:
			# normal prior fitted to the raw rewards, scaled by the average payout share
			m = int(PRIOR_SAMPLES * (1 - n_real / PRIOR_FADE)) + 1
			n = len(self.raw)
			mean = self.raw_sum / n
			sd = math.sqrt(max(sum((x - mean) ** 2 for x in self.raw) / (n - 1), 1e-12))
			share = sum(phis[k] for _, k in seq) / n_real if n_real else self.phi(1)
			prior = NormalDist(mean, sd)
			vals += [share * prior.inv_cdf((i + 0.5) / m) for i in range(m)]
		vals.sort()
		return vals

	def build_dp(self, H):
		xs = self.samples()
		n = len(xs)
		suf = [0.0] * (n + 1)
		for i in range(n - 1, -1, -1):
			suf[i] = suf[i + 1] + xs[i]
		L = self.L
		V = [0.0] * (H + 1)
		c = [0.0] * (H + 1)
		for s in range(1, H + 1):
			after = V[s - 1 - L] if s - 1 - L >= 0 else 0.0
			cs = V[s - 1] - after
			i = bisect_right(xs, cs)
			gain = (suf[i] - cs * (n - i)) / n
			V[s] = V[s - 1] + gain
			c[s] = cs
		self.dp_c = c

	def threshold(self, t, T):
		s = T - t + 1
		H = min(max(s, 1), DP_MAX_H)
		need = self.dp_c is None or len(self.dp_c) <= H or t - self.dp_at >= DP_EVERY or self.eff_n < 50
		if need:
			self.build_dp(min(max(T - t + 1, 1), DP_MAX_H))
			self.dp_at = t
		self.last_thr = CUTOFF_MULT * self.dp_c[min(s, len(self.dp_c) - 1)]
		return self.last_thr

	# ---------- entry point ----------
	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t <= self.last_t or T != self.T or t == 1:
				self.reset(T)
			if lockout and lockout > 0:
				self.L = int(lockout)
			other_scores = list(other_scores or [])

			if t == self.last_t + 1:
				self.observe(t, your_score, other_scores, is_locked)
			else:
				# gap in calls: drop stale per-turn state
				self.opps = []
			self.last_t = t
			insort(self.raw, reward)
			self.raw_sum += reward
			if len(self.pool) >= 30 and len(self.pool) % OPP_REFIT == 0:
				self.pool_fit = _fit_threshold(self.pool)

			accept = False
			if not is_locked and len(self.raw) >= MIN_SAMPLES:
				share = self.expected_share(t, reward, other_scores) if self.opps else self.phi(1)
				received = reward * share
				if MIN_ACCEPT is not None and T - t + 1 > END_TURNS:
					accept = reward >= MIN_ACCEPT
				else:
					accept = received > 0 and received >= self.threshold(t, T)

			self.prev_reward = reward
			self.prev_accepted = accept
			self.prev_my_score = your_score
			if accept:
				self.my_accept_t = t
			return bool(accept)
		except Exception:
			# never crash on the server: fall back to the last threshold we computed
			return (not is_locked) and self.last_thr is not None and reward > 0 and reward >= self.last_thr


play = Solution().play
