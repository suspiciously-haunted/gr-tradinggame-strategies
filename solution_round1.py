import math
from bisect import bisect_left, bisect_right, insort

# Strategy overview (Round 3: Exponential(mean 10) rewards, collisions score zero)
# -------------------------------------------------------------------------------
# If two or more teams accept on the same turn, all of them score zero (and are
# still locked out).  So a reward is only worth what we expect to *keep*:
#     reward * P(no other free team accepts it this turn).
#
# 1. Watching the field: other_scores reports every team's (score, is_locked) at
#    t-1.  A score rise means the team took that turn's reward alone; becoming
#    locked with no score rise means it accepted and collided.  The lag between
#    a score rise and the lock flag is learned from solo accepts, so collisions
#    are dated correctly whichever way the server reports the lock.
# 2. Opponent model: from the turns each team was free we fit a cutoff
#    (P(accept) is p_hi above it, p_lo below it).  Teams with few observations
#    use a pooled fit; before any data, a Round 1 style cutoff of 2 x mean.
#    A team is free now if its last accept was at least lockout + 1 turns ago.
# 3. Value of waiting: optimal stopping with lockout on what we *would have
#    kept* on each past turn (the reward if nobody else accepted, else 0),
#    padded early on with an Exponential(mean 10) prior scaled by the fraction
#    of uncontested turns.  Accept iff expected kept reward >= V(s-1) - V(s-G).
#    The cutoff falls near the end of the game automatically.

MEAN = 10.0               # known reward mean for this round
H_MAX = 600               # DP horizon; the cutoff is flat well before this many turns left
DP_EVERY = 10             # recompute the DP every N turns
PRIOR_N = 200             # prior pseudo-samples at the start ...
PRIOR_FADE = 400          # ... fading to none after this many real turns
PRIOR_ALONE = 0.7         # prior fraction of turns nobody else accepts
PRIOR_THETA = 2.0 * MEAN  # prior opponent cutoff (Round 1 optimum was about 2 x mean)
PRIOR_HI, PRIOR_LO = 0.8, 0.1
OPP_MIN_OBS = 15          # own observations before an opponent gets its own fit
OPP_WINDOW = 400          # recent observations used per fit
OPP_REFIT = 10
DELAY = 3                 # turns to wait before a turn's accepts are final (lock flag may lag)


def _fit_threshold(obs):
	"""Best single cutoff for (reward, accepted) pairs; returns (theta, p_hi, p_lo)."""
	pts = sorted(obs)
	n = len(pts)
	total_acc = sum(1 for _, a in pts if a)
	if total_acc == 0:
		return float("inf"), 0.0, 0.5 / (n + 2)
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


class Opponent:
	__slots__ = ("prev_score", "prev_locked", "accepts", "pending", "obs", "fit", "n_fit")

	def __init__(self):
		self.prev_score = None
		self.prev_locked = False
		self.accepts = []         # sorted turns this team accepted on
		self.pending = None       # solo accept still waiting for its lock flag
		self.obs = []             # (reward, accepted) on turns this team was free
		self.fit = None
		self.n_fit = 0


class Solution:
	def __init__(self):
		self.T = None
		self.reset(None)

	def reset(self, T):
		self.T = T
		self.last_t = 0
		self.G = 16
		self.opps = []
		self.rewards = {}             # turn -> reward
		self.kept = []                # sorted: what we would have kept on each finished turn
		self.alone = 0                # finished turns nobody else accepted
		self.lock_off = [0, 0]        # lock flag seen 0 / 1 reports after the score rise
		self.pool_fit = None
		self.pool_at = 0
		self.dp_c = None
		self.dp_at = -10**9
		self.last_thr = None

	# ---------- watching the field ----------
	def add_accept(self, op, turn):
		i = bisect_left(op.accepts, turn)
		if i == len(op.accepts) or op.accepts[i] != turn:
			op.accepts.insert(i, turn)

	def observe(self, t, other_scores):
		if len(other_scores) != len(self.opps):
			self.opps = [Opponent() for _ in other_scores]
		pt = t - 1                    # the turn the reports refer to
		off = 1 if self.lock_off[1] > self.lock_off[0] else 0
		for op, (score, locked) in zip(self.opps, other_scores):
			if op.prev_score is not None:
				rose = abs(score - op.prev_score) > 1e-9
				flip = locked and not op.prev_locked
				if rose:
					self.add_accept(op, pt)
					if flip:
						self.lock_off[0] += 1
						op.pending = None
					else:
						op.pending = pt
				elif flip:
					if op.pending == pt - 1:
						self.lock_off[1] += 1         # lock flag one report after the score rise
					elif pt - off >= 1:
						self.add_accept(op, pt - off)  # accepted but scored nothing: a collision
					op.pending = None
				elif op.pending is not None and op.pending < pt - 1:
					op.pending = None
			op.prev_score, op.prev_locked = score, locked

	def finalise(self, turn):
		"""Record what happened on `turn` once its accepts are known."""
		r = self.rewards.pop(turn, None)
		if r is None:
			return
		taken = False
		for op in self.opps:
			i = bisect_left(op.accepts, turn)
			acc = i < len(op.accepts) and op.accepts[i] == turn
			free = i == 0 or turn - op.accepts[i - 1] >= self.G
			if free:
				op.obs.append((r, acc))
			taken = taken or acc
		insort(self.kept, 0.0 if taken else r)
		if not taken:
			self.alone += 1

	# ---------- opponent prediction ----------
	def accept_prob(self, op, r):
		if len(op.obs) >= OPP_MIN_OBS:
			if op.fit is None or len(op.obs) - op.n_fit >= OPP_REFIT:
				op.fit = _fit_threshold(op.obs[-OPP_WINDOW:])
				op.n_fit = len(op.obs)
			theta, hi, lo = op.fit
		elif self.pool_fit is not None:
			theta, hi, lo = self.pool_fit
		else:
			theta, hi, lo = PRIOR_THETA, PRIOR_HI, PRIOR_LO
		return hi if r >= theta else lo

	def p_alone(self, t, r, other_scores):
		p = 1.0
		for op, (_, locked) in zip(self.opps, other_scores):
			if op.accepts:
				free = t - op.accepts[-1] >= self.G
			else:
				free = not locked
			if free:
				p *= 1.0 - self.accept_prob(op, r)
		return p

	# ---------- value of waiting ----------
	def samples(self):
		xs = list(self.kept)
		n = len(xs)
		m = int(PRIOR_N * max(0.0, 1.0 - n / PRIOR_FADE))
		if m > 0:
			share = (self.alone + PRIOR_ALONE * 20) / (n + 20)
			xs += [-MEAN * math.log(1.0 - (i + 0.5) / m) * share for i in range(m)]
			xs.sort()
		return xs

	def build_dp(self, H):
		xs = self.samples()
		n = len(xs)
		suf = [0.0] * (n + 1)
		for i in range(n - 1, -1, -1):
			suf[i] = suf[i + 1] + xs[i]
		G = self.G
		V = [0.0] * (H + 1)
		c = [0.0] * (H + 1)
		for s in range(1, H + 1):
			cs = V[s - 1] - (V[s - G] if s >= G else 0.0)
			i = bisect_right(xs, cs)
			V[s] = V[s - 1] + (suf[i] - cs * (n - i)) / n
			c[s] = cs
		self.dp_c = c

	def threshold(self, t, T):
		s = max(T - t + 1, 1)
		H = min(s, H_MAX)
		if self.dp_c is None or len(self.dp_c) <= H or t - self.dp_at >= DP_EVERY or len(self.kept) < 50:
			self.build_dp(H)
			self.dp_at = t
		return self.dp_c[min(s, len(self.dp_c) - 1)]

	# ---------- entry point ----------
	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t <= self.last_t or T != self.T or t == 1:
				self.reset(T)
			self.G = max(int(lockout or 0), 0) + 1
			other_scores = list(other_scores or [])

			if t == self.last_t + 1:
				self.observe(t, other_scores)
			else:
				self.opps = []        # gap in calls: drop stale per-team state
			self.last_t = t
			self.rewards[t] = reward
			self.finalise(t - DELAY)
			if t - self.pool_at >= 25:
				pool = [o for op in self.opps for o in op.obs[-OPP_WINDOW:]]
				if len(pool) >= 30:
					self.pool_fit = _fit_threshold(pool)
				self.pool_at = t

			if is_locked:
				return False
			kept = reward * self.p_alone(t, reward, other_scores)
			self.last_thr = self.threshold(t, T)
			return bool(kept > 0 and kept >= self.last_thr)
		except Exception:
			# never crash on the server: fall back to the last threshold we computed
			return (not is_locked) and self.last_thr is not None and reward > 0 and reward >= self.last_thr


play = Solution().play
