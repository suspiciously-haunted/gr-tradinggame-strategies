import math
import random
from bisect import bisect_right
from collections import deque

# Strategy overview (Round 3: Exponential(mean 10) rewards, collisions score zero)
# -------------------------------------------------------------------------------
# Everything is learned live, inside the game, from what the other teams do.
#
# 1. Watching.  other_scores reports each team's (score, is_locked) at t-1.  A score
#    rise is a clean accept of that turn's reward; the lock flag switching on with
#    no score rise is an accept that collided.  The lag between a score rise and its
#    lock flag is learned from clean accepts, so collisions are dated correctly.
#    We see every turn's reward ourselves, so for every turn a team was free we know
#    the reward on offer and whether it took it.
# 2. Profiling.  Per team, recency-weighted counts of (reward bin, taken?) on its free
#    turns (old turns fade with half-life HALF_LIFE free turns, so teams that change
#    their rule mid-game are re-learned).  The team's rule is fitted as one reward
#    interval [lo, hi) it takes (maximum-subarray fit of taken minus skipped).  This
#    covers every rule seen so far: greedy (lo = 0), cutoff (hi = inf) and bands.
#    Bins never observed next to the interval edge count as 50/50: we do not know
#    which side of the edge they are on.  Teams with little data borrow the pooled
#    profile of all teams.
# 3. Opposing.  The chance a reward is ours alone is the product over the teams free
#    right now of (1 - their chance of taking it).  So we automatically
#      - avoid any band a free team is known to take,
#      - follow the teams that win by taking the largest rewards: right after such a
#        team accepts it is locked for the lockout, and its range opens up for us,
#      - use the gaps when every team that wants a reward is locked.
#    (Copying a winning team's rule outright would collide with it every time.)
# 4. Our cutoff.  Optimal stopping with lockout on what we would really have kept on
#    recent turns (the reward if no other team took it, else 0), padded early with an
#    Exponential(mean 10) prior.  Accept iff reward * P(ours alone) >= V(s-1) - V(s-G).
#    The cutoff falls near the end of the game automatically.
# 5. Calibration (what this version adds).  The chance in step 3 multiplies one
#    guess per free team, so with 15-20 teams small per-team errors compound: in the
#    18:47 run (20 teams) the uncalibrated version decided almost every reward was
#    contested and made only 12 accepts.  So every turn we also record what the model
#    predicted for that turn's reward, and once the turn is final whether anybody
#    else really took it.  The prediction actually used is the recent hit rate of
#    turns with a similar prediction (bins of width 1 / CAL_BINS, recency-weighted
#    with half-life CAL_HALF_LIFE turns), shrunk towards the raw prediction while
#    data is thin.  Too gloomy, too hopeful, wrong lock timing: all get corrected.
# 6. Follow what works (also new).  We know every past reward and whether anybody else
#    took it, so we can replay the last REPLAY_WINDOW turns and score, with lockouts,
#    (a) the model decisions of steps 3-5, raw and calibrated, and (b) every flat
#    cutoff on a grid.  Every REPLAY_EVERY turns the best of these over the window is
#    adopted (the model is kept unless another rule beats it by REPLAY_MARGIN).  In a
#    crowded field where the winners simply take every reward above ~5, this copies
#    whatever level is actually paying; in a field the model reads well, it keeps the
#    model.  A flat cutoff falls near the end like the optimal no-competition cutoff.
#    The replay ignores how other teams would have reacted to us.
# 7. Clones.  Two copies of a deterministic strategy make the same call every turn and
#    collide every time.  The cutoff gets a random per-game scale (+-JITTER) so a copy
#    of this file in another team drifts out of step instead.

MEAN = 10.0               # known reward mean for this round
DEFAULT_LOCKOUT = 15
EDGES = tuple(float(e) for e in range(41)) + (45.0, 50.0, 60.0, 80.0)
NB = len(EDGES)           # bin b holds rewards in [EDGES[b], EDGES[b + 1])
HALF_LIFE = 60.0          # free-turn observations after which an old one counts half
DECAY = 0.5 ** (1.0 / HALF_LIFE)
MIN_OBS = 8.0             # weighted free turns before a team gets its own profile
SHRINK = 2.0              # pseudo-counts pulling a bin's rate towards its fitted level
POOL_W = 3.0              # weight of the prior in the pooled profile
PRIOR_P = tuple(0.3 if e < 10.0 else 0.45 if e < 20.0 else 0.6 for e in EDGES)
REFIT = 10                # refit profiles every N turns
DELAY = 3                 # turns before a turn's accepts are final (lock flag may lag)
H_MAX = 600               # DP horizon; the cutoff is flat well before this
DP_EVERY = 10
KEPT_WINDOW = 600         # recent turns used for the value of waiting
PRIOR_N = 200             # Exponential prior samples at the start ...
PRIOR_FADE = 300          # ... gone after this many real turns
PRIOR_ALONE = 0.6         # prior share of turns nobody else takes
JITTER = 0.08             # random per-game scale on our cutoff, against clones
CAL_BINS = 10             # calibration bins over the predicted chance of being alone
CAL_HALF_LIFE = 300.0     # turns after which an old calibration turn counts half
CAL_DECAY = 0.5 ** (1.0 / CAL_HALF_LIFE)
CAL_PRIOR = 4.0           # pseudo-turns at the raw prediction in each bin
REPLAY_WINDOW = 600       # recent final turns replayed to pick the rule
REPLAY_EVERY = 25
REPLAY_MIN = 150          # final turns needed before the replay is trusted
REPLAY_MARGIN = 0.15      # another rule must beat the model by this fraction (sim: 0.05 switched on noise)
FLAT_GRID = tuple(0.5 * i for i in range(81))   # flat cutoffs 0 .. 40


def _bin(r):
	return max(bisect_right(EDGES, r) - 1, 0)


def _unit_cut(gap, H):
	"""Optimal cutoffs for Exponential(mean 1) with no competition, c[s] for s = 0..H."""
	W = [0.0] * (H + 1)
	c = [0.0] * (H + 1)
	for s in range(1, H + 1):
		c[s] = W[s - 1] - (W[s - gap] if s >= gap else 0.0)
		W[s] = W[s - 1] + math.exp(-c[s])
	return c


def _replay(history, rule, gap):
	"""Score of `rule(entry) -> bool` over history entries (r, kept, raw_acc, cal_acc)."""
	score, free_at = 0.0, 0
	for i, e in enumerate(history):
		if i >= free_at and rule(e):
			score += e[1]
			free_at = i + gap
	return score


def _fit(acc, n):
	"""Profile from weighted counts: per-bin chance a free team takes a reward."""
	best, cur, lo, best_lo, best_hi = 0.0, 0.0, 0, 0, 0
	for b in range(NB):
		d = acc[b] - (n[b] - acc[b])
		if cur <= 0.0:
			cur, lo = d, b
		else:
			cur += d
		if cur > best:
			best, best_lo, best_hi = cur, lo, b + 1
	a_in = sum(acc[best_lo:best_hi])
	n_in = sum(n[best_lo:best_hi])
	a_out = sum(acc) - a_in
	n_out = sum(n) - n_in
	p_in = (a_in + 0.5) / (n_in + 1.0)
	p_out = (a_out + 0.1) / (n_out + 2.0)
	prof = []
	for b in range(NB):
		level = p_in if best_lo <= b < best_hi else p_out
		prof.append((acc[b] + SHRINK * level) / (n[b] + SHRINK))
	# unobserved bins just outside the interval: the edge could be anywhere in there
	if best_hi > best_lo:
		mid = 0.5 * (p_in + p_out)
		b = best_lo - 1
		while b >= 0 and n[b] < 0.5:
			prof[b] = max(prof[b], mid)
			b -= 1
		b = best_hi
		while b < NB and n[b] < 0.5:
			prof[b] = max(prof[b], mid)
			b += 1
	return prof


class Team:
	__slots__ = ("prev_score", "prev_locked", "accepts", "pending", "acc", "n", "w", "prof", "wins")

	def __init__(self):
		self.prev_score = None
		self.prev_locked = False
		self.accepts = deque(maxlen=8)   # recent accept turns
		self.pending = None              # clean accept still waiting for its lock flag
		self.acc = [0.0] * NB            # recency-weighted taken counts per bin
		self.n = [0.0] * NB              # recency-weighted free-turn counts per bin
		self.w = 0.0                     # total weight
		self.prof = None
		self.wins = 0.0                  # points scored this game


class Solution:
	def __init__(self, seed=None):
		self.rng = random.Random(seed)
		self.T = None
		self.reset(None)

	def reset(self, T):
		self.T = T
		self.jit = 1.0 + JITTER * (2.0 * self.rng.random() - 1.0)
		self.last_t = 0
		self.G = DEFAULT_LOCKOUT + 1
		self.teams = []
		self.rewards = {}
		self.lag_votes = [0, 0]
		self.kept = deque(maxlen=KEPT_WINDOW)
		self.n_turns = 0
		self.alone = 0
		self.pool = None
		self.fit_at = -10**9
		self.dp_c = None
		self.dp_at = -10**9
		self.last_thr = None
		self.cal_alone = [0.0] * CAL_BINS
		self.cal_n = [0.0] * CAL_BINS
		self.history = deque(maxlen=REPLAY_WINDOW)   # (r, kept, raw model accepted, calibrated accepted)
		self.rule = "cal"             # "raw", "cal" or a flat cutoff
		self.replay_at = 0
		self.unit = None

	# ---------- watching ----------
	def lag(self):
		return 1 if self.lag_votes[1] > self.lag_votes[0] else 0

	def mark(self, tm, turn):
		if turn >= 1 and turn not in tm.accepts:
			tm.accepts.append(turn)

	def observe(self, t, others):
		if len(others) != len(self.teams):
			self.teams = [Team() for _ in others]
		pt = t - 1
		lag = self.lag()
		for tm, (score, locked) in zip(self.teams, others):
			locked = bool(locked)
			if tm.prev_score is not None:
				d = score - tm.prev_score
				flip = locked and not tm.prev_locked
				if abs(d) > 1e-9:
					self.mark(tm, pt)
					tm.wins += d
					if flip:
						self.lag_votes[0] += 1
						tm.pending = None
					else:
						tm.pending = pt
				elif flip:
					if tm.pending == pt - 1:
						self.lag_votes[1] += 1           # flag one report after the score rise
					else:
						self.mark(tm, pt - lag)          # took it and collided
					tm.pending = None
				elif tm.pending is not None and tm.pending < pt - 1:
					tm.pending = None
			tm.prev_score, tm.prev_locked = score, locked

	def finalise(self, turn):
		entry = self.rewards.pop(turn, None)
		if entry is None:
			return
		r, raw, raw_acc, cal_acc = entry
		b = _bin(r)
		taken = False
		for tm in self.teams:
			acc = turn in tm.accepts
			taken = taken or acc
			prior = [a for a in tm.accepts if a < turn]
			if not prior or turn - max(prior) >= self.G:
				# free on that turn: one more observation, older ones fade
				for k in range(NB):
					tm.acc[k] *= DECAY
					tm.n[k] *= DECAY
				tm.w = tm.w * DECAY + 1.0
				tm.n[b] += 1.0
				if acc:
					tm.acc[b] += 1.0
		self.kept.append(0.0 if taken else r)
		self.history.append((r, 0.0 if taken else r, raw_acc, cal_acc))
		self.n_turns += 1
		if raw is not None:
			k = min(int(raw * CAL_BINS), CAL_BINS - 1)
			for j in range(CAL_BINS):
				self.cal_alone[j] *= CAL_DECAY
				self.cal_n[j] *= CAL_DECAY
			self.cal_n[k] += 1.0
			if not taken:
				self.cal_alone[k] += 1.0
		if not taken:
			self.alone += 1

	# ---------- profiling ----------
	def refit(self):
		pa = [POOL_W * PRIOR_P[b] for b in range(NB)]
		pn = [POOL_W] * NB
		for tm in self.teams:
			for b in range(NB):
				pa[b] += tm.acc[b]
				pn[b] += tm.n[b]
		self.pool = [pa[b] / pn[b] for b in range(NB)]
		for tm in self.teams:
			tm.prof = _fit(tm.acc, tm.n) if tm.w >= MIN_OBS else None

	def p_take(self, tm, b):
		if tm.prof is not None:
			return tm.prof[b]
		return self.pool[b] if self.pool is not None else PRIOR_P[b]

	def free_now(self, tm, t):
		if tm.accepts:
			return t - max(tm.accepts) >= self.G
		return not tm.prev_locked

	def p_alone(self, t, r):
		b = _bin(r)
		p = 1.0
		for tm in self.teams:
			if self.free_now(tm, t):
				p *= 1.0 - self.p_take(tm, b)
		return p

	def calibrated(self, raw):
		k = min(int(raw * CAL_BINS), CAL_BINS - 1)
		return (self.cal_alone[k] + CAL_PRIOR * raw) / (self.cal_n[k] + CAL_PRIOR)

	def pick_rule(self):
		h = self.history
		G = self.G
		best_rule = "cal"
		best = _replay(h, lambda e: e[3], G)
		raw = _replay(h, lambda e: e[2], G)
		if raw > best * (1.0 + REPLAY_MARGIN):
			best_rule, best = "raw", raw
		bar = best * (1.0 + REPLAY_MARGIN)
		for k in FLAT_GRID:
			sc = _replay(h, lambda e, k=k: e[0] >= k, G)
			if sc > bar:
				best_rule, bar = k, sc
		self.rule = best_rule

	def flat_cut(self, k, s):
		if self.unit is None or len(self.unit) != H_MAX + 1:
			self.unit = _unit_cut(self.G, H_MAX)
		return k * self.unit[min(s, H_MAX)] / self.unit[H_MAX]

	# ---------- our cutoff ----------
	def samples(self):
		xs = list(self.kept)
		m = int(PRIOR_N * max(0.0, 1.0 - self.n_turns / PRIOR_FADE))
		if m > 0:
			share = (self.alone + PRIOR_ALONE * 20) / (self.n_turns + 20)
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

	def cutoff(self, t, T):
		s = max(T - t + 1, 1)
		H = min(s, H_MAX)
		if self.dp_c is None or len(self.dp_c) <= H or t - self.dp_at >= DP_EVERY or self.n_turns < 50:
			self.build_dp(H)
			self.dp_at = t
		return self.dp_c[min(s, len(self.dp_c) - 1)]

	# ---------- entry point ----------
	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t != self.last_t + 1 or T != self.T:
				self.reset(T)
			self.last_t = t
			self.G = (int(lockout) if lockout and lockout > 0 else DEFAULT_LOCKOUT) + 1
			others = list(other_scores or [])
			if t > 1:
				self.observe(t, others)
			if t - self.fit_at >= REFIT:
				self.refit()
				self.fit_at = t
			raw = self.p_alone(t, reward) if self.teams else None
			self.last_thr = self.jit * self.cutoff(t, T)
			p_raw = raw if raw is not None else 1.0
			p_cal = self.calibrated(raw) if raw is not None else 1.0
			raw_acc = reward > 0 and reward * p_raw >= self.last_thr
			cal_acc = reward > 0 and reward * p_cal >= self.last_thr
			self.rewards[t] = (reward, raw, raw_acc, cal_acc)
			self.finalise(t - DELAY)
			if len(self.history) >= REPLAY_MIN and t - self.replay_at >= REPLAY_EVERY:
				self.pick_rule()
				self.replay_at = t

			if is_locked or reward <= 0:
				return False
			if self.rule == "cal":
				return bool(cal_acc)
			if self.rule == "raw":
				return bool(raw_acc)
			return bool(reward >= self.jit * self.flat_cut(self.rule, max(T - t + 1, 1)))
		except Exception:
			# never crash on the server: fall back to the last cutoff we computed
			return (not is_locked) and self.last_thr is not None and reward > 0 and reward >= self.last_thr


play = Solution().play
