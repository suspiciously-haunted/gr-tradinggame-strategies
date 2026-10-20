import math
import random
from bisect import bisect_right, insort
from collections import deque

# Strategy overview (Round 3: Exponential(mean 10) rewards, collisions score zero)
# -------------------------------------------------------------------------------
# If two or more teams accept on the same turn, all of them score zero (and are
# still locked out).  Ranking is by latest score, so the goal is the highest
# expected score against the field (15-20 teams).
#
# 1. Opponent tracking.  other_scores gives each team's score and lock flag.  An
#    accept shows up as a score change, or for a collision (score unchanged) as the
#    lock flag switching on.  Whether the flag appears on the same report as the
#    score change or one report later is learned from clean accepts.  From the
#    accept turns we know when every team is free again.
# 2. Opponent model.  For every turn a team was free we record (reward bin,
#    accepted?).  p_i(r) = team i's acceptance rate in r's bin, shrunk towards the
#    pooled rate of all teams, which is shrunk towards a Round 1 style prior
#    (cutoff about 20).  Bins also capture bands and other odd rules.
# 3. Building blocks, combined by the presets below:
#    sole      value of a reward = r * P(no free team accepts it)
#    band      accept only between our cutoff and the lowest cutoff of any free team
#    window    only accept while every other team is locked
#    crowd     cutoff from the rewards we would really have kept (r if nobody else
#              took it, else 0), so waiting is not over-valued in a crowded game
#    endscale  assume opponents' cutoffs fall like the optimal one near the end,
#              when everybody's bar drops and collisions pile up
#    window_n  opponent model uses only each team's last N free turns
#    jitter    random per-game scale on the cutoff, to fall out of step with clones
#    spoil     late in the game, collide on purpose with the leader if behind
#    hybrid    every team locked: take anything above the crowd cutoff (use the
#              gap).  Otherwise take r only if r * P(nobody else takes it) beats the
#              crowd cutoff: the band is learned, not fixed, so it lands wherever the
#              field leaves room (in the 18:26 run that was 20+, not 10-20).  A hard
#              band edge breaks when a free team takes almost anything (edge ~0).
#              Cutoff jittered +-8% per game, endgame and recent-data model on.
#    hybrid_bands  same gaps and jitter, but with several fixed price bands spread
#              over GeneralRelativity's range (wins 12.5-83.5, median 24.9) instead
#              of one crowded 10-20 band.  Each turn a band is open if its midpoint
#              times P(nobody else takes it) beats our cutoff, so we work off
#              whichever bands the free teams are leaving alone.  Band edges are
#              jittered per game and fall with the cutoff near the end.
#    best      the learned rule (sole + gaps + crowd cutoff + endgame + recent data
#              + jitter) plus calibration: teams running the same rule accept
#              together, so multiplying their chances one by one overstates the
#              risk.  For every reward bin we compare the predicted chance nobody
#              else takes it with how often nobody actually did this game, and
#              scale the prediction by that ratio.  No fixed band floor: in the
#              18:26 run rewards under 10 were rarely contested, so cheap rewards
#              are worth taking when the cutoff falls that low.
#              Tuned after the 18:42 run, where the field had moved again (Test won
#              on rewards under 10, the 0-10 win rate rose from 10% to 24%, and an
#              earlier version of this rule won only 10 times, all 13.7+): the start
#              assumes the live field rather than a Round 1 game (see PRIOR_KEPT),
#              and every estimate forgets old turns faster, since teams change
#              bands between runs and sometimes within one.
#
# sim_round3.py plays every preset against the Round 1 field and against itself.

MEAN = 10.0               # announced mean of the Round 1 distribution
DEFAULT_LOCKOUT = 15
H_MAX = 800               # cutoffs are flat long before this many turns left
BIN_EDGES = (0.0, 4.0, 8.0, 11.0, 13.0, 15.0, 17.0, 18.5, 20.0, 21.5, 23.0, 25.0, 28.0, 32.0, 40.0)
NB = len(BIN_EDGES)
# Chance nobody else takes a reward, by reward size, guessed from the 18:42 Round 3 run
# (share of offered rewards someone won: 0-10 24%, 10-15 50%, 15-20 52%, 20-25 33%,
# 25-30 33%, 30-40 49%, 40+ 33%; unwon high rewards are mostly collisions).  Used twice
# before there is data: spread over about 12 free teams as each team's chance of
# accepting, p = 1 - P(nobody) ** (1 / 12), and to thin the Exponential samples the
# starting cutoff is built from (PRIOR_KEPT).  Data from the game replaces both fast.
_PRIOR_NOBODY = ((40.0, 0.05), (30.0, 0.12), (25.0, 0.15), (20.0, 0.15), (15.0, 0.25), (10.0, 0.30), (0.0, 0.65))


def _prior_nobody(r):
	return next(q for lo, q in _PRIOR_NOBODY if r >= lo)


PRIOR_P = tuple(1.0 - _prior_nobody(e) ** (1.0 / 12.0) for e in BIN_EDGES)
CAL_K = 5.0               # calibration: pseudo-counts pulling the ratio towards 1
CAL_DECAY = 0.98          # calibration: per-observation decay within a bin (memory ~50)
POOL_W = 5.0              # prior weight of the Round 1 guess in the pooled rate
OPP_W = 4.0               # prior weight of the pooled rate in each team's rate
END_ZONE = 100            # endscale: turns left where opponents' cutoffs are scaled down
PRIOR_N = 200             # crowd: prior samples before real data
PRIOR_FADE = 150          # crowd: real samples after which the prior is gone
# crowd prior: Exponential(mean 10) quantiles, each kept with the chance nobody else
# takes it in the live field, else 0 (deterministic thinning with a golden-ratio
# sequence).  Starts the cutoff near where a crowded game puts it, not at about 20.
PRIOR_KEPT = tuple(
	x if ((i * 0.6180339887) % 1.0) < _prior_nobody(x) else 0.0
	for i, x in enumerate(-MEAN * math.log(1.0 - (k + 0.5) / PRIOR_N) for k in range(PRIOR_N))
)
DP_EVERY = 10             # crowd: rebuild the cutoff table every N turns
SPOIL_ZONE = 300
SPOIL_P = 0.8
BAND_P = 0.5              # band: a team's cutoff = lowest bin it accepts with this probability
# hybrid_bands: price bands (lo, hi), priced like GeneralRelativity's accepts in the 18:26 run
BANDS = ((12.5, 17.0), (17.0, 22.0), (22.0, 28.0), (28.0, 40.0), (40.0, float("inf")))

BASE = dict(mode="sole", cut="blind", endscale=False, window_n=0, jitter=0.0, spoil=False, calibrate=False,
			live_prior=False)
PRESETS = {
	"blind": dict(mode="blind"),                 # Round 1 rule, ignores other teams
	"A_sole": dict(),
	"B_band": dict(mode="band"),
	"C_window": dict(mode="window", cut="crowd"),
	"D_crowd": dict(cut="crowd"),
	"E_endgame": dict(endscale=True),
	"F_recent": dict(window_n=200),
	"G_jitter": dict(jitter=0.10),
	"H_spoiler": dict(spoil=True),
	"combo": dict(cut="crowd", endscale=True, window_n=300, jitter=0.03),
	"hybrid": dict(mode="hybrid", cut="crowd", endscale=True, window_n=300, jitter=0.08),
	"hybrid_bands": dict(mode="bands", cut="crowd", endscale=True, window_n=300, jitter=0.08),
	"best": dict(mode="sole", cut="crowd", endscale=True, window_n=120, jitter=0.05, calibrate=True,
				 live_prior=True),
}
ACTIVE = "best"

_UNIT = {}


def _unit_cut(gap):
	"""c[s] = optimal cutoff with s turns left for Exponential(mean 1), nobody else playing."""
	c = _UNIT.get(gap)
	if c is None:
		W = [0.0] * (H_MAX + 1)
		c = [0.0] * (H_MAX + 1)
		for s in range(1, H_MAX + 1):
			cs = W[s - 1] - (W[s - gap] if s >= gap else 0.0)
			c[s] = cs
			W[s] = W[s - 1] + math.exp(-cs)
		_UNIT[gap] = c
	return c


def _dp_cut(xs, gap, H):
	"""Cutoffs for rewards drawn from the sorted samples xs."""
	n = len(xs)
	suf = [0.0] * (n + 1)
	for i in range(n - 1, -1, -1):
		suf[i] = suf[i + 1] + xs[i]
	V = [0.0] * (H + 1)
	c = [0.0] * (H + 1)
	for s in range(1, H + 1):
		cs = V[s - 1] - (V[s - gap] if s >= gap else 0.0)
		i = bisect_right(xs, cs)
		V[s] = V[s - 1] + (suf[i] - cs * (n - i)) / n
		c[s] = cs
	return c


def _bin(r):
	return max(bisect_right(BIN_EDGES, r) - 1, 0)


class Opp:
	__slots__ = ("prev_score", "prev_flag", "trusted", "accepts", "probe", "obs", "acc", "n")

	def __init__(self, window_n):
		self.prev_score = None
		self.prev_flag = False
		self.trusted = False      # lock state known (seen unlocked at least once)
		self.accepts = deque(maxlen=4)
		self.probe = None         # call at which a score change was seen without the flag
		self.obs = deque(maxlen=window_n or None)
		self.acc = [0] * NB
		self.n = [0] * NB


class Solution:
	def __init__(self, preset=None, seed=None):
		self.cfg = dict(BASE)
		self.cfg.update(PRESETS[preset or ACTIVE])
		self.rng = random.Random(seed)
		self.T = None
		self.reset(None)

	def reset(self, T):
		self.T = T
		self.last_t = 0
		self.gap = DEFAULT_LOCKOUT + 1
		self.unit = _unit_cut(self.gap)
		self.rew = {}
		self.opps = []
		self.pool_acc = [0] * NB
		self.pool_n = [0] * NB
		self.lag_votes = [0, 0]
		self.last_final = 0
		self.eff = []             # sorted rewards we would have kept (0 if someone else took it)
		self.cal_n0 = [0.0] * NB  # calibration: turns per bin nobody else took (decayed)
		self.cal_pred = [0.0] * NB  # calibration: predicted chance of that, summed (decayed)
		self.crowd_c = None
		self.crowd_at = -10**9
		j = self.cfg["jitter"]
		self.jit = 1.0 + j * (2.0 * self.rng.random() - 1.0)
		self.band_jit = 1.0 + j * (2.0 * self.rng.random() - 1.0)
		self.last_thr = None

	# ---------- opponent tracking ----------
	def lag(self):
		return 1 if self.lag_votes[1] > self.lag_votes[0] else 0

	def register(self, op, tau):
		if tau >= 1 and tau not in op.accepts:
			op.accepts.append(tau)

	def was_free(self, op, tau):
		prior = [a for a in op.accepts if a < tau]
		return not prior or tau - max(prior) >= self.gap

	def free_now(self, op, t):
		if op.accepts:
			return t - max(op.accepts) >= self.gap
		return not op.prev_flag

	def add_obs(self, op, b, a):
		if op.obs.maxlen is not None and len(op.obs) == op.obs.maxlen:
			ob, oa = op.obs.popleft()
			op.n[ob] -= 1
			self.pool_n[ob] -= 1
			if oa:
				op.acc[ob] -= 1
				self.pool_acc[ob] -= 1
		op.obs.append((b, a))
		op.n[b] += 1
		self.pool_n[b] += 1
		if a:
			op.acc[b] += 1
			self.pool_acc[b] += 1

	def obs_bin(self, r, s):
		if self.cfg["endscale"] and s <= END_ZONE:
			cs = self.unit[s]
			r = r * self.unit[H_MAX] / cs if cs > 1e-9 else 1e9
		return _bin(r)

	def finalise(self, tau):
		r = self.rew[tau]
		b = self.obs_bin(r, self.T - tau + 1)
		free = [op.trusted and self.was_free(op, tau) for op in self.opps]
		taken = any(tau in op.accepts for op in self.opps)
		if self.cfg["calibrate"] and any(free):
			pred = 1.0
			for op, f in zip(self.opps, free):
				if f:
					pred *= 1.0 - self.p_acc(op, b)   # prediction made before seeing this turn
			self.cal_pred[b] = CAL_DECAY * self.cal_pred[b] + pred
			self.cal_n0[b] = CAL_DECAY * self.cal_n0[b] + (0.0 if taken else 1.0)
		for op, f in zip(self.opps, free):
			if f:
				self.add_obs(op, b, tau in op.accepts)
		insort(self.eff, 0.0 if taken else r)

	def observe(self, t, others):
		if not others:
			return
		if len(self.opps) != len(others):
			self.opps = [Opp(self.cfg["window_n"]) for _ in others]
			self.pool_acc = [0] * NB
			self.pool_n = [0] * NB
		lag = self.lag()
		for op, (score, flag) in zip(self.opps, others):
			flag = bool(flag)
			if op.prev_score is None:
				op.prev_score, op.prev_flag, op.trusted = score, flag, not flag
				continue
			rising = flag and not op.prev_flag
			if abs(score - op.prev_score) > 1e-9:
				self.register(op, t - 1)
				if rising:
					self.lag_votes[0] += 1
					op.probe = None
				else:
					op.probe = t if not flag else None
			elif rising:
				if op.probe == t - 1:
					self.lag_votes[1] += 1        # flag one report after the score change
				else:
					self.register(op, t - 1 - lag)  # collision: lock without a score change
				op.probe = None
			elif op.probe is not None and op.probe < t:
				op.probe = None
			if not flag:
				op.trusted = True
			op.prev_score, op.prev_flag = score, flag
		last = t - 1 - self.lag()
		for tau in range(max(self.last_final + 1, last - 2), last + 1):
			if tau in self.rew:
				self.finalise(tau)
		self.last_final = max(self.last_final, last)

	# ---------- opponent prediction ----------
	def p_acc(self, op, b):
		pool = (self.pool_acc[b] + POOL_W * PRIOR_P[b]) / (self.pool_n[b] + POOL_W)
		return (op.acc[b] + OPP_W * pool) / (op.n[b] + OPP_W)

	def p_sole(self, t, r, s):
		b = self.obs_bin(r, s)
		p = 1.0
		n = 0
		for op in self.opps:
			if self.free_now(op, t):
				p *= 1.0 - self.p_acc(op, b)
				n += 1
		if self.cfg["calibrate"] and n:
			# correct for teams that accept together (same rule), checked against this game
			p *= (self.cal_n0[b] + CAL_K) / (self.cal_pred[b] + CAL_K)
		return min(p, 1.0)

	def n_free(self, t):
		return sum(1 for op in self.opps if self.free_now(op, t))

	def band_cap(self, t, s):
		cap = float("inf")
		for op in self.opps:
			if not self.free_now(op, t):
				continue
			for k in range(NB):
				if self.p_acc(op, k) >= BAND_P:
					cap = min(cap, BIN_EDGES[k])
					break
		if self.cfg["endscale"] and s <= END_ZONE and cap < float("inf"):
			cap *= self.unit[s] / self.unit[H_MAX]   # bins are in mid-game units near the end
		return cap

	def band_open(self, t, s, r, base):
		"""hybrid_bands: is the fixed price band holding r open this turn?"""
		scale = self.band_jit
		if self.cfg["endscale"] and s <= END_ZONE:
			scale *= self.unit[s] / self.unit[H_MAX]   # bands fall with everybody's cutoff
		for lo, hi in BANDS:
			lo, hi = lo * scale, (hi * scale if hi != float("inf") else hi)
			if lo <= r < hi:
				mid = r if hi == float("inf") else 0.5 * (lo + hi)
				return mid * self.p_sole(t, mid, s) >= base
		return False

	# ---------- our cutoff ----------
	def cutoff(self, t, s):
		if self.cfg["cut"] != "crowd":
			return MEAN * self.unit[min(s, H_MAX)]
		if self.crowd_c is None or t - self.crowd_at >= DP_EVERY:
			m = int(PRIOR_N * max(0.0, 1.0 - len(self.eff) / PRIOR_FADE))
			if self.cfg["live_prior"]:
				prior = [PRIOR_KEPT[k * PRIOR_N // m] for k in range(m)]   # m picks spread over all sizes
			else:
				prior = [-MEAN * math.log(1.0 - (i + 0.5) / m) for i in range(m)]
			xs = sorted(self.eff + prior)
			if not xs:
				return MEAN * self.unit[min(s, H_MAX)]
			self.crowd_c = _dp_cut(xs, self.gap, min(max(self.T, 1), H_MAX))
			self.crowd_at = t
		return self.crowd_c[min(s, len(self.crowd_c) - 1)]

	# ---------- entry point ----------
	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t != self.last_t + 1 or T != self.T:
				self.reset(T)
			self.last_t = t
			gap = (int(lockout) if lockout and lockout > 0 else DEFAULT_LOCKOUT) + 1
			if gap != self.gap:
				self.gap = gap
				self.unit = _unit_cut(gap)
				self.crowd_c = None
			others = list(other_scores or [])
			if t > 1:
				self.observe(t, others)
			self.rew[t] = reward
			self.rew.pop(t - 8, None)

			s = min(max(T - t + 1, 1), H_MAX * 1000)
			base = self.jit * self.cutoff(t, s)
			self.last_thr = base
			if is_locked or reward <= 0:
				return False
			mode = self.cfg["mode"]
			if mode == "blind":
				accept = reward >= base
			elif mode == "band":
				accept = base <= reward < self.band_cap(t, s)
			elif mode == "window":
				accept = reward >= base and self.n_free(t) == 0
			elif mode == "bands":
				if self.n_free(t) == 0:
					accept = reward >= base                 # gap: everybody else is locked
				else:
					accept = reward >= base and self.band_open(t, s, reward, base)
			elif mode == "hybrid":
				if self.n_free(t) == 0:
					accept = reward >= base                 # gap: everybody else is locked
				else:
					# learned band: whatever range the free teams leave alone
					accept = reward * self.p_sole(t, reward, s) >= base
			else:
				accept = reward * self.p_sole(t, reward, s) >= base

			if not accept and self.cfg["spoil"] and s <= SPOIL_ZONE and self.opps:
				i = max(range(len(others)), key=lambda k: others[k][0])
				leader = self.opps[i]
				if (others[i][0] > your_score and self.free_now(leader, t)
						and self.p_acc(leader, self.obs_bin(reward, s)) >= SPOIL_P):
					accept = True
			return bool(accept)
		except Exception:
			# never crash on the server: fall back to the last threshold we computed
			return (not is_locked) and self.last_thr is not None and reward > 0 and reward >= self.last_thr


play = Solution().play
