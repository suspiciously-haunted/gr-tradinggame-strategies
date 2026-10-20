from collections import deque

# Strategy overview (Round 3: collisions score zero)
# --------------------------------------------------
# Like solution_above25.py, only rewards above 25 are of interest, but most teams
# take those too: in the 18:42 run only a third of 25-30 and 40+ rewards were won by
# anyone, the rest mostly collided.  So a big reward is only taken when it is likely
# to be ours alone.
#
# 1. Watching.  other_scores gives each team's (score, is_locked) for the previous
#    turn.  A score rise is a clean accept; the lock flag switching on without a score
#    rise is an accept that collided.  Whether the flag shows on the same report as
#    the score rise or one report later is learned from clean accepts.  From the
#    accept turns we know exactly when every team is free again.
# 2. High takers.  For every turn a team was free and the reward was above FLOOR,
#    record whether it accepted.  p_i = its recent acceptance rate there (old turns
#    fade, half-life HALF_LIFE such turns), shrunk to the pooled rate of all teams.
# 3. Decision.  P(alone) = product over free teams of (1 - p_i).  Accept a reward
#    above FLOOR iff reward * P(alone) >= VALUE_BAR, so a 30 needs about a 50% chance
#    of being ours alone and a 60 about 25%.  When every high taker is locked the
#    chance is 1 and anything above FLOOR is taken (the gaps).
# 4. End of game.  Over the last END_TURNS turns the floor and bar fall linearly to 0,
#    since a reward left on the table at the end is worth nothing.

FLOOR = 25.0              # only rewards above this are considered
VALUE_BAR = 15.0          # reward * P(alone) must reach this
END_TURNS = 50            # floor and bar fall to 0 over this many final turns
DEFAULT_LOCKOUT = 15
HALF_LIFE = 20.0          # high-reward free turns after which an old one counts half
DECAY = 0.5 ** (1.0 / HALF_LIFE)
PRIOR_P = 0.6             # prior chance a free team takes a reward above FLOOR
PRIOR_W = 3.0             # pseudo-counts of the prior in the pooled rate
POOL_W = 3.0              # pseudo-counts of the pooled rate in each team's rate


class Opp:
	__slots__ = ("prev_score", "prev_flag", "trusted", "accepts", "probe", "acc", "n")

	def __init__(self):
		self.prev_score = None
		self.prev_flag = False
		self.trusted = False      # lock state known (seen unlocked at least once)
		self.accepts = deque(maxlen=4)
		self.probe = None         # report where a score rise came without the flag
		self.acc = 0.0            # decayed accepts on free high-reward turns
		self.n = 0.0              # decayed free high-reward turns


class Solution:
	def __init__(self):
		self.T = None
		self.reset(None)

	def reset(self, T):
		self.T = T
		self.last_t = 0
		self.gap = DEFAULT_LOCKOUT + 1
		self.rew = {}
		self.opps = []
		self.lag_votes = [0, 0]
		self.last_final = 0

	def lag(self):
		return 1 if self.lag_votes[1] > self.lag_votes[0] else 0

	def register(self, op, tau):
		if tau >= 1 and tau not in op.accepts:
			op.accepts.append(tau)

	def was_free(self, op, tau):
		before = [a for a in op.accepts if a < tau]
		return not before or tau - max(before) >= self.gap

	def free_now(self, op, t):
		if op.accepts:
			return t - max(op.accepts) >= self.gap
		return not op.prev_flag

	def observe(self, t, others):
		if not others:
			return
		if len(self.opps) != len(others):
			self.opps = [Opp() for _ in others]
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
					self.lag_votes[1] += 1        # flag one report after the score rise
				else:
					self.register(op, t - 1 - lag)  # collision: locked without a score rise
				op.probe = None
			elif op.probe is not None and op.probe < t:
				op.probe = None
			if not flag:
				op.trusted = True
			op.prev_score, op.prev_flag = score, flag
		# turns whose accepts are now known: update the high-taker rates
		last = t - 1 - self.lag()
		for tau in range(max(self.last_final + 1, last - 2), last + 1):
			r = self.rew.get(tau)
			if r is None or r <= FLOOR:
				continue
			for op in self.opps:
				if op.trusted and self.was_free(op, tau):
					op.acc = DECAY * op.acc + (1.0 if tau in op.accepts else 0.0)
					op.n = DECAY * op.n + 1.0
		self.last_final = max(self.last_final, last)

	def p_alone(self, t):
		pool_acc = sum(op.acc for op in self.opps)
		pool_n = sum(op.n for op in self.opps)
		pool = (pool_acc + PRIOR_W * PRIOR_P) / (pool_n + PRIOR_W)
		p = 1.0
		for op in self.opps:
			if self.free_now(op, t):
				p *= 1.0 - (op.acc + POOL_W * pool) / (op.n + POOL_W)
		return p

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t != self.last_t + 1 or T != self.T:
				self.reset(T)
			self.last_t = t
			self.gap = (int(lockout) if lockout and lockout > 0 else DEFAULT_LOCKOUT) + 1
			if t > 1:
				self.observe(t, list(other_scores or []))
			self.rew[t] = reward
			self.rew.pop(t - 8, None)
			if is_locked or reward <= 0:
				return False
			scale = min(1.0, (T - t) / END_TURNS) if END_TURNS > 0 else 1.0
			if reward <= FLOOR * scale:
				return False
			return bool(reward * self.p_alone(t) >= VALUE_BAR * scale)
		except Exception:
			# never crash on the server: fall back to the plain rule
			return (not is_locked) and reward > FLOOR


play = Solution().play
