import math
import random
from bisect import bisect_right, insort
from statistics import NormalDist

# Strategy overview
# -----------------
# Rewards are modelled as Normal(mu, sigma) where mu and sigma are themselves
# random: drawn once per game from an unknown hyper-distribution.
#
# 1. Belief: a Normal-Inverse-Gamma posterior over (mu, sigma^2), updated with
#    every reward we see (one per turn, even while locked).
# 2. Hyper-distribution: when the same process plays several games, the (mean,
#    variance) of each finished game is kept and the prior for the next game is
#    fitted to them (empirical Bayes).  Without history the prior is very weak.
# 3. Decision: with s turns left the optimal threshold for a *known* (mu, sigma)
#    is the value lost by being locked, V(s-1) - V(s-1-L).  Since we keep
#    learning (mu, sigma) for free, the threshold used is that lock cost averaged
#    over posterior particles (not the threshold of the blurred predictive
#    distribution, which would treat parameter doubt as per-draw noise).
#    By scale invariance V for (mu, sigma) is sigma * W(mu / sigma), so only a
#    one-parameter family of tables W is needed; they are cached on a z grid.
# 4. Regime check: if the last REGIME_WINDOW rewards are wildly inconsistent with
#    the posterior, mu/sigma are assumed to have been re-drawn and the posterior
#    restarts from those rewards.
# 5. Shape check: if the rewards are clearly not normal (Jarque-Bera), fall back
#    to the distribution-free DP on the empirical rewards.
#
# Payouts are not shared on this server (every accepter gets the full reward),
# so other teams cannot change our score and are ignored.

K = 64                    # posterior particles
MIN_SAMPLES = 10          # rewards to see before accepting anything
Z_STEP = 0.05             # grid for mu / sigma in the cached value tables
Z_MAX = 6.0
H_MAX = 600               # tables are stationary well before this many turns left
TABLE_BUDGET = 6          # new value tables built per call at most (keeps every call fast)
WEAK_PRIOR = (0.0, 0.01, 1.0, 0.01)  # m0, k0, a0, b0 for Normal-Inverse-Gamma
REGIME_WINDOW = 60
REGIME_Z = 5.0            # restart if the recent mean is this many standard errors off
JB_LIMIT = 25.0           # Jarque-Bera above this (n >= 100) -> not normal
EMP_EVERY = 10            # recompute the empirical DP every N turns

_ND = NormalDist()
_RNG = random.Random(12345)
_U = [_ND.inv_cdf((k + 0.5) / K) for k in range(K)]   # stratified normals for sigma
_E = _U[:]
_RNG.shuffle(_E)                                       # independent stratified normals for mu
_TABLES = {}              # (gap, z index) -> W table, shared across games
_BUILT = {}               # gap -> sorted z indices already in _TABLES
_BUDGET = [TABLE_BUDGET]
_HISTORY = []             # (mean, variance) of finished games in this process


def _w_table(gap, zi):
	"""W[s] = optimal value with s turns left for rewards Normal(z, 1), z = zi * Z_STEP."""
	key = (gap, zi)
	W = _TABLES.get(key)
	if W is None and _BUDGET[0] <= 0 and _BUILT.get(gap):
		# out of budget this call: use the nearest table already built
		built = _BUILT[gap]
		i = bisect_right(built, zi)
		near = min(built[max(i - 1, 0):i + 1], key=lambda b: abs(b - zi))
		return _TABLES[(gap, near)]
	if W is None:
		_BUDGET[0] -= 1
		z = zi * Z_STEP
		W = [0.0] * (H_MAX + 1)
		for s in range(1, H_MAX + 1):
			c = W[s - 1] - (W[s - gap] if s >= gap else 0.0)
			d = z - c
			# E[max(Y, 0)] for Y ~ Normal(d, 1)
			W[s] = W[s - 1] + d * 0.5 * (1.0 + math.erf(d / math.sqrt(2.0))) + math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
		_TABLES[key] = W
		insort(_BUILT.setdefault(gap, []), zi)
	return W


def _lock_cost(gap, z, s):
	"""W(s-1) - W(s-gap) for Normal(z, 1), linearly interpolated on the z grid."""
	z = min(max(z, -Z_MAX), Z_MAX)
	f = z / Z_STEP
	lo = math.floor(f)
	w = f - lo
	s = min(s, H_MAX)
	out = 0.0
	for zi, wt in ((lo, 1.0 - w), (lo + 1, w)):
		if wt <= 0.0:
			continue
		W = _w_table(gap, zi)
		out += wt * (W[s - 1] - (W[s - gap] if s >= gap else 0.0))
	return out


def _gamma_quantile(a, u):
	"""Wilson-Hilferty approximation to the Gamma(a, 1) quantile at normal score u."""
	x = a * (1.0 - 1.0 / (9.0 * a) + u / (3.0 * math.sqrt(a))) ** 3
	return max(x, 1e-12)


def _prior_from_history():
	if len(_HISTORY) < 2:
		return WEAK_PRIOR
	means = [m for m, _ in _HISTORY]
	vars_ = [v for _, v in _HISTORY]
	n = len(_HISTORY)
	m0 = sum(means) / n
	vbar = sum(vars_) / n
	var_mu = max(sum((m - m0) ** 2 for m in means) / (n - 1), 1e-9)
	k0 = min(max(vbar / var_mu, 0.01), 100.0)   # prior strength of mu, in pseudo-observations
	var_v = sum((v - vbar) ** 2 for v in vars_) / (n - 1)
	a0 = min(max(vbar * vbar / var_v + 2.0, 1.0), 50.0) if var_v > 0 else 50.0
	b0 = vbar * (a0 - 1.0)
	# shrink towards the weak prior while there are only a few games
	if n < 5:
		k0 *= n / 5.0
		a0 = 1.0 + (a0 - 1.0) * n / 5.0
		b0 = vbar * max(a0 - 1.0, 0.01)
	return m0, k0, a0, b0


class Solution:
	def __init__(self):
		self.T = None
		self.reset(None)

	def reset(self, T):
		if self.T is not None and getattr(self, "n", 0) >= 100:
			_HISTORY.append((self.mean, self.m2 / (self.n - 1)))
		self.T = T
		self.last_t = 0
		self.L = 15
		self.prior = _prior_from_history()
		self.n = 0                # rewards in the current regime
		self.mean = 0.0
		self.m2 = 0.0
		self.m3 = 0.0
		self.m4 = 0.0
		self.recent = []          # every reward of the current regime, in order
		self.sorted = []          # same, sorted (empirical fallback)
		self.normal_ok = True
		self.emp_c = None
		self.emp_at = -10**9
		self.last_thr = None

	# ---------- belief ----------
	def add(self, x):
		# running moments (Welford / Terriberry)
		n1 = self.n
		self.n += 1
		n = self.n
		d = x - self.mean
		dn = d / n
		dn2 = dn * dn
		t1 = d * dn * n1
		self.mean += dn
		self.m4 += t1 * dn2 * (n * n - 3 * n + 3) + 6 * dn2 * self.m2 - 4 * dn * self.m3
		self.m3 += t1 * dn * (n - 2) - 3 * dn * self.m2
		self.m2 += t1
		self.recent.append(x)
		insort(self.sorted, x)

	def restart_from(self, xs):
		self.n = 0
		self.mean = self.m2 = self.m3 = self.m4 = 0.0
		self.recent, self.sorted = [], []
		self.emp_c = None
		for x in xs:
			self.add(x)

	def posterior(self):
		m0, k0, a0, b0 = self.prior
		n = self.n
		kn = k0 + n
		mn = (k0 * m0 + n * self.mean) / kn
		an = a0 + 0.5 * n
		bn = b0 + 0.5 * self.m2 + k0 * n * (self.mean - m0) ** 2 / (2.0 * kn)
		return mn, kn, an, bn

	def check_regime(self):
		"""Restart the posterior if the recent rewards clearly come from new parameters."""
		w = REGIME_WINDOW
		if self.n < 3 * w:
			return
		old = self.recent[:-w]
		new = self.recent[-w:]
		mo = sum(old) / len(old)
		vo = sum((x - mo) ** 2 for x in old) / (len(old) - 1)
		mn = sum(new) / w
		vn = sum((x - mn) ** 2 for x in new) / (w - 1)
		if vo <= 0:
			return
		z_mean = abs(mn - mo) / math.sqrt(vo / w + vo / len(old))
		# log variance ratio has sd about sqrt(2 / (w - 1))
		z_var = abs(math.log(max(vn, 1e-12) / vo)) / math.sqrt(2.0 / (w - 1))
		if z_mean > REGIME_Z or z_var > REGIME_Z:
			self.restart_from(new)

	def check_shape(self):
		if self.n < 100 or self.m2 <= 0:
			self.normal_ok = True
			return
		n = self.n
		skew = math.sqrt(n) * self.m3 / self.m2 ** 1.5
		kurt = n * self.m4 / (self.m2 * self.m2) - 3.0
		jb = n / 6.0 * (skew * skew + 0.25 * kurt * kurt)
		self.normal_ok = jb < JB_LIMIT

	# ---------- thresholds ----------
	def bayes_threshold(self, s):
		mn, kn, an, bn = self.posterior()
		gap = self.L + 1
		total = 0.0
		for u, e in zip(_U, _E):
			var = bn / _gamma_quantile(an, u)    # sigma^2 ~ Inverse-Gamma(an, bn)
			sd = math.sqrt(var)
			mu = mn + e * sd / math.sqrt(kn)     # mu | sigma^2 ~ Normal(mn, sigma^2 / kn)
			total += sd * _lock_cost(gap, mu / sd, s)
		return total / K

	def empirical_threshold(self, t, s):
		if self.emp_c is None or t - self.emp_at >= EMP_EVERY:
			xs = self.sorted
			n = len(xs)
			suf = [0.0] * (n + 1)
			for i in range(n - 1, -1, -1):
				suf[i] = suf[i + 1] + xs[i]
			L = self.L
			H = min(self.T, H_MAX)
			V = [0.0] * (H + 1)
			c = [0.0] * (H + 1)
			for k in range(1, H + 1):
				after = V[k - 1 - L] if k - 1 - L >= 0 else 0.0
				ck = V[k - 1] - after
				i = bisect_right(xs, ck)
				V[k] = V[k - 1] + (suf[i] - ck * (n - i)) / n
				c[k] = ck
			self.emp_c = c
			self.emp_at = t
		return self.emp_c[min(s, len(self.emp_c) - 1)]

	# ---------- entry point ----------
	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t <= self.last_t or T != self.T or t == 1:
				self.reset(T)
			self.last_t = t
			if lockout and lockout > 0:
				self.L = int(lockout)
			self.add(reward)
			if t % 10 == 0:
				self.check_regime()
				self.check_shape()

			if self.n < 3:
				return False
			# computed every turn, even when locked, so the table cache warms up early
			_BUDGET[0] = TABLE_BUDGET
			s = T - t + 1
			thr = self.bayes_threshold(s) if self.normal_ok else self.empirical_threshold(t, s)
			self.last_thr = thr
			return bool(not is_locked and self.n >= MIN_SAMPLES and reward > 0 and reward >= thr)
		except Exception:
			# never crash on the server: fall back to the last threshold we computed
			return (not is_locked) and self.last_thr is not None and reward > 0 and reward >= self.last_thr


play = Solution().play
