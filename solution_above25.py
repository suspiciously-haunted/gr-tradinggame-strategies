# Strategy overview (Round 3)
# ---------------------------
# Accept any reward above 25 whenever we are not locked out; skip everything else.
# Ignores the other teams entirely.

THRESHOLD = 20.0


def play(reward, lockout, t, T, your_score, other_scores, is_locked):
	return bool(not is_locked and reward > THRESHOLD)
