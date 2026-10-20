# Strategy overview (Round 3) 
# Won Round 3

THRESHOLD = 20.0


def play(reward, lockout, t, T, your_score, other_scores, is_locked):
	return bool(not is_locked and reward > THRESHOLD)
