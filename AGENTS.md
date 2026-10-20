# AGENTS.md - Trading Game Client

## Project Overview
This repository contains the participant client library, local testing harness, and submission script for the G-Research Trading Game.

## Key Conventions & Constraints

### 1. Function Parameter Signature (Strict)
All client strategies MUST implement the exact 7-parameter signature:
```python
def play(reward: float, lockout: int, t: int, T: int, your_score: float, other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
```

- `reward` (`float`): Value drawn from the distribution this round.
- `lockout` (`int`): Number of rounds the player is locked out if they accept (default 15).
- `t` (`int`): Current round index ($1 \le t \le T$).
- `T` (`int`): Total rounds in simulation ($2000$).
- `your_score` (`float`): Player cumulative score up to round $t-1$.
- `other_scores` (`list[tuple[float, bool]]`): List of `(score, is_locked)` tuples for other teams based on $t-1$. Empty list `[]` on round $t=1$.
- `is_locked` (`bool`): True if the player is currently in lockout at round $t$.

### 2. Testing & Submissions
- Run local unit tests: `uv run python test.py`.
- Submit to server: `uv run python submit.py`.
- Configuration (team name, password, server endpoint) is stored in `config.py`. Keep the checked-in team name and password empty; users fill them in for their submissions.
- Current EC2 submission endpoint: `ec2-54-217-141-255.eu-west-1.compute.amazonaws.com:5000`.

### 3. Guidelines
- Do NOT use emojis in code, commits, or documentation.
