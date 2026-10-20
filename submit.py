from gr_tradinggame.coding.client import Client
from config import TEAM_NAME, PASSWORD, SERVER

if not TEAM_NAME or not PASSWORD or not SERVER:
    raise SystemExit("ERROR: Edit config.py with your team name, password, and server")


# Password is set on first submission and required for all future submissions.
# Choose any password you like on your first submit. Don't forget it.
client = Client(TEAM_NAME, password=PASSWORD, server=SERVER)
exec(open('solution.py').read())
client.test(play)
client.submit(open('solution.py').read())
