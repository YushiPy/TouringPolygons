"""
Calculates the total time spent on commits in the current Git repository,
considering only the intervals between commits that are less than or equal to 
a specified threshold (3 hours in this case).
"""

from datetime import datetime, timedelta
from itertools import accumulate
import re

import subprocess
from pathlib import Path

THRESHOLD_TIME = timedelta(hours=3)

def get_git_history(repo_path: str | Path) -> str:

	repo_path = Path(repo_path)

	if not (repo_path / ".git").exists():
		raise ValueError(f"{repo_path} is not a Git repository (no .git directory found)")
	
	result = subprocess.run(
		["git", "-C", str(repo_path), "log", "--pretty=fuller"],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		check=True
	)
	
	return result.stdout

history = get_git_history("..")
times = re.findall(r'Date:\s+(.+)', history)

ts = [datetime.strptime(t, '%a %b %d %H:%M:%S %Y %z') for t in times][::-1]

START_DATE = datetime(2025, 7, 1, tzinfo=ts[0].tzinfo)
END_DATE = datetime(2026, 11, 20, tzinfo=ts[0].tzinfo)

ts = [t for t in ts if START_DATE <= t <= END_DATE]

diffs1 = [b - a for a, b in zip(ts, ts[1:])]
diffs2 = [d for d in diffs1 if d <= THRESHOLD_TIME]

d = sum(diffs2, timedelta())

print("Total time spent on commits (considering only intervals <= 3 hours):")
print(f"- Hours: {round(d.total_seconds() / 3600, 2)}h")
print("- Datetime:", d)

import matplotlib.pyplot as plt


x = [i.timestamp() for i in ts]
x = [int(i - x[0]) / (3600 * 24 * 30) for i in x]

y = [0]

for a, b in zip(ts, ts[1:]):
	diff = b - a
	if diff <= THRESHOLD_TIME:
		y.append(y[-1] + diff.total_seconds() / 3600)
	else:
		y.append(y[-1])

y.pop()

plt.grid()
plt.title("Tempo gasto em commits ao longo do tempo")
plt.xlabel("Tempo (meses)")
plt.ylabel("Horas gastas em commits")
plt.plot(x[1:], y)
plt.tight_layout()

plt.savefig("time_on_commits.png", dpi=250)