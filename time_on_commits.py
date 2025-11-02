"""
Calculates the total time spent on commits in the current Git repository,
considering only the intervals between commits that are less than or equal to 
a specified threshold (3 hours in this case).
"""

from datetime import datetime, timedelta
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

history = get_git_history(".")
times = re.findall(r'Date:\s+(.+)', history)

ts = [datetime.strptime(t, '%a %b %d %H:%M:%S %Y %z') for t in times][::-1]

diffs = [b - a for a, b in zip(ts, ts[1:])]
diffs = [d for d in diffs if d <= THRESHOLD_TIME]

d = sum(diffs, timedelta())

print("Total time spent on commits (considering only intervals <= 3 hours):")
print(f"- Hours: {round(d.total_seconds() / 3600, 2)}h")
print("- Datetime:", d)
