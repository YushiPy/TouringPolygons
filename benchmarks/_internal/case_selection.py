"""Which cases of a campaign to run: ``65,66,130-131`` (numbered from 1, as ``tpp.py live`` shows them)."""

from __future__ import annotations

import re

CHARACTERS = "0123456789,- "
_ITEM = re.compile(r"^(\d+)(?:-(\d+))?$")


def parse_case_selection(text: str, total: int | None = None) -> list[int]:
	"""The 0-based indices a selection names, sorted and without repeats.

	Items are numbers or inclusive ranges (``130``, ``64-66``) separated by commas or spaces.
	Numbers start at 1, the numbering of ``tpp.py live`` and ``tpp.py stop``. With ``total``,
	a number past the campaign's last case is rejected.
	"""
	items = [item for item in re.split(r"[,\s]+", text.strip()) if item]
	if not items:
		raise ValueError("name at least one case, such as 65,66,130-131")
	selected: set[int] = set()
	for item in items:
		match = _ITEM.match(item)
		if not match:
			raise ValueError(f"{item!r} is not a case number or a range like 64-66")
		first = int(match[1])
		last = int(match[2]) if match[2] else first
		if first < 1 or last < first:
			raise ValueError(f"{item!r}: cases are numbered from 1 and ranges go upwards")
		if total is not None and last > total:
			raise ValueError(f"case {last} does not exist: the campaign has {total}")
		selected.update(range(first - 1, last))
	return sorted(selected)


def describe_selection(indices: list[int], limit: int = 12) -> str:
	"""``65, 66, 130`` (1-based) for a selection, shortened when long."""
	numbers = [str(index + 1) for index in indices[:limit]]
	return ", ".join(numbers) + (f", ... ({len(indices)} in all)" if len(indices) > limit else "")
