"""Colour ramps shared by the slide-1 instances: letters/regions by visit order, route by progress."""
import math

# Deck palette: one teal family for the letters, one orange family for the route.
LETTER_RAMP = ["3C9F90", "1F7A73", "142D38"]          # first -> last visited region
ROUTE_RAMP = ["F59A4A", "E27A33", "C2501A"]           # start -> end of the route


def ramp(stops, t):
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2); f = t - i
    a, b = (tuple(int(h[k:k + 2], 16) for k in (0, 2, 4)) for h in stops[i:i + 2])
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def rgb(c):
    return "{rgb,255:red,%d;green,%d;blue,%d}" % c


