import matplotlib.pyplot as plt
import textwrap
import numpy as np
from pygame import Color
plt.rcParams.update({'font.size': 13})


# Now each task maps to a list of (start, end) tuples
tasks = {
    "TPP Irrestrito": [(0, 2.5)],
    "TPP Restrito": [(2.5, 5)],
    "Visualização": [(0, 5)],
    "Pseudocódigo": [(1, 2.5), (3.5, 5)],
    "Complexidade": [(1, 2.5), (3.5, 5)],
    "Relatório": [(1.5, 2.5), (4, 7)],
    "TPP Não Convexo": [(7, 9)],
    "Solvers Comercias": [(9, 10.5)],
    "Análise": [(10.5, 11.5)],
    "Relatório Final": [(10.5, 12)]
}

# Ensure that zero-start tasks are slightly offset to avoid matplotlib rendering quirks
for k, v in tasks.items():
    tasks[k] = [(s + 0.01 * (s == 0), e) for s, e in v]

# Reverse order (so last defined task appears on top)
tasks = dict(reversed(list(tasks.items())))

# Color utilities
sk = lambda x: Color("#87CEEB").lerp("white", x)
dk = lambda x: Color("#87CEEB").lerp("black", x)
lc = lambda x: Color("lightcoral").lerp("white", x)
dc = lambda x: Color("lightcoral").lerp("black", x)

# Custom color palette
colors = [dk(0.2), dk(0.2), sk(0.0), sk(0.15), sk(0.15), sk(0.3)] + [dc(0.05), lc(0), lc(0.2), lc(0.3)]
colors.reverse()
colors = [i if isinstance(i, str) else "#" + "".join(hex(x)[2:].zfill(2) for x in i) for i in colors]

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor("lightgrey")

wrapped_tasks = [textwrap.fill(name, 12) for name in tasks.keys()]
bar_height = 0.8
spacing_factor = 1
y_positions = np.arange(len(tasks)) * spacing_factor

# Draw bars — now multiple per task if needed
for i, (task, intervals) in enumerate(tasks.items()):
    color = colors[i % len(colors)]
    for start, end in intervals:
        ax.barh(
            y_positions[i],
            end - start,
            left=start,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            height=bar_height
        )

ax.set_yticks(y_positions)
ax.set_yticklabels(wrapped_tasks)

ax.set_xlim(0, 12)
ax.set_xlabel("Meses")
ax.set_xticks(range(0, 13))
ax.set_title("Cronograma do Projeto de Pesquisa")
ax.grid(axis='x', alpha=0.2, color="black", linewidth=2)

plt.tight_layout()
plt.savefig("project_timeline.png")
plt.show()