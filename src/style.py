from dataclasses import dataclass

import matplotlib.pyplot as plt

# Each entity keeps the same color in every notebook. The gold of SOLAR (index 1)
# has low contrast on white, so it goes to GRU, drawn last and with thick lines.
ENTITY_COLORS: dict[str, int] = {"actual": 0, "GRU": 1, "simple RNN": 2, "LSTM": 3, "dense": 4}


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]  # one color per entity (actual data, then each model), in fixed order
    muted: str               # raw values and baselines, drawn behind the main lines
    ink: str                 # axes, ticks and text
    grid: str
    highlight: str           # shaded spans, such as the zoom area or the test period
    
    def color(self, entity: str) -> str:
        return self.series[ENTITY_COLORS[entity]]


# Sun-themed palette. Orange, gold and red alone are too close to tell apart, so purple
# and sky blue complete it. Checked with simulated protanopia and deuteranopia.
SOLAR = Palette(
    series=("#e8590c", "#e0a100", "#b0233a", "#6a3d9a", "#3f8fd2"),
    muted="#b5a99b",
    ink="#4a3f35",
    grid="#efe8dc",
    highlight="#fdf1cf",
)

# Okabe-Ito, the standard palette designed for colorblind readers.
OKABE_ITO = Palette(
    series=("#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9"),
    muted="#a8a8a8",
    ink="#4d4d4d",
    grid="#e8e8e8",
    highlight="#e6eef6",
)


def apply_style(colorblind_safe: bool = False) -> Palette:
    palette = OKABE_ITO if colorblind_safe else SOLAR
    # Light grid and soft ink so the data stands out more than the frame.
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": palette.grid,
        "grid.linewidth": 0.8,
        "axes.edgecolor": palette.ink,
        "axes.labelcolor": palette.ink,
        "axes.titlecolor": palette.ink,
        "xtick.color": palette.ink,
        "ytick.color": palette.ink,
        "legend.labelcolor": palette.ink,
    })
    return palette