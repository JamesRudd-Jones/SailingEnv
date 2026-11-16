"""
Shared by the testing scripts: PASS/FAIL reporting that shows the numbers, and a consistent look for the plots.

Each script in this folder covers one section of the model and can be run alone, for example
    pdm run python -m project_name.sailing.testing.foil_testing
or all together with
    pdm run python -m project_name.sailing.testing.run_all
Plots are saved to testing/plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PLOTS = Path(__file__).parent / "plots"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed order, so a colour is never reused for another meaning
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({"figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID,
                     "axes.labelcolor": TEXT_SECONDARY,
                     "axes.titlecolor": TEXT,
                     "axes.titlesize": 12,
                     "axes.titlelocation": "left",
                     "axes.spines.top": False,
                     "axes.spines.right": False,
                     "axes.grid": True,
                     "grid.color": GRID,
                     "grid.linewidth": 0.8,
                     "xtick.color": TEXT_SECONDARY,
                     "ytick.color": TEXT_SECONDARY,
                     "text.color": TEXT,
                     "legend.frameon": False,
                     "lines.linewidth": 2.0,
                     "font.size": 10,
                     })


class Checks:
    """Collects results for one section and prints each one as it happens, along with the measured numbers."""

    def __init__(self, section: str):
        self.section = section
        self.results = []
        print(f"\n=== {section} ===")

    def check(self, description: str, passed, detail: str = "") -> None:
        passed = bool(passed)
        self.results.append((description, passed))
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}" + (f"  ({detail})" if detail else ""))

    @property
    def failed(self) -> list:
        return [description for description, passed in self.results if not passed]

    def finish(self, figure=None, name: str = "") -> "Checks":
        if figure is not None:
            PLOTS.mkdir(exist_ok=True)
            path = PLOTS / f"{name}.png"
            figure.tight_layout()
            figure.savefig(path, dpi=130)
            plt.close(figure)
            print(f"  Plot: {path}")
        print(f"  {len(self.results) - len(self.failed)}/{len(self.results)} passed")
        return self
