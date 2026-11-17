"""Dashboard chrome every dataset shares: the header, the progress bar and step labels.

Chrome is the frame around the data: it says what is shown and how far along the run is, and stays
visually quiet (see ``animkit.theme``).
"""

from matplotlib.figure import Figure

from animkit import theme


def format_step(step: float) -> str:
    """A training step's value as a label: whole numbers with thousands separators, others to two decimals.

    Args:
        step: The step's value (an epoch, a tree count, a row count, a parameter value).

    Returns:
        The label, with a typographic minus sign (−) for a negative value.
    """
    text = f"{int(step):,}" if float(step).is_integer() else f"{step:.2f}"
    return text.replace("-", "−")  # typographic minus, matching the "−log10" labels


def add_header(figure: Figure, title: str, subtitle: str):
    """Writes a dashboard's bold title and its subtitle in the top-left corner of the figure.

    Args:
        figure: The dashboard's figure.
        title: The headline.
        subtitle: One line under it.
    """
    figure.text(0.04, 0.955, title, fontsize=theme.FONT_TITLE, weight="bold", color=theme.INK_PRIMARY)
    figure.text(0.04, 0.922, subtitle, fontsize=theme.FONT_SUBTITLE, color=theme.INK_SECONDARY)


class ProgressBar:
    """A thin horizontal bar across the dashboard that fills from left to right as the run progresses.

    Attributes:
        ax: The bar's own axes, with its frame switched off.
    """

    def __init__(self, figure: Figure, rect=(0.04, 0.895, 0.92, 0.01)):
        """Adds the bar's axes to the figure, with an empty grey track.

        Args:
            figure: The dashboard's figure.
            rect: Where the bar goes, as ``(left, bottom, width, height)`` in figure fractions.
        """
        self.ax = figure.add_axes(rect)
        self.ax.barh([0], [1.0], color=theme.GRIDLINE, height=1.0)
        (self._bar,) = self.ax.barh([0], [0.0], color=theme.SERIES_1, height=1.0)
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_axis_off()

    def update(self, fraction: float) -> None:
        """Fills the bar to ``fraction`` of its length.

        Args:
            fraction: How far along the run is, 0 to 1; values outside are clamped.
        """
        self._bar.set_width(min(max(fraction, 0.0), 1.0))
