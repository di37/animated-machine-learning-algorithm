"""Algorithm diagrams for the side column, drawn from a model's real parameters at that moment.

A diagram shows how a model turns inputs into an answer: the weights on its edges, the trees it
holds, the steps it takes. All helpers draw into unit axes coordinates of one Axes, so a lesson can
redraw them every frame. Text is kept inside the card: titles shrink to fit, results anchor to the
right edge.

One helper per shape of computation:

- ``weighted_sum``: inputs joined to a Σ by edges that follow their weights (linear models).
- ``tree_grid``: thumbnails of many trees feeding an average (forests).
- ``chain``: a starting score plus a correction from each tree (boosting).
- ``flow``: a left-to-right pipeline of steps (kNN's lookup).
- ``factor_rows``: per class, factors combined and fed to a normalising node (naive Bayes).
"""

from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from animkit.algorithms.common import draw_tree
from animkit.lessons import theme
from animkit.lessons.lesson import plural
from animkit.lessons.stage import reset_axes

TEXT_SCALE = 1.4  # the diagram card is about twice the height of the panels below it: its drawing is scaled up
TITLE_Y = 0.9
TITLE_X = 0.02
TITLE_SIZE = theme.PANEL_TITLE_SIZE  # the same as the panel and metric titles beneath it
INPUT_ROW = 0.62  # many inputs sit on this row, the Σ hub below them
HUB = (0.33, 0.24)
FEW = 3  # up to this many inputs are stacked on the left, with their weights written on the edges
ROW_START, ROW_END = 0.07, 0.61  # more inputs sit in a row, their weights carried by edge width alone
ZERO_SHARE = 1e-9  # a weight this small next to the scale is zero to rounding error
ROW_NAME_SIZE = 9.0 * TEXT_SCALE
FEW_X = 0.36  # their nodes; names (and values) are right-aligned in the column before them
FEW_HUB = (0.56, 0.4)
SMALL = 10.5 * TEXT_SCALE
RIGHT = 0.985  # right-hand text anchor, inside the card
GRID_COLUMNS, GRID_ROWS = 5, 2
GRID_TREES = GRID_COLUMNS * GRID_ROWS  # thumbnails a tree grid draws
CHAIN_TREES = 4  # newest trees a boosting chain draws
REGULAR_EM = 0.55  # average regular glyph width, in ems, for spacing boxes
BOX_PAD_PT = 7.0 * TEXT_SCALE  # a box's padding (both sides) plus a little air, in points


def _pt(size: float) -> float:
    """A size in points (font, marker, arrow head), scaled to the diagram card."""
    return size * TEXT_SCALE


def _area(area: float) -> float:
    """A scatter marker's area in points², scaled to the diagram card."""
    return area * TEXT_SCALE ** 2


def frame(ax: Axes, title: str) -> None:
    """Blank card plus a title that shrinks, never overflows, to fit the card's width.

    The title is measured, not guessed, with the figure's own renderer.

    Args:
        ax: The diagram's axes; it is cleared first.
        title: The diagram's title.
    """
    reset_axes(ax)
    label = ax.text(TITLE_X, TITLE_Y, title, fontsize=TITLE_SIZE, weight="bold", color=theme.NAVY, va="center")
    renderer = ax.figure.canvas.get_renderer()
    room = ax.get_window_extent(renderer).width * (1.0 - 2 * TITLE_X)
    width = label.get_window_extent(renderer).width
    if width > room:
        label.set_fontsize(TITLE_SIZE * room / width)


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], shrink: float = 12.0) -> None:
    """A grey arrow from ``start`` to ``end``, its ends pulled back by ``shrink`` points."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=_pt(12), color=theme.INK_MUTED,
                                 linewidth=1.4, shrinkA=shrink, shrinkB=shrink, zorder=2))


def _width_pt(ax: Axes) -> float:
    """The axes' width, in points."""
    return ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0


def _box(ax: Axes, x: float, y: float, text: str, *, active: bool = True, size: float = _pt(11)) -> None:
    """A rounded text box at (x, y): blue-edged when active, greyed out when not."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=theme.NAVY if active else theme.INK_MUTED,
            zorder=5, bbox=dict(boxstyle="round,pad=0.35", facecolor=theme.PANEL if active else theme.CANVAS,
                                edgecolor=theme.MODEL if active else theme.INACTIVE, linewidth=1.3))


def _edge_style(weight: float, scale: float, decades: float | None = None) -> tuple[str, float, object]:
    """Colour, width and dash for one edge.

    ``decades`` puts widths on a log scale spanning that many decades below ``scale``, for weights that
    range over orders of magnitude. A non-finite weight is a term not there (yet) and is drawn faint; a
    weight of zero, or zero to rounding error, is dashed.
    """
    if not np.isfinite(weight):
        return theme.INACTIVE, 0.8, "solid"  # a term the model does not have (yet), or not revealed yet
    share = min(abs(weight) / scale, 1.0)
    if share <= ZERO_SHARE:
        return theme.INK_MUTED, 1.0, (0, (3, 3))  # switched off: zero, or zero to rounding error
    if decades:
        share = max(0.0, 1.0 + float(np.log10(share)) / decades)
    return (theme.MODEL if weight > 0 else theme.ERROR), 0.8 + 4.2 * share, "solid"


def _positions(count: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Input node coordinates and the Σ hub: a left column for few inputs, a top row for many."""
    if count <= FEW:
        ys = np.linspace(0.66, 0.14, count) if count > 1 else np.array([FEW_HUB[1]])
        return np.full(count, FEW_X), ys, FEW_HUB
    return np.linspace(ROW_START, ROW_END, count), np.full(count, INPUT_ROW), HUB


def _check_lengths(count: int, **sequences: Sequence | None) -> None:
    """Refuses any given per-input sequence whose length is not ``count``."""
    for label, sequence in sequences.items():
        if sequence is not None and len(sequence) != count:
            raise ValueError(f"{label} has {len(sequence)} entries for {count} inputs")


def weighted_sum(ax: Axes, names: Sequence[str], weights: np.ndarray, bias_text: str, outputs: Sequence[str], *,
                 title: str, fmt: Callable[[float], str], scale: float | None = None,
                 decades: float | None = None, values: Sequence[str] | None = None,
                 pushes: np.ndarray | None = None, result: str | None = None, reveal: float = 1.0) -> None:
    """Inputs joined to Σ by edges whose width and colour follow their weights.

    During training the edges follow the weights on a fixed ``scale``, so growth shows; Σ adds the bias
    and feeds the output chain. At inference (``pushes`` given) the edges follow weight × input instead,
    on their own scale: they light up one by one as ``reveal`` grows, with ``values`` labelling the
    inputs, and ``result`` appears over the last output at the end.

    Up to ``FEW`` inputs are stacked in a column with their weights written on the edges; more sit in a
    row, their weights shown by edge width alone.

    Args:
        ax: The diagram's axes.
        names: The input names.
        weights: One weight per input; a non-finite one is a term the model does not have yet.
        bias_text: The bias, as written beside the Σ.
        outputs: The boxes after Σ, left to right.
        title: The diagram's title.
        fmt: Formats the number written on an edge.
        scale: The weight drawn at full width during training; ``None`` fits the current weights.
        decades: Puts edge widths on a log scale over that many decades below the scale.
        values: Each input's value at inference, written with its name.
        pushes: Each input's weight × value at inference; giving them switches to inference.
        result: The prediction, shown above the last output box once ``reveal`` reaches 1.
        reveal: Inference progress, 0 to 1: the share of inputs lit so far.

    Raises:
        ValueError: If ``weights``, ``values`` or ``pushes`` does not have one entry per input.
    """
    _check_lengths(len(names), weights=weights, values=values, pushes=pushes)
    frame(ax, title)
    weights = np.asarray(weights, dtype=float)
    inference = pushes is not None
    drive = np.where(np.isfinite(weights), np.asarray(pushes, dtype=float), np.nan) if inference else weights
    finite = np.abs(drive[np.isfinite(drive)])
    if inference or scale is None:
        scale = float(finite.max()) if finite.size else 1.0
    xs, ys, hub = _positions(len(names))
    few = len(names) <= FEW
    slot_pt = (ROW_END - ROW_START) / max(len(names) - 1, 1) * _width_pt(ax)  # room per name in a row
    shown = int(np.ceil(len(names) * reveal)) if inference else len(names)
    for index, (x, y, name) in enumerate(zip(xs, ys, names)):
        known, pending = bool(np.isfinite(weights[index])), index >= shown
        value = np.nan if pending else drive[index]
        colour, width, style = _edge_style(value, max(scale, 1e-12), decades)
        ax.plot([x, hub[0]], [y, hub[1]], color=colour, linewidth=width, linestyle=style, zorder=2,
                alpha=0.9 if known else 0.5)
        ax.scatter([x], [y], s=_area(150 if few else 110), zorder=4, color=theme.PANEL,
                   edgecolors=theme.NAVY if known else theme.INACTIVE, linewidths=1.2)
        _input_label(ax, x, y, name, None if values is None or pending else values[index], few=few, known=known,
                     slot_pt=slot_pt)
        if few and known and not pending:
            _edge_label(ax, (x + hub[0]) / 2, (y + hub[1]) / 2, fmt(float(value)))
    ax.scatter([hub[0]], [hub[1]], s=_area(520), zorder=4, color=theme.PANEL, edgecolors=theme.NAVY,
               linewidths=1.6)
    ax.text(*hub, "Σ", ha="center", va="center", fontsize=_pt(16), weight="bold", color=theme.NAVY, zorder=5)
    ax.text(hub[0] + 0.05, hub[1] - 0.13, bias_text, ha="left", va="center", fontsize=SMALL,
            color=theme.INK_SECONDARY)
    _chain_outputs(ax, outputs, result if reveal >= 1.0 else None, hub)


def _edge_label(ax: Axes, x: float, y: float, text: str) -> None:
    """A number written on its edge, on a pale backing so the edge does not cross it."""
    ax.text(x, y, text, ha="center", va="center", fontsize=SMALL, color=theme.NAVY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=theme.PANEL, edgecolor="none", alpha=0.85))


def _input_label(ax: Axes, x: float, y: float, name: str, value: str | None, *, few: bool, known: bool,
                 slot_pt: float) -> None:
    """Labels one input node.

    Few inputs: name (and its value on a second line) right-aligned before the node. Many: the name
    above the node, shrunk if needed to fit its ``slot_pt`` share of the row.
    """
    text = name if not value else f"{name}\n{value}"
    colour = theme.NAVY if known else theme.INK_MUTED
    if few:
        ax.text(x - 0.035, y, text, ha="right", va="center", fontsize=SMALL, color=colour, zorder=6,
                linespacing=1.15)
        return
    widest = max(len(line) for line in text.split("\n")) * REGULAR_EM
    size = min(ROW_NAME_SIZE, 0.85 * slot_pt / max(widest, 1e-9))
    ax.text(x, y + 0.08, text, ha="center", va="bottom", fontsize=size, color=colour, zorder=6)


def _chain_outputs(ax: Axes, outputs: Sequence[str], result: str | None, hub: tuple[float, float]) -> None:
    """Arrows from the Σ hub through each output box in turn, with the result above the last one."""
    previous = hub
    xs = np.linspace(hub[0] + 0.17, 0.89, len(outputs)) if len(outputs) > 1 else np.array([0.8])
    for x, label in zip(xs, outputs):
        _arrow(ax, previous, (x, hub[1]))
        _box(ax, x, hub[1], label)
        previous = (x, hub[1])
    if result is not None:
        ax.text(RIGHT, hub[1] + 0.17, result, ha="right", va="bottom", fontsize=_pt(12), weight="bold",
                color=theme.NAVY, zorder=6)


def tree_grid(ax: Axes, trees: Sequence, colour: Callable[[np.ndarray], object], *, title: str, total: int,
              summary: str, paths: Sequence | None = None, votes: Sequence[str] | None = None,
              depth: int = 3, columns: int = GRID_COLUMNS, rows: int = GRID_ROWS) -> int:
    """Thumbnails of the first trees feeding an average; returns how many were drawn.

    Each thumbnail shows its tree's top ``depth`` levels (with a path highlighted at inference) and keeps
    a strip beneath it for its vote.

    Args:
        ax: The diagram's axes.
        trees: The tree structures, first tree first; at most ``columns × rows`` are drawn.
        colour: Maps a node's value row to its colour.
        title: The diagram's title.
        total: How many trees the model has, for the "+ n more trees" note.
        summary: The text in the average box.
        paths: Each drawn tree's decision path for the query, at inference.
        votes: Each drawn tree's vote, written under it.
        depth: How many levels of each tree to draw.
        columns: Thumbnails per row.
        rows: Rows of thumbnails.

    Returns:
        How many trees were drawn.

    Raises:
        ValueError: If ``paths`` or ``votes`` does not have one entry per drawn tree.
    """
    shown = list(trees)[:columns * rows]
    _check_lengths(len(shown), paths=paths, votes=votes)
    frame(ax, title)
    width, height = 0.72 / columns, 0.74 / rows
    for index, tree in enumerate(shown):
        row, column = divmod(index, columns)
        bottom = 0.8 - (row + 1) * height
        box = (0.02 + column * width + 0.008, bottom + 0.16, width - 0.022, height - 0.2)
        draw_tree(ax, tree, box, depth, path=None if paths is None else paths[index], node_size=_area(14.0),
                  node_colour=colour)
        if votes is not None:
            ax.text(box[0] + box[2] / 2, bottom + 0.06, votes[index], ha="center", va="center", fontsize=_pt(9.5),
                    weight="bold", color=theme.NAVY)
    _arrow(ax, (0.75, 0.42), (0.84, 0.42))
    _box(ax, 0.9, 0.42, summary, size=_pt(10))
    if total > len(shown):
        ax.text(0.9, 0.16, f"+{plural(total - len(shown), 'more tree')}", ha="center", va="center", fontsize=SMALL,
                color=theme.INK_SECONDARY)
    return len(shown)


def chain(ax: Axes, trees: Sequence, colour: Callable[[np.ndarray], object], *, title: str, start: str, end: str,
          step_label: str, total: int, earlier: str = "", notes: Sequence[str] | None = None,
          paths: Sequence | None = None, depth: int = 2) -> int:
    """F₀, then '+ step_label ×' each of the newest trees (thumbnails), then the running total.

    ``earlier`` summarises the trees not drawn (e.g. their summed correction); ``notes`` carry each
    drawn tree's correction at inference.

    Args:
        ax: The diagram's axes.
        trees: Every tree structure, oldest first; the newest ``CHAIN_TREES`` are drawn.
        colour: Maps a node's value row to its colour.
        title: The diagram's title.
        start: The first box's text (F₀ and its value).
        end: The last box's text (the running total).
        step_label: Written above each tree, such as "0.1 ×" for the learning rate.
        total: How many trees the model has, for the "… n earlier" note.
        earlier: Added to that note, such as the earlier trees' summed correction.
        notes: Each drawn tree's correction, written under it at inference.
        paths: Each drawn tree's decision path for the query, at inference.
        depth: How many levels of each tree to draw.

    Returns:
        How many trees were drawn.

    Raises:
        ValueError: If ``notes`` or ``paths`` does not have one entry per drawn tree.
    """
    shown = list(trees)[-CHAIN_TREES:]
    _check_lengths(len(shown), notes=notes, paths=paths)
    frame(ax, title)
    _box(ax, 0.07, 0.4, start, size=_pt(10))
    xs = (0.27, 0.42, 0.57, 0.72)[:len(shown)]
    if total > len(shown):
        ax.text(TITLE_X, 0.7, f"… {total - len(shown)} earlier{earlier}", ha="left", va="center",
                fontsize=_pt(9.5), color=theme.INK_SECONDARY)
    for index, (x, tree) in enumerate(zip(xs, shown)):
        ax.text(x - 0.075, 0.4, "+", ha="center", va="center", fontsize=_pt(14), color=theme.NAVY)
        ax.text(x, 0.75, step_label, ha="center", va="center", fontsize=_pt(9.5), color=theme.INK_SECONDARY)
        draw_tree(ax, tree, (x - 0.055, 0.17, 0.11, 0.48), depth, node_size=_area(16.0), node_colour=colour,
                  path=None if paths is None else paths[index])
        if notes is not None:
            ax.text(x, 0.07, notes[index], ha="center", va="center", fontsize=_pt(9.5), weight="bold", color=theme.NAVY)
    _arrow(ax, (0.79, 0.4), (0.84, 0.4))
    _box(ax, 0.91, 0.4, end, size=_pt(10))
    return len(shown)


def flow(ax: Axes, steps: Sequence[str], *, title: str, values: Sequence[str] | None = None,
         reveal: float = 1.0) -> None:
    """A left-to-right pipeline of boxes joined by arrows; at inference each box's value appears beneath it.

    The boxes are spaced by their widths (and their values' widths), in one font size that fits them all,
    and the values appear in turn as ``reveal`` grows.

    Args:
        ax: The diagram's axes.
        steps: The boxes' labels, in order.
        title: The diagram's title.
        values: Each step's value at inference, written under its box.
        reveal: Inference progress, 0 to 1: the share of values shown so far.

    Raises:
        ValueError: If ``values`` does not have one entry per step.
    """
    _check_lengths(len(steps), values=values)
    frame(ax, title)
    sized = steps if values is None else [f"{step}\n{value}" for step, value in zip(steps, values)]
    spans, size = _spread(sized, TITLE_X, 1.0 - TITLE_X, _width_pt(ax), _pt(10.0), gap_em=2.5)
    shown = int(np.ceil(len(steps) * reveal))
    for index, ((left, right), step) in enumerate(zip(spans, steps)):
        x = (left + right) / 2
        _box(ax, x, 0.5, step, size=size)
        if index:
            _arrow(ax, (spans[index - 1][1], 0.5), (left, 0.5), shrink=2.0)
        if values is not None and index < shown:
            ax.text(x, 0.24, values[index], ha="center", va="center", fontsize=min(SMALL, size + 0.5),
                    weight="bold", color=theme.NAVY)


def _spread(labels: Sequence[str], start: float, end: float, width_pt: float, size: float,
            gap_em: float = 1.6) -> tuple[list[tuple[float, float]], float]:
    """Lays out a row of boxes and picks the font size that fits them.

    Returns (left, right) of each box in a row that fills ``start``..``end`` (unit coordinates of a
    card ``width_pt`` wide), and the font size, at most ``size``, that leaves at least ``gap_em``
    between boxes. Widths are estimated from each label's longest line.
    """
    widths = [max(len(line) for line in label.split("\n")) * REGULAR_EM for label in labels]  # in ems
    room = (end - start) * width_pt
    size = min(size, (room - BOX_PAD_PT * len(labels)) / max(sum(widths) + gap_em * (len(labels) - 1), 1e-9))
    boxes = [width * size + BOX_PAD_PT for width in widths]
    gap = (room - sum(boxes)) / (len(boxes) - 1) if len(boxes) > 1 else 0.0
    spans, left = [], start * width_pt
    for box in boxes:
        spans.append((left / width_pt, (left + box) / width_pt))
        left += box + gap
    return spans, size


def factor_rows(ax: Axes, rows: Sequence[tuple[str, Sequence[str]]], *, title: str, merge: str,
                joiner: str = "×", result: str | None = None) -> None:
    """One row of combined factors per class, every row feeding a normalising node.

    The factors are a prior × likelihoods, or their logs joined by '+'.

    Args:
        ax: The diagram's axes.
        rows: Per class, its label and its factors' texts.
        title: The diagram's title.
        merge: The normalising node's text.
        joiner: The symbol between factors: "×", or "+" for logs.
        result: The prediction, shown above the normalising node; ``None`` for none.
    """
    frame(ax, title)
    width_pt = _width_pt(ax)
    ys = np.linspace(0.62, 0.22, len(rows))
    for y, (label, factors) in zip(ys, rows):
        ax.text(0.02, y, label, ha="left", va="center", fontsize=SMALL, weight="bold", color=theme.NAVY)
        spans, size = _spread(factors, 0.17, 0.74, width_pt, _pt(9.5))
        for (left, right), factor in zip(spans, factors):
            _box(ax, (left + right) / 2, y, factor, size=size)
        for (_, right), (left, _) in zip(spans, spans[1:]):
            ax.text((right + left) / 2, y, joiner, ha="center", va="center", fontsize=_pt(12), color=theme.NAVY)
        _arrow(ax, (0.76, y), (0.82, 0.42))
    _box(ax, 0.9, 0.42, merge, size=_pt(10))
    if result is not None:
        ax.text(RIGHT, 0.58, result, ha="right", va="bottom", fontsize=_pt(12), weight="bold", color=theme.NAVY)
