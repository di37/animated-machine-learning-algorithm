"""Layout checks for a drawn card: every text stays inside its axes and no two texts collide.

The lessons' tests run these on each algorithm diagram, at every state the video shows. The checks
measure the box matplotlib's renderer really gives each text (with any background box around it), so
they catch what a viewer would see, not what the layout code meant to draw.
"""

from itertools import combinations

from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

SLACK_PX = 1.5  # anti-aliasing tolerance at the frame's edge


def _extents(ax: Axes) -> list[tuple[str, Bbox]]:
    """Each visible, non-empty text with its rendered box in display pixels, its background box included."""
    ax.figure.draw_without_rendering()  # lays out text boxes so their extents are final
    renderer = ax.figure.canvas.get_renderer()
    boxes = []
    for text in ax.texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        patch = text.get_bbox_patch()
        boxes.append((text.get_text(), box if patch is None else Bbox.union([box, patch.get_window_extent(renderer)])))
    return boxes


def overflowing_texts(ax: Axes) -> tuple[str, ...]:
    """Texts whose rendered box (with any background box) leaves the axes.

    Args:
        ax: An axes that has been drawn into.

    Returns:
        The text of each one that crosses the axes' edge by more than ``SLACK_PX``.
    """
    frame = ax.get_window_extent()
    return tuple(label for label, box in _extents(ax)
                 if box.x0 < frame.x0 - SLACK_PX or box.x1 > frame.x1 + SLACK_PX
                 or box.y0 < frame.y0 - SLACK_PX or box.y1 > frame.y1 + SLACK_PX)


def colliding_texts(ax: Axes) -> tuple[tuple[str, str], ...]:
    """Pairs of texts whose rendered boxes overlap by more than the slack.

    Args:
        ax: An axes that has been drawn into.

    Returns:
        The ``(first, second)`` texts of each pair whose boxes overlap by more than ``SLACK_PX`` both
        across and up.
    """
    def overlap(a: Bbox, b: Bbox) -> bool:
        """Whether two boxes overlap by more than the slack, both across and up.

        Args:
            a: One text's box, in display pixels.
            b: The other's.

        Returns:
            True when the shared width and the shared height both exceed ``SLACK_PX``.
        """
        return (min(a.x1, b.x1) - max(a.x0, b.x0) > SLACK_PX) and (min(a.y1, b.y1) - max(a.y0, b.y0) > SLACK_PX)
    return tuple((first, second) for (first, a), (second, b) in combinations(_extents(ax), 2) if overlap(a, b))
