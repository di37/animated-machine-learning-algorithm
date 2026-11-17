"""Survival colours on top of the shared animkit palette.

Outcome is a polarity around the decision threshold, so it uses the diverging pair: blue for
survived, red for did not survive, a neutral grey for "could go either way".

A score becomes a colour in two steps. A ``Normalize`` maps it onto 0 … 1 (a probability as it is,
a margin from ``-MARGIN_COLOUR_LIMIT`` to ``+MARGIN_COLOUR_LIMIT``), then ``SURVIVAL_CMAP`` turns
that into red, grey or blue. So both decision thresholds, a probability of 0.5 and a margin of 0,
are the same grey.
"""

from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

from animkit import theme as base
from titanic_anim.config import MARGIN_COLOUR_LIMIT, SCORE_MARGIN

SURVIVED = base.POSITIVE
DIED = base.NEGATIVE
UNDECIDED = "#dddcd5"
OUTCOME_COLOURS = (DIED, SURVIVED)  # indexed by class label


def survival_colormap() -> LinearSegmentedColormap:
    """The red → grey → blue colormap: did not survive, undecided, survived.

    Returns:
        A matplotlib colormap over 0 … 1.
    """
    return LinearSegmentedColormap.from_list("survival", (DIED, UNDECIDED, SURVIVED))


SURVIVAL_CMAP = survival_colormap()
PROBABILITY_NORM = Normalize(0.0, 1.0)
MARGIN_NORM = Normalize(-MARGIN_COLOUR_LIMIT, MARGIN_COLOUR_LIMIT)


def score_norm(score_kind: str) -> Normalize:
    """How a score of one kind maps onto the colormap's 0 … 1.

    Args:
        score_kind: ``SCORE_PROBABILITY`` or ``SCORE_MARGIN``.

    Returns:
        ``MARGIN_NORM`` (``±MARGIN_COLOUR_LIMIT`` onto 0 … 1) for a margin, ``PROBABILITY_NORM``
        (0 … 1 as it is) for anything else.
    """
    return MARGIN_NORM if score_kind == SCORE_MARGIN else PROBABILITY_NORM


def survival_rgba(probability: float):
    """The colour of a probability of surviving.

    Args:
        probability: P(survive), from 0 to 1.

    Returns:
        An RGBA tuple: red near 0, grey at 0.5, blue near 1.
    """
    return SURVIVAL_CMAP(PROBABILITY_NORM(probability))


def score_rgba(score: float, score_kind: str):
    """The colour of a score of either kind.

    Args:
        score: P(survive), or a signed margin.
        score_kind: ``SCORE_PROBABILITY`` or ``SCORE_MARGIN``: which of the two ``score`` is.

    Returns:
        An RGBA tuple; a margin beyond ``±MARGIN_COLOUR_LIMIT`` takes the end colour.
    """
    return SURVIVAL_CMAP(score_norm(score_kind)(score))


def outcome_rgba(label: int, alpha: float = 1.0):
    """The colour of what really happened to a passenger.

    Args:
        label: 1 (survived, blue) or 0 (did not survive, red).
        alpha: The opacity, from 0 to 1.

    Returns:
        An RGBA tuple.
    """
    return to_rgba(OUTCOME_COLOURS[int(label)], alpha)
