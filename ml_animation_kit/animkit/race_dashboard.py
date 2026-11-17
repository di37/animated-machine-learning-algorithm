"""Styling every race dashboard shares: sibling textures, podium ordinals and leaderboard easing.

A race dashboard shows several models training side by side. Two models may share a hue (see
``animkit.race``); the second of the pair is told apart by the textures named here, a hatched bar and
a dashed curve. When the leaderboard reorders, ``ease`` moves the bars smoothly to their new ranks.
"""

TEXTURE_HATCH = "///"
TEXTURE_LINESTYLE = "--"
ORDINALS = ("1st", "2nd", "3rd")


def ease(t: float) -> float:
    """Smoothstep: bars start and stop gently instead of jumping between ranks.

    Args:
        t: Progress through the move, 0 to 1; values outside are clamped.

    Returns:
        The eased progress, 0 to 1: 3t² − 2t³, which is flat at both ends.
    """
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
