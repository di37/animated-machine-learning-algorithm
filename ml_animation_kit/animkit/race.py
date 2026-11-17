"""Colours for a race of models: colour follows the model, never its rank.

In a race the leaderboard reorders as the models train. If colour followed rank, a model would change
colour mid-video, so each model gets a fixed palette slot from its place in the list instead. Two
variants of one algorithm can be paired as siblings: they share a hue, and the second is textured.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from animkit.models import ModelSpec

NO_SIBLINGS: Mapping[str, str] = MappingProxyType({})


def colour_slots(specs: Sequence[ModelSpec],
                 siblings: Mapping[str, str] = NO_SIBLINGS) -> tuple[tuple[int, bool], ...]:
    """(palette slot, textured) per model, in order; colour follows the model, never its rank.

    The palette has eight validated hues. A model listed in ``siblings`` (key -> sibling key) shares
    its sibling's hue and is told apart by texture (hatched bar, dashed curve) and its name label.

    Args:
        specs: The racing models, in the order they are listed.
        siblings: Maps a model's key to the key of the model whose hue it borrows. An entry whose sibling
            is not in the race is ignored.

    Returns:
        One ``(slot, textured)`` pair per model, in ``specs`` order. Only models with a hue of their own
        take a new slot, so the slots stay consecutive.

    Example:
        With models ``a``, ``b`` and ``c``, where ``c`` borrows ``a``'s hue::

            colour_slots(specs, {"c": "a"})  # ((0, False), (1, False), (0, True))
    """
    keys = [spec.key for spec in specs]
    borrows = {key: siblings[key] for key in keys if siblings.get(key) in keys}
    slot_of = {key: index for index, key in enumerate(k for k in keys if k not in borrows)}
    return tuple((slot_of[borrows[k]], True) if k in borrows else (slot_of[k], False) for k in keys)
