"""The voice-over says the Titanic captions' probability notation in words."""

import pytest

from animkit.lessons.speech_text import spoken
from titanic_anim.lessons.scene import CLASSIFICATION


@pytest.mark.parametrize(("caption", "said"), [
    ("First guess: all weights are 0, so every passenger gets P(survive) = 50% and the maps are grey.",
     "First guess: all weights are 0, so every passenger gets a 50% chance of survival and the maps are grey."),
    ("The boundary is where P(survive) = 50%.", "The boundary is where the chance of survival is 50%."),
    ("Height is the model's P(survive), one sheet per sex.",
     "Height is the model's chance of survival, one sheet per sex."),
])
def test_probability_notation_is_said_in_words(caption, said):
    assert spoken(caption, CLASSIFICATION.spoken_phrases) == said
