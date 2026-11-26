"""The narration script: what the voice says in each chapter, taken from what that chapter shows."""

from types import SimpleNamespace

from animkit.lessons.narration import Cue, narration_script
from animkit.lessons.timing import CHAPTERS


def stand_ins():
    story = SimpleNamespace(
        title="Linear Regression", method="trained by batch gradient descent",
        objective="Watch a straight line learn θ₀ and θ₁.", initial="First guess: θ₀ = θ₁ = 0.",
        training="Each step moves downhill.", final="Train RMSE $83k.", inference="An unseen district.",
        takeaway="Gradient descent steps downhill.", limitation="a straight line cannot bend.")
    lesson = SimpleNamespace(story=story)
    scene = SimpleNamespace(data_caption=lambda lesson: "Each dot is a district (x).",
                            exam_caption=lambda lesson, exam: f"Final exam on {exam} districts.")
    extras = SimpleNamespace(exam=4128, surface=SimpleNamespace(caption="A tilted plane."),
                             comparison=SimpleNamespace(headline="More inputs cut the error to $75k.",
                                                        note="95% interval: $4.4k to $12.1k."))
    return lesson, scene, extras


def test_every_chapter_has_one_cue_in_order():
    script = narration_script(*stand_ins())
    assert tuple(cue.chapter for cue in script) == CHAPTERS
    assert all(isinstance(cue, Cue) for cue in script)


def test_each_cue_reads_its_chapters_main_text():
    screen = {cue.chapter: cue.screen for cue in narration_script(*stand_ins())}
    assert screen == {
        "title": "Linear Regression. Watch a straight line learn θ₀ and θ₁.",
        "data": "Each dot is a district (x).",
        "initial": "First guess: θ₀ = θ₁ = 0.",
        "training": "Each step moves downhill.",
        "final": "Train RMSE $83k.",
        "inference": "An unseen district.",
        "exam": "Final exam on 4128 districts.",
        "surface": "A tilted plane.",
        "features": "More inputs cut the error to $75k.",
        "takeaway": "Gradient descent steps downhill. Limitation: a straight line cannot bend.",
    }


def test_the_spoken_text_is_the_screen_text_in_words():
    cues = {cue.chapter: cue for cue in narration_script(*stand_ins())}
    assert cues["initial"].spoken == "First guess: theta zero equals theta one equals 0."
    assert cues["final"].spoken == "Train root mean squared error 83 thousand dollars."
    assert cues["data"].spoken == "Each dot is a district."


def test_the_fine_print_stays_on_screen():
    spoken = " ".join(cue.spoken for cue in narration_script(*stand_ins()))
    assert "interval" not in spoken and "batch gradient descent" not in spoken


def test_a_scene_can_say_its_own_notation():
    lesson, scene, extras = stand_ins()
    phrased = SimpleNamespace(**vars(scene), spoken_phrases=(("θ₀ = θ₁ = 0", "both numbers at zero"),))
    cues = {cue.chapter: cue for cue in narration_script(lesson, phrased, extras)}
    assert cues["initial"].spoken == "First guess: both numbers at zero."
    assert cues["initial"].screen == "First guess: θ₀ = θ₁ = 0."
