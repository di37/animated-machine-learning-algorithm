"""Turning on-screen text into words a voice can say: symbols, money, notation and acronyms."""

import pytest

from animkit.lessons.speech_text import spoken, unexpanded_acronyms

CASES = [
    ("First guess: θ₀ = θ₁ = 0, a flat line at $0.",
     "First guess: theta zero equals theta one equals 0, a flat line at 0 dollars."),
    ("train RMSE $83k, validation $84k",
     "train root mean squared error 83 thousand dollars, validation 84 thousand dollars"),
    ("the same line to within $0.08k.", "the same line to within 80 dollars."),
    ("from $12.1k to $500k", "from 12.1 thousand dollars to 500 thousand dollars"),
    ("Almost no penalty (α = 1.0e-5):", "Almost no penalty (alpha equals 10 to the minus 5):"),
    ("Validation chose α = 4.6e-1:", "Validation chose alpha equals 0.46:"),
    ("α = 1.6e-2", "alpha equals 0.016"),
    ("α = 2.5e-4", "alpha equals 2.5 times 10 to the minus 4"),
    ("squared error + α × (sum of squared weights)", "squared error plus alpha times (sum of squared weights)"),
    ("F_m = F_m−1 + 0.15 × h_m.", "F m equals F m minus 1 plus 0.15 times h m."),
    ("P = σ(F).", "P equals sigma of F."),
    ("(price − F₀)", "(price minus F zero)"),
    ("the dashed lines are scores −1 and +1.", "the dashed lines are scores minus 1 and plus 1."),
    ("(x³ and x⁵ move together)", "(x cubed and x to the 5th move together)"),
    ("x²", "x squared"),
    ("x, x², … x¹²", "x, x squared, up to x to the 12th"),
    ("median income (x) and median house value (y).", "median income and median house value."),
    ("age (x) and fare paid (y, log scale);", "age and fare paid on a log scale;"),
    ("|score| < 1 means", "a score between minus 1 and plus 1 means"),
    ("hinge loss + 0.0003 × ½|weights|².", "hinge loss plus 0.0003 times half the squared weights."),
    ("sum of |weights|, which", "sum of absolute weights, which"),
    ("(○ on the right)", "(open circles on the right)"),
    ("more passengers (partial_fit). The", "more passengers. The"),
    ("a speckled 0% / 100% patchwork", "a speckled 0% or 100% patchwork"),
    ("never leaves the 0–100% range", "never leaves the 0 to 100% range"),
    ("yes/no questions about 0/1 sex", "yes-or-no questions about 0-or-1 sex"),
    ("blue = worth more than predicted", "blue means worth more than predicted"),
    ("1 → 8 → 8 → 1", "1 to 8 to 8 to 1"),
    ("SGD on hinge loss; a linear SVM; the MSE",
     "Stochastic gradient descent on hinge loss; a linear support vector machine; the mean squared error"),
    ("Neural Network (MLP). A small", "Neural Network (multi-layer perceptron). A small"),
    ("an MLP, an SVM and an RMSE", "a multi-layer perceptron, a support vector machine and a root mean squared error"),
    ("An MLP learns", "A multi-layer perceptron learns"),
    ("two MLPs and three SVMs", "two multi-layer perceptrons and three support vector machines"),
    ("The linear SVM's score", "The linear support vector machine's score"),
    ("kNN predicts by analogy", "k-nearest neighbours predicts by analogy"),
    ("The network's ReLU units fold the sheet", "The network's rectified linear units fold the sheet"),
    ("ReLU neurons switch on", "Rectified linear units switch on"),
    ("83k. ReLU units fold it. MLP learns. kNN predicts.",
     "83k. Rectified linear units fold it. Multi-layer perceptron learns. k-nearest neighbours predicts."),
    ("(every ReLU outputs 0)", "(every rectified linear unit outputs 0)"),
    ("a ReLU and an AUC of 0.86", "a rectified linear unit and an area under the curve of 0.86"),
    ("PCA, then an RBF kernel", "Principal component analysis, then a radial basis function kernel"),
]


@pytest.mark.parametrize("abbreviation", ["MLP", "RMSE", "SVM", "SGD", "kNN", "ReLU", "MSE"])
def test_jargon_is_said_by_its_full_name_never_spelled_out(abbreviation):
    said = spoken(f"the {abbreviation} here")
    assert abbreviation not in said and "-".join(abbreviation.upper()) not in said


def test_an_abbreviation_with_no_full_name_is_found_in_what_is_said():
    assert unexpanded_acronyms(spoken("The XYZ score, the RMSE and the QR code.")) == ("QR", "XYZ")
    assert unexpanded_acronyms(spoken("Train RMSE $83k with an MLP.")) == ()


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_symbols_become_words(text, expected):
    assert spoken(text) == expected


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_what_is_left_is_plain_ascii(text, expected):
    assert spoken(text).isascii()


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_saying_it_twice_changes_nothing(text, expected):
    assert spoken(spoken(text)) == spoken(text)


def test_plain_sentences_are_untouched():
    text = "Each dot is one California district. Showing 900 of the 12,384 districts, 95% of the time."
    assert spoken(text) == text


@pytest.mark.parametrize(("exponent", "word"), [("¹", "1st"), ("⁴", "4th"), ("¹¹", "11th"), ("²¹", "21st"),
                                                 ("²²", "22nd"), ("¹³", "13th")])
def test_powers_are_ordinals(exponent, word):
    assert spoken(f"x{exponent}") == f"x to the {word}"


def test_spaces_are_tidied():
    assert spoken("a  ( b )  , c") == "a (b), c"


def test_a_projects_own_phrases_are_said_its_way_first():
    phrases = (("P(win) = 50%", "an even chance of winning"), ("P(win)", "the chance of winning"))
    assert spoken("each team gets P(win) = 50%; P(win) rises", phrases) == \
        "each team gets an even chance of winning; the chance of winning rises"


def test_without_its_phrases_the_notation_is_still_said_symbol_by_symbol():
    assert spoken("P(win) = 50%") == "P(win) equals 50%"


@pytest.mark.parametrize(("text", "expected"), [
    ("$1", "1 dollar"),
    ("$1.5M", "1.5 million dollars"),
    ("an 80/20 split", "an 80/20 split"),
    ("10⁻⁵", "10 to the minus 5"),
    ("10²", "10 squared"),
    ("2½ steps", "2 and a half steps"),
    ("k ≤ 5, k ≥ 2, about ≈ 3, ±1", "k is at most 5, k is at least 2, about about 3, plus or minus 1"),
    ("train $83k · validation $84k", "train 83 thousand dollars, validation 84 thousand dollars"),
])
def test_less_common_notation_is_said_too(text, expected):
    assert spoken(text) == expected
    assert spoken(text).isascii()
