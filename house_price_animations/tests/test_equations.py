import warnings
from dataclasses import FrozenInstanceError, replace

import matplotlib.pyplot as plt
import pytest
from sklearn.dummy import DummyRegressor

from animkit.equation_panel import EquationPanel
from house_anim.equations import (
    HEADING,
    KINDS,
    MATH,
    EquationCard,
    EquationLine,
    prediction_card,
    training_card,
)
from house_anim.models import get_spec, model_keys
from house_anim.trainers import train_silently


def small_spec(key):
    spec = get_spec(key)
    return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:3])


@pytest.fixture(scope="module")
def trained(splits):
    return {key: (small_spec(key), *train_silently(small_spec(key), splits)) for key in model_keys()}


def render(card: EquationCard):
    figure = plt.figure(figsize=(5, 7))
    panel = EquationPanel(figure.add_axes((0, 0, 1, 1)))
    panel.update(card)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.canvas.draw()  # raises if any mathtext fails to parse
    return panel, [str(w.message) for w in caught if "Glyph" in str(w.message)]


@pytest.mark.parametrize("key", model_keys())
def test_training_card_shows_model_objective_and_step(key, trained, splits):
    spec, model, frame = trained[key]
    card = training_card(model, frame, spec, splits)
    kinds = {line.kind for line in card.lines}
    assert kinds <= set(KINDS) and MATH in kinds
    assert sum(line.kind == HEADING for line in card.lines) >= 2
    assert spec.short_title in card.title


@pytest.mark.parametrize("key", model_keys())
def test_prediction_card_reproduces_model_predict(key, trained, splits):
    _, model, _ = trained[key]
    x = splits.X_test[3]
    card = prediction_card(model, x, house_number=4, splits=splits)
    assert card.value == pytest.approx(model.predict(x[None, :])[0], abs=1e-9)
    assert "✓" in card.lines[-1].text
    assert "#4" in card.title


@pytest.mark.parametrize("key", model_keys())
def test_cards_render_without_math_errors_or_missing_glyphs(key, trained, splits):
    spec, model, frame = trained[key]
    for card in (training_card(model, frame, spec, splits),
                 prediction_card(model, splits.X_test[0], house_number=1, splits=splits)):
        _, missing = render(card)
        assert missing == []


def test_unknown_models_get_a_generic_card(splits):
    model = DummyRegressor().fit(splits.X_train, splits.y_train)
    _, frame = train_silently(small_spec("decision_tree"), splits)
    card = training_card(model, frame, get_spec("decision_tree"), splits)
    assert "DummyRegressor" in " ".join(line.text for line in card.lines)
    prediction = prediction_card(model, splits.X_test[0], house_number=1, splits=splits)
    assert prediction.value == pytest.approx(model.predict(splits.X_test[:1])[0])


def test_prediction_needs_a_single_house(trained, splits):
    _, model, _ = trained["linear_regression"]
    with pytest.raises(ValueError, match="one house"):
        prediction_card(model, splits.X_test[:2], house_number=1, splits=splits)


def test_cards_are_frozen():
    card = EquationCard("t", (EquationLine("x", MATH),))
    with pytest.raises(FrozenInstanceError):
        card.title = "u"


def test_linear_card_lists_every_contribution(trained, splits):
    _, model, _ = trained["linear_regression"]
    card = prediction_card(model, splits.X_test[0], house_number=1, splits=splits)
    text = "\n".join(line.text for line in card.lines)
    for name in splits.feature_names:
        assert name in text


def test_tree_card_walks_the_decision_path(trained, splits):
    _, model, _ = trained["decision_tree"]
    card = prediction_card(model, splits.X_test[0], house_number=1, splits=splits)
    text = "\n".join(line.text for line in card.lines)
    assert "≤" in text or ">" in text
    assert "leaf" in text


@pytest.mark.parametrize("alpha, expected", [(10.0, "set the other 8"), (1e-6, "too weak")])
def test_lasso_note_matches_how_many_weights_survive(splits, alpha, expected):
    from sklearn.linear_model import Lasso

    model = Lasso(alpha=alpha).fit(splits.X_train, splits.y_train)
    _, frame = train_silently(small_spec("lasso"), splits)
    card = training_card(model, frame, get_spec("lasso"), splits)
    assert expected in " ".join(line.text for line in card.lines)
