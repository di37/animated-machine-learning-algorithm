from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from animkit.equation_panel import EquationPanel
from titanic_anim.equations import prediction_card, training_card
from titanic_anim.equations.kernels import kernel_terms, log_likelihood_ratios
from titanic_anim.equations.trees import initial_log_odds, leaf_note
from titanic_anim.models import SCORE_MARGIN, get_spec, model_keys
from titanic_anim.trainers import train_silently


def short(key):
    spec = get_spec(key)
    return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:4])


@pytest.fixture(scope="module")
def trained(splits):
    return {key: (short(key), *train_silently(short(key), splits)) for key in model_keys()}


def model_answer(model, spec, x):
    if spec.score_kind == SCORE_MARGIN:
        return float(model.decision_function(x[None, :])[0])
    return float(model.predict_proba(x[None, :])[0, 1])


@pytest.mark.parametrize("key", model_keys())
def test_worked_equation_equals_the_models_answer(key, trained, splits):
    spec, model, _ = trained[key]
    for index in range(5):
        x = splits.X_test[index]
        card = prediction_card(model, x, index + 1, splits)
        assert card.value == pytest.approx(model_answer(model, spec, x), abs=1e-6)
        assert "✓" in card.lines[-1].text


@pytest.mark.parametrize("key", model_keys())
def test_cards_render_as_mathtext(key, trained, splits):
    spec, model, frame = trained[key]
    figure = plt.figure(figsize=(6, 9))
    panel = EquationPanel(figure.add_axes((0.02, 0.02, 0.96, 0.96)))
    panel.update(training_card(model, frame, spec, splits))
    figure.canvas.draw()  # raises if any mathtext fails to parse
    panel.update(prediction_card(model, splits.X_test[0], 1, splits))
    figure.canvas.draw()
    assert panel.title.get_text().startswith("Passenger #1")


def test_training_cards_name_the_model(trained, splits):
    spec, model, frame = trained["logistic_regression"]
    card = training_card(model, frame, spec, splits)
    assert card.title == "What Logistic Regression is computing"
    assert any("TRAINING MINIMISES" == line.text for line in card.lines)


def test_kernel_terms_rebuild_the_svm_margin(trained, splits):
    _, model, _ = trained["svm"]
    x = splits.X_test[3]
    assert kernel_terms(model, x).sum() + model.intercept_[0] == pytest.approx(model.decision_function(x[None])[0])
    survived = model.dual_coef_[0] > 0
    np.testing.assert_array_equal(survived, splits.y_train[model.support_] == 1)


def test_naive_bayes_log_odds_match_predict_proba(trained, splits):
    _, model, _ = trained["naive_bayes"]
    x = splits.X_test[2]
    log_odds = np.log(model.class_prior_[1] / model.class_prior_[0]) + log_likelihood_ratios(model, x).sum()
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(model.predict_proba(x[None])[0, 1])


def test_boosting_starts_at_the_base_rate(trained, splits):
    _, model, _ = trained["gradient_boosting"]
    rate = splits.y_train.mean()
    assert initial_log_odds(model) == pytest.approx(np.log(rate / (1 - rate)))


def test_probability_rule_shows_which_side_of_the_threshold():
    from titanic_anim.equations.cards import probability_rule

    assert probability_rule(0.7) == "0.700 > 0.5"
    assert probability_rule(0.5004) == "0.500400 > 0.5"
    assert probability_rule(0.5) == "0.500000 ≤ 0.5"


def test_leaf_note_wording():
    assert "single training passenger here survived" in leaf_note(1, 1)
    assert "did not survive" in leaf_note(0, 1)
    assert leaf_note(3, 10) == "leaf: 3 of the 10 training passengers here survived"


def test_prediction_card_rejects_wrong_shapes(trained, splits):
    _, model, _ = trained["knn"]
    with pytest.raises(ValueError):
        prediction_card(model, splits.X_test[:2], 1, splits)


def test_generic_fallback_for_unknown_models(trained, splits):
    from sklearn.dummy import DummyClassifier

    _, _, frame = trained["knn"]
    model = DummyClassifier(strategy="prior").fit(splits.X_train, splits.y_train)
    card = prediction_card(model, splits.X_test[0], 1, splits)
    assert card.value == pytest.approx(splits.y_train.mean())
    assert training_card(model, frame, get_spec("knn"), splits).lines
