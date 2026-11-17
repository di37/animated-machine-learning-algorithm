"""Equation cards: what a model computes, as lines of mathtext and numbers, ready to draw.

A card is a title and a list of lines, and each line has a kind that sets how it is drawn: an
upper-case section heading, a line of matplotlib mathtext (TeX-like maths between dollar signs), a
row of aligned numbers in a monospaced font, or a plain note. The helpers here build each kind of
line, plus pieces several models' cards share: a linear model's terms, a value on a log scale, and a
standardised threshold turned back into real units. Projects build the cards; ``animkit.equation_panel``
draws them.
"""

import math
from dataclasses import dataclass

import numpy as np

from animkit.data import ScaledSplits

HEADING = "heading"
MATH = "math"  # matplotlib mathtext, e.g. r"$\hat{y} = b + \sum_j w_j x_j$"
MONO = "mono"  # aligned numbers
NOTE = "note"
KINDS = (HEADING, MATH, MONO, NOTE)
PREDICT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class EquationLine:
    """One line of an equation card.

    Attributes:
        text: What the line says; for ``MATH`` it is mathtext between dollar signs.
        kind: How it is drawn: one of ``KINDS``.
    """

    text: str
    kind: str


@dataclass(frozen=True)
class EquationCard:
    """What a model computes, as a title and lines ready to draw.

    Attributes:
        title: The card's title.
        lines: Its lines, top to bottom.
        value: For prediction cards, the value the worked equation arrives at; ``None`` otherwise.
    """

    title: str
    lines: tuple[EquationLine, ...]
    value: float | None = None  # for prediction cards: the value the worked equation arrives at


def heading(text: str) -> EquationLine:
    """A section heading line, in capitals.

    Args:
        text: The heading.

    Returns:
        A ``HEADING`` line, its text upper-cased.
    """
    return EquationLine(text.upper(), HEADING)


def math_line(text: str) -> EquationLine:
    """A line of mathtext.

    Args:
        text: matplotlib mathtext, with the maths between dollar signs.

    Returns:
        A ``MATH`` line, its text unchanged.
    """
    return EquationLine(text, MATH)


def mono(text: str) -> EquationLine:
    """A line of aligned numbers, in a monospaced font.

    Dollar signs are escaped so matplotlib does not read them as maths, and hyphens become typographic
    minus signs.

    Args:
        text: The numbers, already spaced into columns.

    Returns:
        A ``MONO`` line.
    """
    return EquationLine(text.replace("$", "\\$").replace("-", "−"), MONO)


def note(text: str) -> EquationLine:
    """A plain note line; dollar signs are escaped so they print as themselves.

    Args:
        text: The note.

    Returns:
        A ``NOTE`` line.
    """
    return EquationLine(text.replace("$", "\\$"), NOTE)


def power_of_ten(value: float) -> str:
    """Mathtext for a value on a log schedule, e.g. 10^{-1.50}.

    Args:
        value: A positive value, such as a regularisation strength.

    Returns:
        ``10^{n}``, with a whole exponent when the value is an exact power of ten and two decimals
        otherwise. It has no dollar signs, so it can go inside a longer formula.
    """
    exponent = math.log10(value)
    shown = f"{exponent:.0f}" if abs(exponent - round(exponent)) < 1e-9 else f"{exponent:.2f}"
    return f"10^{{{shown}}}"


def subscript_name(name: str) -> str:
    """Mathtext for a feature as a variable: x, with the feature's name as an upright subscript.

    Args:
        name: The feature's name.

    Returns:
        The mathtext, without dollar signs.
    """
    return f"x_{{\\mathrm{{{name}}}}}"


def linear_terms(intercept: float, weights: np.ndarray, names: tuple[str, ...],
                 per_line: int = 3, max_terms: int = 6) -> tuple[EquationLine, ...]:
    """'= b + w1 x_a - w2 x_b ...' wrapped over lines, biggest weights first.

    Zero weights are left out; past ``max_terms`` the rest are counted in one "+ … (n more)" term.

    Args:
        intercept: The model's intercept, b.
        weights: One weight per feature.
        names: The feature names, in the same order.
        per_line: How many terms fit on one line.
        max_terms: How many terms to write out.

    Returns:
        The ``MATH`` lines: the first starts with "= b", the others are indented continuations.
    """
    order = [i for i in np.argsort(-np.abs(weights)) if weights[i] != 0.0][:max_terms]
    terms = [f"{'+' if weights[i] >= 0 else '-'} {abs(weights[i]):.2f}\\,{subscript_name(names[i])}"
             for i in order]
    hidden = int(np.count_nonzero(weights)) - len(order)
    if hidden > 0:
        terms.append(f"+ \\ldots ({hidden}\\ \\mathrm{{more}})")
    chunks = [terms[i:i + per_line] for i in range(0, len(terms), per_line)] or [[]]
    first = [math_line(f"$= {intercept:.2f}\\ {' '.join(chunks[0])}$")]
    rest = [math_line(f"$\\quad {' '.join(chunk)}$") for chunk in chunks[1:]]
    return tuple(first + rest)


def raw_value(splits: ScaledSplits, feature: int, scaled: float) -> float:
    """Undo standardisation so thresholds read in the feature's real units.

    Args:
        splits: The project's splits, which hold each feature's training mean and scale.
        feature: The feature's column index.
        scaled: A standardised value of that feature, such as a tree's split threshold.

    Returns:
        The value in the feature's own units: scaled × scale + mean.
    """
    return float(scaled * splits.feature_scale[feature] + splits.feature_mean[feature])


def validate_row(x: np.ndarray, n_features: int, noun: str = "row") -> np.ndarray:
    """One feature row as floats; ``noun`` names what a row is in the error message.

    Args:
        x: The row a prediction card explains.
        n_features: How many features the model takes.
        noun: What one row is, for the error message.

    Returns:
        The row as a float array.

    Raises:
        ValueError: If the row does not have shape ``(n_features,)``.
    """
    row = np.asarray(x, dtype=float)
    if row.shape != (n_features,):
        raise ValueError(f"prediction cards explain one {noun}: expected shape ({n_features},), got {row.shape}")
    return row
