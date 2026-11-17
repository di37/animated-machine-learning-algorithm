"""Plain-language wording for passengers, features and tree splits.

The models read standardised numbers, but the dashboards speak in real units. These helpers take a
raw row (encoded but unscaled, as ``TitanicSplits.raw_test`` holds it) and put it in words, such as
"3rd class · man · age 22", and turn a tree's threshold into a question, such as "fare ≤ £15?".
``COLUMN`` maps a feature's name to its column in a row.
"""

import math

import numpy as np

from titanic_anim.config import CLASS_NAMES
from titanic_anim.data import FEATURE_COLUMNS

CHILD_AGE = 16
ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}
PORT_NAMES = {"embarked_C": "Cherbourg", "embarked_Q": "Queenstown"}
BINARY_FEATURES = ("female", "embarked_C", "embarked_Q")
COLUMN = {name: index for index, name in enumerate(FEATURE_COLUMNS)}


def ordinal(pclass: float) -> str:
    """A class of travel as an ordinal.

    Args:
        pclass: The class, 1, 2 or 3; it is rounded first, so a float works.

    Returns:
        ``"1st"``, ``"2nd"`` or ``"3rd"``, or ``"<n>th"`` for any other number.
    """
    return ORDINALS.get(int(round(pclass)), f"{int(round(pclass))}th")


def outcome(label: int) -> str:
    """What a label means, in words.

    Args:
        label: 0 or 1.

    Returns:
        ``"did not survive"`` or ``"survived"``.
    """
    return CLASS_NAMES[int(label)]


def person(raw: np.ndarray, age_known: bool = True) -> str:
    """Woman, man, girl or boy, from a passenger's raw features.

    Args:
        raw: One passenger's encoded, unscaled features.
        age_known: False when the age was missing and filled in; such a passenger is never a child.

    Returns:
        ``"girl"`` or ``"boy"`` under ``CHILD_AGE``, otherwise ``"woman"`` or ``"man"``.
    """
    female, age = raw[COLUMN["female"]] >= 0.5, raw[COLUMN["age"]]
    child = age_known and age < CHILD_AGE
    return ("girl" if child else "woman") if female else ("boy" if child else "man")


def age_text(age: float) -> str:
    """An age in words; a baby under a year old is given in months.

    Args:
        age: The age in years; the data gives babies a fraction, such as 0.17.

    Returns:
        For example ``"age 22"``, or ``"age 2 months"``.
    """
    return f"age {age:.0f}" if age >= 1 else f"age {max(round(age * 12), 1)} months"


def describe(raw: np.ndarray, age_known: bool = True) -> str:
    """One passenger, in a line of plain words.

    Args:
        raw: The passenger's encoded, unscaled features.
        age_known: False when the age was missing and filled in; the line then says "age unknown".

    Returns:
        For example ``"3rd class · man · age 22 · fare £7.25 · 1 sibling/spouse, 0 parent/child aboard"``.
    """
    age = age_text(raw[COLUMN["age"]]) if age_known else "age unknown"
    return (f"{ordinal(raw[COLUMN['pclass']])} class · {person(raw, age_known)} · {age} · "
            f"fare £{raw[COLUMN['fare']]:.2f} · {int(raw[COLUMN['sibsp']])} sibling/spouse, "
            f"{int(raw[COLUMN['parch']])} parent/child aboard")


def split_label(name: str, threshold: float) -> str:
    """A tree's split as a yes/no question in real units; the left branch is always "yes".

    scikit-learn's trees send ``x <= threshold`` to the left. A 0/1 feature splits at 0.5, so its left
    branch holds the zeros: ``female`` asks "male?" and a port flag asks "not Cherbourg?". Class and
    family counts are whole numbers, so their thresholds are rounded down.

    Args:
        name: The feature's name.
        threshold: The split's threshold, in real (unscaled) units.

    Returns:
        For example ``"male?"``, ``"class ≤ 2?"``, ``"fare ≤ £15?"`` or ``"age ≤ 9.5?"``.
    """
    if name == "female":
        return "male?"
    if name in PORT_NAMES:
        return f"not {PORT_NAMES[name]}?"
    if name == "pclass":
        return f"class ≤ {math.floor(threshold)}?"
    if name == "fare":
        return f"fare ≤ £{threshold:.0f}?"
    if name in ("sibsp", "parch"):
        return f"{name} ≤ {math.floor(threshold)}?"
    return f"{name} ≤ {threshold:.1f}?"


def feature_text(name: str, raw_value: float) -> str:
    """One feature's value in real units.

    Args:
        name: The feature's name.
        raw_value: Its value, encoded but unscaled.

    Returns:
        For example ``"female = yes"``, ``"fare = £7.25"``, ``"age = 30"`` or ``"sibsp = 1"``.
    """
    if name in BINARY_FEATURES:
        return f"{name} = {'yes' if raw_value >= 0.5 else 'no'}"
    if name == "fare":
        return f"fare = £{raw_value:.2f}"
    if name == "age":
        return age_text(raw_value).replace("age ", "age = ")
    return f"{name} = {raw_value:.0f}"
