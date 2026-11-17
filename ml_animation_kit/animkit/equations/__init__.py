"""Equation cards: what a model computes, as lines of mathtext and numbers, ready to draw.

This package re-exports the card types and the line kinds from ``cards``; ``animkit.equation_panel``
draws a card.
"""

from animkit.equations.cards import HEADING, KINDS, MATH, MONO, NOTE, EquationCard, EquationLine

__all__ = ["HEADING", "KINDS", "MATH", "MONO", "NOTE", "EquationCard", "EquationLine"]
