"""Narrated 1440p lesson videos: one regression algorithm learning house prices from median income.

Each lesson trains a real model on one input, a California district's median income, and records
the states it passes through. animkit's lesson director (``animkit.lessons.director``) then plays them
as chapters: the data, the first guess, training, the final model, a prediction for an unseen
district, a final exam on every test district, the model in 3-D over two inputs, the same algorithm
with all eight features, and a takeaway.

The parts, in reading order: ``data`` (the one-feature view of the splits), ``lesson`` (the
``Snapshot`` and ``Lesson`` contract), ``stories`` (one module per algorithm family, each building a
``Lesson``), ``plots`` and ``lesson_diagrams`` (the drawing), ``regression_scene`` (animkit's ``Scene``
for regression, which routes each chapter to ``plots``, ``exam``, ``surface_view`` or
``more_features``), then ``director``, ``summary`` and ``cli``.
"""
