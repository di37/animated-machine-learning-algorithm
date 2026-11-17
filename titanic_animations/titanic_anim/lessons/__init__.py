"""Narrated 1440p lesson videos: one classifier learning who survived the Titanic from age, fare and sex.

Each video follows one classifier through the ten chapters of animkit's lesson director
(``animkit.lessons.director``): title, the data, a first guess, training, the final model, a prediction for
one unseen passenger, a final exam on every test passenger, the model in 3-D, the same algorithm with every
input, and a takeaway. The director lays out the frame, paces the chapters and writes the MP4; this package
supplies everything that is about the Titanic:

- ``data``: the passengers as age, log fare and sex, split into training, validation and test rows.
- ``lesson``: the lesson contract, ``ClassLesson``, built from snapshots of a real model as it learns.
- ``stories``: one module per classifier family; each trains its model and writes its animkit ``Story``.
- ``scene``: ``ClassificationScene``, this project's implementation of animkit's ``Scene`` protocol. It
  draws the age × fare maps (``maps``), the final exam (``exam``), the 3-D view (``surface_view``) and the
  all-inputs comparison (``more_features``).
- ``cli``: the command line, ``python -m titanic_anim.lessons``, and ``summary``, the CSV it writes.
"""
