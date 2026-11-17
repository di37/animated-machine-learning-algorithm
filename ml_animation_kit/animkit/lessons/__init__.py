"""Narrated 1440p lesson videos: the director, canvas, encoder, pacing and the lesson contract.

A lesson teaches one algorithm in ten chapters: title, data, first guess, training, final model, a
prediction on an unseen example, a final exam on the test set, a 3-D view, more features, takeaway.

- ``director`` plays the chapters, ``stage`` owns the 16:9 canvas, ``encoder`` pipes its frames
  into ffmpeg, and ``timing`` and ``config`` set the pacing.
- ``lesson`` is the contract: the words of the video, and a worked prediction checked against the
  model.
- ``diagrams`` and ``model_diagrams``, ``surface``, ``plots``, ``exam``, ``evidence`` and ``fitting``
  are the chapter parts every project shares; ``fit`` checks their layout in tests.
- ``cli`` is the command line every lesson project shares.

A project supplies a ``Scene`` (what its plots look like) and the lessons themselves.
"""
