"""Dataset-neutral engine for scikit-learn training and inference animations.

A project brings its own data and its own task-specific drawing; animkit brings everything that stays
the same whatever the data is. It has two halves:

- The live dashboards. A ``ModelSpec`` (``models``) says how one estimator is trained step by step.
  The project's training loop yields a frame per step; ``runner.play`` draws each frame into a
  dashboard the moment it exists, and the frame sinks (``recorder``) show it in a window, append it
  to a video file, or both. ``lockstep`` advances several models together for a race. The pieces the
  dashboards share live here too: the theme, the header and progress bar, the weight bars, the
  equation cards and their panel, and the algorithm diagrams (``animkit.algorithms``).
- The narrated lessons (``animkit.lessons``): one ten-chapter video per algorithm, drawn off screen
  and piped straight into ffmpeg.

Nothing here knows which dataset it is drawing: names, units and wording all come from the project.
"""
