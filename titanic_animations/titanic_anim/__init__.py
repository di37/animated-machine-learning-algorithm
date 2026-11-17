"""Live training and inference animations for scikit-learn classifiers on the Titanic passenger list.

The package runs real training and shows it as it happens. ``data`` loads and splits the
passengers; ``models`` says how each classifier trains step by step; ``trainers`` and
``inference`` do that work and yield an immutable frame after every real step. The dashboards draw
each frame the moment it arrives, and animkit (the shared engine in ``ml_animation_kit``) shows it
in a live window and records it to video. ``lessons`` holds the narrated lesson videos.
"""
