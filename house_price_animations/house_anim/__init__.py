"""Live training and inference animations for scikit-learn regressors on California Housing.

Each frame is drawn right after a real piece of training work (a ``partial_fit`` chunk, more trees,
one level deeper, more training rows), then again while the trained model predicts the held-out
test houses batch by batch.

The modules follow the data through the program:

- ``data`` loads the dataset and makes scaled, read-only train/validation/test splits.
- ``models`` lists the nine regressors and the schedule each one trains on.
- ``trainers`` and ``inference`` are loops that yield one immutable snapshot (``frames``) per step.
- ``training_dashboard`` and ``inference_dashboard`` draw a snapshot, with the algorithm diagrams
  in ``algorithms`` and the live equation cards in ``equations``.
- ``race`` and ``map_scene`` (with their dashboards) train several models side by side.
- ``runner``, ``scenes`` and ``cli`` record the videos and run the command line.

The drawing engine (recorder, panels, tree and network diagrams, lockstep) is the shared ``animkit``
package. ``lessons`` holds the separate narrated lesson videos.
"""
