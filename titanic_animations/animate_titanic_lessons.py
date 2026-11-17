"""Entry script for the narrated lesson videos: ``python animate_titanic_lessons.py --model decision_tree``.

It puts the shared engine, animkit in the sibling ``ml_animation_kit`` folder, on ``sys.path``
first, because the lessons are rendered by animkit's lesson director. Then it runs
``titanic_anim.lessons.cli`` and exits with its status. ``python -m titanic_anim.lessons`` does the
same, and ``--help`` lists every option.

Example:
    From the ``titanic_animations`` folder::

        python animate_titanic_lessons.py --list              # the nine lessons
        python animate_titanic_lessons.py --resolution preview
"""

import sys

from titanic_anim.kit_path import add_kit_to_path

add_kit_to_path()

from titanic_anim.lessons.cli import main  # noqa: E402  (animkit must be on sys.path first)

if __name__ == "__main__":
    sys.exit(main())
