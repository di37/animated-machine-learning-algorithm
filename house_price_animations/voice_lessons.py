"""Voice the narrated lessons: each chapter's caption, spoken, laid over the rendered silent video.

Run (from this folder):
    python voice_lessons.py --script-only                  # read what the voice will say; free
    python voice_lessons.py --model linear_regression      # voice one lesson with OpenAI (needs OPENAI_API_KEY)
    python voice_lessons.py --engine say                   # every lesson with macOS's offline voice

The voiced videos go to outputs/california_housing/lessons_voiced/; see --help for every option.
"""

import sys

from house_anim.kit_path import add_kit_to_path

add_kit_to_path()

from sklearn.datasets import fetch_california_housing  # noqa: E402

from animkit.lessons import voice_cli  # noqa: E402  (animkit must be on sys.path first)
from house_anim.lessons.cli import project  # noqa: E402

if __name__ == "__main__":
    sys.exit(voice_cli.run(project(), fetcher=fetch_california_housing))
