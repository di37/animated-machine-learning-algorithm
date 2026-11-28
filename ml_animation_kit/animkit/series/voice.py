"""The voice-over for series videos: each chapter's words spoken and dubbed over the rendered video.

It is the lessons' voice-over (``animkit.lessons.voice_cli``) applied to a series: the words come from
the chapters file written beside the silent video, are said as the voice should say them (symbols as
words, every abbreviation by its full name; a chapter with an abbreviation that has no full name is
refused before anything is paid for), and a chapter's last frame is held wherever its voice runs long.
The narration file beside the voiced video has the lessons' layout, so the YouTube pack reads it too.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from animkit.lessons.cli import FAILED, INTERRUPTED
from animkit.lessons.lesson import LessonError
from animkit.lessons.narration import Cue
from animkit.lessons.speech_text import spoken
from animkit.lessons.tts import ServiceError, SpeechCache, TtsError, audio_seconds
from animkit.lessons.voice_cli import (BYTES_PER_MB, DEFAULT_CACHE_DIR, ENGINES, LESSON_ERRORS, EngineFactory,
                                       cue_rows, unexplained, write_narration)
from animkit.lessons.voiceover import DubError, dub, plan_voiceover
from animkit.series.chapters import read_chapters
from animkit.series.render import CHAPTERS_SUFFIX, chapters_path

VOICED_DIR = "series_voiced"  # next to the silent series folder
NARRATION_SUFFIX = ".narration.json"


def series_cues(record: Mapping[str, Any], phrases: Sequence[tuple[str, str]] = ()) -> tuple[Cue, ...]:
    """Each chapter's words, as the voice should say them.

    Args:
        record: A chapters file's record.
        phrases: The project's own (written, spoken) phrases, tried first.

    Returns:
        One cue per chapter: its key, its label (as the screen names it) and the words to say.
    """
    return tuple(Cue(row["chapter"], row["label"], spoken(row["narration"], phrases)) for row in record["chapters"])


def voice_series(source: Path, out: Path, cache: SpeechCache,
                 phrases: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """Speaks a series video's chapters and dubs them over it.

    Args:
        source: The silent series video; its chapters file must be beside it.
        out: The voiced video to write.
        cache: Makes each line's clip, or reads it back.
        phrases: The project's own (written, spoken) phrases.

    Returns:
        The narration record, also written beside ``out``.

    Raises:
        DubError: If ``out`` would replace ``source``, or the source is missing.
        LessonError: If the words hold an abbreviation with no full name (nothing is spoken then).
        TtsError: If a clip cannot be made.
    """
    source, out = Path(source), Path(out)
    if out.resolve() == source.resolve():
        raise DubError(f"the voiced video would overwrite its silent source {source}")
    if not source.is_file():
        raise DubError(f"source video not found: {source}")
    record = read_chapters(chapters_path(source))
    cues = series_cues(record, phrases)
    if unknown := unexplained(cues):
        raise LessonError(f"no full name for {', '.join(unknown)}: add it to speech_text.JARGON, so the voice "
                          f"says it in full instead of spelling it out")
    fps = record["fps"]
    clips = {cue.chapter: cache.clip(cue.spoken) for cue in cues}
    seconds = {chapter: audio_seconds(path) for chapter, path in clips.items()}
    slots = plan_voiceover(tuple((row["chapter"], row["frames"]) for row in record["chapters"]), seconds, fps=fps)
    dub(source, slots, clips, out, fps=fps)
    narration = {"series": record["series"], "title": record.get("title", ""), "engine": cache.engine.identity,
                 "source": str(source), "video": str(out), "seconds": sum(slot.frames for slot in slots) / fps,
                 "chapters": cue_rows(cues, slots, fps)}
    write_narration(out.with_suffix(NARRATION_SUFFIX), narration)
    return narration


def build_parser(engines: Mapping[str, EngineFactory]) -> argparse.ArgumentParser:
    """The series voice-over's options."""
    parser = argparse.ArgumentParser(description="Voice the series videos (introduction, conclusion).")
    parser.add_argument("--source-dir", type=Path, required=True, help="the rendered, silent series videos")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"where the voiced videos go (default: {VOICED_DIR}/ next to the source folder)")
    parser.add_argument("--engine", default=next(iter(engines)), choices=tuple(engines),
                        help="the voice engine (default: %(default)s)")
    parser.add_argument("--voice", default=None, help="the engine's voice")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="where voice clips are kept (default: %(default)s)")
    return parser


def run_series_voice(argv: list[str] | None = None, *, engines: Mapping[str, EngineFactory] = ENGINES,
                     phrases: Sequence[tuple[str, str]] = ()) -> int:
    """Voices every series video in the source folder, one after another.

    Args:
        argv: The arguments, without the program name; None reads ``sys.argv``.
        engines: The engines ``--engine`` chooses from; tests pass a stand-in.
        phrases: The project's own (written, spoken) phrases.

    Returns:
        0 if every video was voiced, ``FAILED`` if none was found, the voice service failed or a video
        failed, ``INTERRUPTED`` if stopped.
    """
    args = build_parser(engines).parse_args(argv)
    videos = sorted(path.with_name(path.name.removesuffix(CHAPTERS_SUFFIX) + ".mp4")
                    for path in Path(args.source_dir).glob(f"*{CHAPTERS_SUFFIX}"))
    if not videos:
        print(f"error: no series video (with its {CHAPTERS_SUFFIX}) in {args.source_dir}", file=sys.stderr)
        return FAILED
    output_dir = args.output_dir or Path(args.source_dir).parent / VOICED_DIR
    cache = SpeechCache(engines[args.engine](args.voice), args.cache_dir)
    failed = 0
    for video in videos:
        print(f"{video.stem} …", flush=True)
        try:
            record = voice_series(video, output_dir / video.name, cache, phrases)
        except KeyboardInterrupt:
            print("\nStopped; the unfinished video was discarded.", file=sys.stderr)
            return INTERRUPTED
        except ServiceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return FAILED
        except (*LESSON_ERRORS, TtsError) as exc:
            print(f"error: {video.stem}: {exc}", file=sys.stderr)
            failed += 1
            continue
        out = Path(record["video"])
        held = sum(row["hold_s"] for row in record["chapters"])
        print(f"    saved {out} · {record['seconds']:.1f} s ({held:.1f} s held for the voice) · "
              f"{out.stat().st_size / BYTES_PER_MB:.1f} MB", flush=True)
    return FAILED if failed else 0
