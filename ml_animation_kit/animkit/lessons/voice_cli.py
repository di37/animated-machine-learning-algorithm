"""The voice-over command every lesson project shares: script, speak and dub each chosen lesson.

For each lesson it builds the narration from the lesson's own captions, asks the engine for each line
(the cache answers any line it has heard before), and lays the clips over the rendered silent video,
holding a chapter's last frame wherever its voice runs long. Next to each voiced video it writes
``<key>_lesson.narration.json``: what each chapter shows, what the voice says, and when.

``--script-only`` writes just the narration, with no voice and no cost, to read before paying for any.
A lesson that cannot be voiced is reported and the rest go on; a voice-service failure (no key, no
credit, no connection) stops the run, since every later lesson would fail the same way.
"""

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from animkit.cli import silence_spurious_matmul_warnings
from animkit.lessons.cli import FAILED, INTERRUPTED, LessonProject
from animkit.lessons.director import prepare_extras
from animkit.lessons.lesson import LessonError
from animkit.lessons.narration import Cue, narration_script
from animkit.lessons.speech_text import unexpanded_acronyms
from animkit.lessons.timing import CHAPTERS, Timing
from animkit.lessons.tts import (ElevenLabsSpeech, OpenAISpeech, SaySpeech, ServiceError, SpeechCache,
                                 SpeechEngine, TtsError, audio_seconds)
from animkit.lessons.voiceover import DubError, Slot, dub, plan_voiceover

VOICED_DIR = "lessons_voiced"  # next to the silent lessons folder
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "animkit" / "tts"
WORDS_PER_SECOND = 2.6  # for the script-only estimate
BYTES_PER_MB = 1e6
LESSON_ERRORS = (DubError, LessonError, TtsError, OSError, ValueError)  # these fail one lesson; ServiceError, all

EngineFactory = Callable[[str | None], SpeechEngine]
ENGINES: dict[str, EngineFactory] = {  # the first is the default
    "elevenlabs": lambda voice: ElevenLabsSpeech(voice=voice) if voice else ElevenLabsSpeech(),
    "openai": lambda voice: OpenAISpeech(voice=voice) if voice else OpenAISpeech(),
    "say": lambda voice: SaySpeech(voice=voice) if voice else SaySpeech(),
}


def build_parser(project: LessonProject, engines: Mapping[str, EngineFactory]) -> argparse.ArgumentParser:
    """The voice-over options, with the project's lessons and its silent lessons folder.

    Args:
        project: The lesson project.
        engines: Engine name: makes the engine for a voice name (or None for its default voice).

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=f"Voice the lessons. {project.description}")
    parser.add_argument("--model", default="all", choices=(*project.lesson_keys(), "all"),
                        help="which lesson to voice (default: all)")
    parser.add_argument("--engine", default=next(iter(engines)), choices=tuple(engines),
                        help="elevenlabs (online, needs ELEVENLABS_API_KEY; the default), openai (online, needs "
                             "OPENAI_API_KEY) or say (macOS, offline and free)")
    parser.add_argument("--voice", default=None,
                        help="the engine's voice (elevenlabs: a voice ID; openai: marin, cedar, coral, ...)")
    parser.add_argument("--source-dir", type=Path, default=project.parser().get_default("output_dir"),
                        help="the rendered, silent lessons (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"where the voiced lessons go (default: {VOICED_DIR}/ next to the source folder)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="where voice clips are kept (default: %(default)s)")
    parser.add_argument("--script-only", action="store_true",
                        help="write the narration only: no voice, no cost")
    return parser


def _cue_rows(cues: tuple[Cue, ...], slots: tuple[Slot, ...] | None, fps: int) -> list[dict[str, Any]]:
    """The narration file's chapter rows; with the plan, when each chapter and its voice start."""
    rows = [{"chapter": cue.chapter, "screen": cue.screen, "spoken": cue.spoken} for cue in cues]
    if slots is None:
        return rows
    return [{**row, "start_s": round(slot.start / fps, 3), "voice_at_s": round(slot.voice_at, 3),
             "voice_s": round(slot.voice_seconds, 3), "hold_s": round(slot.hold / fps, 3)}
            for row, slot in zip(rows, slots)]


def _write_narration(path: Path, record: dict[str, Any]) -> None:
    """Writes the narration file, as readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _unexplained(cues: tuple[Cue, ...]) -> tuple[str, ...]:
    """The abbreviations the narration would spell out, having no full name to say."""
    return tuple(sorted({acronym for cue in cues for acronym in unexpanded_acronyms(cue.spoken)}))


def _script_only(key: str, cues: tuple[Cue, ...], output_dir: Path, fps: int) -> None:
    """Writes the narration of one lesson and prints how long it would take to say."""
    words = sum(len(cue.spoken.split()) for cue in cues)
    _write_narration(output_dir / f"{key}_lesson.narration.json",
                     {"lesson": key, "engine": None, "chapters": _cue_rows(cues, None, fps)})
    print(f"    {words} words, about {words / WORDS_PER_SECOND:.0f} s of speech", flush=True)
    if unknown := _unexplained(cues):
        print(f"    no full name for {', '.join(unknown)}: this lesson would not be voiced", flush=True)


def _voice_one(key: str, cues: tuple[Cue, ...], args: argparse.Namespace, cache: SpeechCache,
               output_dir: Path, timing: Timing) -> None:
    """Speaks one lesson's narration and dubs its silent video."""
    source = args.source_dir / f"{key}_lesson.mp4"
    out = output_dir / f"{key}_lesson.mp4"
    # checked before any voice is paid for: a lesson that cannot be dubbed costs nothing
    if out.resolve() == source.resolve():
        raise DubError(f"the voiced video would overwrite its silent source {source}; choose another --output-dir")
    if not source.is_file():
        raise DubError(f"source video not found: {source}")
    if unknown := _unexplained(cues):
        raise LessonError(f"no full name for {', '.join(unknown)}: add it to speech_text.JARGON (or the scene's "
                          f"spoken_phrases), so the voice says it in full instead of spelling it out")
    clips = {cue.chapter: cache.clip(cue.spoken) for cue in cues}
    seconds = {chapter: audio_seconds(path) for chapter, path in clips.items()}
    chapters = tuple((chapter, timing.frames(chapter)) for chapter in CHAPTERS)
    slots = plan_voiceover(chapters, seconds, fps=timing.fps)
    dub(source, slots, clips, out, fps=timing.fps)
    total = sum(slot.frames for slot in slots) / timing.fps
    held = sum(slot.hold for slot in slots) / timing.fps
    _write_narration(output_dir / f"{key}_lesson.narration.json",
                     {"lesson": key, "engine": cache.engine.identity, "source": str(source), "video": str(out),
                      "seconds": total, "chapters": _cue_rows(cues, slots, timing.fps)})
    print(f"    saved {out} · {total:.1f} s ({held:.1f} s held for the voice) · "
          f"{out.stat().st_size / BYTES_PER_MB:.1f} MB", flush=True)


def run(project: LessonProject, argv: list[str] | None = None, *, fetcher: Callable,
        engines: Mapping[str, EngineFactory] = ENGINES, timing: Timing = Timing(), label: str = "") -> int:
    """Voices the chosen lessons, one after another.

    Args:
        project: The lesson project.
        argv: The arguments, without the program name; None reads ``sys.argv``.
        fetcher: Loads the project's data, as rendering does (the captions' numbers come from it).
        engines: The engines ``--engine`` chooses from; tests pass a stand-in.
        timing: The chapter lengths the silent videos were rendered with.
        label: Added after each lesson's key on its progress line, for example " (Manim)".

    Returns:
        0 if every lesson was voiced, ``FAILED`` if the data, the voice service or a lesson failed,
        ``INTERRUPTED`` if stopped.
    """
    args = build_parser(project, engines).parse_args(argv)
    silence_spurious_matmul_warnings()
    output_dir = args.output_dir or args.source_dir.parent / VOICED_DIR
    cache = SpeechCache(engines[args.engine](args.voice), args.cache_dir)
    try:
        data = project.load_data(fetcher=fetcher, max_rows=None)  # all rows: the videos were rendered on all
    except project.data_errors as exc:
        print(f"error: {exc}", file=sys.stderr)
        return FAILED
    keys = project.lesson_keys() if args.model == "all" else (args.model,)
    failed: tuple[str, ...] = ()
    for index, key in enumerate(keys, start=1):
        print(f"[{index}/{len(keys)}] {key}{label} …", flush=True)
        try:
            lesson = project.build_lesson(key, data)
            cues = narration_script(lesson, project.scene, prepare_extras(lesson, project.scene))
            if args.script_only:
                _script_only(key, cues, output_dir, timing.fps)
            else:
                _voice_one(key, cues, args, cache, output_dir, timing)
        except KeyboardInterrupt:
            print("\nStopped; the unfinished video was discarded.", file=sys.stderr)
            return INTERRUPTED
        except ServiceError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return FAILED
        except LESSON_ERRORS as exc:
            print(f"error: {key}: {exc}", file=sys.stderr)
            failed = (*failed, key)
    if failed:
        print(f"{len(failed)} of {len(keys)} lessons were not voiced: {', '.join(failed)}", file=sys.stderr)
        return FAILED
    return 0
