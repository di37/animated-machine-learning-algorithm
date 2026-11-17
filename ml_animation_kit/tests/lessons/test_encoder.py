import shutil

import pytest

from animkit.lessons import encoder
from animkit.lessons.encoder import EncoderError, Mp4Encoder, probe

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")


def solid(width: int, height: int, value: int) -> bytes:
    return bytes((value, value, value, 255)) * (width * height)


@needs_ffmpeg
def test_encoder_writes_h264_yuv420p_with_exact_frame_count(tmp_path):
    path = tmp_path / "nested" / "clip.mp4"
    with Mp4Encoder(path, 64, 36, fps=10) as video:
        video.write(solid(64, 36, 10))
        video.write(solid(64, 36, 200), repeat=4)
    info = probe(path)
    assert (info.codec, info.pix_fmt, info.width, info.height) == ("h264", "yuv420p", 64, 36)
    assert info.frames == 5 == video.frames_written
    assert info.fps == pytest.approx(10.0)
    assert info.duration_s == pytest.approx(0.5, abs=0.05)


@needs_ffmpeg
def test_wrong_frame_size_fails_and_leaves_no_file(tmp_path):
    path = tmp_path / "clip.mp4"
    with pytest.raises(ValueError, match="bytes"):
        with Mp4Encoder(path, 64, 36, fps=10) as video:
            video.write(solid(32, 36, 10))
    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
def test_an_error_mid_render_removes_the_partial_video(tmp_path):
    path = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        with Mp4Encoder(path, 64, 36, fps=10) as video:
            video.write(solid(64, 36, 10))
            raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
def test_write_rejects_a_non_positive_repeat(tmp_path):
    with pytest.raises(ValueError, match="repeat"):
        with Mp4Encoder(tmp_path / "clip.mp4", 64, 36, fps=10) as video:
            video.write(solid(64, 36, 10), repeat=0)


def test_encoder_requires_even_dimensions(tmp_path):
    with pytest.raises(ValueError, match="even"):
        Mp4Encoder(tmp_path / "clip.mp4", 63, 36, fps=10)


def test_encoder_reports_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(encoder.shutil, "which", lambda name: None)
    with pytest.raises(EncoderError, match="ffmpeg"):
        with Mp4Encoder(tmp_path / "clip.mp4", 64, 36, fps=10):
            pass


@needs_ffmpeg
def test_probe_rejects_a_missing_file(tmp_path):
    with pytest.raises(EncoderError, match="ffprobe"):
        probe(tmp_path / "missing.mp4")


def fake_ffmpeg(tmp_path, body: str) -> str:
    script = tmp_path / "fake-ffmpeg"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize("body", ["echo broken codec >&2\nexit 1\n",
                                  "cat > /dev/null\necho broken codec >&2\nexit 1\n"])
def test_a_failing_ffmpeg_raises_with_its_message_and_leaves_no_file(tmp_path, monkeypatch, body):
    monkeypatch.setattr(encoder.shutil, "which", lambda name: fake_ffmpeg(tmp_path, body))
    out = tmp_path / "out"
    with pytest.raises(EncoderError, match="broken codec"):
        with Mp4Encoder(out / "clip.mp4", 64, 36, fps=10) as video:
            video.write(solid(64, 36, 10))
    assert list(out.iterdir()) == []
