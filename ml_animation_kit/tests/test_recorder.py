import matplotlib.pyplot as plt
import pytest
from matplotlib import animation
from PIL import Image

from animkit.recorder import LiveSink, RenderingStopped, SinkGroup, VideoSink

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


def tiny_figure():
    figure = plt.figure(figsize=(2.0, 1.5))
    figure.add_subplot().plot([0, 1], [0, 1])
    return figure


def test_gif_sink_writes_repeated_frames(tmp_path):
    figure = tiny_figure()
    path = tmp_path / "nested" / "clip.gif"
    sink = VideoSink(figure, path, fps=5, dpi=40)
    sink.emit()
    sink.emit(repeat=3)
    sink.close()
    with Image.open(path) as gif:
        assert gif.n_frames >= 1
    assert sink.frames_written == 4


@pytest.mark.skipif(not animation.writers.is_available("ffmpeg"), reason="ffmpeg not installed")
def test_mp4_sink_writes_file(tmp_path):
    figure = tiny_figure()
    path = tmp_path / "clip.mp4"
    sink = VideoSink(figure, path, fps=5, dpi=40)
    sink.emit(repeat=2)
    sink.close()
    assert path.stat().st_size > 0


@pytest.mark.skipif(not animation.writers.is_available("ffmpeg"), reason="ffmpeg not installed")
def test_mp4_sink_handles_odd_pixel_dimensions(tmp_path):
    figure = plt.figure(figsize=(15.0, 8.4))  # the inference figure; 15 x 33 dpi = 495 px wide
    figure.add_subplot().plot([0, 1], [0, 1])
    path = tmp_path / "odd.mp4"
    sink = VideoSink(figure, path, fps=5, dpi=33)
    sink.emit(repeat=2)
    sink.close()
    assert path.stat().st_size > 0


def test_video_sink_survives_figure_resize(tmp_path):
    figure = tiny_figure()
    sink = VideoSink(figure, tmp_path / "clip.gif", fps=5, dpi=40)
    sink.emit()
    figure.set_size_inches(4.0, 4.0)
    sink.emit()
    sink.close()
    assert sink.frames_written == 2


def test_hold_converts_seconds_to_frames(tmp_path):
    sink = VideoSink(tiny_figure(), tmp_path / "clip.gif", fps=4, dpi=40)
    sink.hold(seconds=0.5)
    sink.close()
    assert sink.frames_written == 2


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError, match=".avi"):
        VideoSink(tiny_figure(), tmp_path / "clip.avi", fps=5, dpi=40)


def test_invalid_repeat_raises(tmp_path):
    sink = VideoSink(tiny_figure(), tmp_path / "clip.gif", fps=5, dpi=40)
    with pytest.raises(ValueError, match="repeat"):
        sink.emit(repeat=0)
    sink.close()
    assert not sink.path.exists()  # no frames -> no half-written file


def test_live_sink_stops_when_window_closed():
    figure = tiny_figure()
    sink = LiveSink(figure, pause_s=0.0)
    sink.emit()
    plt.close(figure)
    with pytest.raises(RenderingStopped):
        sink.emit()


class RecordingSink:
    def __init__(self, fail_on_close: bool = False):
        self.emitted, self.held, self.closed = [], [], False
        self.fail_on_close = fail_on_close

    def emit(self, repeat: int = 1) -> None:
        self.emitted.append(repeat)

    def hold(self, seconds: float) -> None:
        self.held.append(seconds)

    def close(self) -> None:
        self.closed = True
        if self.fail_on_close:
            raise RuntimeError("writer crashed")


def test_group_fans_out_and_closes_every_sink_even_on_failure():
    broken, healthy = RecordingSink(fail_on_close=True), RecordingSink()
    group = SinkGroup((broken, healthy))
    group.emit(repeat=2)
    group.hold(1.5)
    with pytest.raises(RuntimeError, match="writer crashed"):
        group.close()
    assert broken.emitted == healthy.emitted == [2]
    assert broken.held == healthy.held == [1.5]
    assert broken.closed and healthy.closed


def test_group_works_as_context_manager():
    sink = RecordingSink()
    with SinkGroup((sink,)) as group:
        group.emit()
    assert sink.closed


def test_live_windows_shrink_to_fit_the_screen_but_videos_keep_full_size(tmp_path):
    from animkit.recorder import fit_to_screen

    figure = plt.figure(figsize=(25.6, 14.4), dpi=100)
    fit_to_screen(figure, max_width_px=1500)
    assert figure.get_dpi() * 25.6 == pytest.approx(1500)
    small = plt.figure(figsize=(4, 3), dpi=100)
    fit_to_screen(small, max_width_px=1500)
    assert small.get_dpi() == 100

    figure.add_subplot().plot([0, 1], [0, 1])
    sink = VideoSink(figure, tmp_path / "big.gif", fps=2, dpi=20)
    sink.emit()
    sink.close()
    with Image.open(tmp_path / "big.gif") as gif:
        assert gif.size == (512, 288)
