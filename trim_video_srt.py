"""
Trim a video and its matching SRT subtitle file to a time range, then
re-base subtitle timestamps to start at 00:00:00 for the new clip.

Requires: moviepy, pysrt
See requirements.txt (optional section for this script).
"""
from __future__ import annotations

import sys
from pathlib import Path
# -----------------------------------------------------------------------------
# --- Configuration: edit these values for each trim job ----------------------
# -----------------------------------------------------------------------------

# Source video in ./data/ (e.g. "lecture_part1.mp4")
VIDEO_FILENAME: str = "my_video.mp4"

# Time window on the *original* video timeline, aligned with MoviePy: [START, END)
#   (start included, end exclusive — the same half-open range used for SRT overlap).
# Use "HH:MM:SS", "H:MM:SS", "MM:SS", "M:SS", or optional ",mmm" milliseconds.
START_TIME: str = "00:00:10"
END_TIME: str = "00:01:30"

# Project paths (relative to this script)
DATA_DIR: Path = Path(__file__).resolve().parent / "data"
EXPORTS_DIR: Path = Path(__file__).resolve().parent / "exports"
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"

# Optional: video encode settings (passed to write_videofile when supported)
VIDEO_CODEC: str = "libx264"
AUDIO_CODEC: str = "aac"
OUTPUT_VIDEO_SUFFIX: str = ""  # e.g. "_trim" if you do not want to overwrite by name

# -----------------------------------------------------------------------------


def parse_time_to_seconds(s: str) -> float:
    """
    Parse a time string to seconds (float).
    Accepts: "SS", "M:SS", "MM:SS", "H:MM:SS", "HH:MM:SS" (and optional ,mmm ms).
    """
    raw = s.strip()
    if not raw:
        raise ValueError("Empty time string")

    if "," in raw:
        raw, ms_part = raw.split(",", 1)
        ms = float("0." + ms_part) if ms_part.isdigit() else int(ms_part) / 1000.0
    else:
        ms = 0.0

    parts = raw.split(":")
    if len(parts) == 1:
        return float(parts[0]) + ms
    if len(parts) == 2:
        m, sec = int(parts[0]), float(parts[1])
        return m * 60.0 + sec + ms
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600.0 + m * 60.0 + sec + ms
    raise ValueError(f"Unrecognized time format: {s!r}")


def _import_video_file_clip():
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:  # MoviePy 2.x removed moviepy.editor
        from moviepy import VideoFileClip
    return VideoFileClip


def trim_video(
    input_path: Path,
    output_path: Path,
    t_start: float,
    t_end: float,
    video_codec: str = VIDEO_CODEC,
    audio_codec: str = AUDIO_CODEC,
) -> None:
    """
    Load video, cut [t_start, t_end] in seconds, write to output_path.
    Works with MoviePy 1.x (subclip) and 2.x (with_subclip).
    """
    VideoFileClip = _import_video_file_clip()
    clip = VideoFileClip(str(input_path))
    sub = None
    try:
        if hasattr(clip, "subclip"):
            sub = clip.subclip(t_start, t_end)
        else:
            sub = clip.with_subclip(t_start, t_end)
        sub.write_videofile(
            str(output_path),
            codec=video_codec,
            audio_codec=audio_codec,
        )
    finally:
        if sub is not None:
            try:
                sub.close()
            except Exception:
                pass
        try:
            clip.close()
        except Exception:
            pass


def _seconds_to_pysrt_time(total_seconds: float):
    import pysrt

    if total_seconds < 0:
        total_seconds = 0.0
    total_ms = int(round(total_seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = int(total_s % 60)
    total_m = total_s // 60
    m = int(total_m % 60)
    h = int(total_m // 60)
    return pysrt.SubRipTime(h, m, s, ms)


def _subrip_time_to_seconds(t) -> float:
    return (
        t.hours * 3600.0
        + t.minutes * 60.0
        + t.seconds
        + t.milliseconds / 1000.0
    )


def trim_and_resync_srt(
    srt_path: Path,
    out_path: Path,
    range_start: float,
    range_end: float,
    offset_seconds: float,
    encoding: str = "utf-8",
) -> int:
    """
    Keep only cues that overlap the half-open interval [range_start, range_end) on
    the *source* timeline (matches MoviePy subclip / with_subclip semantics).
    Cues are clipped to that window, then shifted by -offset_seconds so t=0
    is the new clip start. Returns number of subtitle blocks written.
    """
    import pysrt

    subs = pysrt.open(str(srt_path), encoding=encoding)
    new_file = pysrt.SubRipFile()
    index = 1

    for item in subs:
        s = _subrip_time_to_seconds(item.start)
        e = _subrip_time_to_seconds(item.end)
        if e <= range_start or s >= range_end:
            continue
        new_s = max(s, range_start)
        new_e = min(e, range_end)
        if new_e <= new_s:
            continue
        out_s = new_s - offset_seconds
        out_e = new_e - offset_seconds
        new_item = pysrt.SubRipItem(
            index,
            _seconds_to_pysrt_time(out_s),
            _seconds_to_pysrt_time(out_e),
            item.text,
        )
        new_file.append(new_item)
        index += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file.save(str(out_path), encoding=encoding)
    return len(new_file)


def _resolve_paths() -> tuple[Path, Path, Path, Path]:
    """Return input video, input SRT, output video, output SRT paths."""
    stem = Path(VIDEO_FILENAME).stem
    ext = Path(VIDEO_FILENAME).suffix
    video_in = DATA_DIR / VIDEO_FILENAME
    srt_in = EXPORTS_DIR / f"{stem}.srt"
    out_name = f"{stem}{OUTPUT_VIDEO_SUFFIX}{ext}"
    video_out = OUTPUT_DIR / out_name
    srt_out = OUTPUT_DIR / f"{Path(out_name).stem}.srt"
    return video_in, srt_in, video_out, srt_out


def main() -> int:
    try:
        t0 = parse_time_to_seconds(START_TIME)
        t1 = parse_time_to_seconds(END_TIME)
    except ValueError as e:
        print(f"Invalid time configuration: {e}", file=sys.stderr)
        return 1

    if t1 <= t0:
        print("END_TIME must be greater than START_TIME.", file=sys.stderr)
        return 1

    video_in, srt_in, video_out, srt_out = _resolve_paths()

    if not video_in.is_file():
        print(f"Video not found: {video_in}", file=sys.stderr)
        return 1
    if not srt_in.is_file():
        print(f"SRT not found: {srt_in}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Trimming video: {video_in} -> {video_out} [{t0:.3f}s, {t1:.3f}s]")
    try:
        trim_video(video_in, video_out, t0, t1)
    except Exception as e:
        print(f"Video trim failed: {e}", file=sys.stderr)
        return 1

    try:
        n = trim_and_resync_srt(srt_in, srt_out, t0, t1, t0)
    except Exception as e:
        print(f"SRT processing failed: {e}", file=sys.stderr)
        return 1

    print(f"Wrote SRT: {srt_out} ({n} blocks)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
