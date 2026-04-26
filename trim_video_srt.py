"""
Trim a video or audio file and its matching SRT subtitle file to a time range,
then re-base subtitle timestamps to start at 00:00:00 for the new clip.

With parts/parts.json: processes every part by id (e.g. all five); each output is
``output/{source_basename}/`` + file named only from the part's JSON ``title``
(sanitized) and the media extension. Existing project folders are reused. Use
``--part N`` to run a single part only.

If no parts config is found, SOURCE_FILENAME / START_TIME / END_TIME in this file apply.

Requires: moviepy, pysrt
See requirements.txt (optional section for this script).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# --- Configuration -----------------------------------------------------------
# If parts/parts.json or parts.json is present, all parts are processed unless --part
# is given. Otherwise these defaults apply.
# -----------------------------------------------------------------------------

# Source file in ./data/ — video (e.g. .mp4) or audio (e.g. .m4a, .mp3, .wav)
SOURCE_FILENAME: str = "How_To_Stop_Revenge_Bedtime_Procrastination.m4a"

# Time window on the *original* media timeline, aligned with MoviePy: [START, END)
#   (start included, end exclusive — the same half-open range used for SRT overlap).
# Use "HH:MM:SS", "H:MM:SS", "MM:SS", "M:SS", or optional ",mmm" milliseconds.
START_TIME: str = "00:04:11"
END_TIME: str = "00:04:36"

# Project paths (relative to this script)
DATA_DIR: Path = Path(__file__).resolve().parent / "data"
EXPORTS_DIR: Path = Path(__file__).resolve().parent / "exports"
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"

# Optional: encode settings
VIDEO_CODEC: str = "libx264"  # used for video output
AUDIO_CODEC: str = "aac"  # used for both video audio track and audio-only export
OUTPUT_FILE_SUFFIX: str = ""  # e.g. "_trim" (legacy only)

# -----------------------------------------------------------------------------

# Extensions treated as audio-only (AudioFileClip + write_audiofile)
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
)


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


def _sanitize_title_for_filename(title: str, max_len: int = 100) -> str:
    """
    Build a single path segment from JSON `title` (spaces → underscores, strip illegal chars).
    Empty after sanitization → "".
    """
    s = title.strip()
    for c in '\\/:*?"<>|':
        s = s.replace(c, "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def _prepare_output_subdir_for_source(source_filename: str) -> Path:
    """
    ``output/{source_stem}/`` for all trims from that source.
    Reuses an existing directory; creates it if missing. If a *file* already
    exists at that path, raises ValueError.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_filename).stem
    d = OUTPUT_DIR / stem
    if d.is_file():
        raise ValueError(
            f"Cannot use project folder {d!s}: a file exists with that name; "
            f"remove or rename it, or pick another media filename."
        )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _import_video_file_clip():
    try:
        from moviepy.editor import VideoFileClip  # type: ignore[import-not-found]
    except ImportError:  # MoviePy 2.x removed moviepy.editor
        from moviepy import VideoFileClip
    return VideoFileClip


def _import_audio_file_clip():
    try:
        from moviepy.editor import AudioFileClip  # type: ignore[import-not-found]
    except ImportError:
        from moviepy import AudioFileClip
    return AudioFileClip


def _subclip(clip, t_start: float, t_end: float):
    """MoviePy 1.x: subclip; 2.x: subclipped; some builds: with_subclip."""
    if hasattr(clip, "subclip"):
        return clip.subclip(t_start, t_end)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(t_start, t_end)
    if hasattr(clip, "with_subclip"):
        return clip.with_subclip(t_start, t_end)
    raise TypeError("Clip does not support subclip / subclipped / with_subclip")


def trim_media(
    input_path: Path,
    output_path: Path,
    t_start: float,
    t_end: float,
    *,
    audio_only: bool,
    video_codec: str = VIDEO_CODEC,
    audio_codec: str = AUDIO_CODEC,
) -> None:
    """
    Load media, cut [t_start, t_end) in seconds, write to output_path.
    Uses AudioFileClip for audio-only extensions, else VideoFileClip.
    """
    if audio_only:
        AudioFileClip = _import_audio_file_clip()
        clip = AudioFileClip(str(input_path))
    else:
        VideoFileClip = _import_video_file_clip()
        clip = VideoFileClip(str(input_path))

    sub = None
    try:
        sub = _subclip(clip, t_start, t_end)
        if audio_only:
            sub.write_audiofile(str(output_path), codec=audio_codec)
        else:
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


def _resolve_paths(
    source_filename: str,
    output_file_suffix: str = "",
    *,
    output_dir: Path,
    title_stem: str | None = None,
    title_only: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """
    Return input media, input SRT, output media, output SRT paths.
    Outputs are written under ``output_dir`` (e.g. ``output/{source_stem}/``).
    * Legacy: ``{source_stem}{suffix}{ext}`` when title_only is false and title_stem is None.
    * Parts JSON: ``title_only`` true — output is ``{title_stem}{suffix}{ext}`` (title from JSON only).
    """
    stem = Path(source_filename).stem
    ext = Path(source_filename).suffix
    media_in = DATA_DIR / source_filename
    srt_in = EXPORTS_DIR / f"{stem}.srt"
    if title_only and title_stem:
        out_stem = title_stem
    elif title_stem:
        out_stem = f"{stem}_{title_stem}"
    else:
        out_stem = stem
    out_name = f"{out_stem}{output_file_suffix}{ext}"
    media_out = output_dir / out_name
    srt_out = output_dir / f"{Path(out_name).stem}.srt"
    return media_in, srt_in, media_out, srt_out


def load_parts_config_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _config_path_from_args(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent
    if args.config is not None:
        return args.config
    candidate = script_dir / "parts" / "parts.json"
    return candidate if candidate.is_file() else (script_dir / "parts.json")


def _process_one_part(
    *,
    source_filename: str,
    part: dict[str, Any],
    part_id: int,
    output_dir: Path,
) -> int:
    """Trim one part; output basename = sanitized title only. Returns 0 on success."""
    try:
        t0 = parse_time_to_seconds(str(part["time_start"]))
        t1 = parse_time_to_seconds(str(part["time_end"]))
    except (KeyError, ValueError) as e:
        print(f"Part {part_id}: bad time: {e}", file=sys.stderr)
        return 1
    if t1 <= t0:
        print(f"Part {part_id}: end time must be after start time.", file=sys.stderr)
        return 1

    title = str(part.get("title", ""))
    title_stem = _sanitize_title_for_filename(title) if title else ""
    if not title_stem:
        print(
            f"Part {part_id}: title required for output filename (empty after sanitize).",
            file=sys.stderr,
        )
        return 1

    out_suffix = str(part.get("output_suffix", ""))
    tag = part.get("tag", "")
    print(f"Part {part_id}: {title}" + (f"  ({tag})" if tag else ""))
    print(
        f"  Time: {part['time_start']} – {part['time_end']}  "
        f"[{t0:.3f}s, {t1:.3f}s)  (~{(t1 - t0) / 60.0:.2f} min)",
    )
    h = str(part.get("hook", ""))
    if h:
        short = h if len(h) <= 120 else h[:117] + "..."
        print(f"  Hook: {short}")

    media_in, srt_in, media_out, srt_out = _resolve_paths(
        source_filename,
        out_suffix,
        output_dir=output_dir,
        title_stem=title_stem,
        title_only=True,
    )
    ext_lower = Path(source_filename).suffix.lower()
    audio_only = ext_lower in AUDIO_EXTENSIONS

    kind = "audio" if audio_only else "video"
    print(
        f"Trimming {kind}: {media_in} -> {media_out} [{t0:.3f}s, {t1:.3f}s)",
    )
    try:
        trim_media(media_in, media_out, t0, t1, audio_only=audio_only)
    except Exception as e:
        print(f"Part {part_id} media trim failed: {e}", file=sys.stderr)
        return 1
    try:
        n = trim_and_resync_srt(srt_in, srt_out, t0, t1, t0)
    except Exception as e:
        print(f"Part {part_id} SRT failed: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {media_out} and {srt_out} ({n} subtitle blocks)")
    return 0


def _run_from_parts_config(config_path: Path, args: argparse.Namespace) -> int:
    data = load_parts_config_file(config_path)
    media = data.get("media") or {}
    source_filename = str(media.get("source_filename", SOURCE_FILENAME))
    stem = Path(source_filename).stem
    media_in = DATA_DIR / source_filename
    srt_in = EXPORTS_DIR / f"{stem}.srt"
    if not media_in.is_file():
        print(f"File not found: {media_in}", file=sys.stderr)
        return 1
    if not srt_in.is_file():
        print(f"SRT not found: {srt_in}", file=sys.stderr)
        return 1

    try:
        out_dir = _prepare_output_subdir_for_source(source_filename)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    all_parts: list[dict[str, Any]] = sorted(
        data.get("parts", []), key=lambda p: int(p.get("id", 0))
    )
    if not all_parts:
        print("No parts in config.", file=sys.stderr)
        return 1
    if args.part is not None:
        to_run = [p for p in all_parts if int(p.get("id", 0)) == int(args.part)]
        if not to_run:
            print(f"Unknown part id: {args.part} in {config_path}", file=sys.stderr)
            return 1
    else:
        to_run = all_parts

    for part in to_run:
        part_id = int(part.get("id", 0))
        rc = _process_one_part(
            source_filename=source_filename,
            part=part,
            part_id=part_id,
            output_dir=out_dir,
        )
        if rc != 0:
            return rc
    print("Done.")
    return 0


def _run_legacy() -> int:
    try:
        t0 = parse_time_to_seconds(START_TIME)
        t1 = parse_time_to_seconds(END_TIME)
    except ValueError as e:
        print(f"Invalid time configuration: {e}", file=sys.stderr)
        return 1
    if t1 <= t0:
        print("End time must be after start time.", file=sys.stderr)
        return 1
    ext_lower = Path(SOURCE_FILENAME).suffix.lower()
    audio_only = ext_lower in AUDIO_EXTENSIONS
    media_in = DATA_DIR / SOURCE_FILENAME
    srt_in = EXPORTS_DIR / f"{Path(SOURCE_FILENAME).stem}.srt"
    if not media_in.is_file():
        print(f"File not found: {media_in}", file=sys.stderr)
        return 1
    if not srt_in.is_file():
        print(f"SRT not found: {srt_in}", file=sys.stderr)
        return 1
    try:
        out_dir = _prepare_output_subdir_for_source(SOURCE_FILENAME)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    media_in, srt_in, media_out, srt_out = _resolve_paths(
        SOURCE_FILENAME,
        OUTPUT_FILE_SUFFIX,
        output_dir=out_dir,
        title_stem=None,
        title_only=False,
    )
    kind = "audio" if audio_only else "video"
    print(
        f"Trimming {kind} (legacy): {media_in} -> {media_out} [{t0:.3f}s, {t1:.3f}s)",
    )
    try:
        trim_media(media_in, media_out, t0, t1, audio_only=audio_only)
    except Exception as e:
        print(f"Media trim failed: {e}", file=sys.stderr)
        return 1
    try:
        n = trim_and_resync_srt(srt_in, srt_out, t0, t1, t0)
    except Exception as e:
        print(f"SRT processing failed: {e}", file=sys.stderr)
        return 1
    print(f"Wrote SRT: {srt_out} ({n} blocks)")
    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim media and SRT: with parts.json, runs every part (or one via --part); output name = title only.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to parts.json (default: parts/parts.json or parts.json next to this script).",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=None,
        help="If set, only this part id from JSON; otherwise all parts are processed.",
    )
    args = parser.parse_args()

    config_path = _config_path_from_args(args)
    if not config_path.is_file():
        if args.config is not None:
            print(f"Config file not found: {config_path}", file=sys.stderr)
            return 1
        return _run_legacy()
    return _run_from_parts_config(config_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
