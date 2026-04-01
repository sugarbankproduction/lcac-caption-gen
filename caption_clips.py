#!/usr/bin/env python3
"""
Auto-caption video clips using Google Gemini Flash.
Uploads each clip to Gemini Files API, generates a descriptive caption,
and saves results to a JSONL file alongside individual .txt sidecar files.

Usage:
    python caption_clips.py --clips-dir /path/to/clips --output captions.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3-flash-preview"

CAPTION_PROMPT = """You are annotating military training footage of LCAC (Landing Craft Air Cushion) hovercraft operations for an AI model training dataset.

Watch this 5-second clip and write a single, dense descriptive caption (2-4 sentences) covering:
- What is happening (action, motion, activity)
- Camera angle and framing (wide shot, close-up, tracking shot, etc.)
- Environmental conditions (water, beach, dock, time of day, weather if visible)
- Any notable details (personnel, wake patterns, ramp position, cargo, etc.)

Be specific and factual. Do not speculate. Do not reference time codes."""

UPLOAD_MIME = "video/mp4"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_clip(client: genai.Client, path: Path):
    """Upload a video file to Gemini Files API and wait for processing."""
    print(f"  Uploading {path.name}...", end="", flush=True)
    video_file = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type=UPLOAD_MIME),
    )

    # Poll until the file is ACTIVE
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"File upload failed with state: {video_file.state.name}")

    print(" done.", flush=True)
    return video_file


def generate_caption(client: genai.Client, model: str, video_file) -> str:
    """Generate a caption for an uploaded video file."""
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(parts=[
                types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type=UPLOAD_MIME)),
                types.Part(text=CAPTION_PROMPT),
            ])
        ],
    )
    return response.text.strip()


def delete_file(client: genai.Client, video_file):
    """Delete an uploaded file from Gemini Files API."""
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass  # Non-fatal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-caption video clips with Gemini Flash")
    parser.add_argument("--clips-dir", required=True, help="Directory containing .mp4 clip files")
    parser.add_argument("--output", default="captions.jsonl", help="Output JSONL file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--no-sidecar", action="store_true", help="Skip writing individual .txt sidecar files")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip clips that already have sidecar .txt files (default: on)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided. Use --api-key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        print(f"ERROR: Clips directory not found: {clips_dir}")
        sys.exit(1)

    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        print(f"No .mp4 files found in {clips_dir}")
        sys.exit(0)

    client = genai.Client(api_key=args.api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing JSONL entries to build a resume set
    done_files = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done_files.add(entry["file"])
                except (json.JSONDecodeError, KeyError):
                    pass

    to_process = [c for c in clips if c.name not in done_files]
    skipped = len(clips) - len(to_process)

    print(f"Found {len(clips)} clips, {skipped} already captioned, {len(to_process)} to process.")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}\n")

    results = []
    failed = []

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, clip in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] {clip.name}")

            video_file = None
            caption = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    video_file = upload_clip(client, clip)
                    print(f"  Generating caption...", end="", flush=True)
                    caption = generate_caption(client, args.model, video_file)
                    print(" done.", flush=True)
                    break
                except Exception as e:
                    print(f"\n  Attempt {attempt} failed: {e}")
                    if attempt < MAX_RETRIES:
                        print(f"  Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"  Giving up on {clip.name}.")
                        failed.append(clip.name)
                finally:
                    if video_file:
                        delete_file(client, video_file)
                        video_file = None

            if caption is None:
                continue

            entry = {"file": clip.name, "caption": caption}
            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            results.append(entry)

            # Write sidecar .txt
            if not args.no_sidecar:
                sidecar = clip.with_suffix(".txt")
                sidecar.write_text(caption, encoding="utf-8")

            print(f"  Caption: {caption[:120]}{'...' if len(caption) > 120 else ''}\n")

    print("=" * 60)
    print(f"Done. {len(results)} captions written to {output_path}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
