#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
import time
from pprint import pformat

import numpy as np
import sounddevice as sd
import soundfile as sf


DEFAULT_SR = 16000
DEFAULT_CHANNELS = 1


def load_turns(turns_path: str):
    spec = importlib.util.spec_from_file_location("turns_module", turns_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {turns_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "turns"):
        raise RuntimeError(f"{turns_path} does not define a 'turns' variable")
    turns_list = getattr(module, "turns")
    if not isinstance(turns_list, list):
        raise RuntimeError("'turns' must be a list")
    return turns_list


def write_turns(turns_path: str, turns):
    # Atomic-ish write
    tmp_path = turns_path + ".tmp"
    content = "turns = " + pformat(turns, width=100, sort_dicts=False) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, turns_path)


def countdown(seconds: int = 3):
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}… ", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r", flush=True)


def record_until_enter(samplerate=DEFAULT_SR, channels=DEFAULT_CHANNELS):
    frames = []

    def callback(indata, frame_count, time_info, status):  # noqa: ARG001
        if status:
            # Print once per status occurrence
            print(f"[audio] {status}")
        frames.append(indata.copy())

    print("Recording… Press Enter to stop.")
    with sd.InputStream(samplerate=samplerate, channels=channels, dtype="int16", callback=callback):
        try:
            # Block until user presses Enter
            sys.stdin.readline()
        except KeyboardInterrupt:
            print("Interrupted — stopping recording.")

    if frames:
        data = np.concatenate(frames, axis=0)
    else:
        data = np.zeros((0, channels), dtype=np.int16)
    return data


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Record audio for each user input in turns.py")
    parser.add_argument("--turns-path", default="turns.py", help="Path to turns.py")
    parser.add_argument("--output-dir", default="turns_audio", help="Directory to save WAV files")
    parser.add_argument("--start-index", type=int, default=0, help="Index to start from")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR, help="Sample rate (Hz)")
    args = parser.parse_args()

    turns_path = args.turns_path
    out_dir = args.output_dir
    start = max(0, args.start_index)
    sr = args.sample_rate

    ensure_dir(out_dir)

    turns = load_turns(turns_path)
    num_turns = len(turns)

    if start >= num_turns:
        print(f"Start index {start} is >= number of turns {num_turns}")
        sys.exit(1)

    print(f"Loaded {num_turns} turns from {turns_path}. Output → {out_dir}")

    for i in range(start, num_turns):
        turn = turns[i]
        user_input = turn.get("input", "")

        print("-" * 80)
        print(f"Turn {i}")
        print("User input:")
        print(user_input)

        countdown(3)
        audio = record_until_enter(samplerate=sr, channels=DEFAULT_CHANNELS)

        if audio.size == 0:
            print("No audio captured — skipping save and leaving this turn unchanged.")
            continue

        filename = f"turn_{i:03d}.wav"
        filepath = os.path.join(out_dir, filename)

        # Save 16-bit PCM WAV
        sf.write(filepath, audio, sr, subtype="PCM_16")
        print(f"Saved {filepath} ({len(audio)/sr:.2f}s)")

        # Update data structure and write back to file immediately
        turn["audio_file"] = os.path.relpath(filepath)
        try:
            write_turns(turns_path, turns)
            print(f"Updated {turns_path} with audio_file for turn {i}.")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to update {turns_path}: {e}")

    print("-" * 80)
    print("All done.")


if __name__ == "__main__":
    main()
