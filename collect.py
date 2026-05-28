"""collect.py — record gameplay sessions for COD-AI training.

Run while playing with the DualSense connected via USB-C to the PC:
    python collect.py
    python collect.py --player "friend"
    python collect.py --device 1          # if Elgato is not device 0

Press Ctrl+C to stop. Data is saved to data/session_YYYYMMDD_HHMMSS.h5
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

from frame_source import CaptureDeviceFrameSource

TARGET_FPS   = 20
FRAME_SIZE   = 224          # resize to 224×224 before saving
CHUNK_FRAMES = 256          # h5py chunk size along frame axis
DATA_DIR     = Path(__file__).parent / "data"

# DualSense axis indices under SDL/pygame (USB HID mode)
AXIS_LX, AXIS_LY = 0, 1
AXIS_RX, AXIS_RY = 2, 3
AXIS_L2, AXIS_R2 = 4, 5

# Button indices (cross, circle, square, triangle, L1, R1, L2, R2,
#                 share, options, L3, R3, PS, touchpad)
BUTTON_INDICES = list(range(14))
NUM_BUTTONS    = len(BUTTON_INDICES)


def _init_pygame_joystick() -> object:
    import pygame
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("ERROR: No controller detected by pygame.")
        print("  Make sure the DualSense is connected via USB-C and your")
        print("  Python interpreter is in the HidHide allow-list.")
        sys.exit(1)

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"  Controller: {joy.get_name()}  axes={joy.get_numaxes()}  buttons={joy.get_numbuttons()}")
    return joy


def _read_controller(joy, pygame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pygame.event.pump()

    def axis(i: int) -> float:
        return joy.get_axis(i) if i < joy.get_numaxes() else 0.0

    def btn(i: int) -> float:
        return float(joy.get_button(i)) if i < joy.get_numbuttons() else 0.0

    left_stick  = np.array([axis(AXIS_LX), axis(AXIS_LY)], dtype=np.float32)
    right_stick = np.array([axis(AXIS_RX), axis(AXIS_RY)], dtype=np.float32)
    # Triggers come in as -1..1 from SDL; normalise to 0..1
    triggers    = np.array([(axis(AXIS_L2) + 1.0) / 2.0,
                             (axis(AXIS_R2) + 1.0) / 2.0], dtype=np.float32)
    buttons     = np.array([btn(i) for i in BUTTON_INDICES], dtype=np.float32)

    return left_stick, right_stick, triggers, buttons


def _create_h5(path: Path, chunk: int = CHUNK_FRAMES) -> h5py.File:
    f = h5py.File(path, "w")
    opts = dict(compression="lzf", chunks=True)
    f.create_dataset("frames",      shape=(0, FRAME_SIZE, FRAME_SIZE, 3),
                     maxshape=(None, FRAME_SIZE, FRAME_SIZE, 3),
                     dtype="uint8",    **opts)
    f.create_dataset("left_stick",  shape=(0, 2),  maxshape=(None, 2),  dtype="float32", **opts)
    f.create_dataset("right_stick", shape=(0, 2),  maxshape=(None, 2),  dtype="float32", **opts)
    f.create_dataset("triggers",    shape=(0, 2),  maxshape=(None, 2),  dtype="float32", **opts)
    f.create_dataset("buttons",     shape=(0, NUM_BUTTONS), maxshape=(None, NUM_BUTTONS),
                     dtype="float32", **opts)
    return f


def _append_batch(f: h5py.File,
                  frames: list, left: list, right: list,
                  triggers: list, buttons: list) -> None:
    n = len(frames)
    if n == 0:
        return
    for ds_name, data in (
        ("frames",      np.stack(frames)),
        ("left_stick",  np.stack(left)),
        ("right_stick", np.stack(right)),
        ("triggers",    np.stack(triggers)),
        ("buttons",     np.stack(buttons)),
    ):
        ds = f[ds_name]
        old = ds.shape[0]
        ds.resize(old + n, axis=0)
        ds[old:] = data


def main() -> None:
    parser = argparse.ArgumentParser(description="Record COD-AI training data.")
    parser.add_argument("--player",  default="me",    help="Player tag (default: me)")
    parser.add_argument("--device",  type=int, default=0, help="Elgato capture device index")
    parser.add_argument("--backend", default="dshow", help="OpenCV backend (dshow/msmf)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    session_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path     = DATA_DIR / f"session_{session_ts}.h5"
    meta_path   = DATA_DIR / f"session_{session_ts}.json"

    print(f"\nCOD-AI — Data Collection")
    print(f"  Player:  {args.player}")
    print(f"  Session: {h5_path.name}")
    print(f"  Target:  {TARGET_FPS} FPS  |  frame size: {FRAME_SIZE}×{FRAME_SIZE}")
    print(f"\nInitialising controller...")

    import pygame
    joy = _init_pygame_joystick()

    print(f"Initialising Elgato capture (device {args.device})...")
    src = CaptureDeviceFrameSource(device_index=args.device, backend=args.backend)
    src.start()

    # Warm up — discard first few frames
    for _ in range(5):
        src.read()

    print("\nRecording — press Ctrl+C to stop.\n")

    h5_file   = _create_h5(h5_path)
    buf_frames: list   = []
    buf_left:   list   = []
    buf_right:  list   = []
    buf_trig:   list   = []
    buf_btns:   list   = []
    total      = 0
    FLUSH_EVERY = 200  # frames between h5 flushes

    interval   = 1.0 / TARGET_FPS
    next_tick  = time.monotonic()

    stopped = False

    def _on_sigint(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _on_sigint)

    t0 = time.monotonic()
    try:
        while not stopped:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += interval

            raw = src.read()
            if raw is None:
                continue

            # Resize to 224×224, convert BGR→RGB
            small = cv2.resize(raw, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            ls, rs, tr, bt = _read_controller(joy, pygame)

            buf_frames.append(rgb)
            buf_left.append(ls)
            buf_right.append(rs)
            buf_trig.append(tr)
            buf_btns.append(bt)
            total += 1

            if len(buf_frames) >= FLUSH_EVERY:
                _append_batch(h5_file, buf_frames, buf_left, buf_right, buf_trig, buf_btns)
                buf_frames.clear(); buf_left.clear(); buf_right.clear()
                buf_trig.clear();   buf_btns.clear()
                h5_file.flush()

            elapsed = time.monotonic() - t0
            actual_fps = total / elapsed if elapsed > 0 else 0.0
            print(f"\r  frames={total:>6}  fps={actual_fps:4.1f}  elapsed={elapsed:5.0f}s", end="", flush=True)

    finally:
        # Flush remaining buffer
        if buf_frames:
            _append_batch(h5_file, buf_frames, buf_left, buf_right, buf_trig, buf_btns)
        h5_file.flush()
        h5_file.close()
        src.stop()
        pygame.quit()

        meta = {
            "player": args.player,
            "recorded_at": datetime.now().isoformat(),
            "frame_count": total,
            "used_in_training": False,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        elapsed = time.monotonic() - t0
        print(f"\n\nSaved {total} frames ({elapsed/60:.1f} min) to:\n  {h5_path}")
        print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
