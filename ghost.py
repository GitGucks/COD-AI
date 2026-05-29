"""ghost.py — run the trained COD-AI ghost via virtual DS4 controller.

Usage:
    python ghost.py               # live mode — sends real controller input
    python ghost.py --dry-run     # prints predicted actions, no controller output
    python ghost.py --model models/ghost_v003.pt  # use a specific checkpoint
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from frame_source import CaptureDeviceFrameSource
from model import GhostModel, IMAGENET_MEAN, IMAGENET_STD
from paths import MODELS_DIR
TARGET_FPS  = 20
FRAME_SIZE  = 224

STICK_DEADZONE   = 0.05   # stick values below this are zeroed
BUTTON_THRESHOLD = 0.65   # sigmoid output above this = button pressed

# DS4 button order matching collect.py / model output
# cross, circle, square, triangle, L1, R1, L2, R2, share, options, L3, R3, PS, touchpad
BUTTON_NAMES = [
    "cross", "circle", "square", "triangle",
    "L1", "R1", "L2_btn", "R2_btn",
    "share", "options", "L3", "R3", "PS", "touchpad",
]

# Buttons that must never fire during gameplay regardless of model output.
# share/options/PS/touchpad are meta buttons that open menus or take screenshots.
BUTTON_BLACKLIST = {"share", "options", "PS", "touchpad"}

# vgamepad DS4 button constants in the same order
_DS4_BUTTON_MAP = None  # lazy-loaded only in live mode


def _load_ds4_button_map():
    import vgamepad as vg
    return [
        vg.DS4_BUTTONS.DS4_BUTTON_CROSS,
        vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE,
        vg.DS4_BUTTONS.DS4_BUTTON_SQUARE,
        vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE,
        vg.DS4_BUTTONS.DS4_BUTTON_SHOULDER_LEFT,
        vg.DS4_BUTTONS.DS4_BUTTON_SHOULDER_RIGHT,
        vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_LEFT,
        vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_RIGHT,
        vg.DS4_BUTTONS.DS4_BUTTON_SHARE,
        vg.DS4_BUTTONS.DS4_BUTTON_OPTIONS,
        vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT,
        vg.DS4_BUTTONS.DS4_BUTTON_THUMB_RIGHT,
        None,   # PS button — DS4_SPECIAL_BUTTON, skip for now
        None,   # touchpad — skip for now
    ]


def _apply_deadzone(val: float, dz: float = STICK_DEADZONE) -> float:
    return 0.0 if abs(val) < dz else float(val)


def _preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    small = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    t     = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean  = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std   = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t     = (t - mean) / std
    return t.unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the COD-AI ghost.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print predicted actions without sending controller input.")
    parser.add_argument("--model",    default=str(MODELS_DIR / "ghost_latest.pt"),
                        help="Path to model checkpoint.")
    parser.add_argument("--device",   type=int, default=0,
                        help="Elgato capture device index.")
    parser.add_argument("--backend",  default="dshow")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("  Run train.py first to create a checkpoint.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCOD-AI — Ghost")
    print(f"  Model:   {model_path}")
    print(f"  Device:  {device}")
    print(f"  Mode:    {'DRY RUN (no controller output)' if args.dry_run else 'LIVE — virtual DS4 active'}")

    # Load model
    model = GhostModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  Model loaded.")

    # Set up virtual controller (live mode only)
    pad = None
    ds4_buttons = None
    if not args.dry_run:
        import vgamepad as vg
        pad = vg.VDS4Gamepad()
        pad.reset()
        pad.update()
        ds4_buttons = _load_ds4_button_map()
        print("  Virtual DS4 connected.")

    # Set up capture
    src = CaptureDeviceFrameSource(device_index=args.device, backend=args.backend)
    src.start()
    for _ in range(5):
        src.read()
    print("\nRunning — press Ctrl+C to stop.\n")

    stopped = False

    def _on_sigint(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _on_sigint)

    prev_buttons = [False] * 14
    interval  = 1.0 / TARGET_FPS
    next_tick = time.monotonic()

    try:
        while not stopped:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += interval

            raw = src.read()
            if raw is None:
                continue

            with torch.no_grad():
                inp  = _preprocess(raw, device)
                pred = model(inp)

            ls  = pred["left_stick"][0].cpu().numpy()
            rs  = pred["right_stick"][0].cpu().numpy()
            tr  = pred["triggers"][0].cpu().numpy()
            btn = pred["buttons"][0].cpu().numpy()

            lx = _apply_deadzone(ls[0])
            ly = _apply_deadzone(ls[1])
            rx = _apply_deadzone(rs[0])
            ry = _apply_deadzone(rs[1])
            l2 = float(tr[0])
            r2 = float(tr[1])
            pressed = [
                (b > BUTTON_THRESHOLD) and (BUTTON_NAMES[i] not in BUTTON_BLACKLIST)
                for i, b in enumerate(btn)
            ]

            if args.dry_run:
                active_btns = [BUTTON_NAMES[i] for i, p in enumerate(pressed) if p]
                print(f"\r  LS=({lx:+.2f},{ly:+.2f})  RS=({rx:+.2f},{ry:+.2f})  "
                      f"L2={l2:.2f} R2={r2:.2f}  btns={active_btns}      ", end="", flush=True)
            else:
                pad.left_joystick_float(x_value_float=lx,  y_value_float=ly)
                pad.right_joystick_float(x_value_float=rx, y_value_float=ry)
                pad.left_trigger_float(value_float=l2)
                pad.right_trigger_float(value_float=r2)

                for i, (now_p, was_p) in enumerate(zip(pressed, prev_buttons)):
                    const = ds4_buttons[i]
                    if const is None:
                        continue
                    if now_p and not was_p:
                        pad.press_button(button=const)
                    elif not now_p and was_p:
                        pad.release_button(button=const)

                pad.update()
                prev_buttons = pressed

    finally:
        src.stop()
        if pad is not None:
            pad.reset()
            pad.update()
        print("\n\nGhost stopped.")


if __name__ == "__main__":
    main()
