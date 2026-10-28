"""Assemble a folder of frames (sorted by the number in the file name) into an animated GIF.

Example:
    python scripts/make_gif.py --frames outputs/frames --output assets/DO-FAM.gif --duration 1.0
"""
import argparse
import os
import re

from PIL import Image


def frame_key(fname):
    numbers = re.findall(r'\d+', fname)
    return (int(numbers[-1]) if numbers else 0, fname)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', required=True, help='folder with .png/.jpg frames')
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=1.0, help='seconds per frame')
    parser.add_argument('--width', type=int, default=None, help='optional resize width')
    args = parser.parse_args()

    files = sorted((f for f in os.listdir(args.frames) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), key=frame_key)
    frames = []
    for f in files:
        im = Image.open(os.path.join(args.frames, f)).convert('RGB')
        if args.width:
            im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
        frames.append(im)
    if not frames:
        raise SystemExit(f'No frames found in {args.frames}')
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=int(args.duration * 1000), loop=0)
    print(f'Saved {len(frames)} frames to {args.output}')


if __name__ == '__main__':
    main()
