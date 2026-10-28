"""HyperStyle inversion: aligned face images -> W+ latent codes (+ generator weight offsets).

Outputs for every image ``<name>``:
    <output>/latents/<name>.pt          [1, 18, 512] latent code
    <output>/weight_deltas/<name>.pt    list of 26 offsets (~200 MB in fp32, use --half_deltas to halve)
    <output>/latents.npy                all codes stacked in file-name order (training data for DOLL)

Training DOLL only needs the latent codes (``--no_deltas``); editing needs both.

Example:
    python scripts/invert.py --hyperstyle pretrained_models/hyperstyle_ffhq.pt \
        --w_encoder pretrained_models/faces_w_encoder.pt --input examples --output inversion/examples
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dofam.data import ImageFolderDataset, encoder_transforms, list_images  # noqa: E402
from dofam.editor import save_weight_deltas  # noqa: E402
from dofam.hyperstyle import load_hyperstyle, run_inversion  # noqa: E402
from dofam.utils import get_device, make_row  # noqa: E402


def align_inputs(input_path, predictor_path, out_dir):
    import dlib

    from dofam.alignment import align_face

    predictor = dlib.shape_predictor(predictor_path)
    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(list_images(input_path), desc='Aligning'):
        try:
            name = os.path.splitext(os.path.basename(path))[0]
            align_face(path, predictor).save(os.path.join(out_dir, name + '.png'))
        except RuntimeError as err:
            print(f'Skip {path}: {err}')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hyperstyle', default='pretrained_models/hyperstyle_ffhq.pt')
    parser.add_argument('--w_encoder', default='pretrained_models/faces_w_encoder.pt')
    parser.add_argument('--input', required=True, help='image file or folder of aligned faces')
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_iters', type=int, default=5, help='HyperStyle iterative refinement steps')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no_deltas', action='store_true', help='only save latent codes')
    parser.add_argument('--half_deltas', action='store_true', help='store offsets in fp16')
    parser.add_argument('--save_reconstruction', action='store_true')
    parser.add_argument('--align', action='store_true', help='align in-the-wild faces with dlib first')
    parser.add_argument('--shape_predictor', default='pretrained_models/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    input_path = args.input
    if args.align:
        input_path = align_inputs(input_path, args.shape_predictor, os.path.join(args.output, 'aligned'))

    net = load_hyperstyle(args.hyperstyle, args.w_encoder, device=device)
    dataset = ImageFolderDataset(input_path, encoder_transforms(train=False), return_path=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    dirs = {k: os.path.join(args.output, k) for k in ('latents', 'weight_deltas', 'reconstructions')}
    os.makedirs(dirs['latents'], exist_ok=True)
    if not args.no_deltas:
        os.makedirs(dirs['weight_deltas'], exist_ok=True)
    if args.save_reconstruction:
        os.makedirs(dirs['reconstructions'], exist_ok=True)

    all_codes, names = [], []
    for images, paths in tqdm(loader, desc='Inverting'):
        images = images.to(device)
        recon, codes, deltas = run_inversion(images, net, n_iters=args.n_iters)
        for i, path in enumerate(paths):
            name = os.path.splitext(os.path.basename(path))[0]
            torch.save(codes[i:i + 1].cpu(), os.path.join(dirs['latents'], f'{name}.pt'))
            all_codes.append(codes[i].cpu().numpy())
            names.append(name)
            if not args.no_deltas:
                save_weight_deltas([None if d is None else d[i] for d in deltas],
                                   os.path.join(dirs['weight_deltas'], f'{name}.pt'), half=args.half_deltas)
            if args.save_reconstruction:
                make_row([images[i], recon[i]]).save(os.path.join(dirs['reconstructions'], f'{name}.jpg'))

    np.save(os.path.join(args.output, 'latents.npy'), np.stack(all_codes).astype(np.float32))
    with open(os.path.join(args.output, 'names.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(names))
    print(f'Inverted {len(names)} images into {args.output}')


if __name__ == '__main__':
    main()
