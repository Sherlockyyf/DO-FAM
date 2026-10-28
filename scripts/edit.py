"""Facial attribute editing with DO-FAM.

From images (HyperStyle inversion is run on the fly):
    python scripts/edit.py --images examples --doll_dir pretrained_models/DOLL \
        --attr Smiling --alphas -1.5 -1 -0.5 0.5 1 1.5 --output outputs/smiling

From pre-computed inversions (``scripts/invert.py`` or the released test data):
    python scripts/edit.py --latents test_data/latent_codes --deltas test_data/weight_deltas \
        --doll_dir pretrained_models/DOLL --attr Eyeglasses --alphas 1 2 3 4 --output outputs/eyeglasses

Sequential editing (each column adds one attribute):
    python scripts/edit.py --images examples --doll_dir pretrained_models/DOLL \
        --edits "Smiling=1.5,Eyeglasses=2,Young=-3,Male=3" --output outputs/sequential

Attribute names follow CelebA (Smiling, Male, Eyeglasses, Young, ...); "Gender" and "Age" are
accepted as aliases of Male and Young. ``--save_individual`` additionally writes every edited
image to ``<output>/<attr>_<alpha>/<name>.jpg`` for ``scripts/evaluate.py``.
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dofam.data import ImageFolderDataset, encoder_transforms  # noqa: E402
from dofam.editor import DOLLEditor, load_weight_deltas, pair_latents_and_deltas, parse_edits  # noqa: E402
from dofam.models import load_generator  # noqa: E402
from dofam.utils import get_device, load_latent, make_row, tensor2im  # noqa: E402


def iterate_inputs(args, device):
    """Yields ``(name, input_image_or_None, w, weight_deltas)``."""
    if args.latents:
        for name, latent_path, delta_path in pair_latents_and_deltas(args.latents, args.deltas):
            deltas = load_weight_deltas(delta_path, device) if delta_path else None
            yield name, None, load_latent(latent_path, device), deltas
        return
    from dofam.hyperstyle import load_hyperstyle, run_inversion
    net = load_hyperstyle(args.hyperstyle, args.w_encoder, device=device)
    dataset = ImageFolderDataset(args.images, encoder_transforms(train=False), return_path=True)
    for image, path in dataset:
        image = image.unsqueeze(0).to(device)
        _, codes, deltas = run_inversion(image, net, n_iters=args.n_iters)
        deltas = [None if d is None else d[0] for d in deltas]
        yield os.path.splitext(os.path.basename(path))[0], image[0], codes, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', help='aligned face image or folder')
    src.add_argument('--latents', help='folder of latent codes (.pt)')
    parser.add_argument('--deltas', default=None, help='folder of HyperStyle weight offsets (.pt / .npy)')
    parser.add_argument('--hyperstyle', default='pretrained_models/hyperstyle_ffhq.pt',
                        help='HyperStyle checkpoint (its decoder is the generator)')
    parser.add_argument('--w_encoder', default='pretrained_models/faces_w_encoder.pt')
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--doll_dir', required=True, help='folder with DOLL_<Attribute>.pth(.tar) files')
    parser.add_argument('--attr', default=None)
    parser.add_argument('--alphas', type=float, nargs='+', default=[-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
    parser.add_argument('--edits', default=None, help='sequential edits, e.g. "Smiling=1.5,Eyeglasses=2"')
    parser.add_argument('--edit_layers', type=int, default=11, help='only the first N style layers are edited')
    parser.add_argument('--max_loaded', type=int, default=2, help='DOLL modules kept in memory (~2 GB each)')
    parser.add_argument('--no_origin', action='store_true', help='do not put the reconstruction in the grid')
    parser.add_argument('--save_individual', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if (args.attr is None) == (args.edits is None):
        parser.error('specify exactly one of --attr or --edits')

    device = get_device(args.device)
    generator = load_generator(args.hyperstyle).to(device)
    editor = DOLLEditor(args.doll_dir, device=device, edit_layers=args.edit_layers, max_loaded=args.max_loaded)
    os.makedirs(args.output, exist_ok=True)

    @torch.no_grad()
    def synth(w, deltas):
        return generator([w], input_is_latent=True, randomize_noise=False, weights_deltas=deltas)[0][0]

    for name, image, w, deltas in tqdm(iterate_inputs(args, device), desc='Editing'):
        row = [] if image is None else [image]
        if not args.no_origin:
            row.append(synth(w, deltas))
        if args.attr:
            for alpha in args.alphas:
                edited = synth(editor.edit(w, args.attr, alpha), deltas)
                row.append(edited)
                if args.save_individual:
                    out_dir = os.path.join(args.output, f'{args.attr}_{alpha:g}')
                    os.makedirs(out_dir, exist_ok=True)
                    tensor2im(edited).save(os.path.join(out_dir, f'{name}.jpg'))
        else:
            edits = parse_edits(args.edits)
            for k in range(1, len(edits) + 1):
                row.append(synth(editor.sequential_edit(w, edits[:k]), deltas))
        make_row(row).save(os.path.join(args.output, f'{name}.jpg'))
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
