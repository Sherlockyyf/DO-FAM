"""Train one DOLL module per attribute (the latent classifier must be pre-trained first).

Example:
    python scripts/train_doll.py --latents inversion/celeba_hq_train/latents.npy \
        --labels data/celeba_hq/train_labels.npy --classifier pretrained_models/latent_classifier.pth \
        --exp_dir experiments/doll --attributes Smiling Male Eyeglasses Young
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dofam.data import LatentAttributeDataset  # noqa: E402
from dofam.editor import load_weight_deltas  # noqa: E402
from dofam.models import load_classifier, load_generator  # noqa: E402
from dofam.trainers import DOLLTrainer  # noqa: E402
from dofam.utils import add_config_args, get_device, load_config, load_latent, seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, 'configs/doll.yaml')
    parser.add_argument('--latents', required=True, help='stacked latents (.npy/.pt) or folder of per-image codes')
    parser.add_argument('--labels', required=True, help='[N, 40] attribute labels aligned with the latents')
    parser.add_argument('--classifier', required=True)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--attributes', nargs='*', default=None, help='overrides the list in the config')
    parser.add_argument('--preview_generator', default=None, help='HyperStyle checkpoint (decoder) for previews')
    parser.add_argument('--preview_latent', default=None, help='latent code (.pt) of a preview image')
    parser.add_argument('--preview_deltas', default=None, help='weight offsets of the preview image')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=402)
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    seed_everything(args.seed)
    device = get_device(args.device)

    dataset = LatentAttributeDataset(args.latents, args.labels, 'train', cfg['train_ratio'])
    print(f'{len(dataset)} training codes')
    classifier = load_classifier(args.classifier).to(device)

    preview = None
    if args.preview_generator and args.preview_latent:
        generator = load_generator(args.preview_generator).to(device)
        deltas = load_weight_deltas(args.preview_deltas, device) if args.preview_deltas else None
        preview = (generator, load_latent(args.preview_latent, device), deltas)

    for attr in args.attributes or cfg['attributes']:
        trainer = DOLLTrainer(cfg, attr, classifier, dataset.labels.numpy(), args.exp_dir, device)
        trainer.train(dataset, preview=preview)


if __name__ == '__main__':
    main()
