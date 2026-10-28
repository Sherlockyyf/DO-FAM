"""Gradio demo of DO-FAM.

    python app.py --doll_dir pretrained_models/DOLL            # http://127.0.0.1:7860
    python app.py --doll_dir pretrained_models/DOLL --share    # public link

Upload an FFHQ-aligned face (or tick "align face", requires dlib), move the sliders and press Run.
"""
import argparse
import os
import tempfile

import gradio as gr
import numpy as np
import torch

from dofam.data import encoder_transforms
from dofam.editor import DOLLEditor
from dofam.hyperstyle import load_hyperstyle, run_inversion
from dofam.utils import make_row

ATTRIBUTES = [  # (attribute, slider label, min, max, default, step)
    ('Smiling', 'Smiling', -2.0, 2.0, 1.5, 0.1),
    ('Eyeglasses', 'Eyeglasses', -5.0, 5.0, 0.0, 0.5),
    ('Young', 'Age (Young)', -10.0, 10.0, 0.0, 0.5),
    ('Male', 'Gender (Male)', -5.0, 5.0, 0.0, 0.5),
]


def build_demo(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_hyperstyle(args.hyperstyle, args.w_encoder, device=device)
    editor = DOLLEditor(args.doll_dir, device=device, edit_layers=args.edit_layers, max_loaded=args.max_loaded)
    transform = encoder_transforms(train=False)
    predictor = None

    def align(image):
        nonlocal predictor
        import dlib

        from dofam.alignment import align_face
        if predictor is None:
            predictor = dlib.shape_predictor(args.shape_predictor)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'input.png')
            image.save(path)
            return align_face(path, predictor)

    @torch.no_grad()
    def process(image, do_align, *alphas):
        if image is None:
            raise gr.Error('Please upload a face image.')
        image = image.convert('RGB')
        if do_align:
            image = align(image)
        x = transform(image).unsqueeze(0).to(device)
        recon, w, deltas = run_inversion(x, net, n_iters=args.n_iters)
        deltas = [None if d is None else d[0] for d in deltas]
        edits = [(attr, a) for (attr, *_), a in zip(ATTRIBUTES, alphas) if a != 0]
        w_edit = editor.sequential_edit(w, edits) if edits else w
        edited, _ = net.decoder([w_edit], input_is_latent=True, randomize_noise=False, weights_deltas=deltas)
        return np.asarray(make_row([recon[0], edited[0]], size=512))

    with gr.Blocks(title='DO-FAM') as demo:
        gr.Markdown('# DO-FAM: Disentangled Non-Linear Latent Navigation for Facial Attribute Manipulation\n'
                    'Left: HyperStyle reconstruction. Right: edited result.')
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type='pil', label='Input face (FFHQ-aligned)')
                do_align = gr.Checkbox(label='align face (dlib)', value=False)
                sliders = [gr.Slider(minimum=lo, maximum=hi, value=v, step=s, label=label)
                           for _, label, lo, hi, v, s in ATTRIBUTES]
                run = gr.Button('Run', variant='primary')
            with gr.Column():
                out = gr.Image(type='numpy', label='Reconstruction | Edited')
        run.click(fn=process, inputs=[inp, do_align] + sliders, outputs=out)
        if args.examples and os.path.isdir(args.examples):
            files = sorted(os.path.join(args.examples, f) for f in os.listdir(args.examples)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            gr.Examples(examples=[[f] for f in files], inputs=[inp])
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperstyle', default='pretrained_models/hyperstyle_ffhq.pt')
    parser.add_argument('--w_encoder', default='pretrained_models/faces_w_encoder.pt')
    parser.add_argument('--doll_dir', default='pretrained_models/DOLL')
    parser.add_argument('--shape_predictor', default='pretrained_models/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--examples', default='examples')
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--edit_layers', type=int, default=11)
    parser.add_argument('--max_loaded', type=int, default=2)
    parser.add_argument('--server_name', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    build_demo(args).queue().launch(server_name=args.server_name, server_port=args.port, share=args.share)
