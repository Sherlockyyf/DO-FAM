<div align="center">

# DO-FAM: Disentangled Non-Linear Latent Navigation for Facial Attribute Manipulation

[**Yifan Yuan**](https://yifanyuan.com)<sup>1*</sup> · **Siteng Ma**<sup>1*</sup> · **Hongming Shan**<sup>2</sup> · **Junping Zhang**<sup>1&dagger;</sup>

<sup>1</sup>Shanghai Key Lab of Intelligent Information Processing, School of Computer Science, Fudan University
<br>
<sup>2</sup>Institute of Science and Technology for Brain-inspired Intelligence, Fudan University
<br>
<sup>*</sup>Equal contribution &emsp; <sup>&dagger;</sup>Corresponding author

**ICASSP 2023**

<a href="https://doi.org/10.1109/ICASSP49357.2023.10095959"><img src="https://img.shields.io/badge/Paper-ICASSP%202023-blue" alt="Paper"></a>
<a href="https://github.com/Sherlockyyf/VR-FAM"><img src="https://img.shields.io/badge/Previous-VR--FAM%20(ICASSP%202022)-orange" alt="VR-FAM"></a>
<a href="#-gradio-演示"><img src="https://img.shields.io/badge/Demo-Gradio-ff7c00" alt="Demo"></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A51.8-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"></a>

[English](README.md) · [简体中文](README_zh.md)

<img src="assets/DO-FAM.gif" width="80%">

</div>

**一句话总结：** 真实人脸的属性编辑既要求**重建质量高**，也要求**无关属性保持得好**。DO-FAM 借助超网络（HyperStyle）为每张图像微调 StyleGAN2 生成器，保证重建忠实；再用 **DOLL**（*Disentangled nOn-Linear Latent transformation module*，解耦非线性潜码变换模块）编辑潜码：把潜码拆成属性相关和属性无关两部分，并在潜码分类器的引导下，只对前者做非线性导航。DO-FAM 是我们 [VR-FAM](https://github.com/Sherlockyyf/VR-FAM)（ICASSP 2022）的后续工作。

## 🔍 方法

<p align="center"><img src="assets/framework.png" width="100%"></p>

DO-FAM = 编码器 + **DOLL** + 微调后的 StyleGAN2 生成器。

- **生成器微调（论文 2.2 节）。** e4e 编码器给出初始编码 $\hat{w}_{init}$；HyperStyle 超网络预测每层的权重偏移 $\Delta_\ell$，使 $G(\hat{w}_{init}; \hat{\theta})$ 忠实重建输入图像（迭代细化，损失为 $\mathcal{L}_2 + \lambda_{perp}\mathcal{L}_{perp} + \lambda_{ID}\mathcal{L}_{ID}$）。
- **DOLL（论文 2.1 节）。** *分解器* $P$ 把潜码分解为 $w_{re} = P(w)$ 和 $w_{un} = w - P(w)$；非线性*潜码变换网络*（LTNet）$T$ 只对属性相关部分做导航，$\tilde{w} = w_{un} + w_{re} + \alpha\,(T(w_{re}) - w_{re})$；预训练的*潜码分类器* $C$ 负责引导导航。DOLL 的训练目标为
  $\mathcal{L}_{DOLL} = \mathcal{L}_{re} + \lambda_{un}\mathcal{L}_{un} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{ort}\mathcal{L}_{ort}$：
  目标属性编辑正确（$\mathcal{L}_{re}$）、无关属性保持不变并按属性相关性加权（$\mathcal{L}_{un}$）、潜码改动尽量小（$\mathcal{L}_{reg}$），以及 $w_{un}$ 与 $w_{re}$、$T(w_{re})$ 相互正交（$\mathcal{L}_{ort}$）。

## 🛠️ 安装

```bash
git clone https://github.com/Sherlockyyf/DO-FAM.git
cd DO-FAM
conda env create -f environment.yml && conda activate dofam      # 或：pip install -r requirements.txt
```

StyleGAN2 的所有算子都用纯 PyTorch 实现，不需要编译 C++/CUDA 扩展。

## 📦 预训练模型与测试数据

把模型放到 `pretrained_models/` 下（DOLL 模块放到 `pretrained_models/DOLL/`）。

| 模型 | 说明 | 下载 |
|:-|:-|:-:|
| DOLL | *Smiling、Gender (Male)、Eyeglasses、Age (Young)* 四个属性的 DOLL 模块 | [Google Drive](https://drive.google.com/drive/folders/1N2B1UFjqwRNK5ZydCIa_6QEYKiH6PGYE) |
| `hyperstyle_ffhq.pt` | [HyperStyle](https://github.com/yuval-alaluf/hyperstyle) 发布的模型（超网络 + StyleGAN2 生成器） | [Google Drive](https://drive.google.com/file/d/1C3dEIIH1y8w1-zQMCyx7rDF0ndswSXh4/view) |
| `faces_w_encoder.pt` | HyperStyle 发布的 W 编码器，用于初始反演 | [Google Drive](https://drive.google.com/file/d/1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP/view) |
| 潜码分类器 | 基于 HyperStyle 潜码的属性分类器（40 个属性），**仅训练时需要** | [Google Drive](https://drive.google.com/drive/folders/1YWxoMKx6k9XCZZfCLJNSIUYOH0JNLgFF) |
| 测试数据 | 30 张 CelebA-HQ 测试图像及其潜码和权重偏移 | [Google Drive](https://drive.google.com/drive/folders/1DkDyM_kHJrKjo4C37irwdrEWnZpNfK2C) |
| `CurricularFace_Backbone.pth` | 身份指标模型（仅评测时需要） | [HyperStyle](https://github.com/yuval-alaluf/hyperstyle#auxiliary-models) |

每个 DOLL 模块是作用在 9216 维 W+ 潜码上的三层 MLP（fp32 下每个约 2 GB）；`--max_loaded` 控制同时驻留内存的模块数量。

## 🚀 快速开始

**编辑发布的测试数据**（预先算好的潜码 + HyperStyle 权重偏移）：

```bash
python scripts/edit.py --latents test_data/latent_codes --deltas test_data/weight_deltas \
    --doll_dir pretrained_models/DOLL --attr Smiling --alphas -1.5 -1 -0.5 0.5 1 1.5 --output outputs/smiling
```

**编辑自己的图像。** 人脸需要按 FFHQ 方式对齐（`scripts/invert.py --align` 会用 dlib 完成对齐），HyperStyle 反演在编辑时实时进行：

```bash
python scripts/edit.py --images path/to/aligned_faces --doll_dir pretrained_models/DOLL \
    --attr Eyeglasses --alphas 1 2 3 4 --output outputs/eyeglasses

# 连续编辑：每一列多叠加一个属性
python scripts/edit.py --images path/to/aligned_faces --doll_dir pretrained_models/DOLL \
    --edits "Smiling=1.5,Eyeglasses=2,Age=3,Gender=3" --output outputs/sequential
```

显存不够时可以把反演和编辑分开运行（`invert.py` 输出潜码和权重偏移，fp32 下每张图约 200 MB，加 `--half_deltas` 可减半）：

```bash
python scripts/invert.py --input path/to/aligned_faces --output inversion/mine --save_reconstruction
python scripts/edit.py --latents inversion/mine/latents --deltas inversion/mine/weight_deltas \
    --doll_dir pretrained_models/DOLL --attr Age --alphas -4 -2 2 4 --output outputs/age
```

<details>
<summary><b>Python API</b></summary>

```python
import torch
from PIL import Image
from dofam.data import encoder_transforms
from dofam.editor import DOLLEditor
from dofam.hyperstyle import load_hyperstyle, run_inversion
from dofam.utils import tensor2im

net = load_hyperstyle('pretrained_models/hyperstyle_ffhq.pt', 'pretrained_models/faces_w_encoder.pt', device='cuda')
editor = DOLLEditor('pretrained_models/DOLL', device='cuda', edit_layers=11)

x = encoder_transforms(train=False)(Image.open('face.png').convert('RGB')).unsqueeze(0).cuda()
recon, w, deltas = run_inversion(x, net, n_iters=5)            # w: [1, 18, 512]
deltas = [None if d is None else d[0] for d in deltas]
w_edit = editor.sequential_edit(w, [('Smiling', 1.5), ('Eyeglasses', 2.0)])
with torch.no_grad():
    image, _ = net.decoder([w_edit], input_is_latent=True, randomize_noise=False, weights_deltas=deltas)
tensor2im(image[0]).save('edited.png')
```
</details>

### 🤗 Gradio 演示

```bash
python app.py --doll_dir pretrained_models/DOLL      # http://127.0.0.1:7860，加 --share 可生成公开链接
```

## 🏋️ 训练

按照论文设置，编码器和生成器直接沿用 HyperStyle 的模型并保持冻结；先训练潜码分类器，再为每个属性各训练一个 DOLL。

```bash
# 1) CelebA-HQ 训练图像的 HyperStyle 潜码（只保存潜码）
python scripts/invert.py --input data/celeba_hq/train --output inversion/celeba_hq_train --no_deltas

# 2) 与潜码一一对应的属性标签（CelebAMask-HQ 标注）
python scripts/prepare_celebahq_labels.py --anno data/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt \
    --images_dir data/celeba_hq/train --output inversion/celeba_hq_train/labels.npy

# 3) 潜码分类器：20 个 epoch，batch 256，Adam 1e-4（也可以直接下载我们的模型）
python scripts/train_classifier.py --latents inversion/celeba_hq_train/latents.npy \
    --labels inversion/celeba_hq_train/labels.npy --exp_dir experiments/classifier --opts train_ratio=1.0

# 4) DOLL：20 个 epoch，batch 16，Adam 1e-4，lambda_un=1，lambda_reg=100，lambda_ort=1（configs/doll.yaml）
python scripts/train_doll.py --latents inversion/celeba_hq_train/latents.npy \
    --labels inversion/celeba_hq_train/labels.npy --classifier experiments/classifier/latent_classifier.pth \
    --exp_dir experiments/doll --attributes Smiling Male Eyeglasses Young
```

`ort_mode: gram`（默认）复现已发布模型所用的正交损失（对 batch 内所有样本对计算内积）；`ort_mode: elementwise` 是论文式 (5) 的逐样本形式。

## 📊 评测

```bash
python scripts/edit.py --latents ... --deltas ... --doll_dir pretrained_models/DOLL --attr Smiling --alphas 1.5 \
    --save_individual --output outputs/eval                    # 写入 outputs/eval/Smiling_1.5/
python scripts/evaluate.py --source_dir data/celeba_hq/test --result_dir outputs/eval/Smiling_1.5 \
    --metrics id msssim fid l2 lpips
```

<details open>
<summary><b>CelebA-HQ 上的定量对比（取自论文）</b></summary>

| 属性 | 方法 | ID ↑ | SSIM (×0.1) ↑ | FID ↓ | L2 (×0.01) ↓ | LPIPS (×0.01) ↓ |
|:-|:-|:-:|:-:|:-:|:-:|:-:|
| Smiling | pSp | 0.52 | 7.03 | 52.61 | 4.89 | 18.56 |
| | e4e | 0.48 | 6.83 | 54.77 | 5.60 | 21.60 |
| | VR-FAM | 0.47 | 6.91 | 55.09 | 5.27 | 21.18 |
| | HyperStyle | 0.67 | 7.74 | 47.76 | 3.44 | 12.04 |
| | HyperStyle + StyleCLIP | 0.68 | 8.01 | 46.69 | **2.82** | **10.96** |
| | **DO-FAM** | **0.70** | **8.07** | **45.21** | 2.85 | 11.00 |
| Gender | HyperStyle + StyleCLIP | 0.56 | 7.42 | 54.25 | 6.15 | 16.68 |
| | **DO-FAM** | **0.62** | **7.44** | **53.06** | **5.09** | **15.20** |
| Eyeglasses | HyperStyle + StyleCLIP | 0.55 | 6.74 | **65.75** | 7.61 | 20.66 |
| | **DO-FAM** | **0.59** | **7.38** | 66.79 | **4.25** | **16.00** |
| Age | HyperStyle + StyleCLIP | 0.47 | 7.47 | 53.31 | **4.66** | 15.08 |
| | **DO-FAM** | **0.59** | **7.49** | **49.02** | 5.32 | **14.12** |

各属性上全部基线的结果见论文表 1。
</details>

<p align="center"><img src="assets/qualitative.jpg" width="75%"><br><em>CelebA-HQ 上的定性对比。</em></p>

<p align="center"><img src="assets/ablation_losses.jpg" width="60%"><br><em>式 (9) 各损失项的消融实验（+ Male）。</em></p>

## 📁 仓库结构

```text
dofam/
├── models/
│   ├── doll.py                # DOLL：分解器 + LTNet
│   ├── latent_classifier.py   # 潜码属性分类器 C
│   └── stylegan2/             # 支持 HyperStyle 权重偏移的 StyleGAN2 生成器（纯 PyTorch 算子）
├── hyperstyle/                # 精简版 HyperStyle 推理（超网络、W 编码器、迭代细化）
├── trainers/doll_trainer.py   # DOLL 训练（式 5–9）
├── losses.py · editor.py · metrics.py · data.py · alignment.py
scripts/                       # invert · prepare_celebahq_labels · train_classifier · train_doll · edit
                               # evaluate · make_gif
configs/                       # doll.yaml · classifier.yaml
app.py                         # Gradio 演示
```

## 📝 引用

```bibtex
@inproceedings{yuan2023dofam,
  title     = {{DO-FAM}: Disentangled Non-Linear Latent Navigation for Facial Attribute Manipulation},
  author    = {Yuan, Yifan and Ma, Siteng and Shan, Hongming and Zhang, Junping},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10095959}
}

@inproceedings{yuan2022vrfam,
  title     = {{VR-FAM}: Variance-Reduced Encoder with Nonlinear Transformation for Facial Attribute Manipulation},
  author    = {Yuan, Yifan and Ma, Siteng and Zhang, Junping},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1755--1759},
  year      = {2022}
}
```

## 📄 许可证

[MIT](LICENSE)。第三方代码保留其原有许可证（见 `licenses/`）。
