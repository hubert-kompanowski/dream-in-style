## ___***Dream-in-Style: Text-to-3D Generation using Stylized Score Distillation***___
<div align="center">

 <a href='https://arxiv.org/abs/2406.18581'><img src='https://img.shields.io/badge/arXiv-2406.18581-b31b1b.svg'></a>&nbsp;
 <a href='https://dream-in-style.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>&nbsp;
 <a href='#citation'><img src='https://img.shields.io/badge/BibTex-Citation-blue.svg'></a>&nbsp;


_**[Hubert Kompanowski](https://kompanowski.com/), [Binh-Son Hua](https://sonhua.github.io/)**_
<br>
International Conference on 3D Vision (3DV) 2025
<br>

<br>
<img src="assets/teaser.gif" width="100%" alt="Dream-in-Style Teaser" loop="infinite">
</div>



## Installation 
(tested on Linux)

### Prerequisites

1. **CUDA 12.1**
   
   Our code is tested on CUDA 12.1. Follow the [official installation guide](https://developer.nvidia.com/cuda-12-1-1-download-archive).

2. **Conda**
   
   If you don't have conda installed, follow the [official instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Setup Steps

1. **Clone Threestudio**

   ```bash
   git clone https://github.com/threestudio-project/threestudio.git
   ```

2. **Clone Dream-in-Style to "custom" directory**
   ```bash
   cd threestudio/custom
   git clone https://github.com/hubert-kompanowski/dream-in-style.git
   ```

3. **Create Conda environment and install dependencies**
   ```bash
   cd dream-in-style
   conda create --name "threestudio" python=3.11 -y
   conda activate threestudio
   
   # Install PyTorch with CUDA support
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu121
   
   # Install remaining dependencies
   pip install ninja
   pip install -r requirements.txt
   ```

## Usage Guide

### Running Stylized 3D Generation

We provide example configuration files that can be modified to suit your specific needs.

From the threestudio root directory, run:

```bash
python launch.py --config custom/dream-in-style/configs/toy-car-fire-style-nfsd.yaml --train --gpu 0
```

### Configuration Options

The configuration files provide extensive customization options for both 3D generation and style application:

- **Content Customization**
  - `prompt_processor.prompt`: Change this to generate different 3D objects (e.g., "a toy car", "a vase", "a chair")
  - `guidance.style_image_path`: Path to your reference style image (e.g., `"custom/dream-in-style/images/a_fire_on_a_black_background.png"`)

- **Style Control Parameters**
  - `guidance.style_ratio_scheduler_type`: Type of style strength scheduling ("constant", "linear", "sqrt", "quadratic")
  - `guidance.style_ratio_start_scale`: Initial style strength (0.0-1.0)
  - `guidance.style_ratio_end_scale`: Final style strength (0.0-1.0)

> **Note**: The method was tested on Nvidia RTX 4090 GPU with 24GB VRAM. If you encounter OOM issues, consider lowering `data.width`, `data.height`, or reducing `num_samples_per_ray` in the config file.

## Acknowledgement
This work was conducted with the financial support of the Research Ireland Centre for Research Training in Digitally Enhanced Reality (d-real) under Grant No. 18/CRT/6224. For the purpose of Open Access, the author has applied a CC BY public copyright licence to any Author Accepted Manuscript version arising from this submission.

This project is supported by Research Ireland under the Research Ireland Frontiers for the Future Programme, award number 22/FFP-P/11522.

Our implementation builds upon [threestudio](https://github.com/threestudio-project/threestudio) and the [Visual-Style-Prompting](https://github.com/naver-ai/Visual-Style-Prompting) framework from NAVER AI. We thank the authors for making their code publicly available.

## Citation
Please consider citing our paper if our code is useful:
```bib
@inproceedings{kompanowski2024dreaminstyle,
    author    = {Hubert Kompanowski and Binh-Son Hua},
    title     = {Dream-in-Style: Text-to-3D Generation using Stylized Score Distillation},
    booktitle = {International Conference on 3D Vision},
    year      = {2025},
}
```