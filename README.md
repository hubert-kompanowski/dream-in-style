## ___***Dream-in-Style: Text-to-3D Generation using Stylized Score Distillation***___
<div align="center">

 <a href='https://arxiv.org/abs/2406.18581'><img src='https://img.shields.io/badge/arXiv-2406.18581-b31b1b.svg'></a>&nbsp;
 <a href='https://dream-in-style.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>&nbsp;
 <a href='#citation'><img src='https://img.shields.io/badge/BibTex-Citation-blue.svg'></a>&nbsp;

_**[Hubert Kompanowski](https://kompanowski.com/), [Binh-Son Hua](https://sonhua.github.io/)**_
<br>
International Conference on 3D Vision (3DV)
<br>

</div>
<br>


# Installation 
(tested for Linux)

### Install Conda (if you havn't done so already)

Install conda following the instructions from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Install CUDA (if you havn't done so already)
Code tested on CUDA 12.1 

To install CUDA 12.1 follow instructions from [here](https://developer.nvidia.com/cuda-12-1-1-download-archive).

### Clone Threestudio

```bash
git clone https://github.com/threestudio-project/threestudio.git
```
### Clone Dream-in-Style to "custom" directory
```bash
cd threestudio/custom
git clone https://github.com/hubert-kompanowski/dream-in-style.git
```
### Create Conda environment and install libraries
```bash
cd dream-in-style
conda create --name "threestudio" python=3.11 -y

conda activate threestudio

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu121

pip install ninja
pip install -r requirements.txt
```


# Quickstart
Note that the method was tested on Nvidia RTX 4090 GPU with 24GB VRAM, if you encounter OOM issues you may consider lowering the data.width and data.height or reducing num_samples_per_ray in config file.

### Running stylized 3D generation
We provide an example config file that you may modify to your needs.

Run following command from threestudio root directory.
```bash
python launch.py --config custom/dream-in-style/configs/toy-car-fire-style-nfsd.yaml --train --gpu 0
```


# Citation
Please consider citing our paper if our code is useful:
```bib
@inproceedings{kompanowski2024dreaminstyle,
    author    = {Hubert Kompanowski and Binh-Son Hua},
    title     = {Dream-in-Style: Text-to-3D Generation using Stylized Score Distillation},
    booktitle = {International Conference on 3D Vision},
    year      = {2025},
}
```