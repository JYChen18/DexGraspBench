# DexGraspBench

A standard and unified simulation benchmark in [MuJoCo](https://github.com/google-deepmind/mujoco/) for dexterous grasping, aimed at **enabling a fair comparison across different grasp synthesis methods**, proposed in *BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization [ICRA 2025]*.

[Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490)

## Introduction
Reproducing the quantitative results of different grasp synthesis methods in the paper.

## Getting Started

### Installation
1. Clone the third-party library [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
```
git submodule update --init --recursive --progress
```
2. Install the python environment via [Anaconda](https://www.anaconda.com/). 
```
conda create -n DGBench python=3.10 
conda activate DGBench
pip install numpy==1.26.4
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install mujoco
pip install trimesh
pip install hydra-core
pip install transforms3d
pip install matplotlib
pip install scikit-learn
pip install usd-core
pip install imageio
pip install 'qpsolvers[clarabel]'
```

### Running

1. DexGraspNet baseline.
generate grasp poses using [DexGraspNet](https://github.com/GFRL/DexGraspNet)

```
bash scripts/test_DexGraspNet_allegro.sh
```

2. SpringGrasp baseline.
generate grasp poses using [SpringGrasp](https://github.com/GFRL/SpringGrasp)

```
bash scripts/test_SpringGrasp_allegro.sh
```

## Changelog
The `main` branch serves as our standard benchmark, with some adjustments to the settings compared to the [BODex](https://arxiv.org/abs/2412.16490) paper, aimed at improving the practicality. Key changes include increasing the object mass from 30g to 100g, raising the hand's kp from 1 to 5, and supporting more diverse object assets.

The original benchmark version is available in the `original` branch. This branch also includes code to test other grasp synthesis baselines, such as [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [FRoGGeR](https://github.com/alberthli/frogger), [SpringGrasp](https://github.com/Stanford-TML/SpringGrasp_release).


### Future Plan
- Incorporate visual/tactile feedback to support **close-loop** evaluation.
- Add support for other physics simulators, such as [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and the [IPC-based simulator](https://dl.acm.org/doi/10.1145/3528223.3530064).

The detailed updating timeline is unsure. 


## Citation
If you find this project useful, please consider citing:
```
@article{chen2024bodex,
  title={BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author={Chen, Jiayi and Ke, Yubin and Wang, He},
  journal={arXiv preprint arXiv:2412.16490},
  year={2024}
}
```
