# DexGraspBench

A simulation benchmark in [MuJoCo](https://github.com/google-deepmind/mujoco/) for dexterous grasping, proposed in *BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization [ICRA 2025]*.

[Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490)

## Highlight

1. **Reproducibility and high-quality physics simulation**, powered by [MuJoCo](https://github.com/google-deepmind/mujoco/).
2. **Comprehensive metrics**, including simulation success rate, analytic grasp quality metrics, penetration depth, contact quality, and data diversity, etc.
3. **Support for multiple settings**, including different hands (Allegro, Shadow, Leap), data formats (motion sequences, static poses), and scenarios (force-closure, tabletop).
5. **Benchmarking diverse methods**, including optimization-based grasp synthesis methods (e.g., [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [FRoGGeR](https://github.com/alberthli/frogger), [SpringGrasp](https://github.com/Stanford-TML/SpringGrasp_release), and [BODex](https://pku-epic.github.io/BODex/)) and data-driven baselines (e.g., CVAE, Diffusion Model, and Normalizing Flow).


## Installation
1. Clone the third-party library [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
```
git submodule update --init --recursive 
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

## Object Asset Preparation
1. Download the object assets of DexGraspNet from [TODO]().
2. Process the objects using [MeshProcess](https://github.com/JYChen18/MeshProcess).
3. Create soft link to `assets`.
```
ln -s ${YOUR_DATA_PATH} assets/DGNObj
ln -s ${YOUR_SPLIT_PATH} assets/DGNObj_splits
```

## Running
1. Synthesize the grasp data with [TODO]() and create a soft link.
```
ln -s ${YOUR_GRASPDATA_PATH} output/debug_shadow/graspdata
```

2. Evaluate the synthesized grasp.   
```
python src/main.py task=eval 
```

3. Calculate statistics after evaluation.
```
python src/main.py task=stat
```

## Visualization
We provide two methods to visualize the synthesized grasps. The first method is through [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD). 
```
python src/main.py task=vusd
```
The other method is to save OBJ files.
```
python src/main.py task=vobj
```


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


## LICENSE
This work are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png