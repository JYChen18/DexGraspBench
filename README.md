# DexGraspBench

This branch contains the early-version benchmark, which is more similar to the setting of BODex's paper but discarded. This branch also supports testing other grasp synthesis baselines, such as [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [FRoGGeR](https://github.com/alberthli/frogger), [SpringGrasp](https://github.com/Stanford-TML/SpringGrasp_release). 

The standard benchmark is in the `main` branch, which is generally recommended.


[Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490)

## Getting Started

### Installation
The Python environment is the same as the `main` branch.

### Running

1. Generating grasp poses: please use our modified [DexGraspNet](https://github.com/GFRL/DexGraspNet), [SpringGrasp](https://github.com/GFRL/SpringGrasp), and [FRoGGeR](https://github.com/GFRL/frogger)
2. Testing:
```
bash scripts/test_DexGraspNet_allegro.sh
bash scripts/test_SpringGrasp_allegro.sh
bash scripts/test_FRoGGeR_allegro.sh
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
