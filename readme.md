This repository contains the official implementation for our paper: [Sequential Stochastic Combinatorial Optimization Using Hierarchal Reinforcement Learning](https://arxiv.org/abs/2502.05537).

In this work, we propose a novel framework — Wake-Sleep Option (WS-Option) — to solve Sequential Stochastic Combinatorial Optimization (SSCO) problems.
Our method leverages hierarchical reinforcement learning to efficiently explore the solution space and handle the combinatorial nature of sequential decisions.

## Getting Started
To reproduce our results, navigate to the specific environment folder (IM or RPP) depending on the task:
```
cd IM  # or cd RPP
```
Then simply run:
```bash
python Main.py
```
Note: Each task environment may have slight differences in configuration or setup.

## Supply-chain recovery (DSR) and LexLaw
`DSR/` adds a third problem, dependency-constrained supply-chain recovery with a hidden compatibility
relation (variants C, Q, A), on which WS-option is trained the same way (`cd DSR; python Main.py --variant C`).
`lexlaw/` implements the text-grounded world-model method described in `method.md` on the same
problem; see `lexlaw/README.md` for its stages, gates and results.

## Citation
If you find this repository helpful in your research, please consider citing our paper:
```
@article{feng2025sequential,
  title={Sequential Stochastic Combinatorial Optimization Using Hierarchical Reinforcement Learning},
  author={Feng, Xinsong and Yu, Zihan and Xiong, Yanhai and Chen, Haipeng},
  journal={arXiv preprint arXiv:2502.05537},
  year={2025}
}
```