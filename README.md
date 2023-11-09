
# Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics

This is the code implementation of the paper [Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics](https://arxiv.org/abs/2305.14286).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Additionally, CC3D version 4.3.0 is use to generate the cellular dynamics data. Please see [this link](https://compucell3d.org/SrcBin#A430) for instructions.

## Training

To train the EPNS or PNS models, run this command, where <config> is a config file in the configs directory:

```train
python train_model.py <config>
```

For example, for training EPNS on celestial dynamics, run:

```train
python train_model.py n_body_dynamics_EPNS
```

In addition, you can easily overwrite parameters in the configuration files from the command line. For example, if you want to train EPNS with one-step training on the cellular dynamics data, simply run:

```train
python train_model.py cell_dynamics_EPNS --training_strategy=one-step
```

Please consult the config files for the relevant parameters.

## Evaluation

To evaluate EPNS on the celestial dynamics data, run:

```eval
python evaluation/run_experiments_n_body.py
```

For cellular dynamics, run:

```eval
python evaluation/run_experiments_cell.py
```

## Generating data

Code for generating the datasets will be added to this repository soon.

---

If you found our work useful, please consider citing:

```
@inproceedings{
minartz2023,
title={Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics},
author={Koen Minartz and Yoeri Poels and Simon Martinus Koop and Vlado Menkovski},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=CCVsGbhFdj}
}
```



