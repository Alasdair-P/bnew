# BNEW
This is the repo containing code accopinaing the paper Training Binarized Neural Networks the Easy Way. If this code is useful please cite as:

```
@Article{paren2022b,
  author       = {Paren, Alasdair and Poudel, Rudra PK},
  title        = {Training Binarized Neural Networks the Easy Way},
  journal      = {British Machine Vision Conference},
  year         = {2022},
}
```

# Abstract

In this work we present a simple but effective method for training Binarized Neural Networks (BNNs). Specifically, models where the majority of both weights and activations are constrained to the set {-1,1}. These models offer significant improvements in memory efficiency, energy usage and inference speed over their floating point counterparts. Our approach to training BNN splits the task into two phases. In the first phase a model with binary activations and floating point weights is trained. In the second, a concave regulariser is added to encourage the weights to become binary. This work is orthogonal to improvements of BNN architectures, and offers an alternative optimisation scheme for these models. Our method doesn't require an auxiliary set of weights during training and can be easily applied to any existing architectures. Finally, we achieve a new state of the art training a BNN on the ImageNet data set.


# Code Requirements and Installation

This code should work for PyTorch >= 1.0 in python3. Please install the necessary packages using:

```
Please set the paths in the data/cli.py
Cifar10 and Cifar100 should download automatically.
```

# Reproducing the Results

Please first complete the code installation as described above. The following command lines assume that the current working directory is "/experiments" . 


To reproduce the experiements from the paper first run, note each script results in 5 runs.

# Cifar10 without Distillation

Run:
```
python reproduce/cifar10/phase_1_10.py
```
Followed buy:
```
python reproduce/cifar10/10bmd.py
python reproduce/cifar10/10ste.py
python reproduce/cifar10/10bop.py
python reproduce/cifar10/10bnew.py
```
Here:
bmd = binary mirror decent - https://arxiv.org/pdf/1910.08237.pdf
ste = straight thought estimate 
bop = binary optimiser - https://arxiv.org/abs/1906.02107
bnew = ours

# Cifar10 with Distillation

Run:
```
python reproduce/cifar10/teachers.py
python reproduce/cifar10/phase_1_dist_10.py
```
Followed buy:
```
python reproduce/cifar10/d10bmd.py
python reproduce/cifar10/d10ste.py
python reproduce/cifar10/d10bop.py
python reproduce/cifar10/d10bnew.py
```

# Cifar100 without Distillation

Run:
```
python reproduce/cifar100/phase_1_100.py
```
Followed buy:
```
python reproduce/cifar100/100bmd.py
python reproduce/cifar100/100ste.py
python reproduce/cifar100/100bop.py
python reproduce/cifar100/100bnew.py
```

# Cifar100 with Distillation

Run:
```
python reproduce/cifar100/teachers.py
python reproduce/cifar100/phase_1_dist_100.py
```
Followed buy:
```
python reproduce/cifar100/d100bmd.py
python reproduce/cifar100/d100ste.py
python reproduce/cifar100/d100bop.py
python reproduce/cifar100/d100bnew.py
```

# Imagenet Expeiments 
to reproduce our experiements, first run the below to train model with binary actiations and real valued parameters
```
cd ImageNet_ReActNet/moblienet/1_step1
./run.sh
```
then change directory and run the binazirtion phase with:
```
ImageNet_ReActNet/moblienet/3_step2_ours
./run.sh
```
finally fine tune the binary parameters with:
```
./fine_tine.sh
```
some paths with the scipts might need to be adjusted to link the data sets or where you want to store logs etc


# acknowledgements

The imagenet code is bases on that of Zechun Liu and is publicly avaible here https://github.com/liuzechun/ReActNet
