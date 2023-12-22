# PAU

## Introduction

The implementation of Aleatoric Uncertainty Quantification for Cross-modal Retrieval.


<!-- PAU Introdcution -->

<img src="assets/Introduction.png" width=80%>

## Requirement

We recommend the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.7.1
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

```bash
pip install requirments.txt
```

## Dataset Preparation

We follow the same split provided by [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip). You can follow the guide of its [Data preparing](https://github.com/ArrowLuo/CLIP4Clip#data-preparing).

### MSRVTT

The official data and video can be found [here](http://ms-multimedia-challenge.com/2017/dataset). 

You can download the splits and captions by:
```bash
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

### MSVD

The raw videos can be found [here](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/).

You can download the splits and captions by:
```bash
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip
```

### DiDeMo

The raw videos can be found [here](https://github.com/LisaAnne/LocalizingMoments). The splits can be found [here](https://github.com/albanie/collaborative-experts/blob/master/misc/datasets/didemo/README.md)


## Checkpoint

We provide the trained model files for evaluation. You can download the model trained on MSRVTT [here](https://drive.google.com/file/d/1ekZMz6S-Vf4RkIQ8SW_OGZnmvTSNCM2q/view?usp=drive_link), model trained on MSVD [here](https://drive.google.com/file/d/1ArI0NUCzqUGkCoVx72k3MRD4VLr0udls/view?usp=drive_link), and model trained on DiDeMo [here](https://drive.google.com/file/d/1scscUxoLeUBVT-6kDkNcRjIw25SO6P3U/view?usp=drive_link).

## Training

**MSR-VTT**

```bash
sh scripts/run_msrvtt.sh
```

**MSVD**

```bash
sh scripts/run_msvd.sh
```

**DiDeMo**

```bash
sh scripts/run_didemo.sh
```

## Evaluation

**MSR-VTT**

```bash
sh scripts/run_msrvtt_eval.sh
```

**MSVD**

```bash
sh scripts/run_msvd_eval.sh
```

**DiDeMo**

```bash
sh scripts/run_didemo_eval.sh
```

