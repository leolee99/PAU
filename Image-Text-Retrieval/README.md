# PAU for Image-text Matching

The implementation on Image-text Matching for NeurIPS 2023 paper of ["Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval."](https://openreview.net/pdf?id=ECRgBK6sk1). It is built on top of the [**project**](https://github.com/leolee99/CLIP_ITM)


## Requirements
We recommend the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.7.1
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

```bash
pip install requirments.txt
```

## Dataset Preparation

### COCO Caption

We follow the same split provided by [VSE++](https://arxiv.org/pdf/1707.05612.pdf).

Dataset images can be found [here](http://www.cs.toronto.edu/~faghri/vsepp/data.tar) or [here](https://cocodataset.org/#download).
Dataset splits and annotations can be found [here](https://drive.google.com/file/d/1JmgzPjW2sagPa6TbWfrNXlO1L4G9ZbQs/view?usp=share_link).

The final data directory tree should be:
```
${DATAPATH}/
├── annotations/
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── coco_train_ids.npy
│   ├── coco_dev_ids.npy
│   ├── coco_test_ids.npy
│   ├──coco_restval_ids.npy
│   └── ...
│          
└── images/ # all images of MS-COCO
```

## Training

You can finetune the model by running:

**ViT-B/32:**
```bash
python main.py --batch_size 128 --epochs 5 --lr 1e-6 --warmup 500 --vision_model ViT-B/32 --K_prototype 8 --uct_weight 100 --var_weight 0.015 --tau 5 --precision fp16
```

If you want to train with noise, add ```--use_noise``` and select ```--noise_ratio``` in ```0.2``` or ```0.5```.


## Evaluation

You can eval the model by running:
```bash
python main.py --eval --vision_model ViT-B/32 --K_prototype 8 --resume ${MODELPATH} 
```

If you want to employ re-rank with specified coefficient, you can add ```--use_rerank``` and run:

```bash
python main.py --eval --vision_model ViT-B/32 --K_prototype 8 --resume ${MODELPATH} --use_rerank
```

If you want to employ re-rank with auto coefficient learning, you can add ```--use_rerank``` and ```--rerank_learn```, then run:

```bash
python main.py --eval --vision_model ViT-B/32 --K_prototype 8 --resume ${MODELPATH} --use_rerank --rerank_learn
```

## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{PAU,
  author    = {Hao Li and
               Jingkuan Song and
               Lianli Gao and
               Xiaosu Zhu and
               Heng Tao Shen},
  title     = {Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval},
  booktitle = {NeurIPS},
  year      = {2023}
}
```
