# CamoNAS: Neural Architecture Search for Camouflaged Object Detection

This is the official PyTorch implementation of **CamoNAS**, a neural architecture search framework designed for camouflaged object detection (COD) across four datasets (CAMO, COD10K, CHAMELEON, NC4K).

## Requirements
* Pytorch version 1.11

* Python 3.8

* tensorboardX

* torchvision

* pycocotools

* tqdm

* numpy

* pandas

* apex

## Installation
```
pip install -r requirements.txt
```
## Architecture Search
### Start Training
```
CUDA_VISIBLE_DEVICES=0 python train_camonas.py
```

### Load and Decode
```
CUDA_VISIBLE_DEVICES=0 python decode_camonas.py --dataset COD10k --resume /path/checkpoint.pth.tar
```

### Retrain
```
python train.py
```

## Pre-computed maps

## Testing Datasets Download

The four COD testing dataset results can be downloaded here:

- **CHAMELEON**  
  [Download Link](https://drive.google.com/file/d/15vKY7iHA_EwCzZGIsPIA65Ia-Kb5vyf6/view?usp=sharing)

- **CAMO**  
  [Download Link](https://drive.google.com/file/d/1MtR1hOpM1arYJcvx9tgcBHbvwinYEPmS/view?usp=sharing)

- **COD10K**  
  [Download Link](https://drive.google.com/file/d/1nlyRV5KbEjWcTk_bHGZfk6yJntsBilg2/view?usp=sharing)

- **NC4K**  
  [Download Link](https://drive.google.com/file/d/1qpxfPih8Zt9fc0ZXP1wu8mODPOb0zkmM/view?usp=sharing)
