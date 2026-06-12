# TACG-Net: Task Aligned Contrastive Gaussian Learning Network

This repository contains the official implementation of the paper **"Task Aligned Contrastive Gaussian Learning Network for Infrared Small Target Detection"**. 

## Introduction

Infrared small target detection is challenging due to the low signal-to-noise ratio and small target size. This work proposes a novel network architecture, **TACG-Net**, that integrates:

- **Task-Aligned Learning**: Aligns task objectives with global soft labels for improved classification
- **Contrastive Learning**: Enhances feature discrimination through contrastive loss


## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.7.0
- CUDA >= 10.2 (for GPU acceleration)

### Install Dependencies

```bash
git clone https://github.com/yourusername/TACG-Net.git
cd TACG-Net
pip install -r requirements.txt
```

## Dataset Preparation

The code supports the following infrared small target detection datasets:
- [IRSTD-1k](https://github.com/RuiZhang97/ISNet/tree/master)
- [NUAA-SIRST](https://github.com/YimianDai/sirst)
- [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
This dataset is the bounding box annotation version of the existing infrared small target public dataset. Download link:[Google Drive](https://drive.google.com/file/d/1goc6D3647xrcDChOvaCycG2op4nfMZpp/view?pli=1)
### Dataset Structure

```
datasets/
├── IRSTD-1k/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── NUAA-sirst/
│   └── ...
└── NUDT-SIRST/
    └── ...
```

### Configuration Files

Dataset configuration files are provided in the root directory:
- `IRSTD-1k.yaml`
- `NUAA-sirst.yaml`
- `NUDT-SIRST.yaml`

Update the paths in these files to point to your dataset location.

## Training

### Training Command

```bash
python train.py \
    --weights weights/tacg-net.pt \
    --cfg cfg/training/tacg-net.yaml \
    --data IRSTD-1k.yaml \
    --hyp hyp-TACG.yaml \
    --epochs 200 \
    --batch-size 12 \
    --name TACG-Net_IRSTD
```


## Testing

### Test Command

```bash
python test.py \
    --weights irstd_best.pt \
    --data IRSTD-1k.yaml 
```

### Evaluation Metrics

The model is evaluated using:
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

## Detection

### Run Detection on Images

```bash
python detect.py \
    --weights runs/train/TACG-Net_IRSTD/weights/best.pt \
    --source path/to/images \
```






## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please contact [your email address].
