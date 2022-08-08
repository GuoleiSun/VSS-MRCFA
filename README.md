# VSS-MRCFA
Official PyTorch implementation of ECCV 2022 paper: Mining Relations among Cross-Frame Affinities for Video Semantic Segmentation

## Abstract
The essence of video semantic segmentation (VSS) is how to leverage temporal information for prediction. Previous efforts are mainly devoted to developing new techniques to calculate the cross-frame affinities such as optical flow and attention. Instead, this paper contributes from a different angle by  mining relations among cross-frame affinities, upon which better temporal information aggregation could be achieved. We explore relations among affinities in two aspects: single-scale intrinsic correlations and multi-scale relations. Inspired by traditional feature processing, we propose Single-scale Affinity Refinement (SAR) and Multi-scale Affinity Aggregation (MAA). To make it feasible to execute MAA, we propose a Selective Token Masking (STM) strategy to select a subset of consistent reference tokens for different scales when calculating affinities, which also improves the efficiency of our method. At last, the cross-frame affinities strengthened by SAR and MAA are adopted for adaptively aggregating temporal information. Our experiments demonstrate that the proposed method performs favorably against state-of-the-art VSS methods.

![block images](https://github.com/GuoleiSun/VSS-MRCFA/blob/main/Figs/diagram.png)

Authors: [Guolei Sun](https://scholar.google.com/citations?hl=zh-CN&user=qd8Blw0AAAAJ), [Yun Liu](https://yun-liu.github.io/), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en), [Ajad Chhatkuli](https://scholar.google.com/citations?user=3BHMHU4AAAAJ&hl=en), [Le Zhang](https://zhangleuestc.github.io), Luc Van Gool.

## Note
This is a preliminary version for early access and I will clean it for better readability.

## Installation
Please follow the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```timm==0.3.0, CUDA11.0, pytorch==1.7.1, torchvision==0.8.2, mmcv==1.3.0, opencv-python==4.5.2```

Download this repository and install by:
```
cd VSS-MRCFA && pip install -e . --user
```

## Usage
### Data preparation
Please follow [VSPW](https://github.com/sssdddwww2/vspw_dataset_download) to download VSPW 480P dataset.
After correctly downloading, the file system is as follows:
```
vspw-480
├── video1
    ├── origin
        ├── .jpg
    └── mask
        └── .png
```
The dataset should be put in ```VSS-MRCFA/data/vspw/```. Or you can use Symlink: 
```
cd VSS-MRCFA
mkdir -p data/vspw/
ln -s /dataset_path/VSPW_480p data/vspw/
```

### Test
1. Download the trained weights from [here](https://drive.google.com/drive/folders/1GIKt21UBYjXqi0Zm_azc6SrrIcK__Lyq?usp=sharing).
2. Run the following commands:
```
# Multi-gpu testing
./tools/dist_test.sh local_configs/mrcfa/B1/mrcfa.b1.480x480.vspw2.160k.py /path/to/checkpoint_file <GPU_NUM> \
--out /path/to/save_results/res.pkl
```

### Training
Training requires 4 Nvidia GPUs, each of which has > 20G GPU memory.
```
# Multi-gpu training
./tools/dist_train.sh local_configs/mrcfa/B1/mrcfa.b1.480x480.vspw2.160k.py 4 --work-dir model_path/vspw2/work_dirs_4g_b1
```

## License
This project is only for academic use. For other purposes, please contact us.

## Acknowledgement
The code is heavily based on the following repositories:
- https://github.com/open-mmlab/mmsegmentation
- https://github.com/NVlabs/SegFormer
- https://github.com/GuoleiSun/VSS-CFFM

Thanks for their amazing works.

## Citation
```
@article{sun2022mining,
  title={Mining Relations among Cross-Frame Affinities for Video Semantic Segmentation},
  author={Sun, Guolei and Liu, Yun and Tang, Hao and Chhatkuli, Ajad and Zhang, Le and Van Gool, Luc},
  journal={arXiv preprint arXiv:2207.10436},
  year={2022}
}
```

## Contact
- Guolei Sun, sunguolei.kaust@gmail.com
- Yun Liu, yun.liu@vision.ee.ethz.ch
