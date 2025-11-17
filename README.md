# VLN-SUSA (AAAI 2026)
Official code for "Agent Journey Beyond RGB: Hierarchical Semantic-Spatial Representation Enrichment for Vision-and-Language Navigation"


> Navigating unseen environments based on natural language instructions remains difficult for egocentric agents in Vision-and-Language Navigation (VLN).
Intuitively, humans inherently ground concrete semantic knowledge within spatial layouts during indoor navigation.
Although previous studies have introduced diverse environmental representations to enhance reasoning, other co-occurrence modalities are often naively concatenated with RGB features, resulting in suboptimal utilization of each modality's distinct contribution.
Inspired by this, we propose a hierarchical Semantic Understanding and Spatial Awareness (SUSA) architecture to enable agents to perceive and ground environments at diverse scales.
Specifically, the Textual Semantic Understanding (TSU) module supports local action prediction by generating view-level descriptions, thereby capturing fine-grained environmental semantics and narrowing the modality gap between instructions and environments. 
Complementarily, the Depth-enhanced Spatial Perception (DSP) module incrementally constructs a trajectory-level depth exploration map, providing the agent with a coarse-grained comprehension of the global spatial layout.
Experiments demonstrate that SUSA’s hierarchical representation enrichment not only boosts the navigation performance of the baseline on discrete VLN benchmarks (REVERIE, R2R, and SOON), but also exhibits superior generalization to the continuous R2R-CE benchmark.

> ![image](https://github.com/user-attachments/assets/57ce4f1c-2cf1-4be2-94ab-0efe69adfdb0)



## 1. Requirements

Environment setup in environment.txt

1. Install Matterport3D simulator for `R2R`, `REVERIE` and `SOON`: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name SUSA python=3.8.5
conda activate SUSA
pip install -r requirements.txt
```

## 2. Data Download

1. Download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0), including processed annotations, features and pretrained models of REVERIE, SOON, R2R and R4R datasets. Put the data in `datasets' directory.

2. Download pretrained lxmert
```
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```
3. Download Clip-based rgb feature and Depth feature (glbson and imagenet) from (链接: [https://pan.baidu.com/s/1lKend8xnwuy1uxn-aIDBtw?pwd=n8gv](https://pan.baidu.com/s/1lKend8xnwuy1uxn-aIDBtw?pwd=n8gv) 提取码: n8gv)
```
python get_depth.py
```
The ground truth depth image (undistorted_depth_images) is obtained from the [Matterport Simulator](https://github.com/peteanderson80/Matterport3DSimulator), and features are extracted through a. The code for each view is referenced from [HAMT](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) and [here](https://github.com/zehao-wang/LAD/tree/main/preprocess)

4. Download Caption R2R view (BLIP-2) from [caption.json](https://www.dropbox.com/scl/fo/v75w2i9rzogru1hkx47jy/ALgwgn3eXd0RzOnV0SdfKYo?dl=0&e=1&preview=captions.json&rlkey=l9ci5ez00j2c78x7kf4inpssz)
5. Download checkpoints on three VLN tasks：SUSA 链接: [https://pan.baidu.com/s/1i5eldIr5kiodl7UUAhytaQ?pwd=yabc](https://pan.baidu.com/s/1i5eldIr5kiodl7UUAhytaQ?pwd=yabc) 提取码: yabc

## 3. Pretraining

The pretrained ckpts for REVERIE, R2R, SOON  is at [here](https://pan.baidu.com/s/1lKend8xnwuy1uxn-aIDBtw?pwd=n8gv). You can also pretrain the model by yourself, just change the pre training RGB of Duet from vit based to clip based. 
Combine behavior cloning and auxiliary proxy tasks in pretraining:
```pretrain
cd pretrain_src
bash run_r2r.sh # (run_reverie.sh, run_soon.sh)
```

## 4. Fine-tuning & Evaluation for `R2R`, `REVERIE` and `SOON`

Before training, hyperparameters can be modified in the bash files.

Use pseudo interative demonstrator to fine-tune the model:
```finetune
cd map_nav_src
bash scripts/run_r2r.sh # (run_reverie.sh, run_soon.sh)
```

**Note**: The experiment found that replacing line 585 (reported in the paper) in the agent-obj.py file with line 584 resulted in better val-seen results.
```
# 584 line (better)
REVERIE: Env name: val_unseen, action_steps: 8.33, steps: 12.10, lengths: 23.45, sr: 54.79, oracle_sr: 60.47, spl: 39.46, rgs: 37.26, rgspl: 27.08
R2R: Env name: val_unseen, action_steps: 7.63, steps: 7.63, lengths: 14.59, nav_error: 3.08, oracle_error: 1.64, sr: 73.86, oracle_sr: 82.08, spl: 62.84, nDTW: 68.25, SDTW: 60.25, CLS: 67.35

# 585 line (paper reported)
REVERIE: Env name: val_unseen, action_steps: 8.11, steps: 11.61, lengths: 22.59, sr: 51.66, oracle_sr: 55.95, spl: 38.78, rgs: 34.90, rgspl: 26.44
R2R: Env name: val_unseen, action_steps: 7.23, steps: 7.23, lengths: 13.77, nav_error: 3.03, oracle_error: 1.59, sr: 73.65, oracle_sr: 81.86, spl: 63.73, nDTW: 69.85, SDTW: 61.31, CLS: 68.90

```
The main training logs and weights at [here](https://pan.baidu.com/s/1ESI3KEG0jcs5k5UxvW0TMg?pwd=6u5c).


## 5. Test
Our report results on the test set are from the official website of Eval.ai. 
R2R: [https://eval.ai/web/challenges/challenge-page/97/submission](https://eval.ai/web/challenges/challenge-page/97/submission)
REVERIE: [https://eval.ai/web/challenges/challenge-page/606/overview](https://eval.ai/web/challenges/challenge-page/606/overview)
SOON: [https://eval.ai/web/challenges/challenge-page/1275/overview](https://eval.ai/web/challenges/challenge-page/1275/overview)
![image](https://github.com/user-attachments/assets/63826c3f-2b85-4989-9a65-7ed8cec092f8)



### 6. Additional Resources 

1) Panoramic trajectory visualization is provided by [Speaker-Follower](https://gist.github.com/ronghanghu/d250f3a997135c667b114674fc12edae).
2) Top-down maps for Matterport3D are available in [NRNS](https://github.com/meera1hahn/NRNS).
3) Instructions for extracting image features from Matterport3D scenes can be found in [VLN-HAMT](https://github.com/cshizhe/VLN-HAMT).




## 7. Citation

```bibtex
@article{zhang2024agent,
  title={Agent Journey Beyond RGB: Hierarchical Semantic-Spatial Representation Enrichment for Vision-and-Language Navigation},
  author={Zhang, Xuesong and Xu, Yunbo and Li, Jia and Liu Ruonan and Hu, Zhenzhen},
  journal={arXiv preprint arXiv:2412.06465},
  year={2025}
}
  ```

## Acknowledgments
Our code is based on [VLN-DUET](https://github.com/cshizhe/VLN-DUET) , partially referenced [Paonogen](https://github.com/jialuli-luka/PanoGen?tab=readme-ov-file) for caption and from [HAMT](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) for extract view features. Thanks for their great works!



